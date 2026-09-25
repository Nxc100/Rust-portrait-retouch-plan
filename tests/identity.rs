//! 恒等性测试（方案 7.1 节）。不依赖模型文件。

use portrait_retouch::color::curve::Curve256;
use portrait_retouch::color::lookup512::{apply_lookup512, identity_lookup512};
use portrait_retouch::color::lut3d::{apply_lut3d, Lut3D};
use portrait_retouch::face::semantic::{FaceBox, FaceKeyPoints};
use portrait_retouch::geom::P;
use portrait_retouch::pipeline::{retouch_impl, RetouchParams, SmoothMode};
use portrait_retouch::skin;
use portrait_retouch::warp::{enlarge, pinch, FaceWarp, WarpParams};
use portrait_retouch::ImgF32;

fn test_image(w: u32, h: u32) -> image::RgbImage {
    let mut img = image::RgbImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let n = (x * 7 + y * 13) % 23;
        // 左半肤色 + 噪声，右半灰蓝背景
        if x < w / 2 {
            *p = image::Rgb([(190 + n) as u8, (140 + n / 2) as u8, (115 + n / 3) as u8]);
        } else {
            *p = image::Rgb([(90 + n) as u8, (110 + n) as u8, (140 + n) as u8]);
        }
    }
    img
}

fn synthetic_face() -> FaceKeyPoints {
    let cx = 300.0;
    let eye_y = 300.0;
    let p = |x: f32, y: f32| P::new(x, y);
    let contour: Vec<P> = (0..33)
        .map(|i| {
            let t = i as f32 / 32.0;
            let ang = std::f32::consts::PI * (1.0 - t);
            p(cx + 140.0 * ang.cos(), eye_y + 20.0 + 190.0 * ang.sin())
        })
        .collect();
    let forehead: Vec<P> = (0..9)
        .map(|i| p(cx + 140.0 - 35.0 * i as f32, eye_y - 150.0))
        .collect();
    FaceKeyPoints {
        bbox: FaceBox {
            x1: 140.0,
            y1: 150.0,
            x2: 460.0,
            y2: 530.0,
            score: 1.0,
        },
        pupil_l: p(cx - 50.0, eye_y),
        pupil_r: p(cx + 50.0, eye_y),
        eye_outer_l: p(cx - 75.0, eye_y),
        eye_inner_l: p(cx - 28.0, eye_y),
        eye_inner_r: p(cx + 28.0, eye_y),
        eye_outer_r: p(cx + 75.0, eye_y),
        eye_top_l: p(cx - 50.0, eye_y - 12.0),
        eye_bot_l: p(cx - 50.0, eye_y + 12.0),
        eye_top_r: p(cx + 50.0, eye_y - 12.0),
        eye_bot_r: p(cx + 50.0, eye_y + 12.0),
        nose_bridge_top: p(cx, eye_y),
        nose_tip: p(cx, eye_y + 90.0),
        nose_bottom: p(cx, eye_y + 105.0),
        nostril_l: p(cx - 18.0, eye_y + 100.0),
        nostril_r: p(cx + 18.0, eye_y + 100.0),
        nose_wing_l: p(cx - 30.0, eye_y + 90.0),
        nose_wing_r: p(cx + 30.0, eye_y + 90.0),
        chin: contour[16],
        jaw_l: [contour[4], contour[9], contour[13]],
        jaw_r: [contour[28], contour[23], contour[19]],
        mouth_l: p(cx - 35.0, eye_y + 150.0),
        mouth_r: p(cx + 35.0, eye_y + 150.0),
        contour,
        forehead,
        yaw_deg: 0.0,
        model: "synthetic".into(),
        raw: vec![],
        score: 1.0,
        ..Default::default()
    }
}

#[test]
fn identity_lookup512_max_error_le_1_255() {
    let lut = identity_lookup512();
    let img = ImgF32::from_rgb8(&test_image(128, 96));
    let mut out = img.clone();
    apply_lookup512(&mut out, &lut, 1.0);
    assert!(out.max_abs_diff(&img) <= 1.0 / 255.0 + 1e-6);
    // 经 PNG 往返后仍为恒等
    let png = lut.to_rgb8();
    let lut2 = ImgF32::from_rgb8(&png);
    let mut out2 = img.clone();
    apply_lookup512(&mut out2, &lut2, 1.0);
    assert!(out2.max_abs_diff(&img) <= 1.0 / 255.0 + 1e-6);
}

#[test]
fn identity_cube_33_max_error_le_1_255() {
    let lut = Lut3D::identity(33);
    let img = ImgF32::from_rgb8(&test_image(128, 96));
    let mut out = img.clone();
    apply_lut3d(&mut out, &lut, 1.0);
    assert!(out.max_abs_diff(&img) <= 1.0 / 255.0 + 1e-6);
}

#[test]
fn smooth_one_without_log_hsb_matches_bilateral_on_skin() {
    let img = ImgF32::from_rgb8(&test_image(200, 120));
    let pre = skin::precompute_faithful(&img, skin::WORK_SHORT_SIDE_A);
    let out = skin::combine_bb(&img, &pre.bilateral, &pre.edge, None, 1.0, false);
    let mut skin_checked = 0;
    for i in 0..img.data.len() {
        let o = img.data[i];
        if skin::is_skin_rgb(o) && pre.edge.data[i] < 0.2 {
            assert!((out.data[i][0] - pre.bilateral.data[i][0]).abs() < 1e-6);
            skin_checked += 1;
        } else if !skin::is_skin_rgb(o) {
            assert_eq!(out.data[i], o);
        }
    }
    assert!(skin_checked > 1000);
}

#[test]
fn all_warp_strength_zero_is_bytewise_identity() {
    let img = test_image(600, 700);
    let f = synthetic_face();
    let p = RetouchParams::identity();
    let out = retouch_impl(&img, &[f], &p);
    assert_eq!(img.as_raw(), out.as_raw());
}

#[test]
fn pinch_and_enlarge_zero_return_original_coordinate() {
    let c = P::new(33.0, 44.0);
    assert_eq!(
        pinch(c, P::new(30.0, 40.0), P::new(90.0, 40.0), 50.0, 0.0),
        c
    );
    assert_eq!(enlarge(c, P::new(30.0, 40.0), 50.0, 0.0), c);
}

#[test]
fn no_face_does_not_panic_and_skips_warp() {
    let img = test_image(120, 90);
    let p = RetouchParams {
        thin_face: 1.0,
        big_eye: 1.0,
        thin_nose: 1.0,
        ..Default::default()
    };
    let out = retouch_impl(&img, &[], &p);
    assert_eq!(out.dimensions(), img.dimensions());
    // 对比：只做磨皮（无形变）的输出应完全一致（无人脸时形变被跳过）
    let p2 = RetouchParams::default();
    let out2 = retouch_impl(&img, &[], &p2);
    assert_eq!(out.as_raw(), out2.as_raw());
}

#[test]
fn background_pixels_unchanged_when_restricted_to_face() {
    // 关闭全图性的 log/HSB，只留磨皮；背景（人脸多边形外）应逐位不变
    let img = test_image(600, 700);
    let f = synthetic_face();
    let p = RetouchParams {
        smooth: 1.0,
        brightness: 1.0,
        saturation: 1.0,
        apply_log_curve: false,
        ..Default::default()
    };
    for mode in [
        SmoothMode::Faithful,
        SmoothMode::FreqSep,
        SmoothMode::GpuPixel,
    ] {
        let p = RetouchParams {
            smooth_mode: mode,
            ..p.clone()
        };
        let out = retouch_impl(&img, std::slice::from_ref(&f), &p);
        // 取远离人脸的像素块（左上角与右下角）
        for (x, y) in [(5u32, 5u32), (10, 690), (590, 5), (595, 695), (20, 40)] {
            assert_eq!(
                img.get_pixel(x, y),
                out.get_pixel(x, y),
                "mode {mode:?} changed background at ({x},{y})"
            );
        }
    }
}

#[test]
fn warp_full_strength_has_no_black_border() {
    let img = test_image(600, 700);
    let f = synthetic_face();
    let p = RetouchParams {
        thin_face: 1.0,
        big_eye: 1.0,
        thin_nose: 1.0,
        ..RetouchParams::identity()
    };
    let out = retouch_impl(&img, std::slice::from_ref(&f), &p);
    let w = FaceWarp::from_face(
        &f,
        &WarpParams {
            thin_face: 1.0,
            big_eye: 1.0,
            thin_nose: 1.0,
            ..Default::default()
        },
    );
    assert!(!w.is_identity());
    let (x0, y0, x1, y1) = (
        w.bbox.0.x as u32,
        w.bbox.0.y as u32,
        w.bbox.1.x as u32,
        w.bbox.1.y as u32,
    );
    for y in y0..y1.min(699) {
        for x in x0..x1.min(599) {
            let p = out.get_pixel(x, y);
            assert!(
                p[0] > 40 || p[1] > 40 || p[2] > 40,
                "dark pixel at ({x},{y})"
            );
        }
    }
}

#[test]
fn preview_and_export_are_consistent() {
    // 缩略图处理结果 vs 全图处理后缩小：PSNR ≥ 40 dB（方案 P5 验收）
    let img = test_image(1200, 900);
    let f = synthetic_face().scaled(2.0);
    let p = RetouchParams {
        thin_face: 0.5,
        big_eye: 0.5,
        whiten: 0.3,
        ..Default::default()
    };
    let full = retouch_impl(&img, std::slice::from_ref(&f), &p);
    let small_in = image::imageops::resize(&img, 600, 450, image::imageops::FilterType::Triangle);
    let small_out = retouch_impl(&small_in, &[f.scaled(0.5)], &p);
    let full_small =
        image::imageops::resize(&full, 600, 450, image::imageops::FilterType::Triangle);
    let mut mse = 0.0f64;
    for (a, b) in small_out.as_raw().iter().zip(full_small.as_raw()) {
        let d = *a as f64 - *b as f64;
        mse += d * d;
    }
    mse /= small_out.as_raw().len() as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
    assert!(psnr >= 40.0, "psnr {psnr:.2}");
}

#[test]
fn curve_identity_and_hsb_identity() {
    let c = Curve256::identity();
    for i in 0..256 {
        assert!((c.eval(i as f32 / 255.0) - i as f32 / 255.0).abs() < 1e-6);
    }
    let mut img = ImgF32::from_rgb8(&test_image(64, 64));
    let before = img.clone();
    portrait_retouch::color::hsb_brightness_saturation(&mut img, 1.0, 1.0);
    assert!(img.max_abs_diff(&before) < 1e-6);
}
