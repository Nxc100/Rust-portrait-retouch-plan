//! 性能基准（方案 7.5 节）。无需模型：用合成 12MP 图与合成关键点。
//! 运行：`cargo bench --bench retouch`

use criterion::{criterion_group, criterion_main, Criterion};
use portrait_retouch::color::lookup512::{apply_lookup512, identity_lookup512};
use portrait_retouch::face::semantic::{FaceBox, FaceKeyPoints};
use portrait_retouch::geom::P;
use portrait_retouch::pipeline::{retouch_with, Precomp, RetouchParams, SmoothMode};
use portrait_retouch::skin;
use portrait_retouch::ImgF32;

fn synthetic_image(w: u32, h: u32) -> image::RgbImage {
    let mut img = image::RgbImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let fx = x as f32 / w as f32;
        let fy = y as f32 / h as f32;
        let n = ((x * 7 + y * 13) % 17) as f32 / 17.0 * 0.05;
        *p = image::Rgb([
            ((0.75 + 0.1 * fx + n) * 255.0) as u8,
            ((0.55 + 0.1 * fy + n) * 255.0) as u8,
            ((0.45 + n) * 255.0) as u8,
        ]);
    }
    img
}

fn synthetic_face(scale: f32) -> FaceKeyPoints {
    let cx = 300.0 * scale;
    let eye_y = 300.0 * scale;
    let p = |x: f32, y: f32| P::new(x, y);
    let contour: Vec<P> = (0..33)
        .map(|i| {
            let t = i as f32 / 32.0;
            let ang = std::f32::consts::PI * (1.0 - t);
            p(
                cx + 140.0 * scale * ang.cos(),
                eye_y + 20.0 * scale + 190.0 * scale * ang.sin(),
            )
        })
        .collect();
    let forehead: Vec<P> = (0..9)
        .map(|i| {
            p(
                cx + (140.0 - 35.0 * i as f32) * scale,
                eye_y - 150.0 * scale,
            )
        })
        .collect();
    FaceKeyPoints {
        bbox: FaceBox {
            x1: 140.0 * scale,
            y1: 150.0 * scale,
            x2: 460.0 * scale,
            y2: 530.0 * scale,
            score: 1.0,
        },
        pupil_l: p(cx - 50.0 * scale, eye_y),
        pupil_r: p(cx + 50.0 * scale, eye_y),
        eye_outer_l: p(cx - 75.0 * scale, eye_y),
        eye_inner_l: p(cx - 28.0 * scale, eye_y),
        eye_inner_r: p(cx + 28.0 * scale, eye_y),
        eye_outer_r: p(cx + 75.0 * scale, eye_y),
        eye_top_l: p(cx - 50.0 * scale, eye_y - 12.0 * scale),
        eye_bot_l: p(cx - 50.0 * scale, eye_y + 12.0 * scale),
        eye_top_r: p(cx + 50.0 * scale, eye_y - 12.0 * scale),
        eye_bot_r: p(cx + 50.0 * scale, eye_y + 12.0 * scale),
        nose_bridge_top: p(cx, eye_y),
        nose_tip: p(cx, eye_y + 90.0 * scale),
        nose_bottom: p(cx, eye_y + 105.0 * scale),
        nostril_l: p(cx - 18.0 * scale, eye_y + 100.0 * scale),
        nostril_r: p(cx + 18.0 * scale, eye_y + 100.0 * scale),
        nose_wing_l: p(cx - 30.0 * scale, eye_y + 90.0 * scale),
        nose_wing_r: p(cx + 30.0 * scale, eye_y + 90.0 * scale),
        chin: contour[16],
        jaw_l: [contour[4], contour[9], contour[13]],
        jaw_r: [contour[28], contour[23], contour[19]],
        mouth_l: p(cx - 35.0 * scale, eye_y + 150.0 * scale),
        mouth_r: p(cx + 35.0 * scale, eye_y + 150.0 * scale),
        contour,
        forehead,
        yaw_deg: 0.0,
        model: "synthetic".into(),
        raw: vec![],
        score: 1.0,
        ..Default::default()
    }
}

fn bench_all(c: &mut Criterion) {
    // 12MP：4000×3000
    let img12 = synthetic_image(4000, 3000);
    let face12 = synthetic_face(4000.0 / 600.0);
    let pre12 = Precomp::from_rgb8(&img12);
    let mut params = RetouchParams {
        thin_face: 0.6,
        big_eye: 0.4,
        thin_nose: 0.5,
        ..Default::default()
    };
    params.style = Some(portrait_retouch::StyleFilter::Lookup512(
        std::sync::Arc::new(identity_lookup512()),
    ));

    let mut g = c.benchmark_group("12MP");
    g.sample_size(10);
    g.bench_function("precompute_faithful", |b| {
        b.iter(|| skin::precompute_faithful(&pre12.orig, skin::WORK_SHORT_SIDE_A))
    });
    g.bench_function("full_pipeline_A_warm_cache", |b| {
        let _ = pre12.faithful();
        b.iter(|| retouch_with(&pre12, std::slice::from_ref(&face12), &params))
    });
    g.bench_function("smooth_only_change_A", |b| {
        let p = RetouchParams::default();
        b.iter(|| retouch_with(&pre12, std::slice::from_ref(&face12), &p))
    });
    g.bench_function("lookup512", |b| {
        let lut = identity_lookup512();
        let mut img = pre12.orig.clone();
        b.iter(|| apply_lookup512(&mut img, &lut, 0.8))
    });
    g.bench_function("warp_only", |b| {
        let p = RetouchParams {
            thin_face: 1.0,
            big_eye: 1.0,
            thin_nose: 1.0,
            ..RetouchParams::identity()
        };
        b.iter(|| retouch_with(&pre12, std::slice::from_ref(&face12), &p))
    });
    g.bench_function("full_pipeline_B_warm_cache", |b| {
        let p = RetouchParams {
            smooth_mode: SmoothMode::FreqSep,
            ..params.clone()
        };
        let _ = pre12.freqsep(p.freqsep_radius);
        b.iter(|| retouch_with(&pre12, std::slice::from_ref(&face12), &p))
    });
    g.finish();

    // 1280 预览
    let img_prev = synthetic_image(1280, 960);
    let face_prev = synthetic_face(1280.0 / 600.0);
    let pre_prev = Precomp::from_rgb8(&img_prev);
    let _ = pre_prev.faithful();
    c.bench_function("preview_1280_full_pipeline_warm", |b| {
        b.iter(|| retouch_with(&pre_prev, std::slice::from_ref(&face_prev), &params))
    });
    let _ = ImgF32::new(1, 1);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
