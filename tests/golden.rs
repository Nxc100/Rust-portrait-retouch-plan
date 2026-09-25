//! 视觉回归（方案 7.3 节）：`PORTRAIT_SAMPLES` 目录下的样张 × 参数组 → golden 图，PSNR < 45 dB 判失败。
//!
//! - 需要模型文件与 ONNX Runtime，否则跳过；
//! - `UPDATE_GOLDEN=1` 重新生成 golden；
//! - golden 存放在 `tests/golden/<样张名>/<参数组>.png`（不入库时首次运行自动生成）。

use portrait_retouch::{Engine, EngineConfig, RetouchParams, SmoothMode};
use std::path::{Path, PathBuf};

fn psnr(a: &image::RgbImage, b: &image::RgbImage) -> f64 {
    let mut mse = 0.0f64;
    for (x, y) in a.as_raw().iter().zip(b.as_raw()) {
        let d = *x as f64 - *y as f64;
        mse += d * d;
    }
    mse /= a.as_raw().len() as f64;
    if mse < 1e-12 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn param_sets() -> Vec<(&'static str, RetouchParams)> {
    vec![
        ("default_A", RetouchParams::default()),
        (
            "smooth1_A",
            RetouchParams {
                smooth: 1.0,
                ..Default::default()
            },
        ),
        (
            "freqsep",
            RetouchParams {
                smooth_mode: SmoothMode::FreqSep,
                ..Default::default()
            },
        ),
        (
            "gpupixel",
            RetouchParams {
                smooth_mode: SmoothMode::GpuPixel,
                ..Default::default()
            },
        ),
        (
            "whiten_warp",
            RetouchParams {
                whiten: 0.6,
                thin_face: 0.7,
                big_eye: 0.5,
                thin_nose: 0.5,
                ..Default::default()
            },
        ),
        (
            "warp_only",
            RetouchParams {
                thin_face: 1.0,
                big_eye: 1.0,
                thin_nose: 1.0,
                ..RetouchParams::identity()
            },
        ),
        (
            "cream_skin",
            portrait_retouch::Preset::cream_skin()
                .to_params()
                .expect("builtin preset"),
        ),
    ]
}

#[test]
fn golden_regression() {
    let samples = match std::env::var("PORTRAIT_SAMPLES") {
        Ok(s) if !s.is_empty() => PathBuf::from(s),
        _ => {
            eprintln!("PORTRAIT_SAMPLES not set; skipping golden test");
            return;
        }
    };
    let cfg = EngineConfig::default();
    if !cfg.models_dir.join("version-RFB-320.onnx").is_file() {
        eprintln!("models missing; skipping golden test");
        return;
    }
    let engine = match Engine::new(cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("engine init failed ({e}); skipping golden test");
            return;
        }
    };
    let update = std::env::var("UPDATE_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false);
    let golden_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden");
    let mut checked = 0;
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(&samples).expect("read samples dir") {
        let path = entry.unwrap().path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
            continue;
        }
        let img = image::open(&path).unwrap().to_rgb8();
        let faces = engine.detect_faces(&img).unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let dir = golden_root.join(&stem);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, p) in param_sets() {
            let out = engine.retouch(&img, &faces, &p);
            let gpath = dir.join(format!("{name}.png"));
            if update || !gpath.is_file() {
                out.save(&gpath).unwrap();
                continue;
            }
            let g = image::open(&gpath).unwrap().to_rgb8();
            let v = psnr(&out, &g);
            checked += 1;
            if v < 45.0 {
                let diff_path = dir.join(format!("{name}.diff.png"));
                let mut diff = image::RgbImage::new(out.width(), out.height());
                for (d, (a, b)) in diff.pixels_mut().zip(out.pixels().zip(g.pixels())) {
                    *d = image::Rgb([
                        (a[0] as i16 - b[0] as i16)
                            .unsigned_abs()
                            .saturating_mul(4)
                            .min(255) as u8,
                        (a[1] as i16 - b[1] as i16)
                            .unsigned_abs()
                            .saturating_mul(4)
                            .min(255) as u8,
                        (a[2] as i16 - b[2] as i16)
                            .unsigned_abs()
                            .saturating_mul(4)
                            .min(255) as u8,
                    ]);
                }
                diff.save(&diff_path).unwrap();
                failures.push(format!(
                    "{stem}/{name}: PSNR {v:.2} dB (diff: {})",
                    diff_path.display()
                ));
            }
        }
    }
    eprintln!("golden: checked {checked} images");
    assert!(
        failures.is_empty(),
        "golden regressions:\n{}",
        failures.join("\n")
    );
}
