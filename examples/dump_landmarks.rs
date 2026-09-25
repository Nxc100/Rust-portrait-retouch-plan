//! 关键点核对工具（方案 7.4 节）：输出带索引编号的可视化图 + JSON。
//!
//! `cargo run --release --example dump_landmarks -- photo.jpg [out_prefix] [2d106|facemesh]`

use portrait_retouch::{debug, Engine, EngineConfig, LandmarkKind};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dump_landmarks <photo> [out_prefix] [2d106|facemesh]");
        std::process::exit(2);
    }
    let out_prefix = args.get(2).cloned().unwrap_or_else(|| "landmarks".into());
    let landmark: LandmarkKind = args.get(3).map(|s| s.parse().unwrap()).unwrap_or_default();
    let img = image::open(&args[1])?.to_rgb8();
    let engine = Engine::new(EngineConfig {
        landmark,
        ..Default::default()
    })?;
    let faces = engine.detect_faces(&img)?;
    println!("{} face(s) with {}", faces.len(), engine.landmark_name());
    let mut vis = img.clone();
    for f in &faces {
        debug::draw_face_debug(&mut vis, f, true);
        let v = f.sanity_check();
        println!(
            "  ed={:.1} yaw={:.1} sanity={}",
            f.eye_distance(),
            f.yaw_deg,
            if v.is_empty() {
                "ok".into()
            } else {
                format!("{v:?}")
            }
        );
    }
    vis.save(format!("{out_prefix}.png"))?;
    std::fs::write(
        format!("{out_prefix}.json"),
        serde_json::to_string_pretty(&debug::faces_json(&faces))?,
    )?;
    println!("saved {out_prefix}.png / {out_prefix}.json");
    Ok(())
}
