//! 关键点可视化与中间结果导出（无字体依赖，内置 3×5 点阵数字）。

use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use image::{Rgb, RgbImage};

pub const GREEN: Rgb<u8> = Rgb([0, 255, 0]);
pub const RED: Rgb<u8> = Rgb([255, 40, 40]);
pub const YELLOW: Rgb<u8> = Rgb([255, 255, 0]);
pub const CYAN: Rgb<u8> = Rgb([0, 220, 255]);
pub const MAGENTA: Rgb<u8> = Rgb([255, 0, 255]);

fn put(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

pub fn draw_point(img: &mut RgbImage, p: P, r: i32, c: Rgb<u8>) {
    let (cx, cy) = (p.x.floor() as i32, p.y.floor() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                put(img, cx + dx, cy + dy, c);
            }
        }
    }
}

pub fn draw_line(img: &mut RgbImage, a: P, b: P, c: Rgb<u8>) {
    let n = a.dist(b).ceil().max(1.0) as i32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p = a.lerp(b, t);
        put(img, p.x.floor() as i32, p.y.floor() as i32, c);
    }
}

pub fn draw_polyline(img: &mut RgbImage, pts: &[P], closed: bool, c: Rgb<u8>) {
    for w in pts.windows(2) {
        draw_line(img, w[0], w[1], c);
    }
    if closed && pts.len() > 2 {
        draw_line(img, pts[pts.len() - 1], pts[0], c);
    }
}

pub fn draw_rect(img: &mut RgbImage, x1: f32, y1: f32, x2: f32, y2: f32, c: Rgb<u8>) {
    let pts = [
        P::new(x1, y1),
        P::new(x2, y1),
        P::new(x2, y2),
        P::new(x1, y2),
    ];
    draw_polyline(img, &pts, true, c);
}

/// 3×5 点阵字体（0-9，'-'，'.'）。
fn glyph(ch: char) -> [u8; 5] {
    match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        _ => [0b111, 0b111, 0b111, 0b111, 0b111],
    }
}

/// 在 (x, y) 处绘制文本（仅数字），`scale` 为像素放大倍数，带黑色描边。
pub fn draw_text(img: &mut RgbImage, x: i32, y: i32, text: &str, scale: i32, c: Rgb<u8>) {
    let mut cx = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + col * scale + sx;
                            let py = y + row as i32 * scale + sy;
                            // 描边
                            for (ox, oy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                                let (qx, qy) = (px + ox, py + oy);
                                if qx >= 0
                                    && qy >= 0
                                    && (qx as u32) < img.width()
                                    && (qy as u32) < img.height()
                                {
                                    let cur = *img.get_pixel(qx as u32, qy as u32);
                                    if cur != c {
                                        img.put_pixel(qx as u32, qy as u32, Rgb([0, 0, 0]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put(img, cx + col * scale + sx, y + row as i32 * scale + sy, c);
                        }
                    }
                }
            }
        }
        cx += 4 * scale;
    }
}

/// 绘制人脸调试信息：框（青）、轮廓 + 额头多边形（品红）、原始点（绿，可带索引）、语义点（红/黄）。
pub fn draw_face_debug(img: &mut RgbImage, f: &FaceKeyPoints, raw_indices: bool) {
    let scale = ((f.eye_distance() / 60.0).round() as i32).clamp(1, 4);
    draw_rect(img, f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2, CYAN);
    let mut poly = f.contour.clone();
    poly.extend(f.forehead.iter().copied());
    draw_polyline(img, &poly, true, MAGENTA);
    for (i, p) in f.raw.iter().enumerate() {
        draw_point(img, *p, scale.max(1), GREEN);
        if raw_indices {
            draw_text(
                img,
                p.x as i32 + 2 * scale,
                p.y as i32 - 3 * scale,
                &i.to_string(),
                scale,
                YELLOW,
            );
        }
    }
    for (_name, p) in f.semantic_points() {
        draw_point(img, p, scale + 1, RED);
    }
    draw_line(img, f.pupil_l, f.pupil_r, YELLOW);
    draw_line(img, f.chin, f.nose_bridge_top, YELLOW);
}

/// 关键点 JSON（语义点 + 原始点 + 校验结果）。
pub fn faces_json(faces: &[FaceKeyPoints]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = faces
        .iter()
        .map(|f| {
            let sem: serde_json::Map<String, serde_json::Value> = f
                .semantic_points()
                .into_iter()
                .map(|(n, p)| (n.to_string(), serde_json::json!([p.x, p.y])))
                .collect();
            serde_json::json!({
                "model": f.model,
                "score": f.score,
                "bbox": [f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2],
                "eye_distance": f.eye_distance(),
                "yaw_deg": f.yaw_deg,
                "gender": f.gender,
                "age": f.age,
                "semantic": sem,
                "contour": f.contour.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                "forehead": f.forehead.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                "raw": f.raw.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                "sanity_violations": f.sanity_check(),
            })
        })
        .collect();
    serde_json::json!({ "faces": arr })
}
