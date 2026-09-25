//! GPUImage 512×512 查找图（`GPUImageLookupFilter` 直译）。
//!
//! 布局：64 个蓝色切片按 8×8 排列，每个切片 64×64，切片内 x = 红、y = 绿；
//! `blue = b * 63`，在相邻两个切片之间按 `fract(blue)` 线性插值。

use crate::buffer::ImgF32;
use rayon::prelude::*;
use std::path::Path;

/// 计算 (r, g) 在蓝色切片 `bi` 中的采样位置（像素中心制，单位：像素，512 网格）。
#[inline]
fn tex_pos(bi: f32, r: f32, g: f32) -> (f32, f32) {
    let qy = (bi / 8.0).floor();
    let qx = bi - qy * 8.0;
    let u = qx * 0.125 + 0.5 / 512.0 + (0.125 - 1.0 / 512.0) * r;
    let v = qy * 0.125 + 0.5 / 512.0 + (0.125 - 1.0 / 512.0) * g;
    (u * 512.0, v * 512.0)
}

/// 单像素查表（不含 intensity 混合）。
#[inline]
pub fn lookup_px(lut: &ImgF32, p: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (
        p[0].clamp(0.0, 1.0),
        p[1].clamp(0.0, 1.0),
        p[2].clamp(0.0, 1.0),
    );
    let blue = b * 63.0;
    let b0 = blue.floor();
    let b1 = blue.ceil();
    let (u0, v0) = tex_pos(b0, r, g);
    let (u1, v1) = tex_pos(b1, r, g);
    let c0 = lut.sample_bilinear(u0, v0);
    let c1 = lut.sample_bilinear(u1, v1);
    let t = blue - b0;
    [
        c0[0] + (c1[0] - c0[0]) * t,
        c0[1] + (c1[1] - c0[1]) * t,
        c0[2] + (c1[2] - c0[2]) * t,
    ]
}

/// 对整幅图应用查找图并按 `intensity` 与原图混合。`lut` 必须为 512×512。
pub fn apply_lookup512(img: &mut ImgF32, lut: &ImgF32, intensity: f32) {
    assert!(lut.w == 512 && lut.h == 512, "lookup image must be 512x512");
    if intensity <= 0.0 {
        return;
    }
    img.data.par_iter_mut().for_each(|p| {
        let n = lookup_px(lut, *p);
        for c in 0..3 {
            p[c] += (n[c] - p[c]) * intensity;
        }
    });
}

/// 生成恒等查找图（设计师在其上调色后即成为风格滤镜）。
pub fn identity_lookup512() -> ImgF32 {
    let mut lut = ImgF32::new(512, 512);
    for y in 0..512 {
        for x in 0..512 {
            let r = (x % 64) as f32 / 63.0;
            let g = (y % 64) as f32 / 63.0;
            let b = ((y / 64) * 8 + x / 64) as f32 / 63.0;
            lut.data[y * 512 + x] = [r, g, b];
        }
    }
    lut
}

/// 从文件加载查找图（PNG/JPEG），校验尺寸。
pub fn load_lookup512(path: impl AsRef<Path>) -> anyhow::Result<ImgF32> {
    let img = image::open(path.as_ref())?.to_rgb8();
    anyhow::ensure!(
        img.width() == 512 && img.height() == 512,
        "lookup image {} must be 512x512, got {}x{}",
        path.as_ref().display(),
        img.width(),
        img.height()
    );
    Ok(ImgF32::from_rgb8(&img))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_lookup_is_identity() {
        let lut = identity_lookup512();
        let mut img = ImgF32::new(64, 64);
        for (i, p) in img.data.iter_mut().enumerate() {
            let x = (i % 64) as f32 / 63.0;
            let y = (i / 64) as f32 / 63.0;
            *p = [x, y, (x * 0.37 + y * 0.61) % 1.0];
        }
        let orig = img.clone();
        apply_lookup512(&mut img, &lut, 1.0);
        let d = img.max_abs_diff(&orig);
        assert!(d <= 1.0 / 255.0, "max diff {d}");
    }
}
