//! 灰度形态学：方形窗口（边长 `2r+1`）的膨胀、腐蚀与闭运算，分离实现（先逐行、再转置后逐行）；
//! 方向线段的闭运算与细长暗线的深度；二值图的 4 连通域。越出图像的部分按边缘像素延拓（方形窗口即只取实际
//! 覆盖的像素）。

use crate::buffer::GrayF32;
use crate::skin::guided::transpose;
use rayon::prelude::*;

/// 逐行滑动窗口（`2r+1`）内的极值，`pick` 为 `f32::max` / `f32::min`。
fn extreme_rows(src: &GrayF32, r: usize, pick: fn(f32, f32) -> f32) -> GrayF32 {
    let w = src.w;
    let mut out = GrayF32::new(w, src.h);
    out.data
        .par_chunks_mut(w)
        .zip(src.data.par_chunks(w))
        .for_each(|(o, s)| {
            for (x, v) in o.iter_mut().enumerate() {
                let (lo, hi) = (x.saturating_sub(r), (x + r).min(w - 1));
                *v = s[lo + 1..=hi].iter().fold(s[lo], |m, &p| pick(m, p));
            }
        });
    out
}

fn separable(m: &GrayF32, r: usize, pick: fn(f32, f32) -> f32) -> GrayF32 {
    if r == 0 || m.w == 0 || m.h == 0 {
        return m.clone();
    }
    let rows = extreme_rows(m, r, pick);
    transpose(&extreme_rows(&transpose(&rows), r, pick))
}

/// 膨胀：窗口内取最大。
pub fn dilate(m: &GrayF32, r: usize) -> GrayF32 {
    separable(m, r, f32::max)
}

/// 腐蚀：窗口内取最小。
pub fn erode(m: &GrayF32, r: usize) -> GrayF32 {
    separable(m, r, f32::min)
}

/// 闭运算（先膨胀后腐蚀）：窄于窗口的暗结构被填成两侧的亮度；台阶与比窗口宽的暗区不变。
/// `close(x) − x` 即灰度黑顶帽（暗结构的深度）。
pub fn close(m: &GrayF32, r: usize) -> GrayF32 {
    erode(&dilate(m, r), r)
}

/// 线段结构元：过原点、半长 `half`、方向 `theta`（弧度）的栅格化线段上的偏移（去重，含原点）。
fn segment_offsets(half: usize, theta: f32) -> Vec<(isize, isize)> {
    let (s, c) = theta.sin_cos();
    let n = 2 * half as isize;
    let mut offs: Vec<(isize, isize)> = (-n..=n)
        .map(|k| {
            let t = k as f32 * 0.5;
            ((t * c).round() as isize, (t * s).round() as isize)
        })
        .collect();
    offs.sort_unstable();
    offs.dedup();
    offs
}

/// 沿线段结构元取极值；越出图像的采样点取最近的边缘像素（按边缘延拓）——只截掉出界的部分会让线段
/// 凭一侧的证据"跨过"边缘处的台阶与暗点。
fn along_segment(src: &GrayF32, offs: &[(isize, isize)], pick: fn(f32, f32) -> f32) -> GrayF32 {
    let (w, h) = (src.w as isize, src.h as isize);
    let mut out = GrayF32::new(src.w, src.h);
    out.data
        .par_chunks_mut(src.w)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as isize;
            for (x, v) in row.iter_mut().enumerate() {
                let x = x as isize;
                *v = offs
                    .iter()
                    .map(|&(dx, dy)| {
                        let (sx, sy) = ((x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1));
                        src.data[(sy * w + sx) as usize]
                    })
                    .fold(src.data[(y * w + x) as usize], pick);
            }
        });
    out
}

/// 细长暗线的深度：在 `directions` 个方向上各用长 `2·half+1` 的线段做闭运算，取各方向结果的最大减最小。
/// 暗线只在横跨它的方向上被填平（最大 = 两侧的亮度），沿着它的方向填不平（最小 = 它本身），差即线深；
/// 比线段短的暗点在每个方向都被填平，差为 0；台阶与比线段宽的暗区在每个方向都不变，也为 0。
/// 图像之外按边缘像素延拓：延伸到边缘的线仍是线，边缘处的台阶与暗点仍为 0；只有斜穿图像角落的台阶，
/// 在两条边都不到 `half` 的角上仍可能被量成线（延拓在角上有歧义）。
pub fn dark_line_depth(m: &GrayF32, half: usize, directions: usize) -> GrayF32 {
    let mut out = GrayF32::new(m.w, m.h);
    if half == 0 || m.w == 0 || m.h == 0 {
        return out;
    }
    let n = m.w * m.h;
    let (mut hi, mut lo) = (vec![f32::MIN; n], vec![f32::MAX; n]);
    let directions = directions.max(2);
    for i in 0..directions {
        let theta = std::f32::consts::PI * i as f32 / directions as f32;
        let offs = segment_offsets(half, theta);
        let closed = along_segment(&along_segment(m, &offs, f32::max), &offs, f32::min);
        hi.par_iter_mut()
            .zip(lo.par_iter_mut())
            .zip(&closed.data)
            .for_each(|((h, l), c)| {
                *h = h.max(*c);
                *l = l.min(*c);
            });
    }
    out.data
        .par_iter_mut()
        .zip(hi.par_iter().zip(&lo))
        .for_each(|(o, (h, l))| *o = h - l);
    out
}

pub(crate) struct Component {
    pub(crate) pixels: Vec<usize>,
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) x1: usize,
    pub(crate) y1: usize,
}

/// 4 连通域（`limit` 为单个连通域的扫描上限，超过即视为大区域并整体丢弃）。
pub(crate) fn components(hit: &[u8], w: usize, h: usize, limit: usize) -> Vec<Component> {
    let mut label = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if hit[start] == 0 || label[start] {
            continue;
        }
        stack.clear();
        stack.push(start);
        label[start] = true;
        let mut comp = Component {
            pixels: Vec::new(),
            x0: w,
            y0: h,
            x1: 0,
            y1: 0,
        };
        let mut too_big = false;
        while let Some(i) = stack.pop() {
            comp.pixels.push(i);
            let (x, y) = (i % w, i / w);
            comp.x0 = comp.x0.min(x);
            comp.x1 = comp.x1.max(x);
            comp.y0 = comp.y0.min(y);
            comp.y1 = comp.y1.max(y);
            let nb = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in nb.into_iter().flatten() {
                if hit[j] != 0 && !label[j] {
                    label[j] = true;
                    stack.push(j);
                }
            }
            if comp.pixels.len() > limit {
                too_big = true;
            }
        }
        if !too_big {
            out.push(comp);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilate_grows_a_point_into_a_square() {
        let mut m = GrayF32::new(9, 7);
        m.data[3 * 9 + 4] = 1.0;
        let d = dilate(&m, 2);
        for y in 0..7 {
            for x in 0..9 {
                let inside = (2..=6).contains(&x) && (1..=5).contains(&y);
                assert_eq!(d.get(x, y), if inside { 1.0 } else { 0.0 }, "({x},{y})");
            }
        }
        assert_eq!(dilate(&m, 0).data, m.data);
    }

    #[test]
    fn erode_removes_what_dilate_added() {
        let mut m = GrayF32::new(12, 10);
        for y in 3..7 {
            for x in 4..9 {
                m.data[y * 12 + x] = 1.0;
            }
        }
        // 方块比窗口大：膨胀再腐蚀恢复原样；腐蚀把边长 5 × 4 的方块缩成 3 × 2
        assert_eq!(erode(&dilate(&m, 1), 1).data, m.data);
        let e = erode(&m, 1);
        let kept: Vec<(usize, usize)> = (0..10)
            .flat_map(|y| (0..12).map(move |x| (x, y)))
            .filter(|&(x, y)| e.get(x, y) > 0.5)
            .collect();
        let expected: Vec<(usize, usize)> =
            (4..6).flat_map(|y| (5..8).map(move |x| (x, y))).collect();
        assert_eq!(kept, expected);
    }

    #[test]
    fn closing_fills_thin_dark_lines_but_keeps_steps() {
        // 亮度 70 的平面上：一条宽 3 的暗线（50）与右侧一个大台阶（40）
        let (w, h) = (40, 8);
        let mut m = GrayF32::from_vec(w, h, vec![70.0; w * h]);
        for y in 0..h {
            for x in 0..w {
                if (8..11).contains(&x) {
                    m.data[y * w + x] = 50.0;
                } else if x >= 24 {
                    m.data[y * w + x] = 40.0;
                }
            }
        }
        let c = close(&m, 2);
        for y in 0..h {
            for x in 0..w {
                let expected = if x >= 24 { 40.0 } else { 70.0 };
                assert_eq!(c.get(x, y), expected, "({x},{y})");
            }
        }
        // 窗口比暗线窄：不填
        assert_eq!(close(&m, 1).data, m.data);
    }

    /// 亮度 70 的平面上画暗 10 的形状。
    fn plane_with(w: usize, h: usize, dark: impl Fn(usize, usize) -> bool) -> GrayF32 {
        let data = (0..w * h)
            .map(|i| if dark(i % w, i / w) { 60.0 } else { 70.0 })
            .collect();
        GrayF32::from_vec(w, h, data)
    }

    #[test]
    fn dark_line_depth_finds_lines_in_any_direction() {
        // 宽 2 的横线与斜线：线心的深度为 10
        let horizontal = plane_with(40, 30, |x, y| (14..16).contains(&y) && (5..35).contains(&x));
        let d = dark_line_depth(&horizontal, 4, 8);
        assert!((d.get(20, 14) - 10.0).abs() < 1e-4, "{}", d.get(20, 14));
        let diagonal = plane_with(40, 40, |x, y| {
            (x as i32 - y as i32).abs() <= 1 && (5..35).contains(&x)
        });
        let d = dark_line_depth(&diagonal, 4, 8);
        assert!((d.get(20, 20) - 10.0).abs() < 1e-4, "{}", d.get(20, 20));
    }

    #[test]
    fn dark_line_depth_ignores_dots_and_steps() {
        // 比线段短的暗点（胡茬、毛孔）与台阶：深度 0，贴着图像边缘与角落的也一样
        let dots = plane_with(40, 40, |x, y| x % 10 < 3 && y % 10 < 3);
        assert!(dark_line_depth(&dots, 4, 8)
            .data
            .iter()
            .all(|v| v.abs() < 1e-4));
        for step in [
            plane_with(40, 30, |x, _| x >= 20),
            plane_with(40, 30, |_, y| y >= 12),
        ] {
            assert!(dark_line_depth(&step, 4, 8)
                .data
                .iter()
                .all(|v| v.abs() < 1e-4));
            assert!(dark_line_depth(&step, 0, 8).data.iter().all(|v| *v == 0.0));
        }
        // 斜的台阶：除了它斜穿的两个角（两条边都不到 half 的地方），处处为 0
        let diagonal = plane_with(40, 40, |x, y| x + y >= 40);
        let d = dark_line_depth(&diagonal, 4, 8);
        let near_corner = |x: usize, y: usize| (x >= 35 && y <= 4) || (x <= 4 && y >= 35);
        for y in 0..40 {
            for x in 0..40 {
                if !near_corner(x, y) {
                    assert!(d.get(x, y).abs() < 1e-4, "({x},{y}): {}", d.get(x, y));
                }
            }
        }
    }

    #[test]
    fn dark_line_depth_keeps_lines_that_run_off_the_image() {
        // 一直延伸到上下边缘的竖线：边缘处仍是线
        let line = plane_with(40, 30, |x, _| (19..21).contains(&x));
        let d = dark_line_depth(&line, 4, 8);
        for y in [0, 1, 15, 28, 29] {
            assert!(
                (d.get(19, y) - 10.0).abs() < 1e-4,
                "row {y}: {}",
                d.get(19, y)
            );
        }
    }
}
