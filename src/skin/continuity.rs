//! 身体遮罩的颜色连续性：遮罩的边界只该落在真正的颜色边缘上。
//!
//! 身体遮罩是"严格肤色规则（`mask::is_skin_strict`）∧ 语义皮肤概率"。两者各有盲区，而遮罩里的皮肤会被
//! 提亮、匀肤、磨皮，遮罩外的保持原样——只要边界落在连续的皮肤中间，就显出一块块"阴影色块"
//! （doc/test_report_shadow_blotches.md）：
//! - **颜色规则漏掉的皮肤**：固定的颜色阈值在有色光下失效——天光照亮的肩膀、胸口、手臂偏蓝（B ≥ G），
//!   阳光直射的高光饱和度过低，红色影棚里被红墙染红的暗部过饱和。语义模型认得它们是皮肤。
//! - **语义模型的孤岛**：语义模型在一大片它不认的同色皮肤里，偶尔只认出一小块（抬起的手臂），
//!   或把一小块同色衣物当成皮肤（领结）。只处理这一小块，就是一块亮斑。
//!
//! 判据：光照造成的偏色、语义模型的误差都不会在交界处留下颜色边缘；物体（玫瑰、指甲、纹身、道具、
//! 围巾）与皮肤之间有。沿交界比较两侧紧邻的平均颜色，色度差与亮度差的中位数都小，交界就只是阈值线。
//! 在身体遮罩的分辨率上：
//! 1. **找回**：颜色规则不通过、语义概率高、人像抠图内、不太暗（头发、蕾丝、指缝的深阴影）的像素，去掉
//!    颜色梯度大的像素（物体边界把它们与皮肤切开）后取 4 连通块；与已接受皮肤之间没有颜色边缘的块整块接受。
//!    反方向（颜色规则通过、语义拒绝）不找回：语义边界本身是模糊的，"已接受"的一侧会混进同色的衬衫、
//!    米色西装、婚纱，从那里量不出边缘。
//! 2. **去孤岛**：面积小的已接受块，一半以上被"颜色规则通过、语义拒绝"的像素包围、且两者之间没有颜色边缘，
//!    就与周围一样不处理。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::rgb_to_lab;
use crate::skin::guided::fast_gaussian;
use crate::skin::masks::bbox_of;
use crate::skin::morph::{components, Component};
use rayon::prelude::*;

/// 找回的候选：语义皮肤概率（已模糊）的下限
const SEMANTIC_MIN: f32 = 0.7;
/// 候选的亮度（L）下限：更暗的是头发、黑色蕾丝、指缝的深阴影
const LIGHTNESS_MIN: f32 = 30.0;
/// 量颜色梯度前的平滑（像素）
const EDGE_SIGMA: f32 = 1.0;
/// 物体边界：亮度梯度（L / 像素）或色度梯度（a、b 合计，/ 像素）超过此值的像素不作候选。
/// 皮肤内部的梯度 99.9% 分位为亮度 5–11、色度 2.3–6（22 张样张），物体边界远大于此
const EDGE_LIGHTNESS: f32 = 8.0;
const EDGE_CHROMA: f32 = 4.0;
/// 量交界两侧颜色的窗口半宽（像素）
const STEP_RADIUS: usize = 3;
/// 交界两侧的色度差（a、b）与亮度差（L）的中位数都小于此值即为连续。偏蓝 / 高光 / 染红的皮肤为 0.4–3.4、
/// 亮度差 ≤ 3.2；玫瑰、指甲、纹身、围巾的色度差 ≥ 4.4，色度差更小的物体（扇柄 2.4）亮度差 ≥ 3.9
const STEP_CHROMA: f32 = 3.5;
const STEP_LIGHTNESS: f32 = 3.5;
/// 小于此面积（像素）的块不判定（留给身体遮罩的形态学清理）
const MIN_AREA: usize = 20;
/// 交界少于此像素数的块不判定
const MIN_BORDER: usize = 8;
/// 孤岛：面积上限（× 瞳距当量²，遮罩分辨率）
const ISLAND_AREA: f32 = 4.0;
/// 孤岛：外圈中"颜色规则通过、语义拒绝"的像素至少占此比例
const ISLAND_SURROUND: f32 = 0.5;

/// 按颜色连续性修正身体遮罩（原地，只改 0/1 判定，羽化在之后）。各输入与 `mask` 同尺寸：`img` 为 RGB 0..1，
/// `person` 为人像 alpha，`skin_prob` 为（模糊后的）语义皮肤概率，`strict` 为严格肤色规则（> 0.5 为通过）；
/// `mask` 当前为"规则 ∧ 语义"的遮罩（> 0.5 为已接受），`ed` 为该分辨率下的瞳距当量。
pub fn reconcile_body_mask(
    mask: &mut GrayF32,
    img: &ImgF32,
    person: &GrayF32,
    skin_prob: &GrayF32,
    strict: &GrayF32,
    ed: f32,
) {
    let (w, h) = (img.w, img.h);
    // 两个线索只有一方认作皮肤的像素：颜色规则拒、语义认（找回的候选）；颜色规则认、语义拒（孤岛的外圈）
    let split: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let open = person.data[i] > 0.5 && mask.data[i] <= 0.5;
            let rule = strict.data[i] > 0.5;
            let hit = open && (rule || skin_prob.data[i] > SEMANTIC_MIN);
            f32::from(u8::from(hit))
        })
        .collect();
    let Some((x0, y0, x1, y1)) = bbox_of(&GrayF32::from_vec(w, h, split), 0.5, STEP_RADIUS + 4)
    else {
        return;
    };
    let r = Region::new(img, x0, y0, x1 - x0, y1 - y0);
    let edge = r.edges();
    let rw = r.w;
    let at = move |i: usize| (y0 + i / rw) * w + x0 + i % rw;
    let n = r.w * r.h;
    // 可以改判的像素：人像内、不太暗、不在物体边界上
    let open: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| person.data[at(i)] > 0.5 && r.lab[i][0] >= LIGHTNESS_MIN && !edge[i])
        .collect();

    // 1. 找回颜色规则漏掉的皮肤
    let acc: Vec<bool> = (0..n).map(|i| mask.data[at(i)] > 0.5).collect();
    let cand: Vec<u8> = (0..n)
        .into_par_iter()
        .map(|i| {
            let g = at(i);
            u8::from(
                open[i] && !acc[i] && strict.data[g] <= 0.5 && skin_prob.data[g] > SEMANTIC_MIN,
            )
        })
        .collect();
    for c in r.continuing(&cand, usize::MAX, |_| true, &acc) {
        for &i in &c.pixels {
            mask.data[at(i)] = 1.0;
        }
    }

    // 2. 去掉被同色的语义拒绝区包围的小孤岛
    let acc: Vec<u8> = (0..n).map(|i| u8::from(mask.data[at(i)] > 0.5)).collect();
    let rule_only: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| open[i] && acc[i] == 0 && strict.data[at(i)] > 0.5)
        .collect();
    let limit = (ISLAND_AREA * ed * ed) as usize;
    let surrounded = |c: &Component| {
        let (mut ring, mut hits) = (0usize, 0usize);
        r.for_each_outer_neighbour(c, &acc, |j| {
            ring += 1;
            hits += usize::from(rule_only[j]);
        });
        ring > 0 && hits as f32 >= ISLAND_SURROUND * ring as f32
    };
    for c in r.continuing(&acc, limit, surrounded, &rule_only) {
        for &i in &c.pixels {
            mask.data[at(i)] = 0.0;
        }
    }
}

/// 包围盒（图像中左上角 `x0, y0`、尺寸 `w × h`）内的 Lab。
struct Region {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    /// 所在图像的尺寸
    image: (usize, usize),
    lab: Vec<[f32; 3]>,
}

impl Region {
    fn new(img: &ImgF32, x0: usize, y0: usize, w: usize, h: usize) -> Self {
        let lab = (0..w * h)
            .into_par_iter()
            .map(|i| rgb_to_lab(img.data[(y0 + i / w) * img.w + x0 + i % w]))
            .collect();
        Self {
            x0,
            y0,
            w,
            h,
            image: (img.w, img.h),
            lab,
        }
    }

    /// 块是否被包围盒截断（碰到包围盒边缘、而那条边不是图像边缘）。
    fn truncates(&self, c: &Component) -> bool {
        (c.x0 == 0 && self.x0 > 0)
            || (c.y0 == 0 && self.y0 > 0)
            || (c.x1 + 1 == self.w && self.x0 + self.w < self.image.0)
            || (c.y1 + 1 == self.h && self.y0 + self.h < self.image.1)
    }

    /// 物体边界：平滑后的亮度或色度梯度（中心差分）超过阈值的像素。
    fn edges(&self) -> Vec<bool> {
        let (w, h) = (self.w, self.h);
        let plane = |c: usize| {
            let data = self.lab.iter().map(|p| p[c]).collect();
            fast_gaussian(&GrayF32::from_vec(w, h, data), EDGE_SIGMA)
        };
        let (l, a, b) = (plane(0), plane(1), plane(2));
        let grad2 = |p: &GrayF32, x: usize, y: usize| {
            let dx = (p.get((x + 1).min(w - 1), y) - p.get(x.saturating_sub(1), y)) * 0.5;
            let dy = (p.get(x, (y + 1).min(h - 1)) - p.get(x, y.saturating_sub(1))) * 0.5;
            dx * dx + dy * dy
        };
        (0..w * h)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % w, i / w);
                grad2(&l, x, y) > EDGE_LIGHTNESS * EDGE_LIGHTNESS
                    || grad2(&a, x, y) + grad2(&b, x, y) > EDGE_CHROMA * EDGE_CHROMA
            })
            .collect()
    }

    /// `hit` 的 4 连通块中（面积 ≤ `limit`、没被包围盒截断、满足 `pick`）与 `other` 之间没有颜色边缘的块。
    fn continuing(
        &self,
        hit: &[u8],
        limit: usize,
        pick: impl Fn(&Component) -> bool + Sync,
        other: &[bool],
    ) -> Vec<Component> {
        let (w, h) = (self.w, self.h);
        let comps = components(hit, w, h, limit);
        let mut label = vec![0u32; w * h];
        for (k, c) in comps.iter().enumerate() {
            for &i in &c.pixels {
                label[i] = k as u32 + 1;
            }
        }
        let keep: Vec<bool> = comps
            .par_iter()
            .enumerate()
            .map(|(k, c)| {
                !self.truncates(c)
                    && c.pixels.len() >= MIN_AREA
                    && pick(c)
                    && self.continues(c, k as u32 + 1, &label, other)
            })
            .collect();
        comps
            .into_iter()
            .zip(keep)
            .filter_map(|(c, k)| k.then_some(c))
            .collect()
    }

    /// 块 `c` 外圈（4 邻接、不属于 `own`）的每个像素调用一次 `f`（可能重复计入）。
    fn for_each_outer_neighbour(&self, c: &Component, own: &[u8], mut f: impl FnMut(usize)) {
        let (w, h) = (self.w, self.h);
        for &i in &c.pixels {
            let (x, y) = (i % w, i / w);
            let nb = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in nb.into_iter().flatten().filter(|&j| own[j] == 0) {
                f(j);
            }
        }
    }

    /// 块 `c`（标号 `id`）与 `other` 之间是否没有颜色边缘：沿交界（块内与 `other` 4 邻接的像素），窗口内块自身
    /// 的平均颜色与 `other` 的平均颜色之差，色度与亮度的中位数都小于阈值。
    fn continues(&self, c: &Component, id: u32, label: &[u32], other: &[bool]) -> bool {
        let (w, h) = (self.w, self.h);
        let (mut chroma, mut lightness) = (Vec::new(), Vec::new());
        for &i in &c.pixels {
            let (x, y) = (i % w, i / w);
            let touches = (x > 0 && other[i - 1])
                || (x + 1 < w && other[i + 1])
                || (y > 0 && other[i - w])
                || (y + 1 < h && other[i + w]);
            if !touches {
                continue;
            }
            let (mut inside, mut outside) = (Mean::default(), Mean::default());
            for yy in y.saturating_sub(STEP_RADIUS)..=(y + STEP_RADIUS).min(h - 1) {
                for xx in x.saturating_sub(STEP_RADIUS)..=(x + STEP_RADIUS).min(w - 1) {
                    let j = yy * w + xx;
                    if label[j] == id {
                        inside.add(self.lab[j]);
                    } else if other[j] {
                        outside.add(self.lab[j]);
                    }
                }
            }
            if let (Some(p), Some(q)) = (inside.get(), outside.get()) {
                chroma.push((p[1] - q[1]).hypot(p[2] - q[2]));
                lightness.push((p[0] - q[0]).abs());
            }
        }
        chroma.len() >= MIN_BORDER
            && median(&mut chroma) < STEP_CHROMA
            && median(&mut lightness) < STEP_LIGHTNESS
    }
}

/// Lab 均值的累加器。
#[derive(Default)]
struct Mean {
    sum: [f32; 3],
    n: usize,
}

impl Mean {
    fn add(&mut self, p: [f32; 3]) {
        for (s, v) in self.sum.iter_mut().zip(p) {
            *s += v;
        }
        self.n += 1;
    }

    fn get(&self) -> Option<[f32; 3]> {
        let n = self.n as f32;
        (self.n > 0).then(|| self.sum.map(|s| s / n))
    }
}

/// 中位数（偶数个时取靠上的一个；会重排 `v`）。
fn median(v: &mut [f32]) -> f32 {
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f32::total_cmp).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skin::mask::is_skin_strict;

    const W: usize = 120;
    const H: usize = 80;
    const ED: f32 = 10.0;
    /// 暖色皮肤（通过严格规则）与天光下偏蓝的同一片皮肤（B > G，不通过）
    const WARM: [f32; 3] = [0.82, 0.64, 0.56];
    const COOL: [f32; 3] = [0.76, 0.66, 0.70];

    struct Scene {
        img: ImgF32,
        person: GrayF32,
        skin_prob: GrayF32,
    }

    /// 全是人像、语义上全是皮肤的场景，`color(x, y)` 给出像素颜色。
    fn scene(color: impl Fn(usize, usize) -> [f32; 3]) -> Scene {
        let mut img = ImgF32::new(W, H);
        for y in 0..H {
            for x in 0..W {
                img.data[y * W + x] = color(x, y);
            }
        }
        Scene {
            img,
            person: GrayF32::from_vec(W, H, vec![1.0; W * H]),
            skin_prob: GrayF32::from_vec(W, H, vec![1.0; W * H]),
        }
    }

    /// 修正结果：严格肤色规则图、修正前（"规则 ∧ 人像 ∧ 语义"，同 `masks::build_skin_masks`）与修正后的遮罩。
    struct Outcome {
        strict: GrayF32,
        before: GrayF32,
        after: GrayF32,
    }

    fn reconcile(s: &Scene) -> Outcome {
        let strict: Vec<f32> = s
            .img
            .data
            .iter()
            .map(|p| f32::from(u8::from(is_skin_strict(*p))))
            .collect();
        let strict = GrayF32::from_vec(W, H, strict);
        let mut mask = strict.clone();
        mask.data
            .iter_mut()
            .zip(s.skin_prob.data.iter().zip(&s.person.data))
            .for_each(|(m, (p, a))| {
                *m *= f32::from(u8::from(*a > 0.5)) * ((*p - 0.15) / 0.25).clamp(0.0, 1.0)
            });
        let before = mask.clone();
        reconcile_body_mask(&mut mask, &s.img, &s.person, &s.skin_prob, &strict, ED);
        Outcome {
            strict,
            before,
            after: mask,
        }
    }

    fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * t)
    }

    /// 从左到右由暖渐变到偏蓝（x 40..80），右侧整片偏蓝。
    fn gradual(x: usize) -> [f32; 3] {
        lerp(WARM, COOL, ((x as f32 - 40.0) / 40.0).clamp(0.0, 1.0))
    }

    #[test]
    fn gradually_tinted_skin_is_recovered() {
        let o = reconcile(&scene(|x, _| gradual(x)));
        assert!(
            o.strict.get(10, 40) > 0.5 && o.strict.get(110, 40) < 0.5,
            "the rule splits the gradient"
        );
        assert!(
            o.after.data.iter().all(|v| *v > 0.5),
            "the whole gradient is skin"
        );
    }

    #[test]
    fn objects_with_a_colour_edge_are_not_recovered() {
        // 暖色皮肤上贴着一块边界清晰的粉色（玫瑰 / 指甲油：B > G，与皮肤的色度差很大）
        let rose = [0.86, 0.45, 0.58];
        let o = reconcile(&scene(|x, y| {
            if (50..80).contains(&x) && (25..55).contains(&y) {
                rose
            } else {
                WARM
            }
        }));
        assert!(o.strict.get(65, 40) < 0.5);
        assert_eq!(o.after.data, o.before.data);
    }

    #[test]
    fn dark_non_semantic_and_detached_candidates_are_not_recovered() {
        // 同一片渐变的偏蓝皮肤：太暗、语义上不是皮肤、或不与已接受的皮肤相接时都不找回。
        // 偏蓝的一侧再逐渐暗到深阴影（L < 30）：交界连续，亮处找回，太暗的不找回
        let dim = |x: usize| 1.0 - 0.75 * ((x as f32 - 40.0) / 80.0).clamp(0.0, 1.0);
        let shaded = scene(|x, _| gradual(x).map(|v| v * dim(x)));
        let o = reconcile(&shaded);
        let dark = |i: &usize| rgb_to_lab(shaded.img.data[*i])[0] < LIGHTNESS_MIN;
        assert!(o.before.get(10, 40) > 0.5, "the warm side is accepted");
        assert!(
            (0..W * H).any(|i| dark(&i)),
            "the shadow is darker than the limit"
        );
        assert!(
            (0..W * H).any(|i| o.after.data[i] > o.before.data[i]),
            "the lit tinted part is recovered"
        );
        assert!((0..W * H)
            .filter(dark)
            .all(|i| o.after.data[i] == o.before.data[i]));

        let mut not_skin = scene(|x, _| gradual(x));
        not_skin.skin_prob.data.iter_mut().for_each(|v| *v = 0.6);
        let o = reconcile(&not_skin);
        assert_eq!(o.after.data, o.before.data);

        // 偏蓝的一块被一条非人像（背景）隔开，碰不到暖色皮肤
        let mut detached = scene(|x, _| if x < 60 { WARM } else { COOL });
        for y in 0..H {
            for x in 55..65 {
                detached.person.data[y * W + x] = 0.0;
            }
        }
        let o = reconcile(&detached);
        assert_eq!(o.after.data, o.before.data);
    }

    /// 语义概率：`on(x, y)` 处为 1，其余为 0。
    fn semantic(s: &mut Scene, on: impl Fn(usize, usize) -> bool) {
        s.skin_prob
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, p)| *p = f32::from(u8::from(on(i % W, i / W))));
    }

    #[test]
    fn small_semantic_islands_in_same_coloured_rejects_are_dropped() {
        // 整片暖色皮肤，语义模型只认出中间一小块（面积 < ISLAND_AREA·ed²）：与周围一样不处理
        let island = |x: usize, y: usize| (55..65).contains(&x) && (35..45).contains(&y);
        let mut s = scene(|_, _| WARM);
        semantic(&mut s, island);
        let o = reconcile(&s);
        assert!(o.before.get(60, 40) > 0.5, "the island was accepted");
        assert!(o.after.data.iter().all(|v| *v == 0.0));

        // 同样的一小块，但与周围之间有颜色边缘（周围是能通过颜色规则的米色衣物）：保留
        let beige = [0.88, 0.78, 0.67];
        let mut s = scene(|x, y| if island(x, y) { WARM } else { beige });
        semantic(&mut s, island);
        let o = reconcile(&s);
        assert!(o.strict.get(20, 20) > 0.5, "beige passes the colour rule");
        assert_eq!(o.after.data, o.before.data);

        // 面积超过上限的已接受区域：保留
        let mut s = scene(|_, _| WARM);
        semantic(&mut s, |x, y| {
            (20..100).contains(&x) && (10..70).contains(&y)
        });
        let o = reconcile(&s);
        assert_eq!(o.after.data, o.before.data);
    }

    #[test]
    fn only_crop_edges_inside_the_image_truncate_a_component() {
        // 包围盒 20×10，图像 40×30：包围盒的边与图像边缘重合时，碰到它的块是完整的
        let region = |x0, y0| Region {
            x0,
            y0,
            w: 20,
            h: 10,
            image: (40, 30),
            lab: vec![[50.0, 0.0, 0.0]; 200],
        };
        let block = |x0, y0, x1, y1| Component {
            pixels: Vec::new(),
            x0,
            y0,
            x1,
            y1,
        };
        assert!(!region(10, 10).truncates(&block(5, 3, 12, 6)));
        for (touching, inside, on_image_edge) in [
            (block(0, 3, 12, 6), (10, 10), (0, 10)),  // 左
            (block(5, 0, 12, 6), (10, 10), (10, 0)),  // 上
            (block(5, 3, 19, 6), (10, 10), (20, 10)), // 右
            (block(5, 3, 12, 9), (10, 10), (10, 20)), // 下
        ] {
            assert!(region(inside.0, inside.1).truncates(&touching));
            assert!(!region(on_image_edge.0, on_image_edge.1).truncates(&touching));
        }
    }
}
