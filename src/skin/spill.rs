//! 身体色调外溢：语义皮肤模型漏掉的相邻皮肤，也做身体色调（提亮 / 降红 / 降黄）。
//!
//! 身体遮罩要经过语义皮肤模型；它偶尔把一片皮肤判成非皮肤（1V3A3127 靠近红礼服的胸口下部，礼服的红色反光）。
//! 色调较轻时（奶油肌：提亮 4.6、黄度拉力 0.19）漏掉的皮肤几乎看不出；色调很强时（婚纱-深色内景：提亮 6.8、
//! 黄度按 0.67 拉向目标）漏掉的一片就与处理过的皮肤形成色块。外溢只补色调——磨皮、祛瑕疵仍只在遮罩内：
//! 遮罩外、人像内、不在脸旁、离已接受的皮肤近、低频颜色与附近已接受皮肤相近的像素，按相似度与邻近度得到权重；
//! 确认的外溢皮肤再加入参考色细化几轮（区域生长），参考色跟着连续的皮肤渐变走。
//! 衣物（白衬衫、米色西装、婚纱、红礼服）与皮肤的色差远大于门限，不受影响。
//! 权重在处理前的颜色上计算，色调遮罩取 `body + (1 − body)·spill`（遮罩的羽化边上不会凹陷）。

use crate::buffer::GrayF32;
use crate::color::lab::LabPlanes;
use crate::skin::guided::fast_gaussian;
use crate::skin::smoothstep;
use rayon::prelude::*;

/// 颜色门限：低频 Lab 距离（a、b 全权，L 减半——同一片皮肤的明暗随光照变化）d ≤ SIM_FULL 全量、≥ SIM_NONE 为 0。
/// 漏掉的皮肤与旁边已接受的皮肤 d ≈ 1–4；白衬衫、米色西装、婚纱、红礼服与皮肤 d ≥ 10
const SIM_FULL: f32 = 4.0;
const SIM_NONE: f32 = 8.0;
/// 被红色衣物的反光映红的皮肤（1V3A3101、3127 胸口贴着红礼服的一条：红度高 5～6，色相 47°～52°）仍是皮肤：
/// 比参考皮肤更红、且本身是肤色色相（Lab 色相角 ≥ HUE_FULL）时，红度差按 REDDER 计。粉色的东西（1V3A2954 的
/// 粉玫瑰、IMG_5785 的扇骨：红度同样高 6，但色相 15°～18°）照常计——色相在 HUE_NONE→HUE_FULL 之间过渡。
/// 红礼服、珊瑚色衣物比皮肤红 20 以上，折让后仍远超门限
const REDDER: f32 = 0.5;
const HUE_NONE: f32 = 25.0;
const HUE_FULL: f32 = 35.0;
/// 参考色的细化：确认的外溢像素（权重 JOIN_FROM→1 渐入、且是肤色色相）加入参考色再算，共 REFINE_ROUNDS 轮——
/// 参考色跟着连续的皮肤渐变走（1V3A3127 胸口上沿的高光比肩颈的已接受皮肤亮 8、红度低 6，只和语义遮罩比时
/// 差得太远）；外溢的范围仍由原始遮罩的邻近度限定，粉色的东西不进参考色（不会顺着花束长过去）
const REFINE_ROUNDS: usize = 2;
const JOIN_FROM: f32 = 0.5;
/// 参考色与邻近度的尺度（× 瞳距当量）
const REACH: f32 = 0.3;
/// 颜色低通（× 瞳距当量）
const COLOR_SIGMA: f32 = 0.03;
/// 附近已接受皮肤的占比（高斯加权）：≤ NEAR_NONE 为 0、≥ NEAR_FULL 全量
const NEAR_NONE: f32 = 0.02;
const NEAR_FULL: f32 = 0.12;
/// 脸旁不外溢（脸由脸部参数处理）：脸部遮罩模糊 FACE_GUARD 瞳距后 > 0.02 的范围
const FACE_GUARD: f32 = 0.1;
/// 计算分辨率：缩到瞳距当量约 WORK_ED 像素
const WORK_ED: f32 = 40.0;
/// 贴近裁剪框边缘（不是图像边缘）时渐隐的宽度（× 瞳距当量），免得在框边截出一道边
const EDGE_FADE: f32 = 0.05;

/// 外溢权重（0..1，与 `planes` 同尺寸）。`body` 为身体遮罩、`faces` 为脸部遮罩的并集、`person` 为人像 alpha，
/// 都与 `planes` 同尺寸（裁剪框坐标）；`ed` 为瞳距当量（像素）；`open_edges` 为裁剪框的左、上、右、下边
/// 是否是图像边缘（是则不渐隐）。
pub fn tone_spill(
    planes: &LabPlanes,
    body: &GrayF32,
    faces: &GrayF32,
    person: &GrayF32,
    ed: f32,
    open_edges: [bool; 4],
) -> GrayF32 {
    spill_refined(planes, body, faces, person, ed, open_edges, REFINE_ROUNDS)
}

/// [`tone_spill`]，参考色细化 `refine` 轮。
fn spill_refined(
    planes: &LabPlanes,
    body: &GrayF32,
    faces: &GrayF32,
    person: &GrayF32,
    ed: f32,
    open_edges: [bool; 4],
    refine: usize,
) -> GrayF32 {
    let (w, h) = (planes.w, planes.h);
    let k = (ed / WORK_ED).floor().max(1.0);
    let (sw, sh) = (
        ((w as f32 / k).round() as usize).max(1),
        ((h as f32 / k).round() as usize).max(1),
    );
    let s_ed = ed * sw as f32 / w as f32;
    let small = |g: &GrayF32| g.resize(sw, sh);
    let low = |g: &GrayF32| fast_gaussian(&small(g), (COLOR_SIGMA * s_ed).max(0.7));
    let (l, a, b) = (low(&planes.l), low(&planes.a), low(&planes.b));
    let body_s = small(body);
    let acc = GrayF32::from_vec(
        sw,
        sh,
        body_s
            .data
            .iter()
            .map(|v| f32::from(u8::from(*v > 0.5)))
            .collect(),
    );
    let reach = (REACH * s_ed).max(1.0);
    let support = fast_gaussian(&acc, reach);
    let face_near = fast_gaussian(&small(faces), (FACE_GUARD * s_ed).max(1.0));
    let person_s = small(person);
    // 肤色色相（0..1）：映红的折让与参考色的细化都只给肤色色相
    let skin_hue: Vec<f32> = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(a, b)| smoothstep(HUE_NONE, HUE_FULL, b.atan2(*a).to_degrees()))
        .collect();
    // 参考色为 `ref_w` 加权的低频颜色（高斯 reach）；ref_w ≥ acc，所以归一化分母不小于 support
    let spill_for = |ref_w: &GrayF32| -> Vec<f32> {
        let weighted = |p: &GrayF32| {
            let data = p.data.iter().zip(&ref_w.data).map(|(v, m)| v * m).collect();
            fast_gaussian(&GrayF32::from_vec(sw, sh, data), reach)
        };
        let norm = fast_gaussian(ref_w, reach);
        let (rl, ra, rb) = (weighted(&l), weighted(&a), weighted(&b));
        (0..sw * sh)
            .into_par_iter()
            .map(|i| {
                let s = support.data[i];
                if s <= NEAR_NONE || face_near.data[i] > 0.02 {
                    return 0.0;
                }
                let n = norm.data[i].max(s);
                let (dl, da, db) = (
                    l.data[i] - rl.data[i] / n,
                    a.data[i] - ra.data[i] / n,
                    b.data[i] - rb.data[i] / n,
                );
                let red_w = if da > 0.0 {
                    1.0 - (1.0 - REDDER) * skin_hue[i]
                } else {
                    1.0
                };
                let d = ((red_w * da).powi(2) + db.powi(2) + (0.5 * dl).powi(2)).sqrt();
                let similar = 1.0 - smoothstep(SIM_FULL, SIM_NONE, d);
                similar * smoothstep(NEAR_NONE, NEAR_FULL, s) * person_s.data[i].clamp(0.0, 1.0)
            })
            .collect()
    };
    let mut spill = spill_for(&acc);
    for _ in 0..refine {
        let ref_w = acc
            .data
            .iter()
            .zip(&spill)
            .zip(&skin_hue)
            .map(|((m, w), hue)| m.max(smoothstep(JOIN_FROM, 1.0, *w) * hue))
            .collect();
        spill = spill_for(&GrayF32::from_vec(sw, sh, ref_w));
    }
    let mut out = GrayF32::from_vec(sw, sh, spill).resize(w, h);
    fade_crop_edges(&mut out, EDGE_FADE * ed, open_edges);
    out
}

/// 在不是图像边缘的裁剪框边上线性渐隐到 0。
fn fade_crop_edges(m: &mut GrayF32, width: f32, open_edges: [bool; 4]) {
    let (w, h) = (m.w, m.h);
    let width = width.max(1.0);
    let ramp = |dist: usize, open: bool| {
        if open {
            1.0
        } else {
            (dist as f32 / width).min(1.0)
        }
    };
    m.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = ramp(y, open_edges[1]).min(ramp(h - 1 - y, open_edges[3]));
        for (x, v) in row.iter_mut().enumerate() {
            *v *= fy * ramp(x, open_edges[0]).min(ramp(w - 1 - x, open_edges[2]));
        }
    });
}

/// 色调遮罩：`body + (1 − body)·spill·strength`。
pub fn with_spill(body: &GrayF32, spill: &GrayF32, strength: f32) -> GrayF32 {
    let data = body
        .data
        .par_iter()
        .zip(&spill.data)
        .map(|(b, s)| b + (1.0 - b) * s * strength.clamp(0.0, 1.0))
        .collect();
    GrayF32::from_vec(body.w, body.h, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 160;
    const H: usize = 120;
    const ED: f32 = 60.0;
    const SKIN: [f32; 3] = [70.0, 12.0, 15.0];

    /// 场景：左半边是已接受的皮肤（颜色 `skin`）；右半边颜色由 `right` 给出，全在人像内、远离脸。
    fn scene(skin: [f32; 3], right: [f32; 3]) -> (LabPlanes, GrayF32) {
        let mut planes = LabPlanes {
            w: W,
            h: H,
            l: GrayF32::new(W, H),
            a: GrayF32::new(W, H),
            b: GrayF32::new(W, H),
        };
        let mut body = GrayF32::new(W, H);
        for y in 0..H {
            for x in 0..W {
                let c = if x < W / 2 { skin } else { right };
                let i = y * W + x;
                planes.l.data[i] = c[0];
                planes.a.data[i] = c[1];
                planes.b.data[i] = c[2];
                body.data[i] = f32::from(u8::from(x < W / 2));
            }
        }
        (planes, body)
    }

    fn spill_of(right: [f32; 3], person: f32, face_right: bool) -> GrayF32 {
        spill_between(SKIN, right, person, face_right)
    }

    fn spill_between(skin: [f32; 3], right: [f32; 3], person: f32, face_right: bool) -> GrayF32 {
        let (planes, body) = scene(skin, right);
        let mut faces = GrayF32::new(W, H);
        if face_right {
            faces
                .data
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = f32::from(u8::from(i % W > W - 10)));
        }
        let person = GrayF32::from_vec(W, H, vec![person; W * H]);
        tone_spill(&planes, &body, &faces, &person, ED, [true; 4])
    }

    #[test]
    fn missed_skin_next_to_accepted_skin_gets_the_tone() {
        // 被红色反光染红的同一片皮肤（红 +3；1V3A3101 胸口贴着红礼服的一条，红 +5、色相 52°）：紧挨着遮罩处
        // 接近全量，远处（超出 REACH）为 0
        for (skin, right) in [
            (SKIN, [69.0, 15.0, 16.0]),
            ([84.0, 13.9, 23.6], [80.4, 19.1, 24.4]),
        ] {
            let s = spill_between(skin, right, 1.0, false);
            assert!(s.get(W / 2 + 3, H / 2) > 0.9, "{}", s.get(W / 2 + 3, H / 2));
            assert!(s.get(W - 1, H / 2) < 0.05, "{}", s.get(W - 1, H / 2));
            assert!(
                s.get(10, H / 2) > 0.9,
                "inside the mask the weight is irrelevant but not suppressed"
            );
        }
    }

    #[test]
    fn clothes_background_and_faces_are_left_alone() {
        for (name, c) in [
            ("white shirt", [92.0, 0.0, 2.0]),
            ("beige suit", [82.0, 3.0, 14.0]),
            ("red dress", [45.0, 60.0, 35.0]),
            ("coral dress", [65.0, 32.0, 24.0]),
            ("pink dress", [78.0, 20.0, 2.0]),
        ] {
            let s = spill_of(c, 1.0, false);
            assert!(
                s.get(W / 2 + 3, H / 2) < 0.05,
                "{name}: {}",
                s.get(W / 2 + 3, H / 2)
            );
        }
        // 暗处偏粉的手旁的粉玫瑰（1V3A2954：红 +8、黄 −3，色相 16°）不享受映红的折让
        let roses = spill_between([56.5, 17.6, 10.2], [57.7, 25.9, 7.5], 1.0, false);
        assert!(
            roses.get(W / 2 + 3, H / 2) < 0.05,
            "{}",
            roses.get(W / 2 + 3, H / 2)
        );
        let outside_person = spill_of([69.0, 15.0, 16.0], 0.0, false);
        assert!(outside_person.data.iter().all(|v| *v == 0.0));
        let near_face = spill_of([69.0, 15.0, 16.0], 1.0, true);
        assert!(near_face.get(W - 12, H / 2) < 0.05);
    }

    #[test]
    fn refinement_follows_a_gradual_highlight_but_not_a_colour_step() {
        // 已接受的皮肤（x < 60）右边渐变成更亮、更不饱和的高光（1V3A3127 胸口上沿）：远处一段只和语义遮罩比时
        // 差得太远，细化后跟着渐变接上；同样的颜色若是突变（衣物边界）仍被挡住
        let highlight = |t: f32| [SKIN[0] + 9.0 * t, SKIN[1] - 6.0 * t, SKIN[2] - 6.0 * t];
        let run = |gradual: bool, refine: usize| {
            let (mut planes, _) = scene(SKIN, SKIN);
            for (i, (l, (a, b))) in planes
                .l
                .data
                .iter_mut()
                .zip(planes.a.data.iter_mut().zip(planes.b.data.iter_mut()))
                .enumerate()
            {
                let x = (i % W) as f32;
                let t = if gradual {
                    ((x - 60.0) / 30.0).clamp(0.0, 1.0)
                } else {
                    f32::from(u8::from(x >= 70.0))
                };
                [*l, *a, *b] = highlight(t);
            }
            let body = GrayF32::from_vec(
                W,
                H,
                (0..W * H)
                    .map(|i| f32::from(u8::from(i % W < 60)))
                    .collect(),
            );
            let person = GrayF32::from_vec(W, H, vec![1.0; W * H]);
            let faces = GrayF32::new(W, H);
            spill_refined(&planes, &body, &faces, &person, ED, [true; 4], refine).get(84, H / 2)
        };
        assert!(run(true, 0) < 0.1, "{}", run(true, 0));
        assert!(
            run(true, REFINE_ROUNDS) > 0.7,
            "{}",
            run(true, REFINE_ROUNDS)
        );
        assert!(
            run(false, REFINE_ROUNDS) < 0.05,
            "{}",
            run(false, REFINE_ROUNDS)
        );
    }

    #[test]
    fn tone_mask_does_not_dip_at_feathered_edges() {
        let body = GrayF32::from_vec(3, 1, vec![1.0, 0.5, 0.0]);
        let spill = GrayF32::from_vec(3, 1, vec![1.0, 1.0, 1.0]);
        let t = with_spill(&body, &spill, 1.0);
        assert_eq!(t.data, vec![1.0, 1.0, 1.0]);
        assert_eq!(with_spill(&body, &spill, 0.0).data, body.data);
    }

    #[test]
    fn crop_edges_fade_unless_they_are_image_edges() {
        let mut m = GrayF32::from_vec(10, 1, vec![1.0; 10]);
        fade_crop_edges(&mut m, 4.0, [false, true, true, true]);
        assert_eq!(m.data[0], 0.0);
        assert!((m.data[2] - 0.5).abs() < 1e-6 && m.data[9] == 1.0);
    }
}
