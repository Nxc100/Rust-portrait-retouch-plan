# Rust 人像修图功能技术方案
## 滤镜 · 磨皮 · 美白 · 瘦脸 / 大眼 / 瘦鼻

> 版本 1.0 · 2026-09-21
> 适用对象：已有 Rust 人像图片处理软件，需要新增"滤镜修图"功能
> 目标：在不引入 Objective-C / OpenGL ES / 闭源 SDK 的前提下，用纯 Rust（CPU 优先、GPU 可选）复现并超越 Meihu-Beautyface-sdk 的修图效果
> 依据：本文所有关于美狐项目的参数、公式、点位均来自对其源码的逐文件阅读（BBGPUImageBeautifyFilter.m、GLImageFaceChangeFilter.m、GPUImageBilateralFilter.m、GPUImageLookupFilter.m、GPUImageHSBFilter.m、filterFile.zip 资源包等），不是凭印象

---

## 目录

0. 结论摘要
1. 参考项目选型与分工
2. 美狐项目源码解读（还原依据）
3. 总体架构
4. 关键算法实现规范（含 Rust 代码）
5. 依赖清单
6. 实施计划与验收标准
7. 验证与测试方案
8. 风险与对策
9. 许可证合规清单
附录 A 关键点语义映射表
附录 B UI 参数映射表
附录 C 参考链接

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 能否直接植入美狐 SDK | 不能。ObjC + UIKit + OpenGL ES + 仅 iOS 的闭源 Face++ 静态库，且 GPL-3.0 + 非商用声明 |
| 能否用 Rust 复现同样效果 | 能。所有像素运算都是确定性公式，本文给出逐项公式与参数；脸型部分效果取决于关键点模型 |
| 主参考项目 | **pixpark/gpupixel**（架构与商用友好的算法实现）+ **美狐源码**（具体参数与点位）+ **YuAo/YUCIHighPassSkinSmoothing**（静态修图更优的磨皮） |
| 人脸关键点 | 开发期用 InsightFace 2d106det（106 点，与 Face++ 布局最接近）；商用发布切换 MediaPipe Face Mesh（Apache-2.0） |
| 实现形态 | 独立 crate `portrait-retouch`，纯 CPU + rayon，处理 12MP 照片全流程 < 2 s；预览走 1280 长边工作副本，交互延迟 < 200 ms |
| 工期估算 | 单人 6 周可交付可用版本（详见第 6 节） |

---

## 1. 参考项目选型与分工

### 1.1 选型原则

1. 商用友好的许可证优先（Apache-2.0 / MIT / BSD）。
2. 算法必须能在 CPU 上用确定性公式复现（不依赖特定 GPU 运行时）。
3. 与美狐项目的点位体系、参数语义可对照，便于"还原"。
4. 项目仍在维护，或算法已被广泛验证。

### 1.2 最终选型

| 角色 | 项目 | 许可证 | 从它取什么 | 不取什么 |
|---|---|---|---|---|
| **主架构参考** | [pixpark/gpupixel](https://github.com/pixpark/gpupixel) | Apache-2.0 | 滤镜链设计（Source → Filter → Target）、磨皮/美白/瘦脸/大眼的 shader 逻辑、参数命名与范围、用开源关键点替代 Face++ 的思路 | 它的 OpenGL 运行时（我们用 CPU/wgpu） |
| **还原依据** | [zhanghao5683934/Meihu-Beautyface-sdk](https://github.com/zhanghao5683934/Meihu-Beautyface-sdk) | GPL-3.0 + 非商用 | 具体数值参数、106 点索引的语义、滤镜资源包格式、Demo 的滤镜串联顺序 | 任何代码文本、任何贴图资源（详见第 9 节） |
| **磨皮质量升级** | [YuAo/YUCIHighPassSkinSmoothing](https://github.com/YuAo/YUCIHighPassSkinSmoothing) | MIT | 频率分离磨皮流程与默认参数 | Core Image 相关代码 |
| **底层滤镜公式** | [BradLarson/GPUImage](https://github.com/BradLarson/GPUImage) | BSD-3 | 双边滤波 9-tap 权重、Sobel、HSB 矩阵、512 查找表采样公式 | — |
| **人脸检测** | [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)（UltraFace） | MIT | `version-RFB-320.onnx`，解码极简 | — |
| **关键点（开发期）** | [deepinsight/insightface](https://github.com/deepinsight/insightface) `2d106det.onnx` | 代码 MIT / **模型仅限非商用** | 106 点，索引语义与 Face++ 最接近，便于对照调参 | 商用发布时不得使用该模型 |
| **关键点（发布期）** | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) Face Mesh（ONNX 版见 PINTO model zoo） | Apache-2.0 | 468/478 点，含虹膜中心，商用无障碍 | — |
| **Rust 基础库** | image / imageproc / rayon / ort / wgpu | MIT/Apache | 解码、基础滤波、并行、ONNX 推理、GPU（可选） | — |

### 1.3 为什么不是"直接 FFI 调 gpupixel"

gpupixel 是 C++ + OpenGL，Windows/Linux 下需要创建 GL 上下文并处理离屏渲染；从 Rust 桥接要写 `cxx` 层并维护两套构建系统。对于**静态修图**场景，CPU 实现更简单、可测试、无平台差异，且单张照片的耗时完全可接受。gpupixel 在本方案里的价值是"算法与架构的对照读本"，不是运行时依赖。若日后要做实时视频美颜，再考虑 FFI 或 wgpu。

---

## 2. 美狐项目源码解读（还原依据）

本节把美狐 Demo 真正在运行的链路和每个数值挖出来，作为第 4 节实现的"标准答案"。

### 2.1 Demo 实际运行的滤镜链

文件：`MHOpenDemo/MHOpenSourceViewController.m`

```
videoCamera ──► BBGPUImageBeautifyFilter ──► GLImageFaceChangeFilterGroup ──► GPUImageView
                （磨皮 + 提亮 + 饱和）        （瘦脸 / 大眼 / 瘦鼻）
```

要点：
- 磨皮先于形变（颜色运算不改变像素位置，关键点在原图检测即可）。
- 风格滤镜（LUT）在 Demo 中通过 `GLImageLutFilter`（512×512 查找图）或曲线滤镜包挂在链尾。
- 相机为 720×1280 前置视频流，这决定了下文所有"以像素为单位"的参数换算基准。

### 2.2 磨皮：`BBGPUImageBeautifyFilter`（Demo 实际使用）

文件：`MHOpenDemo/Third/filter/BBGPUImageBeautifyFilter.m`（来源 Bonway/BBGPUImage）

四个 pass：

| Pass | 滤镜 | 参数 | 说明 |
|---|---|---|---|
| 1 | `GPUImageBilateralFilter` | `distanceNormalizationFactor = 4.0`，`texelSpacingMultiplier = 4.0`（GPUImage 默认） | 可分离两遍（先水平后垂直），每遍 9 tap，tap 间距 4 像素 |
| 2 | `GPUImageSobelEdgeDetectionFilter` | `edgeStrength = 1.0` | 先转灰度（0.2125, 0.7154, 0.0721），再 3×3 Sobel，输出梯度模长 |
| 3 | `GPUImageCombinationFilter`（三输入） | `smoothDegree = 0.5` | 见下方公式 |
| 4 | `GPUImageHSBFilter` | `brightness = 1.1`，`saturation = 1.1` | 4×4 颜色矩阵 |

Pass 3 逐像素公式（原 shader 直译）：

```
skin = edge.r < 0.2
    && r > 0.3725 && g > 0.1568 && b > 0.0784
    && r > b
    && (max(r,g,b) - min(r,g,b)) > 0.0588
    && |r - g| > 0.0588

smooth = skin ? bilateral + (1 - smoothDegree) * (origin - bilateral) : origin

// 全图轻微提亮（log 曲线）
smooth.c = ln(1 + 0.2 * smooth.c) / ln(1.2)      // c ∈ {r,g,b}
```

Pass 1 双边滤波权重（`GPUImageBilateralFilter.m`）：

```
offset:  -4  -3  -2  -1   0  +1  +2  +3  +4   （单位：tap，乘以 spacing 像素）
weight: .05 .09 .12 .15 .18 .15 .12 .09 .05
每个 tap 的实际权重 = weight * (1 - min(‖sample - center‖ * dnf, 1))
输出 = Σ(sample * w) / Σw     （中心 tap 权重固定 0.18）
```

Pass 4 HSB（`GPUImageHSBFilter.m`，矩阵来自 graficaobscura）等价于：

```
lum = 0.3086 r + 0.6094 g + 0.0820 b
c'  = clamp( brightness * ( lum + (c - lum) * saturation ), 0, 1 )
```

另有 `GPURenderKit/.../GPUImageBeautifyFilter.m`（来自 mkil/MagicCamera）使用"双边 + Canny + 高反差强光"的变体，但 Demo 未使用，本文不作为还原目标，仅在 4.3 节作为可选思路提及。

### 2.3 瘦脸 / 大眼 / 瘦鼻：`GLImageFaceChangeFilter`

文件：`GPURenderKit/.../FaceFilters/GLImageFaceChangeFilter.m`
坐标系：纹理归一化坐标 [0,1]，`locArray[106]` 为 Face++ 106 点。

**基础形变函数（局部推挤，反向映射）**

```
warpPositionToUse1(cur, A, B, radius, delta):
    r = |cur - A|                     （y 已按宽高比修正，使 x/y 同尺度）
    if r < radius:
        dir   = normalize(B - A)
        dist  = radius² - r²
        alpha = (dist / (dist + (r - delta)²))²
        cur   = cur - alpha * delta * dir
    return cur
```

**瘦脸 `adjust_thinFace`**（ed = 两瞳距离 `|p74 - p77|`，dir_right = normalize(p77 − p74)）

```
左锚点: L0 = p4  - dir_right*0.13*ed,  L1 = p9  - dir_right*0.33*ed,  L2 = p13 - dir_right*0.33*ed
右锚点: R0 = p28 + dir_right*0.13*ed,  R1 = p23 + dir_right*0.33*ed,  R2 = p19 + dir_right*0.33*ed
x = π/25;  scale = 2*ed;  radius = 0.4*scale = 0.8*ed
delta = [sin(x), sin(2x), sin(2x)] * intensity * 0.15 * scale
for i in 0..3:
    cur = warp(cur, L_i, R_i, radius, delta_i)
    cur = warp(cur, R_i, L_i, radius, delta_i)
```

**大眼 `adjust_eye`**（径向幂次缩放）

```
k = intensity * 0.24
eye_width = |p52 - p55|          （左眼外眼角到内眼角）
radius = eye_width
for center in [p74, p77]:        （左右瞳孔中心，顺序执行，第二次用已修改的坐标）
    d = |cur - center|
    if d < radius:
        w   = pow(max(d, ε) / radius, k)
        cur = center + (cur - center) * w
```

**瘦鼻 `newNarrowNose_2`**

```
scale = 0.28*ed;  delta = intensity * scale * scale;  radius = 0.16（归一化常数！）
锚点: A0 = p48 + dir_up*0.09（归一化常数！）, A1 = p82 ; B0 = p50 + dir_up*0.09, B1 = p83
两轮 warp 同瘦脸
```
注意：瘦鼻里 `radius=0.16`、`dir_up*0.09` 是归一化坐标常数，与分辨率耦合，是原实现的缺陷。第 4.7 节给出以 ed 为尺度的等效换算。

**主函数点位依赖**：`dir_up = normalize(p43 − p16)`（鼻梁顶点 → 下巴），`dir_right = normalize(p77 − p74)`。

### 2.4 关键点索引的语义（来自仓库内 `人脸106个关键点.png`）

| 索引 | 语义（以图像左右为准，非受试者左右） |
|---|---|
| 0–32 | 脸部轮廓，0 = 左太阳穴，16 = 下巴尖，32 = 右太阳穴 |
| 4 / 9 / 13 | 左侧颧骨下方 / 中下颌 / 下颌角附近 |
| 28 / 23 / 19 | 右侧对称点 |
| 43 | 鼻梁顶点（两眼之间） |
| 46 | 鼻尖 |
| 48 / 50 | 左 / 右鼻孔 |
| 82 / 83 | 左 / 右鼻翼外侧 |
| 52 / 55 | 左眼外眼角 / 内眼角 |
| 58 / 61 | 右眼内眼角 / 外眼角 |
| 74 / 77 | 左 / 右瞳孔中心 |
| 84–103 | 嘴部 |

我们的 Rust 实现**不依赖具体索引**，而是定义语义结构体（4.6 节），不同模型各自提供映射。

### 2.5 滤镜资源的两种格式

**格式 A：512×512 查找图（GPUImage Lookup）**
`MHSDK.bundle/filter_*.png`（20 余款）、`white.png`、`skin_lookup.png`（美白）。
采样公式见 4.5 节，`GLImageLutFilter` 带 `intensity` 混合。这是最容易复现、也是行业最通用的格式。

**格式 B：曲线着色器包**
`filterFile.zip` → 每款一个目录：`config.json + fragment.glsl + 若干 PNG`
- `curve.png`（256×2 或 256×N）：按通道的一维曲线
- `mask.png`（320×320）：拉伸到全图的灰度遮罩（暗角类）
- `fragment.glsl`：曲线查表 + 固定色叠加（overlay/screen/multiply）+ 遮罩加权 + `mix(source, result, strength)`
我们不在运行时执行 GLSL，而是把"纯颜色部分"离线烘焙成 3D LUT（.cube），遮罩部分作为独立 op（4.5 节）。

### 2.6 不能直接搬的东西

| 内容 | 原因 | 替代 |
|---|---|---|
| 任何 `.m/.h` 代码文本 | GPL-3.0 传染 + 仓库声明非商用 | 按公式在 Rust 重写（本文已抽取全部公式） |
| `filter_*.png`、`filterFile.zip` 内贴图 | 资源版权，仓库禁止商用 | 自制 LUT（4.5.4 节流程） |
| Face++ 库与模型 | 闭源、需授权、仅 iOS | UltraFace + 2d106det / Face Mesh |
| `GPURenderKit.framework` | iOS 二进制 | — |

---

## 3. 总体架构

### 3.1 Crate 结构

```
portrait-retouch/
├── Cargo.toml
├── models/                       # ONNX 模型（不入库，构建脚本下载或用户放置）
│   ├── version-RFB-320.onnx      # UltraFace 检测（MIT）
│   ├── 2d106det.onnx             # 开发期关键点（非商用）
│   └── face_mesh.onnx            # 发布期关键点（Apache-2.0）
├── luts/                         # 自制 .cube / 512 查找图
├── src/
│   ├── lib.rs                    # pub API: retouch(), detect_faces(), RetouchParams
│   ├── buffer.rs                 # ImgF32：f32 sRGB 缓冲、双线性采样、缩放
│   ├── geom.rs                   # P(点)、向量运算
│   ├── color/
│   │   ├── lookup512.rs          # GPUImage 512 查找图
│   │   ├── lut3d.rs              # .cube 解析 + 三线性
│   │   ├── curve.rs              # 一维曲线（256 表）
│   │   ├── blend.rs              # overlay/screen/multiply/softlight
│   │   └── hsb.rs                # 亮度/饱和矩阵
│   ├── skin/
│   │   ├── bilateral.rs          # GPUImage 风格可分离双边
│   │   ├── edge.rs               # 灰度 + Sobel
│   │   ├── mask.rs               # 颜色肤色规则 + 人脸多边形羽化
│   │   ├── smooth_faithful.rs    # 方案 A：忠实还原 BB 组合
│   │   ├── smooth_freqsep.rs     # 方案 B：频率分离（YUCI）
│   │   └── whiten.rs             # 美白
│   ├── face/
│   │   ├── detector.rs           # UltraFace ONNX
│   │   ├── landmark.rs           # LandmarkModel trait
│   │   ├── lm_2d106.rs           # InsightFace 2d106det 实现 + 语义映射
│   │   ├── lm_facemesh.rs        # MediaPipe Face Mesh 实现 + 语义映射
│   │   └── semantic.rs           # FaceKeyPoints 语义结构体
│   ├── warp/
│   │   ├── pinch.rs              # warpPositionToUse1 移植
│   │   ├── face_warp.rs          # 瘦脸/大眼/瘦鼻步骤生成 + 反向映射
│   │   └── apply.rs              # 并行采样
│   ├── pipeline.rs               # 顺序编排、预览/导出分辨率策略
│   └── debug.rs                  # 关键点可视化、中间结果导出
├── tools/
│   ├── bake_lut.py               # 曲线包 → .cube 烘焙脚本
│   └── oracle_numpy.py           # 数值对照脚本
└── tests/
    ├── identity.rs               # LUT/曲线/形变的恒等测试
    └── golden.rs                 # 视觉回归（PSNR）
```

### 3.2 数据流

```
RgbImage (u8)
   │ from_rgb8
   ▼
ImgF32 (sRGB, 0..1, 不做线性化 —— 与 GPUImage 一致)
   │
   ├─► 工作副本 W（短边 720）──► 双边(spacing=4, dnf=4) ──┐
   │                          └► 灰度+Sobel(step=1) ──────┤ 上采样到全分辨率
   │                                                       ▼
   ├─► [人脸检测 + 关键点]（在原图或 W 上）             组合 (smoothDegree, 肤色规则, 可选人脸遮罩)
   │                                                       │ log 提亮 → HSB(1.1, 1.1)
   │                                                       ▼
   │                                                    美白 (LUT/曲线 × 皮肤遮罩)
   │                                                       ▼
   └─────────── FaceKeyPoints ───────────────────────► 形变（瘦脸/大眼/瘦鼻，反向映射 + 双线性）
                                                           ▼
                                                        风格滤镜（512 查找图 / .cube × intensity）
                                                           ▼
                                                        to_rgb8 → RgbImage
```

顺序与美狐 Demo 一致（颜色 → 形变 → 风格），关键点只在原图上检测一次。

### 3.3 公开 API（lib.rs）

```rust
pub struct RetouchParams {
    // 磨皮
    pub smooth: f32,               // 0..1，对应 smoothDegree，默认 0.5
    pub smooth_mode: SmoothMode,   // Faithful（还原）| FreqSep（频率分离）
    pub restrict_to_face: bool,    // 仅在人脸多边形内磨皮（改进项，默认 true）
    // 提亮/饱和（HSB）
    pub brightness: f32,           // 默认 1.1
    pub saturation: f32,           // 默认 1.1
    pub apply_log_curve: bool,     // 默认 true（还原 BB 的 log 提亮）
    // 美白
    pub whiten: f32,               // 0..1，默认 0.0
    // 形变
    pub thin_face: f32,            // 0..1
    pub big_eye: f32,              // 0..1
    pub thin_nose: f32,            // 0..1
    // 风格
    pub style: Option<StyleFilter>,
    pub style_intensity: f32,      // 0..1
}

pub enum SmoothMode { Faithful, FreqSep }

pub enum StyleFilter {
    Lookup512(std::sync::Arc<ImgF32>),
    Cube(std::sync::Arc<Lut3D>),
}

pub struct FaceKeyPoints { /* 见 4.6 */ }

pub struct Engine { /* 持有 ONNX session 与缓存 */ }

impl Engine {
    pub fn new(cfg: EngineConfig) -> anyhow::Result<Self>;
    /// 在原图上检测所有人脸并返回语义关键点（像素坐标）
    pub fn detect_faces(&mut self, img: &image::RgbImage) -> anyhow::Result<Vec<FaceKeyPoints>>;
    /// 完整修图；faces 为 detect_faces 的结果（可缓存，滑块调整时不必重检）
    pub fn retouch(&self, img: &image::RgbImage, faces: &[FaceKeyPoints], p: &RetouchParams)
        -> image::RgbImage;
}
```

设计要点：
- **检测与修图分离**：用户拖滑块时只调 `retouch`，关键点复用。
- **参数分辨率无关**：所有半径/位移以瞳距 `ed` 为尺度，所有模糊在"短边 720 工作副本"上进行，预览与导出观感一致。
- **纯函数**：`retouch` 不修改输入，便于 undo/redo 与并行。

---

## 4. 关键算法实现规范（含 Rust 代码）

约定：
- 所有颜色在 **sRGB 伽马空间**以 f32（0..1）运算，不做线性化。GPUImage 就是这么做的，线性化反而会偏离原效果。
- 坐标为**像素中心制**：像素 (x, y) 的中心是 (x + 0.5, y + 0.5)，对应 GL 的 `texture2D` 采样习惯。
- 采样等价于 `GL_LINEAR + CLAMP_TO_EDGE`。
- 代码为可直接落地的参考实现（未在本文档环境中编译，以你的工程 `cargo check` 为准；均只依赖 `image`、`rayon` 与标准库）。

### 4.1 图像缓冲与采样（buffer.rs）

```rust
use rayon::prelude::*;

#[derive(Clone)]
pub struct ImgF32 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<[f32; 3]>, // 行优先，sRGB 0..1
}

#[inline]
fn q8(v: f32) -> u8 { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 }

impl ImgF32 {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![[0.0; 3]; w * h] }
    }

    pub fn from_rgb8(img: &image::RgbImage) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let data = img
            .pixels()
            .map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
            .collect();
        Self { w, h, data }
    }

    pub fn to_rgb8(&self) -> image::RgbImage {
        let mut out = image::RgbImage::new(self.w as u32, self.h as u32);
        for (dst, src) in out.pixels_mut().zip(&self.data) {
            *dst = image::Rgb([q8(src[0]), q8(src[1]), q8(src[2])]);
        }
        out
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 3] { self.data[y * self.w + x] }

    /// 等价 GL_LINEAR + CLAMP_TO_EDGE；(x, y) 为像素中心制坐标
    pub fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let fx = (x - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = (y - 0.5).clamp(0.0, (self.h - 1) as f32);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let (p00, p10, p01, p11) = (self.get(x0, y0), self.get(x1, y0), self.get(x0, y1), self.get(x1, y1));
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let a = p00[c] + (p10[c] - p00[c]) * tx;
            let b = p01[c] + (p11[c] - p01[c]) * tx;
            out[c] = a + (b - a) * ty;
        }
        out
    }

    /// 双线性缩放（用于工作副本与上采样）。质量要求更高时换 fast_image_resize。
    pub fn resize(&self, nw: usize, nh: usize) -> ImgF32 {
        let mut out = ImgF32::new(nw, nh);
        let sx = self.w as f32 / nw as f32;
        let sy = self.h as f32 / nh as f32;
        out.data.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            for x in 0..nw {
                row[x] = self.sample_bilinear((x as f32 + 0.5) * sx, (y as f32 + 0.5) * sy);
            }
        });
        out
    }
}
```

### 4.2 磨皮方案 A：忠实还原 BBGPUImageBeautifyFilter

#### 4.2.1 分辨率策略

原 Demo 在 720×1280 视频上跑，tap 间距 4 px、Sobel 步长 1 px。要在任意分辨率上得到**同样观感**：

```
工作副本 W：短边缩放到 720（长边按比例），在 W 上算双边与边缘 → 上采样回原尺寸 → 与原图全分辨率组合
```

好处：忠实、快（12MP 图的双边只需算 ~1MP）、预览和导出一致。

#### 4.2.2 可分离双边滤波（skin/bilateral.rs）

```rust
use crate::buffer::ImgF32;
use rayon::prelude::*;

const W9: [f32; 9] = [0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05];

fn bilateral_pass(src: &ImgF32, spacing: f32, dnf: f32, horizontal: bool) -> ImgF32 {
    let mut out = ImgF32::new(src.w, src.h);
    out.data.par_chunks_mut(src.w).enumerate().for_each(|(y, row)| {
        for x in 0..src.w {
            let c = src.get(x, y);
            let mut sum = [c[0] * 0.18, c[1] * 0.18, c[2] * 0.18];
            let mut wsum = 0.18f32;
            for (i, &wg) in W9.iter().enumerate() {
                if i == 4 { continue; }
                let off = (i as f32 - 4.0) * spacing;
                let (sx, sy) = if horizontal {
                    (x as f32 + 0.5 + off, y as f32 + 0.5)
                } else {
                    (x as f32 + 0.5, y as f32 + 0.5 + off)
                };
                let s = src.sample_bilinear(sx, sy);
                let d = ((s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2) + (s[2] - c[2]).powi(2)).sqrt();
                let w = wg * (1.0 - (d * dnf).min(1.0));
                wsum += w;
                for k in 0..3 { sum[k] += s[k] * w; }
            }
            row[x] = [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum];
        }
    });
    out
}

/// GPUImageBilateralFilter 等价实现：先水平后垂直
pub fn bilateral_gpuimage(src: &ImgF32, spacing: f32, dnf: f32) -> ImgF32 {
    let h = bilateral_pass(src, spacing, dnf, true);
    bilateral_pass(&h, spacing, dnf, false)
}
```

默认：`spacing = 4.0`（在短边 720 的工作副本上），`dnf = 4.0`。

#### 4.2.3 灰度 + Sobel（skin/edge.rs）

```rust
use crate::buffer::ImgF32;
use rayon::prelude::*;

/// 返回与 src 同尺寸的边缘强度图（edgeStrength = 1.0）
pub fn sobel_edge(src: &ImgF32) -> Vec<f32> {
    let (w, h) = (src.w, src.h);
    let lum: Vec<f32> = src.data.iter()
        .map(|p| 0.2125 * p[0] + 0.7154 * p[1] + 0.0721 * p[2])
        .collect();
    let at = |x: isize, y: isize| -> f32 {
        let xi = x.clamp(0, w as isize - 1) as usize;
        let yi = y.clamp(0, h as isize - 1) as usize;
        lum[yi * w + xi]
    };
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y = y as isize;
        for x in 0..w {
            let xi = x as isize;
            let (tl, t, tr) = (at(xi - 1, y - 1), at(xi, y - 1), at(xi + 1, y - 1));
            let (l, r) = (at(xi - 1, y), at(xi + 1, y));
            let (bl, b, br) = (at(xi - 1, y + 1), at(xi, y + 1), at(xi + 1, y + 1));
            let hh = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
            let vv = -bl - 2.0 * l - tl + br + 2.0 * r + tr;
            row[x] = (hh * hh + vv * vv).sqrt();
        }
    });
    out
}

/// 单通道双线性上采样
pub fn resize_gray(src: &[f32], sw: usize, sh: usize, nw: usize, nh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nw * nh];
    let sx = sw as f32 / nw as f32;
    let sy = sh as f32 / nh as f32;
    out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize; let y1 = (y0 + 1).min(sh - 1); let ty = fy - y0 as f32;
        for x in 0..nw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize; let x1 = (x0 + 1).min(sw - 1); let tx = fx - x0 as f32;
            let a = src[y0 * sw + x0] + (src[y0 * sw + x1] - src[y0 * sw + x0]) * tx;
            let b = src[y1 * sw + x0] + (src[y1 * sw + x1] - src[y1 * sw + x0]) * tx;
            row[x] = a + (b - a) * ty;
        }
    });
    out
}
```

#### 4.2.4 组合 + log 提亮 + HSB（skin/smooth_faithful.rs, color/hsb.rs）

```rust
use crate::buffer::ImgF32;
use rayon::prelude::*;

#[inline]
pub fn is_skin_rgb(p: [f32; 3]) -> bool {
    let (r, g, b) = (p[0], p[1], p[2]);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    r > 0.3725 && g > 0.1568 && b > 0.0784 && r > b && (mx - mn) > 0.0588 && (r - g).abs() > 0.0588
}

/// origin/bilateral 为全分辨率；edge 为已上采样到全分辨率的边缘图；
/// face_mask 为可选的人脸羽化遮罩（0..1，None 表示不限制 —— 忠实模式）
pub fn combine_bb(
    origin: &ImgF32,
    bilateral: &ImgF32,
    edge: &[f32],
    face_mask: Option<&[f32]>,
    smooth_degree: f32,
    apply_log: bool,
) -> ImgF32 {
    let ln12 = 1.2f32.ln();
    let mut out = ImgF32::new(origin.w, origin.h);
    out.data.par_iter_mut().enumerate().for_each(|(i, px)| {
        let o = origin.data[i];
        let bl = bilateral.data[i];
        let m = face_mask.map(|m| m[i]).unwrap_or(1.0);
        let mut s = if edge[i] < 0.2 && is_skin_rgb(o) && m > 0.0 {
            // smooth = bilateral + (1 - degree) * (origin - bilateral)，再按遮罩与原图混合
            let mut t = [0.0f32; 3];
            for c in 0..3 {
                let v = bl[c] + (1.0 - smooth_degree) * (o[c] - bl[c]);
                t[c] = o[c] + (v - o[c]) * m;
            }
            t
        } else {
            o
        };
        if apply_log {
            for c in 0..3 { s[c] = (1.0 + 0.2 * s[c]).ln() / ln12; }
        }
        *px = s;
    });
    out
}

/// GPUImageHSBFilter adjustBrightness + adjustSaturation 的等价实现
pub fn hsb_brightness_saturation(img: &mut ImgF32, brightness: f32, saturation: f32) {
    img.data.par_iter_mut().for_each(|p| {
        let lum = 0.3086 * p[0] + 0.6094 * p[1] + 0.0820 * p[2];
        for c in 0..3 {
            let v = lum + (p[c] - lum) * saturation;
            p[c] = (v * brightness).clamp(0.0, 1.0);
        }
    });
}
```

#### 4.2.5 方案 A 组装

```rust
pub fn smooth_faithful(orig: &ImgF32, face_mask: Option<&[f32]>, p: &RetouchParams) -> ImgF32 {
    // 1. 工作副本：短边 720
    let short = orig.w.min(orig.h) as f32;
    let scale = (720.0 / short).min(1.0);
    let (ww, wh) = ((orig.w as f32 * scale).round() as usize, (orig.h as f32 * scale).round() as usize);
    let work = if scale < 1.0 { orig.resize(ww, wh) } else { orig.clone() };

    // 2. 双边 + 边缘（在工作副本上）
    let bl_w = bilateral_gpuimage(&work, 4.0, 4.0);
    let edge_w = sobel_edge(&work);

    // 3. 上采样
    let bl = if scale < 1.0 { bl_w.resize(orig.w, orig.h) } else { bl_w };
    let edge = if scale < 1.0 { resize_gray(&edge_w, ww, wh, orig.w, orig.h) } else { edge_w };

    // 4. 组合 + log + HSB
    let mut out = combine_bb(orig, &bl, &edge, face_mask, p.smooth, p.apply_log_curve);
    hsb_brightness_saturation(&mut out, p.brightness, p.saturation);
    out
}
```

> 注：`apply_log_curve`、`brightness`、`saturation` 在原实现中作用于**全图**。修图软件里建议默认保留（这正是"美颜感"的一部分），但提供关闭开关，因为它会改变背景。

### 4.3 磨皮方案 B：频率分离（静态修图推荐，来自 YUCIHighPassSkinSmoothing）

方案 A 是为 30fps 视频做的取舍（9 tap、颜色阈值肤色）。对静态照片，频率分离能保留毛孔纹理、只压平瑕疵。流程：

```
1. gb = GreenBlueChannelOverlay(src)         // 用 G/B 通道 overlay 自身，突出瑕疵
2. hp = HighPass(gb, radius)                  // hp = gb - gaussian(gb, radius) + 0.5
3. hp = HardLight(hp, hp) ×3                  // 提高遮罩对比
4. toned = ToneCurve(src, curve)              // 轻微提亮肤色的 RGB 曲线，控制点 (0,0) (120/255,146/255) (1,1)
5. out = mix(src, toned, hp) → 再 mix(src, out, amount)
6. out = Sharpen(out, sharpness)              // 轻微锐化找回质感
```

默认参数：`radius = 8 px（相对 ~1000 px 短边，按比例缩放）`，`amount = 0.75`，`sharpness = 0.6`。

Rust 实现只需要：高斯模糊（`imageproc::filter::gaussian_blur_f32` 或自写可分离高斯）、overlay/hardlight 混合、256 点曲线。建议在**同一个人脸羽化遮罩**内应用（4.4.2 节）。

> 两种方案共用 `SmoothMode` 开关；UI 上可命名"自然磨皮"（B）与"直播磨皮"（A）。

### 4.4 美白与皮肤遮罩

#### 4.4.1 美白（skin/whiten.rs）

美狐/gpupixel 都是"美白查找图（512 LUT）× intensity"。三种落地方式，按推荐顺序：

1. **自制 512 查找图**（推荐，效果最可控）：在 Photoshop/Capture One 中打开一张恒等查找图（4.5.4 节脚本生成），对它做你想要的"美白"调色（提亮中间调、轻微降红饱和），另存为 PNG。运行时用 4.5.1 节的采样函数，`intensity = whiten`。
2. **参数化曲线**（零素材）：`c' = pow(c, 1 / (1 + 0.35 * whiten))`，再把结果按皮肤遮罩混合。
3. 在 Lab 空间提 L、降 a（更准确但要写颜色空间转换）。

无论哪种，都乘以皮肤遮罩，否则背景一起发白。

#### 4.4.2 皮肤遮罩（skin/mask.rs）

```
mask_color(p) = is_skin_rgb(p) ? 1 : 0                 // 原实现的颜色规则
mask_face     = fill_polygon(轮廓 0..32 + 额头补点) → gaussian_blur(σ = 0.08 * ed) → clamp
额头补点：把轮廓点 0..32 中眉毛以上的部分沿 dir_up 镜像/外扩 0.6 * 脸高
mask = mask_color * mask_face
```

`fill_polygon` 用 `imageproc::drawing::draw_polygon_mut` 画到 `GrayImage` 再转 f32。多人脸取并集（max）。`restrict_to_face = false` 时 `mask_face ≡ 1`，退化为忠实模式。

### 4.5 风格滤镜 / LUT

#### 4.5.1 512×512 查找图（color/lookup512.rs）—— GPUImageLookupFilter 直译

```rust
use crate::buffer::ImgF32;
use rayon::prelude::*;

/// lut: 512×512 的 ImgF32（从 PNG 读入，第 0 行为顶部）
pub fn apply_lookup512(img: &mut ImgF32, lut: &ImgF32, intensity: f32) {
    debug_assert!(lut.w == 512 && lut.h == 512);
    img.data.par_iter_mut().for_each(|p| {
        let (r, g, b) = (p[0], p[1], p[2]);
        let blue = b * 63.0;
        let (b0, b1) = (blue.floor(), blue.ceil());
        let tex = |bi: f32| -> (f32, f32) {
            let qy = (bi / 8.0).floor();
            let qx = bi - qy * 8.0;
            let u = qx * 0.125 + 0.5 / 512.0 + (0.125 - 1.0 / 512.0) * r;
            let v = qy * 0.125 + 0.5 / 512.0 + (0.125 - 1.0 / 512.0) * g;
            (u * 512.0, v * 512.0) // 归一化 → 像素中心制
        };
        let (u0, v0) = tex(b0);
        let (u1, v1) = tex(b1);
        let c0 = lut.sample_bilinear(u0, v0);
        let c1 = lut.sample_bilinear(u1, v1);
        let t = blue - b0;
        for c in 0..3 {
            let n = c0[c] + (c1[c] - c0[c]) * t;
            p[c] += (n - p[c]) * intensity;
        }
    });
}
```

验证：用恒等查找图跑一遍，输出与输入最大差值应 ≤ 1/255。若出现整体色调反转，说明 PNG 行序与预期相反，对 `v` 做 `512 - v` 即可。

#### 4.5.2 .cube 3D LUT（color/lut3d.rs）

行业通用格式，方便设计师用 DaVinci / Lightroom 导出。

```rust
pub struct Lut3D {
    pub n: usize,
    pub data: Vec<[f32; 3]>, // index = r + n * (g + n * b)，r 变化最快（.cube 规范）
}

impl Lut3D {
    pub fn parse_cube(text: &str) -> Result<Self, String> {
        let mut n = 0usize;
        let mut data = Vec::new();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') || l.starts_with("TITLE") || l.starts_with("DOMAIN") { continue; }
            if let Some(rest) = l.strip_prefix("LUT_3D_SIZE") {
                n = rest.trim().parse().map_err(|_| "bad LUT_3D_SIZE")?;
                continue;
            }
            if l.starts_with("LUT_1D_SIZE") { return Err("1D LUT not supported".into()); }
            let v: Vec<f32> = l.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if v.len() == 3 { data.push([v[0], v[1], v[2]]); }
        }
        if n == 0 || data.len() != n * n * n { return Err("size mismatch".into()); }
        Ok(Self { n, data })
    }

    #[inline]
    fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.data[r + self.n * (g + self.n * b)]
    }

    pub fn lookup(&self, c: [f32; 3]) -> [f32; 3] {
        let m = (self.n - 1) as f32;
        let f = [c[0].clamp(0.0, 1.0) * m, c[1].clamp(0.0, 1.0) * m, c[2].clamp(0.0, 1.0) * m];
        let i0 = [f[0].floor() as usize, f[1].floor() as usize, f[2].floor() as usize];
        let i1 = [(i0[0] + 1).min(self.n - 1), (i0[1] + 1).min(self.n - 1), (i0[2] + 1).min(self.n - 1)];
        let t = [f[0] - i0[0] as f32, f[1] - i0[1] as f32, f[2] - i0[2] as f32];
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
        let c00 = lerp(self.at(i0[0], i0[1], i0[2]), self.at(i1[0], i0[1], i0[2]), t[0]);
        let c10 = lerp(self.at(i0[0], i1[1], i0[2]), self.at(i1[0], i1[1], i0[2]), t[0]);
        let c01 = lerp(self.at(i0[0], i0[1], i1[2]), self.at(i1[0], i0[1], i1[2]), t[0]);
        let c11 = lerp(self.at(i0[0], i1[1], i1[2]), self.at(i1[0], i1[1], i1[2]), t[0]);
        let c0 = lerp(c00, c10, t[1]);
        let c1 = lerp(c01, c11, t[1]);
        lerp(c0, c1, t[2])
    }
}

pub fn apply_lut3d(img: &mut ImgF32, lut: &Lut3D, intensity: f32) {
    use rayon::prelude::*;
    img.data.par_iter_mut().for_each(|p| {
        let n = lut.lookup(*p);
        for c in 0..3 { p[c] += (n[c] - p[c]) * intensity; }
    });
}
```

#### 4.5.3 美狐曲线滤镜包的处理：离线烘焙

`fragment.glsl` 里的运算分两类：
- **纯颜色映射**（曲线查表、固定色 overlay/screen、通道混合）→ 与像素位置无关 → 可以对 33³ 网格逐点求值，导出 `.cube`。
- **空间遮罩**（`mask.png` 拉伸后加权，即暗角）→ 保留为运行时 op：`out = mix(colorPart(src), src, mask(x,y) * k)`。

`tools/bake_lut.py` 做法：用 numpy 手工翻译某款 GLSL 的颜色部分（每款 20–60 行），输入 33³ 网格，输出 .cube。运行时 Rust 只需 4.5.2 的 LUT + 一个可选暗角遮罩。这样把"20 多款 GLSL 的运行时解释器"这个大坑变成了"一次性离线脚本"，可行性大幅提高。

> 合规提醒：烘焙自美狐贴图的 LUT 是其衍生物，只可用于个人学习；商用版本请按 4.5.4 自制。

#### 4.5.4 自制 LUT 的标准流程（设计师侧）

1. 生成恒等 512 查找图或恒等 .cube（10 行 Python 即可：`r = x % 64 / 63, g = y % 64 / 63, b = (y/64*8 + x/64) / 63`）。
2. 在 Photoshop / Lightroom / Capture One 中把调色（曲线、色相、分离色调）应用在**恒等图**上，不要用任何空间滤镜（模糊、锐化、暗角）。
3. 导出 PNG（512 查找图）或用 DaVinci 导出 .cube。
4. 放入 `luts/`，配 `luts/index.json`（名称、缩略图、默认强度）。

这与 gpupixel、美狐、Instagram 系滤镜的制作方式完全一致，效果上限只取决于设计师。

### 4.6 人脸检测与关键点

#### 4.6.1 语义关键点结构体（face/semantic.rs）

```rust
use crate::geom::P;

#[derive(Clone, Debug)]
pub struct FaceKeyPoints {
    pub pupil_l: P,          // 图像左侧眼的瞳孔中心（Face++ 74）
    pub pupil_r: P,          // 图像右侧（77）
    pub eye_outer_l: P,      // 左眼外眼角（52）
    pub eye_inner_l: P,      // 左眼内眼角（55）
    pub eye_inner_r: P,      // 右眼内眼角（58）
    pub eye_outer_r: P,      // 右眼外眼角（61）
    pub nose_bridge_top: P,  // 鼻梁顶点（43）
    pub nose_tip: P,         // 鼻尖（46）
    pub nostril_l: P,        // 左鼻孔（48）
    pub nostril_r: P,        // 右鼻孔（50）
    pub nose_wing_l: P,      // 左鼻翼（82）
    pub nose_wing_r: P,      // 右鼻翼（83）
    pub chin: P,             // 下巴尖（16）
    pub jaw_l: [P; 3],       // 左侧 [4, 9, 13]
    pub jaw_r: [P; 3],       // 右侧 [28, 23, 19]
    pub contour: Vec<P>,     // 完整轮廓（用于皮肤遮罩），左太阳穴 → 下巴 → 右太阳穴
    pub yaw_deg: f32,        // 粗略偏航角，用于大角度时降低/关闭形变
}

pub trait LandmarkModel: Send {
    fn detect(&mut self, img: &image::RgbImage, face_box: &FaceBox) -> anyhow::Result<FaceKeyPoints>;
}

#[derive(Clone, Copy, Debug)]
pub struct FaceBox { pub x1: f32, pub y1: f32, pub x2: f32, pub y2: f32, pub score: f32 }
```

#### 4.6.2 人脸检测：UltraFace RFB-320（face/detector.rs）

- 模型：`version-RFB-320.onnx`（MIT）。
- 输入：`1×3×240×320`，RGB，`(x − 127) / 128`。
- 输出：`scores 1×4420×2`（第 2 列为人脸概率）、`boxes 1×4420×4`（归一化 x1 y1 x2 y2）。
- 后处理：阈值 0.7 → NMS（IoU 0.3）→ 还原到原图像素 → 按面积降序。
- 大图先缩到长边 ≤ 1600 再检测即可，人脸框回乘比例。

ort 2.x 调用骨架（API 以你使用的 ort 版本文档为准）：

```rust
use ort::session::Session;

pub struct UltraFace { session: Session }

impl UltraFace {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?.with_intra_threads(4)?.commit_from_file(path)?;
        Ok(Self { session })
    }

    pub fn detect(&mut self, img: &image::RgbImage) -> anyhow::Result<Vec<FaceBox>> {
        let resized = image::imageops::resize(img, 320, 240, image::imageops::FilterType::Triangle);
        let mut input = ndarray::Array4::<f32>::zeros((1, 3, 240, 320));
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] = (p[c] as f32 - 127.0) / 128.0;
            }
        }
        let outputs = self.session.run(ort::inputs!["input" => input.view()]?)?;
        let scores = outputs["scores"].try_extract_tensor::<f32>()?; // [1, N, 2]
        let boxes  = outputs["boxes"].try_extract_tensor::<f32>()?;  // [1, N, 4]
        let (w, h) = (img.width() as f32, img.height() as f32);
        let mut cands = Vec::new();
        for i in 0..scores.shape()[1] {
            let s = scores[[0, i, 1]];
            if s < 0.7 { continue; }
            cands.push(FaceBox {
                x1: boxes[[0, i, 0]] * w, y1: boxes[[0, i, 1]] * h,
                x2: boxes[[0, i, 2]] * w, y2: boxes[[0, i, 3]] * h, score: s,
            });
        }
        Ok(nms(cands, 0.3))
    }
}
```

#### 4.6.3 关键点（开发期）：InsightFace 2d106det（face/lm_2d106.rs）

来源：insightface `buffalo_l` 模型包中的 `2d106det.onnx`。前后处理与官方 `model_zoo/landmark.py` 一致：

```
输入尺寸 192×192，RGB，像素值 0..255（mean=0, std=1，不做归一化）
裁剪：center = 人脸框中心；scale = 192 / (max(w,h) * 1.5)；rotate = 0
      M = 平移到中心 + 缩放 scale（2×3 仿射），aimg = warpAffine(img, M, 192×192)
输出：pred[212] → reshape (106, 2)
      pred = (pred + 1) * 96          // 映射到 192 像素坐标
      pts  = inverse(M) · pred         // 回到原图像素坐标
```

语义映射：2d106det 的 106 点**顺序与 Face++ 不同**（其下颌轮廓不是 0..32 顺序）。做法：
1. 用 `debug.rs` 把索引画在样张上（`cargo run --example dump_landmarks -- photo.jpg`）。
2. 对照 2.4 节的语义表逐项填 `mapping_2d106.rs` 中的常量。
3. 用附录 A 的恒等性检查（瞳距、眼宽、下巴位置）验证。

这一步是**必须做且只做一次**的人工工作，约 1 小时。

#### 4.6.4 关键点（发布期）：MediaPipe Face Mesh（face/lm_facemesh.rs）

- ONNX 来源：PINTO0309/PINTO_model_zoo `032_FaceMesh`（含带虹膜的 attention 版本，478 点）。
- 输入 192×192，RGB 0..1；需先按人脸框做 1.5× 方形裁剪，并按两眼连线旋转对齐（MediaPipe 的做法，否则侧脸精度下降）。
- 输出 468×3 或 478×3（192 像素坐标）→ 反变换回原图。
- 语义映射候选索引见附录 A（以可视化校验为准）。

两个模型都实现 `LandmarkModel`，`EngineConfig` 里切换，上层零改动。

### 4.7 形变：瘦脸 / 大眼 / 瘦鼻（像素空间移植）

#### 4.7.1 为什么改成像素空间

原 shader 在归一化纹理坐标里用 `aspect_ratio` 反复修正 y、`res_ratio` 修正 x，是因为纹理坐标 x、y 尺度不同。我们在像素空间实现，x、y 天然同尺度，所有修正项消失，公式更短且**数学等价**（半径与位移都以像素为单位，由瞳距 `ed` 换算）。

#### 4.7.2 几何原语（geom.rs）

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct P { pub x: f32, pub y: f32 }

impl P {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn sub(self, o: P) -> P { P::new(self.x - o.x, self.y - o.y) }
    pub fn add(self, o: P) -> P { P::new(self.x + o.x, self.y + o.y) }
    pub fn mul(self, k: f32) -> P { P::new(self.x * k, self.y * k) }
    pub fn len(self) -> f32 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn dist(self, o: P) -> f32 { self.sub(o).len() }
    pub fn norm(self) -> P { let l = self.len(); if l > 1e-6 { self.mul(1.0 / l) } else { P::new(0.0, 0.0) } }
}
```

#### 4.7.3 形变步骤（warp/face_warp.rs）

```rust
use crate::geom::P;
use crate::face::semantic::FaceKeyPoints;

#[derive(Clone, Copy, Debug)]
pub enum Step {
    /// warpPositionToUse1：以 a 为圆心、半径 radius，把采样点沿 a→b 方向反向偏移 delta
    Pinch { a: P, b: P, radius: f32, delta: f32 },
    /// adjust_eye：以 center 为圆心的径向幂次缩放
    Enlarge { center: P, radius: f32, k: f32 },
}

#[derive(Clone, Debug)]
pub struct FaceWarp {
    pub steps: Vec<Step>,
    pub bbox: (P, P), // 所有影响圆的包围盒，用于跳过无关像素
}

pub struct WarpParams { pub thin_face: f32, pub big_eye: f32, pub thin_nose: f32 }

impl FaceWarp {
    pub fn from_face(f: &FaceKeyPoints, p: &WarpParams) -> Self {
        let ed = f.pupil_l.dist(f.pupil_r);              // 瞳距（像素）
        let dir_right = f.pupil_r.sub(f.pupil_l).norm();  // 图像右方向
        let dir_up = f.nose_bridge_top.sub(f.chin).norm();
        let mut steps = Vec::new();

        // ---- 瘦脸：与 adjust_thinFace 逐项对应 ----
        if p.thin_face > 0.0 {
            let offs = [0.13f32, 0.33, 0.33];
            let x = std::f32::consts::PI / 25.0;
            let scale = 2.0 * ed;
            let radius = 0.4 * scale;
            let sins = [x.sin(), (2.0 * x).sin(), (2.0 * x).sin()];
            for i in 0..3 {
                let delta = sins[i] * p.thin_face * 0.15 * scale;
                let l = f.jaw_l[i].sub(dir_right.mul(ed * offs[i]));
                let r = f.jaw_r[i].add(dir_right.mul(ed * offs[i]));
                steps.push(Step::Pinch { a: l, b: r, radius, delta });
                steps.push(Step::Pinch { a: r, b: l, radius, delta });
            }
        }

        // ---- 大眼：与 adjust_eye 逐项对应 ----
        if p.big_eye > 0.0 {
            let radius = f.eye_outer_l.dist(f.eye_inner_l); // 原实现只用左眼宽度
            let k = p.big_eye * 0.24;
            steps.push(Step::Enlarge { center: f.pupil_l, radius, k });
            steps.push(Step::Enlarge { center: f.pupil_r, radius, k });
        }

        // ---- 瘦鼻：原实现常数与分辨率耦合，这里改为以 ed 为尺度的等效值 ----
        // 等效换算（按 720×1280、瞳距≈0.12 宽推导）：radius≈1.3ed，锚点上移≈1.3ed，
        // 位移≈0.04ed·slider（原 setter 已把 slider×2）。系数需在样张上微调。
        if p.thin_nose > 0.0 {
            let radius = 1.3 * ed;
            let delta = 0.04 * ed * p.thin_nose;
            let lift = dir_up.mul(1.3 * ed);
            let a = [f.nostril_l.add(lift), f.nose_wing_l];
            let b = [f.nostril_r.add(lift), f.nose_wing_r];
            for i in 0..2 {
                steps.push(Step::Pinch { a: a[i], b: b[i], radius, delta });
                steps.push(Step::Pinch { a: b[i], b: a[i], radius, delta });
            }
        }

        let bbox = compute_bbox(&steps);
        Self { steps, bbox }
    }

    /// 反向映射：输出像素位置 → 应从原图采样的位置
    #[inline]
    pub fn map(&self, mut pos: P) -> P {
        for s in &self.steps {
            pos = match *s {
                Step::Pinch { a, b, radius, delta } => pinch(pos, a, b, radius, delta),
                Step::Enlarge { center, radius, k } => enlarge(pos, center, radius, k),
            };
        }
        pos
    }
}

#[inline]
pub fn pinch(cur: P, a: P, b: P, radius: f32, delta: f32) -> P {
    let r = cur.dist(a);
    if r >= radius { return cur; }
    let dir = b.sub(a).norm();
    let d2 = radius * radius - r * r;
    let alpha = d2 / (d2 + (r - delta) * (r - delta));
    let alpha = alpha * alpha;
    cur.sub(dir.mul(alpha * delta))
}

#[inline]
pub fn enlarge(cur: P, center: P, radius: f32, k: f32) -> P {
    let d = cur.dist(center);
    if d >= radius { return cur; }
    let w = (d.max(1.0) / radius).powf(k); // 原实现用 (d+0.01)/radius 规避 pow(0)，像素空间用 1px 下限
    center.add(cur.sub(center).mul(w))
}

fn compute_bbox(steps: &[Step]) -> (P, P) {
    let mut mn = P::new(f32::MAX, f32::MAX);
    let mut mx = P::new(f32::MIN, f32::MIN);
    for s in steps {
        let (c, r) = match *s {
            Step::Pinch { a, radius, delta, .. } => (a, radius + delta.abs()),
            Step::Enlarge { center, radius, .. } => (center, radius),
        };
        mn = P::new(mn.x.min(c.x - r), mn.y.min(c.y - r));
        mx = P::new(mx.x.max(c.x + r), mx.y.max(c.y + r));
    }
    (mn, mx)
}
```

#### 4.7.4 应用形变（warp/apply.rs）

```rust
use crate::buffer::ImgF32;
use crate::geom::P;
use crate::warp::face_warp::FaceWarp;
use rayon::prelude::*;

pub fn apply_warps(src: &ImgF32, warps: &[FaceWarp]) -> ImgF32 {
    if warps.is_empty() { return src.clone(); }
    let mut out = src.clone();
    // 所有脸的包围盒并集
    let (mut x0, mut y0, mut x1, mut y1) = (src.w as f32, src.h as f32, 0.0f32, 0.0f32);
    for w in warps {
        x0 = x0.min(w.bbox.0.x); y0 = y0.min(w.bbox.0.y);
        x1 = x1.max(w.bbox.1.x); y1 = y1.max(w.bbox.1.y);
    }
    let (xs, ys) = ((x0.floor().max(0.0)) as usize, (y0.floor().max(0.0)) as usize);
    let (xe, ye) = ((x1.ceil() as usize + 1).min(src.w), (y1.ceil() as usize + 1).min(src.h));
    if xs >= xe || ys >= ye { return out; }

    let width = src.w;
    out.data[ys * width..ye * width]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(dy, row)| {
            let y = ys + dy;
            for x in xs..xe {
                let p = P::new(x as f32 + 0.5, y as f32 + 0.5);
                let mut q = p;
                for w in warps { q = w.map(q); }
                if q != p {
                    row[x] = src.sample_bilinear(q.x, q.y);
                }
            }
        });
    out
}
```

多张脸：各自生成 `FaceWarp`，`map` 依次叠加；两脸重叠时后者覆盖前者，与原实现单脸行为一致。`yaw_deg > 35°` 时建议把 `thin_face`、`thin_nose` 乘以衰减系数（0 ~ 0.3），避免侧脸畸变（原实现没有这一保护）。

### 4.8 流水线编排（pipeline.rs）

```rust
pub fn retouch_impl(input: &image::RgbImage, faces: &[FaceKeyPoints], p: &RetouchParams) -> image::RgbImage {
    let orig = ImgF32::from_rgb8(input);

    // 1. 皮肤遮罩（可选）
    let face_mask = if p.restrict_to_face && !faces.is_empty() {
        Some(skin::mask::face_mask(&orig, faces))
    } else { None };

    // 2. 磨皮 + 提亮/饱和
    let mut img = match p.smooth_mode {
        SmoothMode::Faithful => skin::smooth_faithful(&orig, face_mask.as_deref(), p),
        SmoothMode::FreqSep  => skin::smooth_freqsep(&orig, face_mask.as_deref(), p),
    };

    // 3. 美白
    if p.whiten > 0.0 {
        skin::whiten::apply(&mut img, face_mask.as_deref(), p.whiten);
    }

    // 4. 形变
    if !faces.is_empty() && (p.thin_face > 0.0 || p.big_eye > 0.0 || p.thin_nose > 0.0) {
        let wp = WarpParams { thin_face: p.thin_face, big_eye: p.big_eye, thin_nose: p.thin_nose };
        let warps: Vec<FaceWarp> = faces.iter().map(|f| FaceWarp::from_face(f, &wp)).collect();
        img = warp::apply_warps(&img, &warps);
    }

    // 5. 风格
    match (&p.style, p.style_intensity) {
        (Some(StyleFilter::Lookup512(l)), k) if k > 0.0 => color::apply_lookup512(&mut img, l, k),
        (Some(StyleFilter::Cube(l)), k) if k > 0.0 => color::apply_lut3d(&mut img, l, k),
        _ => {}
    }

    img.to_rgb8()
}
```

**预览策略**：UI 层持有一份长边 1280 的缩略图与按同比例缩放后的 `FaceKeyPoints`（坐标乘以比例即可），滑块变化时对缩略图调 `retouch`；点击导出时对原图调同一函数。因为所有参数以 `ed` 与短边 720 工作副本为尺度，两者观感一致。

**缓存**：`smooth_faithful` 的双边/边缘结果只依赖原图，不依赖 `smooth` 滑块，可缓存；改 `smooth` 时只重跑 `combine_bb` 以后的步骤（12MP 约 150 ms）。
---

## 5. 依赖清单

```toml
[package]
name = "portrait-retouch"
edition = "2021"

[dependencies]
image      = "0.25"      # 解码/编码、RgbImage
imageproc  = "0.25"      # 高斯模糊、多边形填充（皮肤遮罩、方案 B）
rayon      = "1"         # 并行
ndarray    = "0.16"      # ONNX 输入输出张量
ort        = "2"         # ONNX Runtime 绑定（关键点/检测）；按其文档选 rc 版本与 execution provider
serde      = { version = "1", features = ["derive"] }
serde_json = "1"         # luts/index.json
anyhow     = "1"
thiserror  = "1"

[dev-dependencies]
criterion  = "0.5"       # 性能基准

[features]
default = []
gpu = ["dep:wgpu"]       # 阶段 6 可选
```

版本号以 crates.io 当前稳定版为准；`ort` 2.x 的 API（`Session::builder`、`inputs!`、`try_extract_tensor`）在 rc 之间有小改动，以你锁定版本的文档为准。

模型文件（不入库，`build.rs` 或首次运行下载并校验 SHA256）：

| 文件 | 来源 | 大小 | 许可证 |
|---|---|---|---|
| `version-RFB-320.onnx` | Ultra-Light-Fast-Generic-Face-Detector-1MB 仓库 `models/onnx/` | ~1.2 MB | MIT |
| `2d106det.onnx` | insightface `buffalo_l.zip` | ~5 MB | 非商用 |
| `face_mesh.onnx` | PINTO_model_zoo 032_FaceMesh | ~2.5 MB | Apache-2.0 |

---

## 6. 实施计划与验收标准

单人全职估算；有 Rust 图像处理经验可压缩 30%。

| 阶段 | 内容 | 交付物 | 验收标准 | 工期 |
|---|---|---|---|---|
| **P0 骨架** | crate 结构、`ImgF32`、双线性采样、缩放、512 查找图、.cube、恒等测试 | 可加载任意 LUT 并应用到图片的 CLI | 恒等 LUT 最大误差 ≤ 1/255；12MP 应用 LUT < 100 ms | 3 天 |
| **P1 磨皮 A** | 双边、Sobel、组合、log、HSB、工作副本策略、缓存 | `--smooth 0.5` 可出图 | 与 numpy oracle（7.2 节）逐像素误差 ≤ 2/255；12MP < 800 ms | 4 天 |
| **P2 检测 + 关键点** | UltraFace 集成、2d106det 集成、语义映射、可视化工具 | `dump_landmarks` 示例程序 | 100 张样张（正脸/侧脸/多人/戴眼镜）人工核对，语义点定位正确率 ≥ 95% | 5 天 |
| **P3 形变** | pinch/enlarge、步骤生成、并行应用、侧脸衰减 | 瘦脸/大眼/瘦鼻滑块可用 | 强度 0 时输出 == 输入（逐位相等）；强度 1 时无撕裂/黑边；12MP < 300 ms | 4 天 |
| **P4 遮罩 + 美白 + 磨皮 B** | 人脸多边形羽化、美白 LUT/曲线、频率分离 | 两种磨皮模式切换 | 背景像素在 `restrict_to_face=true` 时逐位不变；B 模式 PSNR 对比 A 在皮肤区更高细节保留（人工评审） | 5 天 |
| **P5 集成** | 接入现有软件 UI、预览/导出双分辨率、参数持久化、LUT 索引、错误处理 | 功能上线 | 1280 预览滑块响应 < 200 ms；导出与预览观感一致（PSNR ≥ 40 dB，缩放后对比） | 5 天 |
| **P6 发布模型切换（商用前）** | Face Mesh 集成、语义映射、回归 | 可切换 `LandmarkModel` | 同 P2 验收；与 2d106det 结果在正脸样张上瞳距误差 < 3% | 4 天 |
| **P7 可选 GPU** | wgpu 版双边/组合/LUT/形变（WGSL） | `--features gpu` | 结果与 CPU 版误差 ≤ 2/255；1080p 实时 > 30 fps | 2 周 |

**总计：P0–P6 约 30 个工作日（6 周）。** P7 仅在需要实时预览或视频时投入。

每阶段结束都应有：单元测试、golden 图更新、CHANGELOG 记录参数变更。

---

## 7. 验证与测试方案

### 7.1 恒等性测试（tests/identity.rs）

| 测试 | 断言 |
|---|---|
| 恒等 512 查找图 | `max|out − in| ≤ 1/255` |
| 恒等 .cube（N=33） | 同上 |
| `smooth = 1.0` 且无 log/HSB | 肤色区 == 双边结果，非肤色区 == 原图 |
| 所有形变强度 = 0 | 输出与输入逐字节相等 |
| `pinch(delta = 0)` | 返回原坐标 |
| `enlarge(k = 0)` | 返回原坐标 |
| 无人脸 | `retouch` 不 panic，形变跳过 |

### 7.2 数值对照（tools/oracle_numpy.py）

把 4.2 节的公式用 numpy 独立实现一遍（不到 60 行），对同一输入比较 Rust 输出：

```python
import numpy as np
def combine_bb(origin, bilateral, edge, degree):
    r, g, b = origin[...,0], origin[...,1], origin[...,2]
    mx, mn = origin.max(-1), origin.min(-1)
    skin = (edge < 0.2) & (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) \
         & ((mx - mn) > 0.0588) & (np.abs(r - g) > 0.0588)
    smooth = np.where(skin[...,None], bilateral + (1 - degree) * (origin - bilateral), origin)
    return np.log(1 + 0.2 * smooth) / np.log(1.2)
```

两个实现同时出错的概率极低，误差 > 2/255 即定位问题。同样方法覆盖 HSB、512 查找、pinch。

### 7.3 视觉回归（tests/golden.rs）

- 固定 12 张样张 × 6 组参数 = 72 张 golden 图，存 LFS。
- 每次 CI 计算 PSNR，< 45 dB 判失败并输出差分图。
- 参数刻意变更时用 `UPDATE_GOLDEN=1` 重新生成并在 PR 中附对比图。

### 7.4 关键点核对工具（examples/dump_landmarks.rs）

输出带索引编号的可视化图 + JSON。P2 与 P6 的映射工作全靠它。

### 7.5 性能基准（benches/）

| 用例 | 目标（8 核桌面 CPU） |
|---|---|
| 12MP 全流程（A 模式 + 形变 + LUT） | < 2.0 s |
| 12MP 仅改 `smooth`（命中缓存） | < 300 ms |
| 1280 长边预览全流程 | < 200 ms |
| 检测 + 关键点（单脸） | < 120 ms |

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 2d106det 索引映射填错 | 瘦脸方向错、大眼偏心 | P2 必做可视化核对；附录 A 的恒等性检查写成单元测试 |
| 关键点模型与 Face++ 定位差异 | 同样强度下脸型变化幅度略不同 | 系数（0.13/0.33/0.24 等）留在配置文件里，按 20 张样张微调一次 |
| 侧脸/大角度 | 形变畸变 | `yaw_deg` 衰减；UI 提示"侧脸时效果减弱" |
| 高分辨率下磨皮"变弱/变强" | 观感不一致 | 严格执行"短边 720 工作副本"策略；不要在全分辨率上直接跑 9 tap |
| ort 版本 API 变动 | 编译失败 | `Cargo.lock` 锁定；封装在 `face/` 内，上层不直接依赖 ort |
| 多人脸重叠 | 形变互相覆盖 | 按面积排序，小脸先应用；重叠 > 30% 时只处理最大脸 |
| 曲线滤镜包烘焙不准 | 风格偏差 | 烘焙脚本对比 numpy 直译 GLSL 的结果，误差 ≤ 2/255 |
| 内存（12MP × f32 × 3 通道 ≈ 144 MB/份） | 峰值 ~600 MB | 工作副本策略已把双边/边缘压到 ~1MP；必要时用 f16 或分块 |

---

## 9. 许可证合规清单

> 本节为工程实践建议，不构成法律意见；商用前请法务确认。

| 项目 | 做法 |
|---|---|
| Meihu-Beautyface-sdk（GPL-3.0 + 非商用声明） | **只提取参数与公式**（本文已完成），不复制任何代码行、不打包任何 PNG/zip 资源、不烘焙其贴图用于商用版本 |
| gpupixel（Apache-2.0） | 可参考其 shader 逻辑重写；若直接复制代码段，保留 NOTICE 与版权声明 |
| GPUImage（BSD-3） | 双边/Sobel/HSB/查找表公式属于通用算法；若复制其代码需保留 BSD 声明 |
| YUCIHighPassSkinSmoothing（MIT） | 保留 MIT 声明即可 |
| UltraFace 模型（MIT） | 保留声明 |
| InsightFace 2d106det | **仅开发/内测**；商用版本必须切换到 Face Mesh 或自训练模型；CI 中加检查禁止该文件进入发布包 |
| MediaPipe Face Mesh（Apache-2.0） | 保留 NOTICE |
| 自制 LUT | 你自己拥有版权；若委托设计师制作，合同中明确版权归属 |

发布包内附 `THIRD_PARTY_LICENSES.md` 汇总以上声明。

---

## 附录 A 关键点语义映射表

"图像左"指画面左侧（非镜像时为受试者右脸）。Face Mesh 列为候选索引，**必须**用 7.4 节工具可视化确认后再固化。

| 语义字段 | Face++ 106（美狐用） | 2d106det | Face Mesh 468/478（候选） |
|---|---|---|---|
| `pupil_l` | 74 | P2 核对填写 | 468（attention 版虹膜中心）；无虹膜版取眼轮廓均值 |
| `pupil_r` | 77 | 〃 | 473 |
| `eye_outer_l` | 52 | 〃 | 33 |
| `eye_inner_l` | 55 | 〃 | 133 |
| `eye_inner_r` | 58 | 〃 | 362 |
| `eye_outer_r` | 61 | 〃 | 263 |
| `nose_bridge_top` | 43 | 〃 | 168 |
| `nose_tip` | 46 | 〃 | 1 |
| `nostril_l` / `nostril_r` | 48 / 50 | 〃 | 98 / 327 |
| `nose_wing_l` / `nose_wing_r` | 82 / 83 | 〃 | 129 / 358 |
| `chin` | 16 | 〃 | 152 |
| `jaw_l[0..3]`（颧下 / 中下颌 / 下颌角） | 4 / 9 / 13 | 〃 | 132 / 172 / 150 |
| `jaw_r[0..3]` | 28 / 23 / 19 | 〃 | 361 / 397 / 379 |
| `contour`（左太阳穴→下巴→右太阳穴） | 0..32 | 〃 | 234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454 |

**映射正确性的自动检查**（写成单元测试，任何模型都必须通过）：
1. `pupil_r.x > pupil_l.x`（未镜像输入）。
2. `eye_outer_l.x < eye_inner_l.x < eye_inner_r.x < eye_outer_r.x`。
3. `chin.y > nose_tip.y > nose_bridge_top.y`。
4. `0.8 < |eye_outer_l − eye_inner_l| / |eye_outer_r − eye_inner_r| < 1.25`（正脸样张）。
5. `jaw_l[i].x < chin.x < jaw_r[i].x` 且 `jaw_l[0].y < jaw_l[1].y < jaw_l[2].y`。
6. `1.5 < |pupil_l − pupil_r| / |eye_outer_l − eye_inner_l| < 3.0`。

## 附录 B UI 参数映射表

| UI 滑块 | 范围 | 内部参数 | 美狐对应 | 默认 |
|---|---|---|---|---|
| 磨皮 | 0–100 | `smooth = v/100` | `smoothDegree` | 50 |
| 亮度 | 0–100 | `brightness = 1.0 + 0.2*(v/100)` | `hsb brightness`（1.1 ⇔ 50） | 50 |
| 饱和度 | 0–100 | `saturation = 1.0 + 0.2*(v/100)` | `hsb saturation`（1.1 ⇔ 50） | 50 |
| 美白 | 0–100 | `whiten = v/100` | `white.png` intensity | 0 |
| 瘦脸 | 0–100 | `thin_face = v/100` | `thin_face_param` | 0 |
| 大眼 | 0–100 | `big_eye = v/100` | `eye_param` | 0 |
| 瘦鼻 | 0–100 | `thin_nose = v/100` | `nose_param`（setter 内 ×2 已折入系数） | 0 |
| 滤镜强度 | 0–100 | `style_intensity = v/100` | `GLImageLutFilter.intensity` | 80 |

## 附录 C 参考链接

- 美狐开源版（还原对象）：https://github.com/zhanghao5683934/Meihu-Beautyface-sdk
- gpupixel：https://github.com/pixpark/gpupixel
- YUCIHighPassSkinSmoothing：https://github.com/YuAo/YUCIHighPassSkinSmoothing
- GPUImage：https://github.com/BradLarson/GPUImage
- UltraFace：https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
- InsightFace：https://github.com/deepinsight/insightface
- MediaPipe：https://github.com/google-ai-edge/mediapipe
- PINTO_model_zoo（Face Mesh ONNX）：https://github.com/PINTO0309/PINTO_model_zoo
- ort：https://github.com/pykeio/ort
- image / imageproc：https://github.com/image-rs/image · https://github.com/image-rs/imageproc
- wgpu：https://github.com/gfx-rs/wgpu
