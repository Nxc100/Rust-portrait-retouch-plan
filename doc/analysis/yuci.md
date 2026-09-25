# YuAo/YUCIHighPassSkinSmoothing 解读（MIT，磨皮方案 B）

仓库：`example/YUCIHighPassSkinSmoothing/Sources`。Core Image 实现，默认参数 `inputAmount = 0.75`、
`inputRadius = 8.0`、`inputSharpnessFactor = 0.6`、曲线控制点 `(0,0) (120/255, 146/255) (1,1)`。

## 完整流程（`YUCIHighPassSkinSmoothing.m::outputImage`）

```
mask = MaskGenerator(src, radius):
    e   = CIExposureAdjust(src, EV = -1)                     // 像素 × 0.5
    gb  = GreenBlueChannelOverlayBlend(e)                    // 见下：退化为 2·G·B
    hp  = HighPass(gb, radius) = gb − CIGaussianBlur(gb, radius) + 0.5   // 边界 clampToExtent
    m   = MaskBoost(hp)：对 b 通道做 3 次 hardlight 自混合，再 (x − 75/255) · 255/(164 − 75)
toned = YUCIRGBToneCurve(src, 复合曲线, intensity = amount)   // mix(src, curve(src), amount)
out   = CIBlendWithMask(inputImage = src, background = toned, mask = m)   // mask 白 → src，黑 → toned
out   = CISharpenLuminance(out, sharpness = 0.6 · amount)
```

### YUCIGreenBlueChannelOverlayBlend.cikernel

```glsl
base = vec4(g,g,g,1); overlay = vec4(b,b,b,1);
ba = 2·overlay.b·base.b + overlay.b·(1 − base.a) + base.b·(1 − overlay.a);   // alpha=1 → 2·G·B
```

### YUCIHighPassSkinSmoothingMaskBoost.cikernel

```glsl
h = image.b;
for 3 次: h = h < 0.5 ? 2h² : 1 − 2(1−h)²;
h = (h − 75/255) · 255/89;
```

### YUCIRGBToneCurve（Vivid pod）

与 GPUImageToneCurveFilter 相同：控制点排序 → 转 0..255 → 自然三次样条（`secondDerivative`）→
首点前补 0、末点后补 255 → 每个索引存偏移量 → 组合 `clamp(x + composite[x])`。→ `src/color/curve.rs`。

## 与方案 4.3 节概述的差异

- 方案写 "hp = HardLight(hp, hp) ×3" 之后直接 mix；源码在 hardlight 后还有色阶拉伸，且前置了 −1EV 曝光。
- 方案的 "mix(src, toned, hp)" 方向相反：mask 高处保留原图（高光 / 边缘），低处取提亮后的 toned。
- "再 mix(src, out, amount)" 实际是曲线内部的 intensity 与锐化系数中的 amount，没有整图二次混合。

## 本项目实现（`src/skin/smooth_freqsep.rs`）

- mask 在短边 1000 的工作副本上计算（radius 8 px 对应 ~1000 px 短边），上采样到全分辨率；
- toned / blend 在全分辨率逐像素进行；锐化用 σ = 1 px 的高斯 unsharp mask 近似 `CISharpenLuminance`；
- 额外乘以人脸羽化遮罩（`restrict_to_face`）。
