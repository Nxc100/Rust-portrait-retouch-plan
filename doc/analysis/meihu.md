# Meihu-Beautyface-sdk 源码解读

仓库：`example/Meihu-Beautyface-sdk`（GPL-3.0 + 非商用声明；只提取参数与公式，不复制代码与资源）。

## 1. Demo 实际运行链路（`MHOpenDemo/MHOpenSourceViewController.m`）

```
GPUImageVideoCamera(1280x720 前置, 镜像) ─► BBGPUImageBeautifyFilter ─► GLImageFaceChangeFilterGroup ─► GPUImageView
                                             intensity 初值 0.0            (GLImageFaceChangeFilter + 关键点显示 + ScreenBlend)
```

- `setupFaceConfig` 中还创建了 `GPUImageVignetteFilter(0.3, 0.75)` 并 `addTarget:beautifyFilter`，但相机只 addTarget 到
  beautifyFilter，暗角滤镜没有输入源，实际不生效。
- Face++ 以 `MGFppDetectionModeTrackingFast`、`orientation = 90`、106 点（`GetGetLandmark ... pointsNumber:106`）输出，
  多脸时 `setFacePointsArray` 被循环覆盖，只保留最后一张脸。
- 未授权（`faceServiceBool == NO`）时只有 `BBGPUImageBeautifyFilter`。

### 滑块映射（`MHMeiyanMenusView.m` / `MHBeautyAssembleView.m`）

| 菜单 | 滑块范围 | 传入值 | 生效 |
|---|---|---|---|
| 磨皮 | 0–9 | `level = v/9` | `beautifyFilter.intensity = level`（smoothDegree） |
| 美白 | 0–9 | `level = v/9` | `beautifyFilter.brightness = 1 + level/5`（1..1.2） |
| 红润 | 0–9 | `level = v/9` | `beautifyFilter.saturation = 1 + level`（1..2） |
| 瘦脸 | 0–100 | `v` | `thinFaceParam = v/100` |
| 大眼 | 0–100 | `v` | `eyeParam = v/100` |
| 瘦鼻 | — | — | Demo 菜单未暴露；`setNoseParam` 内部 `×2` 后传给 shader |

## 2. `BBGPUImageBeautifyFilter.m`（来源 Bonway/BBGPUImage）

四个 pass，全部在同一分辨率（720×1280）上：

| Pass | 类 | 参数 |
|---|---|---|
| 1 | `GPUImageBilateralFilter` | `distanceNormalizationFactor = 4.0`（GPUImage 默认 8.0），`texelSpacingMultiplier = 4.0`（默认） |
| 2 | `GPUImageSobelEdgeDetectionFilter` | `edgeStrength = 1.0`，先灰度（0.2125, 0.7154, 0.0721）再 3×3 Sobel |
| 3 | `GPUImageCombinationFilter`（三输入：双边、边缘、原图） | `smoothDegree = intensity`，初值 0.5 |
| 4 | `GPUImageHSBFilter` | `adjustBrightness:1.1` 再 `adjustSaturation:1.1` |

Pass 3 shader 逐句（`kGPUImageBeautifyFragmentShaderString`）：

```glsl
if (canny.r < 0.2 && r > 0.3725 && g > 0.1568 && b > 0.0784 && r > b
    && (max(max(r,g),b) - min(min(r,g),b)) > 0.0588 && abs(r-g) > 0.0588)
    smooth = (1.0 - smoothDegree) * (origin - bilateral) + bilateral;
else
    smooth = origin;
smooth.rgb = log(1.0 + 0.2 * smooth.rgb) / log(1.2);    // 全图 log 提亮
```

→ `src/skin/smooth_faithful.rs::combine_bb`。`setBrightness` / `setSaturation` 先 `reset` 再依次 adjust，
两个矩阵可交换（亮度是各向同性缩放），等价 `c' = clamp(b · (lum + (c − lum)·s))`。

## 3. `GLImageFaceChangeFilter.m`（瘦脸 / 大眼 / 瘦鼻）

坐标：归一化纹理坐标；`locArray[106]` 为 Face++ 106 点（`setFacePointsArray` 里做了 x/y 互换与前置镜像处理）。
主函数：

```glsl
eye_dist   = distance(loc[74], loc[77]);          // 在未修正的纹理坐标中计算（≈ 宽度单位）
aspect     = resolution.y / resolution.x;         // 720x1280 → 1.777
dir_up     = normalize(loc[43] - loc[16]);
dir_right  = normalize(loc[77] - loc[74]);
coord = adjust_thinFace(coord, ...thin_face_param);
coord = adjust_eye(coord, ...eye_param);
coord = newNarrowNose_2(coord, ...nose_param);
```

### warpPositionToUse1

```glsl
cur' = (cur.x, cur.y*aspect + 0.5 - 0.5*aspect);  A' 同理     // y 按宽高比修正，使 x/y 同尺度
r = distance(cur', A');
if (r < radius) { dir = normalize(B - A);          // 注意：dir 在未修正坐标中计算（轻微各向异性）
    dist = radius² - r²; alpha = (dist / (dist + (r - delta)²))²;
    cur = cur - alpha * delta * dir; }
```

r = 0 时 alpha ≈ 1，位移 ≈ delta；r → radius 时位移 → 0。像素空间实现：`src/warp/pinch.rs::pinch`。

### adjust_thinFace

```
L0 = loc[4]  - dir_right*0.13*ed, L1 = loc[9]  - dir_right*0.33*ed, L2 = loc[13] - dir_right*0.33*ed
R0 = loc[28] + dir_right*0.13*ed, R1 = loc[23] + dir_right*0.33*ed, R2 = loc[19] + dir_right*0.33*ed
x = π/25; scale = 2ed; radius = 0.4*scale;  delta_i = sin([x, 2x, 2x]) * intensity * 0.15 * scale
for i: cur = warp(cur, L_i, R_i, radius, delta_i); cur = warp(cur, R_i, L_i, radius, delta_i);
```

最大位移：intensity=1 时约 0.038ed（第一对）/ 0.075ed（后两对）。

### adjust_eye

在 `x·(w/h)` 的等比坐标中：`k = intensity*0.24`，`eye_width = |loc52 − loc55|`（只用左眼），
对 loc74、loc77 依次：`d < radius ⇒ w = pow(d/radius, k); cur = c + (cur − c)·w`
（`d < 0.01` 时用 `(d+0.01)/radius`，是对 pow(0) 的规避，产生一个小的不连续环）。→ `pinch.rs::enlarge`。

### newNarrowNose_2

```
scale = 0.28ed; delta = (intensity*scale)*scale = intensity*0.0784*ed²;   // 归一化单位
radius = 0.16;  A0 = loc48 + dir_up*0.09,  A1 = loc82;  B0 = loc50 + dir_up*0.09,  B1 = loc83
```

`radius = 0.16`、`0.09` 是归一化常数，与分辨率耦合。按 720×1280、ed ≈ 0.12（宽度单位）换算：
radius ≈ 1.33ed，上移 ≈ 1.34ed，delta（含 setter ×2）≈ 0.157·slider·ed² ≈ 0.019ed（ed=0.12）～0.039ed（ed=0.25）。
实现取 `thin_nose_radius = 1.3, thin_nose_lift = 1.3, thin_nose_delta = 0.04`（可配置）。

## 4. Face++ 106 点语义（`人脸106个关键点.png`）

| 索引 | 语义 |
|---|---|
| 0–32 | 轮廓：0 左太阳穴 → 16 下巴尖 → 32 右太阳穴 |
| 33–37 / 38–42 | 左 / 右眉上缘；64–67 / 68–71 眉下缘 |
| 43–46 | 鼻梁：43 两眼之间 → 46 鼻尖 |
| 47–51 | 鼻底：47 左、49 中（鼻小柱）、51 右；48 / 50 鼻孔 |
| 52–57 | 左眼：52 外眼角、55 内眼角、53/54 上睑、56/57 下睑；72/73 上下睑中点；74 瞳孔；104 眼中心 |
| 58–63 | 右眼：58 内眼角、61 外眼角；75/76；77 瞳孔；105 眼中心 |
| 78–79 | 鼻梁两侧；80–81 鼻侧；82–83 鼻翼 |
| 84–103 | 嘴：84 左角、90 右角，85–89 上唇外缘，91–95 下唇外缘，96–103 内缘 |

## 5. 资源

- `MHSDK.bundle`：UI 图标与 20 余款 512×512 查找图（`filter_*.png` 等）、`filterFile.zip`（曲线滤镜包，见 curve_filter_packs.md）。
- `GLImageLutFilter`：`GPUImageLookupFilter` 的封装，`intensity` 直接传给 lookup shader。
- `归档.zip`：mkil/MagicCamera 的 `GPUImageBeautifyFilter`（双边 + Canny + 高反差强光变体），Demo 未使用。
