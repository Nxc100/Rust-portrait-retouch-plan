# 美狐曲线滤镜包（`filterFile.zip`）格式与烘焙

位置：`MHSDK.bundle/filterFile.zip` → 内含 8 个 zip：blackcat、blackwhite、brooklyn、calm、cool、emerald、kevin、romance。
每个解压后为 `config.json + fragment.glsl + 若干贴图`。

## config.json

```json
{"filterList": [{"type": "filter", "name": "calm", "vertexShader": "", "fragmentShader": "fragment.glsl",
  "uniformList": ["maskTexture", "maskTexture1", "curveTexture"],
  "uniformData": {"curveTexture": "curve.bin", "maskTexture": "mask.png", "maskTexture1": "mask1.png"},
  "strength": 1.0, "texelOffset": 0, "audioPath": "", "audioLooping": 1}]}
```

## 贴图格式

| 文件 | 内容 |
|---|---|
| `curve.bin` | 4 字节头 `u16 width(256), u16 height(N)`（小端）+ `256×N` RGBA8；每行一条曲线（R/G/B/A 各一条） |
| `curve.png` | 与 curve.bin 同内容的 PNG（256×N RGBA） |
| `mask.png` / `mask1.png` | 320×320 灰度（RGB 三通道相同），运行时拉伸到全图的空间权重（暗角 / 渐变） |
| `map.png`（kevin） | 256×1 RGBA，一维曲线（按 `.r/.g/.b` 分通道） |
| `map.png`（brooklyn） | 128×128 RGBA 纹理，拉伸到全图后 multiply 混合（空间纹理） |

shader 中曲线采样 `texture2D(curveTexture, vec2(v, y))`，y ∈ {0, 0.5, 1}：GL_LINEAR + CLAMP_TO_EDGE 下
N=2 时 y=0.5 恰好落在两行之间（各 50%），N=3 时 y=0.5 为第 1 行；烘焙脚本按同样规则采样。

## 8 款滤镜的运算结构

| 名称 | 纯颜色部分（可烘焙为 .cube） | 空间部分（运行时 `MaskedLutOp` / 暗角） |
|---|---|---|
| cool | 曲线 r/g/b(y=0) → 曲线 a(y=0) → `×1.25 − 0.12549` → 与原图 `mix(..., 0.451)` | — |
| romance | screen 自混合 → 曲线 r/g/b → 饱和 1.15 | — |
| emerald | 曲线 → RGB→HSL，s×1.5，青绿色相段微调 → 曲线(y=1).r → 曲线(y=1).g | — |
| blackcat | 曲线 → HSV，s×1.2，红 / 橙色相段增饱和 → 曲线(y=1).r → .g | — |
| calm | 去饱和 0.5 → 曲线 r/g/b(y=0)，再 y=0.5 二次查表（b 通道三次）→ overlayer1 = 曲线(y=1).r → overlayer2 = 曲线(y=1).g | `base = mix(overlayer1, base, 1 − mask)`；`out = mix(overlayer2, base, 1 − mask1)` |
| blackwhite | 灰度(0.229, 0.587, 0.114) × 1.33 → 亮度 / 对比度(0, 16, 128) | `calVignette2`：`c −= dist² / 0.5 · 0.5`（解析暗角，无贴图） |
| kevin | 一维 map 曲线 r/g/b | — |
| brooklyn | 曲线 → 饱和 0.88 / 亮度 +0.03 / 对比 0.85 → YIQ 色相 −0.0444 → 与 (0.5647,0.1961,0.0157,α0.14) multiply → 再曲线 | `map.png` 128×128 纹理 multiply（α 0.9）；纹理本身是空间图案，不能烘焙 |

## 烘焙（`tools/bake_lut.py`）

对每款把"纯颜色部分"用 numpy 直译，在 33³ 网格上求值，导出 `.cube`；空间部分：
- calm：额外导出 overlayer1 / overlayer2 两个 `.cube` + 两张 mask，运行时用两个 `MaskedLutOp`（`invert` = 使用 `1 − mask`）；
- blackwhite：暗角为解析式，可用生成的径向 mask（`tools/bake_lut.py --vignette`）近似；
- brooklyn：纹理叠加不烘焙。

### 烘焙误差（`--verify`，与 numpy 直译 shader 逐像素比较，600×767 样张）

| 滤镜 | 33³ 最大误差 | 65³ 最大误差 | 说明 |
|---|---|---|---|
| calm / calm_overlay1 / calm_overlay2 | 1.75 / 0.81 / 1.00 | — | 纯曲线，满足 ≤ 2/255 |
| blackwhite | 1.65 | — | 灰度 + 对比度，满足 |
| brooklyn | 1.17 | — | 颜色部分满足；纹理叠加未烘焙 |
| kevin | 2.78 | 2.05 | 一维曲线较陡，65 网格达标 |
| romance | 3.28 | 2.20 | screen 自混合后曲线较陡 |
| cool | 5.22 | 2.64 | 曲线两次查表 + 线性拉伸 |
| blackcat | 22.4 | 23.1 | HSV 色相分段（0.417 / 0.875 / 0.958 处不连续），网格插值在分段边界附近误差大；均值误差很小 |
| emerald | 18.2 | 17.6 | 同上（0.542 / 0.625 / 0.708 / 0.792 分段） |

结论：曲线类滤镜用 33³ 或 65³ 烘焙即可达标；含硬色相分段的滤镜（blackcat / emerald）应改为
"曲线 LUT + 运行时按色相分段的饱和度调整"，或接受边界附近的小范围误差（视觉上表现为分段边界处色相的轻微平滑）。

合规：烘焙自美狐贴图的 LUT 是其衍生物，只可用于个人学习；商用请按方案 4.5.4 自制。
本仓库不包含烘焙结果，脚本需指向本地解压的 `filterFile` 目录。
