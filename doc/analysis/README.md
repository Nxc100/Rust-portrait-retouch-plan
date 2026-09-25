# 参考项目源码分析笔记

本目录记录对 `example/` 下参考项目的逐文件阅读结论，作为 `src/` 实现的依据。
参考项目通过 `scripts/fetch_examples.sh` 浅克隆到 `example/`（不入库）。

| 文件 | 内容 |
|---|---|
| [meihu.md](meihu.md) | 美狐 Demo 实际链路、`BBGPUImageBeautifyFilter`、`GLImageFaceChangeFilter` 逐行解读、UI 滑块映射、106 点语义 |
| [gpuimage.md](gpuimage.md) | 双边 / Sobel / HSB / 512 查找表的精确公式（含与方案文档不符之处） |
| [gpupixel.md](gpupixel.md) | gpupixel 的磨皮（均值 / 方差自适应）、美白链、形变 shader、资源 LUT |
| [yuci.md](yuci.md) | YUCIHighPassSkinSmoothing 完整流程与 Core Image 语义 |
| [face_models.md](face_models.md) | UltraFace、2d106det、Face Mesh（PINTO ONNX）的 I/O、前后处理、语义索引映射（含投票表） |
| [curve_filter_packs.md](curve_filter_packs.md) | 美狐 `filterFile.zip` 曲线滤镜包格式与离线烘焙方案 |
| [cream_skin.md](cream_skin.md) | 像素蛋糕「奶油肌」预设导出图的逆向分析（纹理衰减剖面、Lab 色调、形变场、瑕疵 / 疤痕处理）与复现方案（四频段、修复画笔） |
| [abpn.md](abpn.md) | ModelScope ABPN 人像美肤模型（瑕疵分割 + 修复 + 匀肤，Apache-2.0）的流水线核对、ONNX 导出、评估与集成；§7 为同仓库皮肤分割图的转换与"身体皮肤遮罩语义门控" |

第二轮（祛疤 / 细腻纹理）额外拉取并分析的项目：`example/FabSoften`（MIT，瑕疵检测 + 动态导向滤波 + 纹理恢复）、
`example/PyPatchMatch`（MIT）、`example/poisson-image-editing`、`example/Blemish-Removal`（OpenCV 修复画笔）、
`example/Wrinkle-detection`、`example/lama` 与 `example/LaMa-OxiONNX`（Apache-2.0，深度修复）、
`example/modelscope-skin-retouching`（ABPN 权重、上游源码、独立复现脚本与评估报告）。

## 与方案文档（doc/rust_portrait_retouch_plan.md）不一致之处（以源码为准）

1. **HSB 亮度权重**：方案 2.2 节写 `lum = 0.3086 r + 0.6094 g + 0.0820 b`。GPUImage 与美狐仓库内副本的
   `GPUImageHSBFilter.m` 实际编译使用 `#define RLUM 0.3f / GLUM 0.59f / BLUM 0.11f`（graficaobscura 的那组被注释）。
   实现采用 0.3 / 0.59 / 0.11（`src/color/hsb.rs`）。
2. **饱和度滑块**：方案附录 B 给出 `saturation = 1.0 + 0.2·(v/100)`。美狐 Demo 的
   `setSaturationValue` 实际为 `1 + level`（level = 滑块 0..9 / 9），即 1..2；亮度为 `1 + level/5`（1..1.2）。
   实现按方案附录 B 的映射（`src/ui_map.rs`），内部参数不限范围。
3. **瘦鼻位移**：方案换算的 `0.04·ed` 对应瞳距约占画面宽 0.25 的大脸自拍；按 720×1280、瞳距 0.12 宽换算约为
   `0.019·ed`。系数保留在 `WarpCoefficients::thin_nose_delta`（默认 0.04）中可调。
4. **2d106det 索引**：方案要求 P2 人工核对。本项目用 Face Mesh 语义点对 2d106det 输出逐点最近邻投票 + 放大可视化确认，
   结果固化在 `src/face/lm_2d106.rs`（见 face_models.md 的投票表）。
5. **YUCI 流程细节**：方案 4.3 节的 6 步概述省略了 `CIExposureAdjust(-1EV)`、GreenBlue overlay 退化为 `2·G·B`、
   `CIBlendWithMask` 的方向（mask 白 = 保留原图）以及 mask 的 `(x − 75/255)·255/89` 色阶拉伸；实现按源码（`src/skin/smooth_freqsep.rs`）。
6. **大眼 `pow(0)` 规避**：原实现 `dis < 0.01` 时用 `(dis + 0.01)/radius`，产生 0.01（归一化）的不连续；像素空间改为 1 px 下限。

## 额外实现（"超越"部分）

- 磨皮方案 C（`SmoothMode::GpuPixel`）：gpupixel 的均值 / 方差自适应磨皮，来自 `beauty_face_unit_filter.cc`。
- 形变风格 `ReshapeStyle::GpuPixel`：gpupixel `face_reshape_filter.cc` 的 curveWarp / enlargeEye。
- 带空间遮罩的 LUT（`MaskedLutOp`）：承载曲线滤镜包中的暗角 / 渐变部分。
- 侧脸衰减、多人脸重叠策略、按图像内容哈希的中间结果缓存。
