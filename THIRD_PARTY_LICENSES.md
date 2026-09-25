# 第三方许可证汇总

本项目代码为原创 Rust 实现（MIT OR Apache-2.0）。以下项目作为算法参考或运行时依赖，按方案第 9 节处理。
本文件为工程实践记录，不构成法律意见；商用前请法务确认。

## 算法参考（未复制代码）

| 项目 | 许可证 | 使用方式 |
|---|---|---|
| [zhanghao5683934/Meihu-Beautyface-sdk](https://github.com/zhanghao5683934/Meihu-Beautyface-sdk) | GPL-3.0 + 仓库声明非商用 | **只提取参数与公式**（见 doc/analysis/meihu.md），未复制任何代码行、未打包任何 PNG / zip 资源。`tools/bake_lut.py` 生成的 LUT 是其贴图的衍生物，仅限个人学习，本仓库不包含烘焙结果。 |
| [pixpark/gpupixel](https://github.com/pixpark/gpupixel) | Apache-2.0 | 参考其磨皮 / 形变 shader 逻辑重写（`src/skin/smooth_gpupixel.rs`、`src/warp/pinch.rs`）。 |
| [BradLarson/GPUImage](https://github.com/BradLarson/GPUImage) | BSD-3-Clause | 双边 / Sobel / HSB / 512 查找表公式属通用算法，按公式重写。 |
| [YuAo/YUCIHighPassSkinSmoothing](https://github.com/YuAo/YUCIHighPassSkinSmoothing) | MIT | 高反差磨皮流程与默认参数（`src/skin/smooth_freqsep.rs`）。 |
| [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | Face Mesh 关键点语义索引、ROI 规则。 |
| [PINTO0309/facemesh_onnx_tensorrt](https://github.com/PINTO0309/facemesh_onnx_tensorrt) | MIT | Face Mesh ONNX（含后处理）的转换与说明。 |
| [Gnimuc/FabSoften](https://github.com/Gnimuc/FabSoften) | MIT | 瑕疵检测 / 动态导向滤波 / 纹理恢复的流程参考（`src/skin/heal.rs`、`src/skin/cream.rs` 按思路重写，未复制代码）。 |
| [vacancy/PyPatchMatch](https://github.com/vacancy/PyPatchMatch) | MIT | PatchMatch 修复的评估参考，未采用。 |
| [cheind/poisson-image-editing](https://github.com/cheind/poisson-image-editing)、Pérez 等 2003 | （仓库无 LICENSE 文件；只阅读） | 调和 / 梯度域插值原理，`src/skin/heal.rs` 按公式自行实现。 |
| [advimman/lama](https://github.com/advimman/lama)、[CloudyTabzy/LaMa-OxiONNX](https://github.com/CloudyTabzy/LaMa-OxiONNX) | Apache-2.0 | 深度修复方案的评估参考，未集成。 |
| [ModelScope iic/cv_unet_skin-retouching](https://www.modelscope.cn/models/iic/cv_unet_skin-retouching)（ABPN，CVPR 2022） | Apache-2.0 | 网络定义复制为 `tools/abpn/*.py`（保留版权头，只改 import），权重导出为 ONNX；流水线逻辑在 `src/skin/ai_blemish.rs` 中用 Rust 重写。 |

## 模型文件（`models/`，不入库，由 `scripts/fetch_models.py` 获取）

| 文件 | 来源 | 许可证 | 备注 |
|---|---|---|---|
| `version-RFB-320.onnx` | [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | MIT | 保留声明 |
| `2d106det.onnx` | [deepinsight/insightface](https://github.com/deepinsight/insightface) `buffalo_l.zip` | 代码 MIT；**模型仅限非商用** | 仅开发 / 内测；发布版本必须切换 Face Mesh（`--landmarks facemesh` / `LandmarkKind::FaceMesh`），CI 应检查发布包不含该文件 |
| `face_mesh.onnx` | PINTO0309/facemesh_onnx_tensorrt release 1.0.0（源自 MediaPipe `face_landmark.tflite`） | Apache-2.0 | 保留 NOTICE |
| `face_parsing_resnet18.onnx` | [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)（BiSeNet，CelebAMask-HQ 训练） | MIT | 奶油肌预设的皮肤 / 五官 / 头发分割 |
| `modnet_photographic.onnx` | [yakhyo/modnet](https://github.com/yakhyo/modnet)（移植自 ZHKKKe/MODNet） | Apache-2.0 | 人像 alpha，身体皮肤门控 |
| `genderage.onnx` | insightface `buffalo_l.zip` | **模型仅限非商用** | 预设按性别分档；商用发布时删除，退化为 unknown 档 |
| `abpn_blemish_detect.onnx`、`abpn_blemish_inpaint.onnx` | ModelScope `iic/cv_unet_skin-retouching`（`joint_20210926.pth`），`tools/abpn/export_onnx.py` 导出 | Apache-2.0 | 奶油肌 AI 瑕疵分割 + 修复；可商用；缺失时自动退化为经典检测 |
| `skin_seg.onnx` | 同一 ModelScope 仓库的皮肤分割图 `tf_graph.pb`，`tools/abpn/convert_skinseg.py`（tf2onnx）转换 | Apache-2.0 | 身体皮肤遮罩的语义门控（排除浅色衣物 / 饰品）；缺失时退化为"抠图 ∧ 肤色规则" |
| `skin_seg_lite.onnx`（轻量备选，`--only skinseg-lite`） | [Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP](https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP) DeepLabV3+（timm-mobilenetv3_large） | MIT | 512×512 皮肤 / 衣物 / 头发三类分割，18 MB；精度低于上一行，作为无 TensorFlow 环境时的备选 |

## 运行时二进制（`runtime/`，不入库）

| 文件 | 来源 | 许可证 |
|---|---|---|
| `onnxruntime.dll` / `onnxruntime_providers_shared.dll` | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) 1.30.0（pip wheel 或 GitHub Release） | MIT |

## Rust 依赖（crates.io）

image（MIT/Apache-2.0）、rayon（MIT/Apache-2.0）、ndarray（MIT/Apache-2.0）、ort（MIT/Apache-2.0）、
serde / serde_json（MIT/Apache-2.0）、anyhow / thiserror（MIT/Apache-2.0）、clap（MIT/Apache-2.0）、criterion（MIT/Apache-2.0，仅测试）、
jpeg-encoder（(MIT OR Apache-2.0) AND IJG，CLI 的 JPEG 输出）。

jpeg-encoder 的 DCT / 量化部分源自 libjpeg，IJG 许可要求在分发二进制时的文档中注明：
**This software is based in part on the work of the Independent JPEG Group.**

## 资源

- `luts/` 下的查找图与 .cube 由 `tools/make_luts.py` 参数化生成，版权属于本项目。
- gpupixel `src/res/lookup_light.png` / `lookup_custom.png`（Apache-2.0）可作为 512 查找图直接使用，本仓库未复制。
- 测试用样张（`example/gpupixel/demo/desktop/demo.png`、insightface `t1.jpg` 等）只在本地验证时使用，不随仓库分发。
