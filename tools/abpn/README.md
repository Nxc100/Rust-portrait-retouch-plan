# tools/abpn — ABPN 人像美肤网络定义与 ONNX 导出

来源：ModelScope `iic/cv_unet_skin-retouching`（阿里达摩院，ABPN CVPR 2022），**Apache License 2.0**。
`*.py` 是上游 `modelscope/models/cv/skin_retouching/` 的独立副本（保留版权头；只改了包内相对 import，
`utils.py` 用等价的 torch reshape 替代 einops），不依赖 modelscope 包。分析见 `doc/analysis/abpn.md`。

| 文件 | 内容 |
|---|---|
| `detection_unet_in.py` / `detection_module.py` | DetectionUNet：瑕疵分割（768 输入 → logits） |
| `inpainting_unet.py` / `gconv.py` | RetouchingNet：门控卷积修复网络（masked RGB + mask → RGB） |
| `unet_deploy.py` | UNet 混合图层生成器（匀肤，本项目未集成） |
| `utils.py` / `weights_init.py` | 上游前后处理与初始化 |
| `export_onnx.py` | 导出 + onnxruntime 数值校验 |
| `convert_skinseg.py` | 同仓库皮肤分割 TF 图 `tf_graph.pb` → `models/skin_seg.onnx`（tf2onnx；身体皮肤遮罩的语义门控，见 doc/analysis/abpn.md §7）。先检查 tensorflow / tf2onnx / onnx 是否可用，`tf_graph.pb` 缺失时下载并校验 SHA256，转换写入临时文件，onnxruntime 自检（无人图输出全 0）通过后才替换目标文件 |

皮肤分割模型（需要 tensorflow-cpu、tf2onnx、onnx，建议单独 venv；Python 3.12 + tensorflow-cpu 2.21 + tf2onnx 1.17 已验证）：

```bash
python -m venv .venv-convert
.venv-convert/Scripts/pip install tensorflow-cpu tf2onnx onnx onnxruntime
.venv-convert/Scripts/python scripts/fetch_models.py --only skinseg     # 下载 tf_graph.pb 到 models/abpn_weights/ 并转换
```

ABPN 瑕疵模型：

```bash
# 需要 torch（CPU 版即可）、onnx、onnxruntime
python scripts/fetch_models.py --only abpn                      # 下载权重到 models/abpn_weights/ 并导出到 models/
python tools/abpn/export_onnx.py --weights-dir <dir> --out models --only blemish   # 手动导出
```

导出结果：`abpn_blemish_detect.onnx`（`image[1,3,H,W]` → `logits[1,1,H,W]`）、
`abpn_blemish_inpaint.onnx`（`image`、`mask` → `inpainted`），H、W 动态（64 的倍数）；Rust 侧实现在 `src/skin/ai_blemish.rs`。
