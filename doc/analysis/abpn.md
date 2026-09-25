# ABPN 人像美肤模型（ModelScope `iic/cv_unet_skin-retouching`）分析与集成

来源：阿里达摩院 ModelScope 模型库 `iic/cv_unet_skin-retouching`（旧名 `damo/cv_unet_skin-retouching`），
论文 ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo（CVPR 2022，Lei 等）。
**模型与代码均为 Apache License 2.0**（模型卡 `license` 字段），可商用。
本地副本：`example/modelscope-skin-retouching/`（权重、上游源码、独立复现脚本 `run_abpn.py`、导出脚本、评估输出 `out_abpn/`、完整评估报告 `REPORT.md`）。

这是目前能找到的、与像素蛋糕"AI 修图"最接近的开源方案：先定位、后编辑的两阶段美肤，
专门针对"痣 / 脂肪粒 / 痘印"做分割 + 修复，并用"混合图层"做匀肤，训练数据来自专业修图师的成对样本。

## 1. 模型组成

| 文件 | 网络 | 参数量 | 作用 |
|---|---|---|---|
| `pytorch_model.pt`（53.6 MB） | `generator`：UNet(3,3) | 13.4 M | 匀肤混合图层：预测逐像素 `mg`，`pred = (1 − 2mg)·x² + 2mg·x`（0.5 = 不变） |
| `joint_20210926.pth`（227.6 MB） | `detection_net`：DetectionUNet(3,1) | 18.0 M | 瑕疵分割（768 输入，sigmoid 概率） |
| 同上 | `inpainting_net`：RetouchingNet(4,3) | 36.1 M | 门控卷积（gated conv）修复网络，填补 mask=0 的洞 |
| （上游另有 TF 皮肤分割图用于美白） | — | — | 未使用（本项目已有 BiSeNet / MODNet） |

## 2. 上游流水线（`skin_retouching_pipeline.py`，逐行核对）

所有网络输入：RGB、NCHW、float32、`(x/255 − 0.5)·2 ∈ [−1, 1]`。

1. **ROI**：`L = int(1.5·max(w, h))`，以人脸框中心取正方形，裁到图像范围（不补边）。
2. **去瑕疵 `retouch_local`**：
   ```
   x768 = bilinear(roi, 768×768, align_corners=True);  p = sigmoid(Det(x768));  p = nearest(p, H×W)
   m = p; m[p ≥ 0.5] = 1; m[p < 0.35] = 0;  keep = 1 − m          # keep ∈ {1} ∪ (0.5, 0.65] ∪ {0}
   roi / keep 右下补 0 到 512 的倍数，四周再补 32 px 0；按 576 窗口、步长 512 切块
   masked = img·keep;  out = Inp(masked, keep);  comp = masked + (1 − keep)·out   # keep = 1 处严格等于原图
   去掉 32 px 边框平铺，裁回 H×W
   ```
   有效区无洞的窗口输出与输入按位相同，可直接跳过（本样张两张脸各 9 个窗口中只有 3 个含洞）。
3. **匀肤 `predict_roi`**：ROI 双线性缩到 512 → `mg` → `(mg − 0.5)·degree + 0.5` → 双线性放回 → 边缘 20/500 渐隐 →
   `pred = (1 − 2mg)·img² + 2mg·img`。上游 `.byte()` 截断会使 1/4 灰阶偏低 1 级，移植时应四舍五入。
4. **美白**：TF 皮肤分割 + 混合图层，跳过。

## 3. ONNX 导出（`tools/abpn/export_onnx.py`，opset 17，TorchScript 导出器，BN 已折叠）

| 文件 | 大小 | 输入 | 输出 | 约束 |
|---|---|---|---|---|
| `abpn_blemish_detect.onnx` | 68.8 MiB | `image` [1,3,H,W] | `logits` [1,1,H,W] | H、W 动态、64 的倍数，标称 768 |
| `abpn_blemish_inpaint.onnx` | 137.6 MiB | `image`（已乘 mask）[1,3,H,W]、`mask`（1 保留 / 0 洞）[1,1,H,W] | `inpainted` [1,3,H,W]，tanh | H、W 动态、64 的倍数，标称 576 |
| `abpn_blend_unet.onnx` | 51.1 MiB | `image` [1,3,H,W] | `blend_mg` [1,3,H,W] ∈ [0,1] | 任意尺寸，建议 512（本项目未集成） |

onnxruntime 1.30 与 PyTorch 输出最大绝对误差 ≤ 6.9e-5（多种尺寸）。
网络定义（`tools/abpn/*.py`）是 ModelScope 源码的独立副本（只改了包内相对 import，`einops` 换成等价的 torch reshape）。

## 4. 在本样张上的评估（`example/modelscope-skin-retouching/out_abpn/`，细节见 `REPORT.md`）

- **检测网（人脸 ROI）**：新郎 9 个连通区（最高 0.98）——太阳穴痣、脸颊暗斑、下巴痣等全是真实瑕疵，
  无一个落在五官 / 发际线上；新娘 5 个（最高 0.74）。偏保守但准确。
- **修复网**：痣与暗斑被干净抹平，亮度 / 颜色与周围一致，毛孔纹理连续，576 窗口之间看不到接缝；
  差分图只在洞内有信号，洞外像素按位不变。
- **匀肤网**：`mg` 0.26–0.76，是低频匀色而非磨皮，毛孔 / 皱纹 / 胡茬 / 发丝完全保留，效果偏轻。
- **尺度敏感**：检测网只认 768 输入里 ≤ ~50 px 的斑点。手臂上 70×40 px 的疤痕在 600 px 裁块里最高概率只有 0.08，
  裁块放大到 1536 / 2048 px 才升到 0.68 / 0.93。给定手动椭圆 mask 时修复网能把疤痕完整、自然地填平
  （洞径 ≤ ~100 px；更大的洞中心开始出现淡紫灰色偏）。
- **CPU 耗时**（i7-13700F，onnxruntime）：检测 768² 约 280–400 ms；修复 576² 窗口约 250–380 ms；匀肤 512² 约 350 ms。

## 5. 本项目的集成（`src/skin/ai_blemish.rs`）

- 只集成"去瑕疵"两段网（分割 + 修复），作为奶油肌流程的**前置步骤**：逐脸取 1.5× ROI，与上游逐步一致地
  检测、切窗、修复，得到**稀疏补丁**（只记录 `keep < 1` 的像素），缓存在 `Precomp` 中（按图像 + 人脸集合键）；
  `SmoothMode::Cream` 分支把补丁按 `RetouchParams::ai_blemish`（0..1）混到原图上，再进入经典流程
  （大尺度疤痕修复 → 小斑 → 三频段 → 匀肤 → 色调）。
- 无洞的窗口跳过（本样张 9 → 3），两张脸合计约 3.0 s（28.5 MP）；洞外像素严格不变。
- 模型缺失（`models/abpn_blemish_*.onnx` 任一不存在）或 `--no-ai` 时自动跳过，退化为纯经典流程；
  `--no-ai-blemish` 只关闭该步骤而不卸载模型。`--debug-dir` 导出 `ai_holes.png`（被修改像素）。
- ORT Session 关闭线程池自旋等待（`session.intra_op.allow_spinning = 0`）：大模型多次调用时曾出现
  与 rayon 争抢 CPU 导致的 50–65 s 停顿，关闭后稳定在 0.4 s。
- 疤痕等大瑕疵不依赖该模型：由 `skin/heal.rs` 的能量判据检测 + 调和插值 / 纹理移植处理（对身体皮肤同样有效）。

## 6. 获取模型

```bash
# 需要 torch（CPU 版即可）；权重从 modelscope.cn 下载（约 280 MB），导出到 models/
python scripts/fetch_models.py --only abpn
# 等价于：
python tools/abpn/export_onnx.py --weights-dir <含 pytorch_model.pt / joint_20210926.pth 的目录> --out models
```

## 7. 同仓库的皮肤分割图 `tf_graph.pb` → `models/skin_seg.onnx`（身体皮肤遮罩的语义门控）

上游流水线的"美白"步骤用一张 TensorFlow 冻结图做**身体皮肤分割**（`sess.run('output_png:0', {'input_image:0': img})`）。
第一轮 / 第二轮没有用它，改用"MODNet 抠图 ∧ 肤色规则"；在中式婚纱样张（`doc/test_report_x04.md`）上，米色缎面、金饰、
粉色团扇全部落在肤色规则内，整条裙子被当成皮肤处理——只有语义模型能区分"肤色的布"和皮肤。完整报告见
`example/modelscope-skin-retouching/SKINSEG_REPORT.md`，转换脚本 `tools/abpn/convert_skinseg.py`。

| 项 | 内容 |
|---|---|
| 图结构 | 2030 节点：`x/127.5 − 1` → 192×192 `detect` 网（Tanh）→ 人体存在判断（`if_person > 50`）→ `CropAndResize` 人物框到 448×448 → `coarse`(256) + `refine`(448，ProGAN 风格 wscale 卷积 + LeakyReLU）→ `((tanh+1)/2)²·255` → 缩放 / Pad 回 `[H,W]` → Cast uint8；无人时输出全 0 |
| 输入 | `input_image:0` float32 `[H,W,3]`，RGB，**0..255**（不归一化、无 batch 维）；上游把长边缩到 800（cv2 INTER_LINEAR） |
| 输出 | `output_png:0` uint8 `[H,W]`，255 = 皮肤（脸、颈、胸、肩、臂、手；眼、眉、唇、头发、衣物、饰品为 0），边缘为软值 |
| 转换 | `tf2onnx --opset 17` 一次成功：`Switch/Merge` → 2 个 `If`，`CropAndResize` → `Loop + Resize`，`Assert` 丢弃；102,556,878 字节 |
| 校验 | onnxruntime 1.30 vs TF 2.21：最大绝对差 1 级（float→uint8 取整），二值化零差异；横 / 竖 / 方 / 小图与无人图均通过 |
| 耗时 | ORT 默认线程 0.16–0.17 s / 张（TF 0.53 s）；4 线程 0.29 s；无人时 0.024 s |
| 评估 | X04：只覆盖两张脸、脖子、胸口、双肩、前臂和相握的手，米色礼服、红裙、金饰、团扇、红墙全部为 0；不足：拿团扇的手几乎漏检。1V3A2861：脸、颈、胸、肩、整条手臂与双手完整覆盖，头发 / 衣物 / 捧花排除 |

获取：`scripts/fetch_models.py --only skinseg`（用装了 tensorflow-cpu + tf2onnx + onnx 的 Python 运行）调用
`tools/abpn/convert_skinseg.py`：下载 `tf_graph.pb` 到 `models/abpn_weights/` 并校验 SHA256
（`6fa1931e…cdda30`），tf2onnx opset 17 转换，写入 H / W 动态维名与来源元数据，onnxruntime 自检（无人图输出全 0）后
原子替换 `models/skin_seg.onnx`。重复转换得到的图可能有等价的常量折叠差异，但输出逐像素相同（`doc/test_report_x04.md` §4）。
无 TensorFlow 时可用 `--only skinseg-lite` 下载 Kazuhito00 的 DeepLabV3+（MIT）到 `models/skin_seg_lite.onnx`；
它更粗，会放行紧贴皮肤的珠绣 / 金饰，只作兜底。

Rust 侧（`src/face/skinseg.rs`）：按输入秩自动识别 `[H,W,3]`（本模型，长边 800）与 `[1,3,H,W]`（ImageNet 归一化的
分割网，如 Kazuhito00 的 DeepLabV3+ 三类模型）两种约定；输出 u8 / f32、秩 2 / 3 / 4 统一映射为 0..1 概率图。
`[H,W,3]` 模型的输入尺寸与上游 `resize_on_long_side` 逐位一致（长边 800，短边 f64 截断）。
`Engine` 依次查找 `skin_seg.onnx`、`skin_seg_lite.onnx`（或 `EngineConfig::skin_seg_model` / CLI `--skinseg-model`），
按图像缓存概率图；`build_skin_masks` 把它作为身体遮罩的门控（半分辨率、σ = 0.8% 短边的高斯扩张、软阈值 0.15–0.40）。
