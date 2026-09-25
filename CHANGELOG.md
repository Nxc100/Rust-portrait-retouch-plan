# CHANGELOG

## 0.5.0 (2026-09-24)

批量处理与第二批样张（7 张原片、14 张脸）的对比优化（`doc/test_report_batch.md`）。

- 新增 `retouch batch` 子命令与库模块 `batch`（`plan` / `run` / `process_photo`）：一个引擎处理整批照片，
  读图预取与编码写盘在后台线程与修图重叠；逐张隔离失败（含 panic）；原子写入 + 已存在即跳过（可续跑）；
  `batch_report.csv`（带 BOM）/ `.json` 报告；`--landmarks-dir` 输出关键点 JSON；`--dry-run`、`-r`、`--ext`、`--format`、`--overwrite`。
- 新增库模块 `photo`：读图时按 EXIF 方向把像素转正，写回的 EXIF 方向置为 1（修复相机直出竖拍照片检测不到人脸），
  ICC / EXIF 写回、原子保存；`apply` / `detect` / `bench` 都改用它。JPEG 改用标准霍夫曼表
  （jpeg-encoder 的优化表在退化分布上会被 zune-jpeg 解错）。
- 引擎 / 流水线的锁改为"中毒后照常取用"（`sync::lock`），批处理里一张照片 panic 不再连累后续照片。
- **更正**：像素蛋糕「奶油肌」不会再应用内嵌的 Camera Raw 设置（7 张新导出与 X04 的重新导出都没有全局变化），
  奶油肌预设 `embedded_develop` 改为 false（`--embedded-develop` 仍可开启）。
- `skin/cream` 色调：低频色度按比例向目标肤色拉（`pull_a/target_a/pull_b/target_b`），提亮与黄度成正比（`lift_per_b`），
  常数由新工具 `tools/skin_tone_fit.py` 在 16 张脸上拟合；`redness` / `yellow` 保留为额外偏移（默认 0）。
- `skin/blemish`：线状结构否决（原始响应上的连通生长）、强暗痕群否决（`BlemishParams::clutter_min_peak`，身体 1.5）；
  `SkinMasks::skin_prob` 携带语义皮肤概率，身体区域的瑕疵 / 疤痕检测只在语义皮肤上进行——修复纹身被当作斑点祛除
  （IMG_5785 笔画损失 11.5% → 0.5%），顺带不再修手指关节褶纹、花瓣纹理。
- 细颗粒 / 中频分界下限 1 px → 0.5 px（小脸的像素级颗粒按中频处理）；眼部局部对比椭圆只留睫毛余量
  （不再锐化泪沟），眼下平滑 女 0.40 / 男 0.35；`fine_smooth` 女 0.26 / 男 0.05；`chroma_smooth` 女 0.33 / 男 0.27；
  瘦脸预设 女 `thin_face` 0.15、`face_narrow` 0.030，男 `thin_face` 0.16、`chin_lift` 0.12、`face_narrow` 0.007。
- 工具：`tools/batch_compare.py`（整批逐脸对比与汇总）、`tools/skin_tone_fit.py`、`tools/sample_montages.py`
  （重新生成文档引用的对比图）；`compare_to_reference.py` /
  `texture_energy.py` 抽出可复用的函数（命令行行为不变）；`debug::faces_json` 增加 gender / age。

## 0.4.0 (2026-09-24)

奶油肌第三轮：以新样张 X04 与海边样张同时对比像素蛋糕导出图（`doc/test_report_x04.md` §6、`doc/test_report_cream.md`）。

- 新增 `color::develop`：读取输入 JPEG 的 XMP Camera Raw 设置（APP1 标准 + 扩展 XMP，属性 / 元素两种写法），并按像素蛋糕的做法
  **再应用一次**（像素蛋糕无视 `crs:AlreadyApplied="True"`，X04 导出图的全局变化与记录的参数逐项吻合）。颜色模型由
  `tools/develop_fit.py` 在 X04 上拟合，包括色调、暗部偏色、HSL 色相 / 饱和度、自然饱和度、饱和度、色温；
  烘焙成 65³ LUT，作用在保边基底上。纹理 + 清晰度为线性光下的 L* 带通增强。没有模拟的设置在横幅中列出。
  奶油肌预设默认开启（预设字段 `embedded_develop`）；CLI 有 `--no-embedded-develop` / `--embedded-develop`；
  `RetouchParams::embedded_develop`。非皮肤区与导出图的 ΔE 3.57 → 3.04。
- CLI 的 JPEG 输出改用 jpeg-encoder（量化正确舍入、4:4:4、优化霍夫曼表），并写回原图的 ICC 配置文件与 EXIF（不写 XMP）。
  image 0.25 的编码器量化前整数截断，会吃掉 4–7% 的最细纹理。`--jpeg-quality` 默认 95 → 98，与像素蛋糕导出的量化表一致。
- `skin/cream`：中频衰减在对数能量上做 smoothstep，纹理处保留 8% 衰减（`KEEP_MAX`）；按修正后的测量重新标定
  （女 `detail_smooth` 0.60、`energy` 1.2–3.5、`fine_smooth` 0.22；男 0.61、0.7–11、0.10；`chroma_smooth` 0.36 / 0.29，身体 0.35）；
  低频色度统一改为向"色度 ~ L 回归线"压缩（修复大面积高光被加黄）；新增 `yellow_bright`（亮部渐增降黄）与
  `eye_contrast`（眼眶局部对比）；参数 `fine_radius` / `fine_eps` 由 `fine_sigma` 取代，新增 `energy_lo` / `energy_hi`。
- `FaceKeyPoints::scale_distance()`：姿态稳健的脸部尺度，取 max(瞳距, 鼻梁顶—下巴距离 / 2)，用于奶油肌的纹理 / 瑕疵尺度。
  侧脸瞳距按 cos(yaw) 缩短，会把频段整体移到过细的尺度。
- `warp/apply`：保纹理重采样。基底用双三次采样，细节层在平滑区用最近邻、在强边缘用双三次；只处理各脸包围盒，
  形变耗时 4.7 s → 0.28 s。`ImgF32::sample_bicubic{,_unclamped}`。
- ORT 会话开启 `session.set_denormal_as_zero` 并关闭线程自旋；AI 瑕疵推理线程设置 FTZ/DAZ（`ort_util::DenormalGuard`），
  消除 ABPN 检测网络偶发的 70 s 以上卡顿。
- 工具：`tools/texture_energy.py`（按局部能量分箱的频段衰减）、`tools/develop_fit.py`（冲印模型拟合）、
  `tools/montage.py`（并排裁剪对比图）；
  `compare_to_reference.py` 改为在未重采样的输出上计算频段比。

## 0.3.1 (2026-09-24)

- 新增 `face/skinseg`：语义皮肤分割作为身体皮肤遮罩的门控。新样张（`doc/test_report_x04.md`，中式婚纱）暴露了
  "抠图 ∧ 肤色规则"把米色缎面婚纱、金饰、团扇当成皮肤并在裙子上"祛斑"364 处的问题；接入 ModelScope 同仓库的皮肤分割
  TF 图（Apache-2.0，`tools/abpn/convert_skinseg.py` 用 tf2onnx 转成 `models/skin_seg.onnx`）后身体区域只剩脸颈胸臂手，
  修图 14.8 s → 4.0 s。封装自动适配 `[H,W,3]` 0..255 与 `[1,3,H,W]` ImageNet 两种模型（轻量备选：Kazuhito00 DeepLabV3+，MIT）。
  `EngineConfig::{skin_seg_model, enable_skin_seg}`、CLI `--no-skinseg` / `--skinseg-model`、`--debug-dir` 导出 `skin_prob.png`。
- 模型获取：`scripts/fetch_models.py --only skinseg`（完整模型，TF 环境下转换）与 `--only skinseg-lite`（轻量模型，
  改存 `models/skin_seg_lite.onnx`，不再与完整模型抢同一个文件名，固定 SHA256）；引擎优先完整模型、缺失时回退轻量模型，
  `retouch apply` 横幅显示实际加载的模型。`--only skinseg` 在缺少 tensorflow / tf2onnx 时立即退出并给出搭建转换环境的命令；
  `convert_skinseg.py` 校验 `tf_graph.pb` 的 SHA256、先写临时文件、onnxruntime 自检通过后才替换目标。新增 `--models-dir`。
- 皮肤分割输入尺寸改为与 ModelScope `resize_on_long_side` 逐位一致（旧的 f32 计算会把 5472 px 长边缩成 799 而不是 800）。
- `skin/blemish`：修复用户指出的"鼻翼被当作斑填平"——滞后阈值（半阈值连通域按面积 / 伸长率 / 填充率整体剔除）、
  与大瑕疵共用眼周 / 眉尾 / 鼻孔 / 鼻翼 / 嘴角保护区及轮廓内缩区。

## 0.3.0 (2026-09-24)

奶油肌第二轮：对标像素蛋糕的祛疤 / 祛斑与"细腻真实"的皮肤质感（`doc/test_report_cream.md`）。

- 新增 `skin/heal`：大尺度瑕疵（疤痕 / 大痣）检测与"修复画笔"式填充——零中心环核 + 暗 / 红 / 能量三判据、
  滞后阈值、PCA 伸长率与边界切断剔除；调和插值（多分辨率 Gauss-Seidel）+ 纹理移植。手臂 0.15 瞳距的疤痕被完整填平。
- 新增 `skin/ai_blemish`：集成 ModelScope ABPN（Apache-2.0）瑕疵分割 + 门控卷积修复网络（`models/abpn_blemish_{detect,inpaint}.onnx`，
  由 `tools/abpn/export_onnx.py` 导出），逐脸 1.5× ROI、576 窗口、稀疏补丁缓存；`RetouchParams::ai_blemish` / 预设 `ai_blemish` /
  CLI `--no-ai-blemish`、`--no-ai`；`--debug-dir` 导出 `ai_holes.png`。
- `skin/cream`：亮度改为四频段（细颗粒 / 中频 / 低中频 / 结构），按像素蛋糕的衰减剖面标定（σ1 21%、σ2–4 12–17%、σ8 4%）；
  眼下椭圆区额外衰减；脸部大尺度修复限制在轮廓多边形内并扣除五官保护区；`cream_skin` 改为接收 `FaceKeyPoints`。
  新参数：`low_smooth`、`fine_smooth`、`fine_radius`、`fine_eps`、`mid_sigma`、`undereye_smooth`、`body_low_smooth`、
  `body_fine_smooth`、`heal`、`body_heal`。
- `skin/blemish`：连通域过滤改为逐峰值分割（密集斑簇逐个祛除）+ PCA 伸长率。
- `face/ort_util`：Session 关闭 intra-op 自旋等待（避免与 rayon 争抢 CPU 造成的数十秒停顿）。
- 文档：`doc/analysis/abpn.md`、`doc/analysis/cream_skin.md` 更新；`scripts/fetch_models.py --only abpn`；
  `scripts/fetch_examples.sh` 增加 FabSoften / PyPatchMatch / poisson-image-editing / LaMa 等参考项目。

## 0.2.0 (2026-09-24)

- 新增「奶油肌」磨皮模式（`SmoothMode::Cream`）与内置预设 `presets/cream_skin.json`（`retouch apply --preset cream`），
  对标像素蛋糕同名预设：逆向分析见 `doc/analysis/cream_skin.md`。
- 新增模块：`color/lab`（sRGB↔Lab）、`skin/guided`（快速导向滤波、盒式近似高斯）、`skin/blemish`（瑕疵检测 / 填充）、
  `skin/masks`（人脸解析 + 人像抠图 + 严格肤色规则的皮肤遮罩）、`skin/cream`、`face/parsing`（BiSeNet）、
  `face/matting`（MODNet）、`face/attribute`（性别年龄）、`preset`（JSON 预设与 `--set` 覆盖）。
- 形变新增 `chin_lift`（缩下巴 / 提升下半脸）与 `face_narrow`（整体收窄），瘦脸权重剖面可配置；
  `RetouchParams` 支持按性别 / 逐脸的美型与奶油肌参数。
- `Engine` 自动加载可选模型（存在即启用），`FaceKeyPoints` 增加 `gender / age / parse` 字段。
- 工具：`tools/compare_to_reference.py`（与参考导出图的定量对比）。

## 0.1.0 (2026-09-23)

按 `doc/rust_portrait_retouch_plan.md` 完成 P0–P6 全部功能：

- **P0 骨架**：`ImgF32` / `GrayF32`、双线性采样、抗混叠缩放、512 查找图、.cube 3D LUT、恒等测试（误差 ≤ 1/255）。
- **P1 磨皮 A**：GPUImage 可分离双边（9 tap、spacing 4、dnf 4）、灰度 + Sobel、BB 组合 shader 直译、log 提亮、
  HSB（亮度权重按源码 0.3/0.59/0.11）、短边 720 工作副本策略、按图像哈希缓存中间结果。
  与 numpy oracle（`tools/oracle_numpy.py`）逐像素误差 ≤ 1/255。
- **P2 检测 + 关键点**：UltraFace RFB-320、InsightFace 2d106det、语义映射（投票 + 可视化核对）、
  `retouch detect` / `examples/dump_landmarks.rs` 可视化与 JSON、附录 A 六条自动检查。
- **P3 形变**：pinch / enlarge 像素空间移植、瘦脸 / 大眼 / 瘦鼻步骤生成（系数可 JSON 配置）、包围盒并行反向映射、
  侧脸衰减、多人脸重叠策略。强度 0 时逐位恒等。
- **P4 遮罩 + 美白 + 磨皮 B**：人脸多边形（轮廓 + 额头补点）羽化遮罩、颜色肤色遮罩、参数化 / 查找图美白、
  YUCI 高反差磨皮（B）；额外实现 gpupixel 磨皮（C）与 gpupixel 风格形变。
- **P5 集成**：`Engine` API（检测与修图分离、纯函数、缓存）、CLI `retouch apply|detect|identity-lut|bench`、
  UI 滑块映射（`ui_map`）、`luts/index.json`、预览 / 导出一致性测试（PSNR ≥ 40 dB）、错误处理。
- **P6 发布模型**：MediaPipe Face Mesh（PINTO ONNX，468 点）实现 `LandmarkModel`，两遍旋转对齐 ROI，
  `EngineConfig.landmark` 一键切换。
- 工具：`tools/bake_lut.py`（曲线滤镜包 → .cube + 遮罩）、`tools/make_luts.py`、`scripts/fetch_*`。
- 文档：`doc/analysis/`（参考项目逐文件解读与方案修正）、`THIRD_PARTY_LICENSES.md`。

未实现：P7 可选 GPU（wgpu）路径，`gpu` feature 仅占位。
