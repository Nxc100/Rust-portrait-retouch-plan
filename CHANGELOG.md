# CHANGELOG

## 0.6.3 (2026-09-26)

「奶油肌」身体皮肤上的阴影色块：修复肩膀、锁骨、胸口、手臂上一块块没被处理的皮肤（doc/test_report_shadow_blotches.md）。

- 原因：身体遮罩 = 人像抠图 ∧ 严格肤色规则 ∧ 语义皮肤概率。严格规则的固定阈值在有色光下漏掉皮肤——天光照亮的
  肩膀、胸口、手臂偏蓝（B ≥ G），阳光直射的高光饱和度过低，红色影棚里染红的暗部过饱和；漏掉的皮肤不被提亮、
  匀肤，交界沿阈值线显出色块。语义模型偶尔在同色皮肤里只认出一小块，同样显出亮斑。
- 新增 `skin::continuity`（`reconcile_body_mask`，身体遮罩的语义门控之后、形态学清理之前）：遮罩的边界只该落在
  真正的颜色边缘上——
  - 找回：颜色规则拒绝、语义模型认作皮肤（概率 > 0.7）、不太暗（L ≥ 30）的像素，在物体边界（颜色梯度大）处切开后
    取连通块，与已接受皮肤的交界两侧色度差、亮度差的中位数都 < 3.5 的整块接受；玫瑰、指甲、纹身、扇柄、
    红围巾与卡片的交界有颜色边缘，照旧排除；
  - 去孤岛：面积 ≤ 4 瞳距当量²、被同色无边缘的"颜色认、语义拒"区域包围的已接受块不处理（1V3A2837 抬起的手臂；
    1V3A3188 被误当成皮肤的米色领结因此恢复原样）；
  - 只在有人像抠图与语义皮肤模型时生效；反方向（颜色认、语义拒）不找回，否则米色西装、白衬衫、婚纱会被当成皮肤。
- 身体降黄改为比例式：低频黄度按 `body_pull_b` 向 `body_target_b` 拉近（女 0.19 / 1.3），`body_yellow` 改作额外偏移
  （女 0，男仍为常数 −1.1、`body_pull_b` 0）——像素蛋糕对偏黄的皮肤降得多、对天光下偏蓝的皮肤略加暖，常数降黄会
  把找回的偏蓝皮肤压得更蓝。女性身体黄蓝轴相对像素蛋糕的 RMS 1.25 → 0.78。`presets/cream_skin.json` 重新生成。
  预设文件里没有 `body_pull_b`（0.6.3 之前导出）时按 0 读入，保持文件里原来的常数降黄；要用比例降黄，
  重新导出预设（`retouch preset`）。
- 22 张照片的身体遮罩空洞（可见面积）：1V3A2922 21.4 万 → 1.3 万像素、1V3A2958 26.3 万 → 0.8 万、1V3A2957 6.7 万 → 0；
  脸部指标、非奶油肌 golden 不变；身体遮罩这一步约增加 0.1–0.2 s。
- 重构：4 连通域标记从 `skin::heal` 移到 `skin::morph`。工具：`tools/body_tone_fit.py` 拟合比例降黄；
  `tools/sample_montages.py blotches` 生成修改前后对比图（`out/blotches/`），前后对比按照片名在两批样张里查找。
- 批量测试报告（doc/test_report_batch_test.md）§3.1 的问题由此修复。
- 图形界面版本号随之改为 0.6.3。

## 0.6.2 (2026-09-26)

「奶油肌」颈纹淡化：修复女性脖子上的皱纹没有像像素蛋糕那样被磨平的问题（doc/test_report_neck.md）。

- 新增 `skin::neck`（奶油肌第 7 步，脸部与身体处理之后）：每张有人脸解析的脸单独处理它的脖子——
  - 区域：解析的脖子类 × 身体遮罩 × 语义皮肤门控 × 异物门控，在处理前的颜色上确定；异物为色度低于皮肤的
    0.55–0.8 或灰度闭运算的黑顶帽深 10–16 L 以上的像素（项链、纹身墨迹、黑发），窄缝连成片（纹身笔画间的淡尾迹）、
    再外扩 0.03 瞳距当量（项链的影子）；
  - 平滑：异物用周围皮肤的归一化平均填上后做导向滤波（半径 0.05 瞳距当量、eps 80），在局部中频能量窗内替换
    （去掉细颗粒后的）亮度——颈纹的暗纹与亮脊一起压平；
  - 细线填平：8 方向线段闭运算量出细长暗线（颈纹的线芯、横过脖子的碎发；胡茬、毛孔、痣等暗点为 0），
    原图上线深 3–6 L 以上（12–20 L 以上的除外）的像素在平滑后填平，细颗粒保留；
  - 两张脸的解析范围盖到同一段脖子时（贴脸、亲吻），脖子像素按与身体色调相同的归属规则分给各人，不重复处理；
    渐隐只在解析图覆盖范围自己的边缘，线深在区域外扩一个线段半长上量、图像外按边缘像素延拓；
  - 参数 `CreamParams::neck`（`strength` 1.0、`radius` 0.05、`eps` 80、`energy_lo` 0.8、`energy_hi` 6、`line_fill` 1.0，
    男女相同，`strength = 0` 关闭）；`presets/cream_skin.json` 重新生成。
- 新娘 1V3A2922 脖子上细线的平均深度 5.51 → 1.61（像素蛋糕 2.30），谷填充 / 脊压低与像素蛋糕相当；
  项链与它的影子、纹身（含笔画的淡尾迹）、胡茬、下颌轮廓保留；变化只在脖子上（每张 ≤ 0.06% 的像素变化 > 3/255）。
  颈纹这一步每张 13–410 ms（瞳距当量约 930 px 的特写约 2.1 s）。
- 调试：`retouch apply --debug-dir` 在奶油肌模式下多导出 `neck_zone.png`（`Engine::neck_weights`）。
- 库：新模块 `skin::morph`（方形窗口的膨胀 / 腐蚀 / 闭运算、方向线段的暗线深度 `dark_line_depth`）；
  `ParseMap::mask_in`；`masked_gaussian` / `subsample_for` 移到 `skin::guided`，`smoothstep` 移到 `skin`。
  `strength = 0` 时 7 张输出与 0.6.1 逐字节相同，非奶油肌参数组的 golden 逐位相同。
- 图形界面："处理身体皮肤"的提示注明颈纹淡化。`tools/sample_montages.py neck` 生成修改前后对比图（`out/neck/`），
  `tools/sample_montages.py batch_test` 生成第三批样张的逐脸 / 脖子对比（没有参考导出时只有原图 | 本程序）。
- 第三批样张 `test/批量测试`（15 张）的批量测试：doc/test_report_batch_test.md（全部成功，平均 7.4 s / 张；
  发现身体遮罩在冷色光、偏粉色调的皮肤上有空洞造成斑驳，未修复）。

## 0.6.1 (2026-09-26)

「奶油肌」身体色调：修复腋下、肘弯等暗褶纹没有被自然抹平（反而被加深）的问题（doc/test_report_body_tone.md）。

- 身体的提亮 / 降黄原来按像素亮度加权（`2.4·clamp((L − 30)/30)`）：L 30–60 的斜率 1.08 把暗褶纹的对比放大约 8%，
  阴影几乎不提亮。改为只看遮罩内的低频亮度（归一化卷积，σ 0.03 瞳距当量），局部对比不变、皮肤边缘不出光圈。
- 提亮形状按像素蛋糕拟合：阴影到中间调是平台，L 64→88 渐弱到 10%（`body_lift` 现为平台值）。
- 身体色调按人：身体像素按到各张脸的距离软分配，混合各人（性别不同）的 `body_lift / body_redness / body_yellow`；
  女（及未知）4.6 / −0.85 / −2.5，男 2.6 / −0.6 / −1.1（原来一律 2.4 / −0.5 / −2.2）。身体的其余处理不变。
- 7 张照片身体色调相对像素蛋糕：新娘 dL RMS 2.62 → 1.70（偏差 −0.90 → +0.01），新郎 db 偏差 −1.25 → −0.26；
  IMG_5785 腋窝凹陷的褶皱频段 0.96 → 0.86（像素蛋糕 0.85）。脸部指标与非奶油肌输出不变。
- 新增 `tools/body_tone_fit.py`（身体色调的拟合与评估）；`presets/cream_skin.json` 重新生成
  （同时补上 0.5.0 起就有的 `blemish.clutter_min_peak`，取值即默认值）。

## 0.6.0 (2026-09-25)

图形界面：手动测试各项修图功能的桌面程序（`gui/`，[gui/README.md](gui/README.md)）。

- 新增工作区成员 `portrait-retouch-gui`（Tauri 2，前端为无构建步骤的原生 ES 模块）：
  - 单张调试：打开 / 拖入照片，在缩小的工作分辨率上调参并自动处理，保存时按原图分辨率重做；
  - 查看：对比分割线、并排、原图 / 结果，按住空格看原图；
  - 调试视图：关键点、皮肤遮罩（脸 / 身体）、人像抠图、皮肤概率、AI 修复区；
  - 批量处理：进度、剩余时间估算、取消、报告；
  - 引擎设置：模型目录、关键点模型、线程数、可选模型开关、加载状态。
  界面与命令行共用参数组装、读写图与批处理代码，保存结果与 `retouch batch` 输出逐字节相同。
  后端 19 个单元测试；`gui/tests/smoke.mjs` 通过 CDP 驱动真实界面做冒烟测试。
- 库：新增 `options::ParamOptions`：用户参数（预设 + 覆盖项 / 手动参数、LUT、形变系数、遮罩 LUT）→ `RetouchParams`
  的唯一组装逻辑。命令行改为 `ParamArgs → ParamOptions → build()`，默认值取自 `ParamOptions::default()`；
  重构前后命令行输出逐位相同。
- 库：`Preset::builtin_names` / `Preset::load_named`（内置名或 JSON 路径）、`Engine::model_tags`；
  `batch::OutputFormat` 可反序列化。
- 库：批处理可取消——`BatchConfig::cancel`（`Arc<AtomicBool>`）置位后不再开始新的照片，已在写盘的照片照常完成，
  其余记为 skipped（"已取消"）；`BatchReport::cancelled`。
- 根目录 `Cargo.toml` 成为工作区（`default-members = ["."]`）：根目录的 `cargo build / test / clippy` 行为不变；
  `Cargo.lock` 中已有的 187 个依赖版本不变。

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
