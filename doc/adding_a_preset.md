# 添加一个预设模板（对标像素蛋糕）：步骤、细节与踩过的坑

「奶油肌」「婚纱-深色内景」两个预设前后调了很多轮，大部分时间花在：不知道像素蛋糕某个滑块实际做了什么、
度量方法选错被误导、一个缺陷被另一个参数"掩盖"后又返工、以及奶油肌里本来就有但看不出的缺陷在更强的预设里暴露。
本文把流程固定下来，按顺序做可以少走这些弯路。实现细节见 [presets.md](presets.md)。

## 快速流程
1. 在像素蛋糕里用新预设导出 `test/原片` 的 7 张，放到 `test/像素蛋糕“<预设名>”预设产物/`（§1）。
2. 跑 `tools/pixcake_param_diff.py` 对照滑块，按分诊表判断哪些只需调数值、哪些要新写功能（§1、§2）。
3. 按 §3、§4 搭骨架、依次校准，每轮一条 `tools/preset_sweep.py run`。
4. 按 §5、§6 做最终对照和回归检查，再按 §7 更新文档。

新预设只用到现有功能时，一般几轮扫参就能完成；遇到问题先查 §8 踩过的坑。

## 0. 约束（每次都适用）
- 只用像素蛋糕的**参数值**（用户分析项目 `D:\cakes\piCake\preset-export` 导出的表）；像素蛋糕的代码、配置、LUT、反编译
  内容一律不进仓库。LUT 与常数都在导出图上自行拟合。仓库是**公开的**。
- `test/`（真人照片）永不推送；`out/` 只收录 `out/对比图/` 与 `out/最终对照组/` 的对比图、分析（用户同意公开，
  见 `.gitignore`），全尺寸批处理输出与 `out/_work/` 不入库。
- 已有预设的输出不能被意外改变：新功能在 `CreamParams` / `Preset` 里加字段、默认关闭，只在新预设里打开；每次改动后
  核对已有预设逐字节不变（§6）。确实要改已有预设（修缺陷、对齐参考）时，先说明影响哪些照片、为什么，再改。
- 草稿、试跑输出放 `out/_work/`（用户会清空），脚本用 `python -X utf8`。

## 1. 准备数据
1. 在像素蛋糕里用新预设导出**同一批原片**：`test/原片/` 的 7 张（两个已有预设都用这 7 张，便于横向比较）→
   `test/像素蛋糕“<预设名>”预设产物/`，文件名与原片相同，不做任何额外调整。注意：导出目录里有子文件夹也没关系
   （工具只取文件），但不要混入别的照片。
2. 批量测试 `test/批量测试/`（15 张，无参考）只用于最后检查泛化与稳定性。
3. 参数表：
   ```bash
   python -X utf8 tools/pixcake_param_diff.py --presets 奶油肌 婚纱-深色内景 <新预设> [--gender 女性]
   ```
   输出修图（女性 / 男性分表）与调色两部分里各预设取值不同的滑块。先看女性，再看男性（`--gender 男性`）。

## 2. 分诊：像素蛋糕参数 → 本项目的功能
对照下表决定工作量。表里有的只需校准数值；表里没有、而新预设取值又不是中性值的，就是要新写的算子（最花时间，先和
用户确认要不要做）。

| 像素蛋糕参数 | 本项目 | 校准方法（§4） |
|---|---|---|
| AI 风格 / 调色参数（曝光、对比、HSL、曲线……） | `Preset.grade`（软黑点 + 全局 LUT + 主体 LUT） | 4.1 `grade_fit.py` |
| 质感磨皮、水润磨皮 | `detail_smooth`、`fine_smooth` | 4.5 `preset_sweep.py run` |
| 中性灰平整 | `low_smooth` | 4.5 |
| 质感保留 | `detail_smooth` 调低（毛孔一档多留） | 4.5 |
| 中性灰立体、高光立体 | `stereo` | 4.5（`band_gain.py`） |
| 肤色美白 / 透亮 / 肤色统一 / 肤色色调 | 脸部色调常数（`midtone_lift`、`pull_*`、`target_*`、`yellow_*`） | 4.3 `skin_tone_fit.py` |
| 身体美白、身体肤色统一 | `body_lift`、`body_redness`、`body_pull_b`、`body_target_b` | 4.4 `body_tone_fit.py` |
| 牙齿美白（明度、祛黄） | `teeth_whiten` | 4.6 `teeth_compare.py` |
| 瘦脸、小脸、下颌、下巴、眼睛大小、瘦鼻 | `reshape.<性别>`、`warp_coeffs.thin_face_profile` | 4.7 `batch_compare.py` 的几何列 |
| 祛颈纹 | `neck`（NeckParams） | 沿用 |
| 祛斑祛痘、身体祛瑕疵 | `blemish`、`heal`、`ai_blemish` | 沿用 |
| 祛双下巴、祛眼周纹、祛碎发、唇纹修整、头发柔顺、牙齿祛瑕疵 / 排齐、AI 面部流畅、祛黑眼圈（单独）、瘦腿…… | **无** | 未实现，记为已知差异 |

经验：像素蛋糕两个预设的"AI 肤色统一""祛斑""祛颈纹"取值相同，这些共用算子一般不用动；变化集中在调色、磨皮三件套 +
立体、色调常数、美型。

## 3. 代码骨架（先搭好，再校准）
1. `src/preset.rs`：
   - 新函数 `pub fn <name>() -> Self`，从最接近的预设改起：`Self { name, description, cream_female, cream_male,
     cream_unknown, reshape, warp_coeffs, grade, ..Self::cream_skin() }`；各性别参数写成独立函数
     （如 `wedding_dark_female()`），在其文档注释里写清常数来源与报告位置。
   - 新功能开关：`face_ownership: true`（新预设一律打开）；色调很强时 `body_tone_spill: 1.0`；有牙齿美白时 `teeth_whiten`。
   - `builtin_names()` 加规范名；`load_named()` 加别名（英文、中文、带不带连字符）。
   - 单元测试：别名、JSON 往返、"只有该调色的预设带 grade"之类的约束（照已有的写）。
2. 有调色时：`luts/<name>.cube`，`src/color/grade.rs` 加常量名与 `builtin_lut` 分支（`include_str!` + `OnceLock`），
   预设里 `lut: "builtin:<name>"`。
3. `tests/golden.rs::param_sets()` 加一组：`("<name>", Preset::<name>().to_params().expect("builtin preset"))`。
4. `tools/sample_montages.py`：`FINAL_PRESETS` 加 `('<预设名>', '<参考目录>')`（最终对照组多两列）；需要单独的完整对比
   时在 `COMPARE_SETS` 加两行。
5. 图形界面自动从 `builtin_names()` 取列表，不用改。

## 4. 校准（按这个顺序做，前一步的结果是后一步的输入）

### 4.1 调色（新预设有调色参数时）
```bash
# 每张原片一个调试目录（matte.png、skin_union.png、landmarks.json）
for f in test/原片/*; do s=$(basename "${f%.*}"); retouch apply -i "$f" -o out/_work/<p>/dbg/$s.jpg --preset cream --debug-dir out/_work/<p>/dbg/$s; done
python -X utf8 tools/grade_fit.py --orig test/原片 --ref "test/像素蛋糕“<预设>”预设产物" --debug out/_work/<p>/dbg --size 17 --smooth 3e-3 --loo
python -X utf8 tools/grade_fit.py ... --cube luts/<name>.cube --title <预设名>        # 定稿后写 LUT
```
- 先看"同一张照片内"的误差（一半像素建表、另一半测试）：小（≈1）说明是逐像素颜色映射；再看留一误差：远大于前者说明
  各张之间不是同一张 LUT，要找自适应规则（深色内景是按 min(RGB) 分位的软黑点，`--search-black` 网格搜索）。
- LUT 取 17³ + 平滑 3e-3：更大的 LUT 在样本上误差更低，但留一更差（记住了每张照片自己的颜色）。
- 主体（人物）与背景分开统计：像素蛋糕可能单独提亮人物（深色内景主体中间调 +6.5）。
- 验收：`tools/image_delta.py` 整图色差中位 ≈ 1.5 以内；逐张看全图五列对比。

### 4.2 只调色的底图（后面几步的基准）
```bash
python -X utf8 tools/preset_sweep.py graded --preset <p> --out out/_work/<p>/graded --ref "test/像素蛋糕“<预设>”预设产物"
```
等价于 `retouch batch --set smooth=0` 并把所有 `reshape.<性别>.<项>` 设 0（**只关磨皮时形变仍会做**），同时写出
`skin_pairs.json`。没有调色的预设跳过这一步，底图就是原片。

### 4.3 脸部色调
```bash
python -X utf8 tools/skin_tone_fit.py out/_work/<p>/graded/skin_pairs.json      # 有调色：底图 → 参考
# 无调色时 pairs.json 的 orig 用原片、faces 用原片的关键点（retouch batch --landmarks-dir）
```
按性别给出 `midtone_lift / highlight_suppress / lift_per_b / pull_a / target_a / pull_b / target_b / yellow_*` 与 R²、逐脸残差。
写进各性别参数函数。逐脸残差 ±1～3 是比例模型的上限（像素蛋糕逐脸自适应），不要为某一张脸去改常数。

### 4.4 身体色调
```bash
# 调试目录直接用 4.1 那组（skin_body.png、landmarks.json 在原图几何上，底图没有形变）
python -X utf8 tools/body_tone_fit.py --orig out/_work/<p>/graded --ref "test/像素蛋糕“<预设>”预设产物" --debug out/_work/<p>/dbg [--heatmap 热图.png]
```
身体色调明显强于奶油肌时（提亮 > 5 或降黄拉力 > 0.3）打开 `body_tone_spill`，否则语义模型漏掉的皮肤会显成色块。

### 4.5 磨皮与立体
```bash
# 基线：磨皮全关时各档的上限（祛斑、形变插值已经削弱了细节，校准目标不能高于它）
python -X utf8 tools/preset_sweep.py run --preset <p> --ref <参考> --name off --base out/_work/<p>/graded \
    --both detail_smooth=0 --both fine_smooth=0 --both low_smooth=0
# 试一组
python -X utf8 tools/preset_sweep.py run --preset <p> --ref <参考> --name v1 --base out/_work/<p>/graded \
    --face detail_smooth=0.33 --face fine_smooth=0.15 --face low_smooth=0.5 --face stereo=0.22 \
    --male detail_smooth=0.33 --male fine_smooth=0.10 --male low_smooth=0.5 --male stereo=0.32
```
每组约 2～3 分钟，只打印汇总：
- 皮肤带通中位比（σ 0.004 / 0.008 / 0.016 / 0.032 / 0.064 瞳距）：第 1 档对应 `fine_smooth`，第 2–4 档对应
  `detail_smooth`（毛孔到斑驳），第 5 档对应 `low_smooth`。以"各档与参考之差的绝对值之和"为目标（两个预设定稿都在
  0.16～0.22），同时看有没有哪一档偏 0.1 以上。
- 大尺度结构增益 α（0.064 / 0.128 / 0.256）：参考 > 1 说明有立体，用 `stereo` 对齐（深色内景女 1.02/1.05/1.06、男
  1.04/1.09/1.11）。
- 调一个参数会牵动相邻档：`fine_smooth` 会连毛孔一档一起压（频段分界 0.005 在第 1、2 档之间）；立体会把斑驳一档
  抬高，要用 `low_smooth` 压回。不建议改 `fine_sigma` / `mid_sigma`（颈纹处理也用它们）。
- 验收：扫参定稿后看 13 张脸的五列对比（§5），数字好但看着"敷粉"、"塑料"、"斑驳"时以目视为准再调。

### 4.6 牙齿
```bash
retouch apply -i test/原片/1V3A3101.JPG -o out/_work/<p>/t.jpg --preset <p> --debug-dir out/_work/<p>/t_dbg
python -X utf8 tools/teeth_compare.py --debug out/_work/<p>/t_dbg --orig test/原片/1V3A3101.JPG \
    --ref "test/像素蛋糕“<预设>”预设产物/1V3A3101.jpg" --ours out/_work/<p>/t.jpg --out out/_work/<p>/teeth.jpg
```
7 张原片里只有 1V3A3101、1V3A2954 两位新娘露齿（3127 新娘闭嘴）。模型只有一个强度（红度、黄度、亮度一起拉），参考的
"明度""祛黄"是两个独立滑块，通常只能对齐红黄、亮度差 2～4。

### 4.7 美型
`tools/batch_compare.py` 的"下颌 x 差 / 下巴 y 差"（光流位移，% 瞳距）。先按滑块比例从已有预设换算初值（如瘦脸 0.15 → 0.25
对应 `thin_face` 0.15 → 0.32），再看下颌上 / 中 / 下三段的位移分布调 `warp_coeffs.thin_face_profile`。男性下巴 y 差
−12～−18 若只出现在有双下巴的照片上，是像素蛋糕"祛双下巴"改了明暗，不是形变，不要去追。

### 4.8 多人与遮罩
```bash
retouch apply -i test/原片/IMG_5785.jpg -o t.jpg --preset <p> --debug-dir dbg
python -X utf8 tools/mask_overlay.py --debug dbg --image test/原片/IMG_5785.jpg --out overlay.jpg
```
报告两张脸遮罩的重叠面积；贴脸、亲吻照（批量测试 1V3A2837、3188、3194，原片 IMG_5785）必看。`face_ownership` 打开后
重叠部分只按自己的参数处理。

## 5. 最终对照组与目检
1. 四批处理（每个预设 × 原片 / 批量测试）→ `out/最终对照组/<预设>/<批>/`（带 `--landmarks-dir`），`python tools/sample_montages.py final`；
2. 指标写进 `out/最终对照组/分析/`：`image_delta.py`（整图）、`batch_compare.py`（逐脸纹理 / 肤色 / 几何）、`skin_bands.py`、
   `band_gain.py`、`face_tone.py`（逐脸与脸 − 身体）；
3. 目检顺序：先把批量测试的全图拼成总览（每张 3 张图纵向拼接、宽 2400）快速扫一遍，再只看数字或总览里可疑的脸 / 脖子
   五列图。检查清单：
   - 脸上成块的颜色或明暗、边界分明的区域 → 先查遮罩（§4.8），不是磨皮问题；
   - 睫毛、眉毛、嘴唇、耳朵颜色异常（被当成皮肤处理了）；
   - 脸和脖子 / 身体不连贯（面具感）、漏检的一片皮肤（色块、黄斑）；
   - 衣物、花束、背景被当成皮肤（颜色被改）；
   - 光晕、接缝、裁切框边线；
   - 牙齿、眼白；侧脸、小脸、低分辨率脸；
   - 性别判错（侧脸新娘判成男性会用男性参数）。

## 6. 回归与门禁
```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo clippy -p portrait-retouch-gui --all-targets -- -D warnings
cargo test && cargo test -p portrait-retouch-gui && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
# 已有预设逐字节不变（新批处理输出与 out/batch*、out/最终对照组 下已发布的逐个 cmp）
for f in out/batch/*.jpg; do cmp -s "$f" "<新输出目录>/$(basename "$f")" || echo "DIFF $f"; done
# golden：只重新生成新预设那一组（删掉它的 PNG 后运行，其余各组照常比对）
rm tests/golden/*/<name>.png; PORTRAIT_SAMPLES=tests/samples cargo test --release --test golden
retouch preset --name <name> -o presets/<name>.json
```

## 7. 文档
- `doc/analysis/<name>.md`：参数构成、每项的逆向分析与拟合数据（照 wedding_dark_interior.md 的结构）；
- `doc/test_report_<name>.md` 与 `doc/test_report_final.md`（加一节）；
- `doc/presets.md` 的参数对照表加一列；`CHANGELOG.md`；`README.md` 功能表一行；`out/README.md`；
- 工具的新用法写进工具自己的文档字符串。

## 8. 踩过的坑

### 度量
1. **按 RMS 纹理能量校准磨皮会磨得太轻**：像素蛋糕的清晰度 / 对比度放大少数强边缘，RMS 被它们主导。用中位幅度
   （`skin_bands.py`）。
2. **均值都对上不代表没有色块**：按亮度分档、按区域的均值都接近时，色块仍可能在空间上存在。看观察层（色度偏差 ×4、
   去大尺度后的亮度 ×6）与遮罩叠加图。
3. **只看平均会掩盖逐脸问题**：先看逐脸行找离群，再看平均。
4. **输出做过形变**：逐像素比较前用 DIS 光流把输出对齐回原图几何（`compare_to_reference.flow_align`）；在输出几何上画差异图时
   要记得调试遮罩是原图几何。
5. **JPEG 会在改动像素周围的 8×8 块里产生小差异**：判断"改动是否只在某区域"用 PNG 输出（`-o x.png`）。
6. **磨皮全关也到不了 1.0**：祛斑、AI 瑕疵、形变插值都会削弱细节；先跑基线，目标不能高于它。
7. **样本少（7 张、13 张脸）**：为一张脸调常数通常会让别的脸更远；以总偏差 + 目视为准，逐脸残差记为已知差异。

### 遮罩与多人
8. **两人脸挨得近时人脸解析会越界**（一个人的裁剪框里有另一个人的脸）：那片按两人的参数各处理一遍。强色调预设里显成
   边界分明的色块、睫毛发金、脸上竖直接缝。新预设一律开 `face_ownership`。用轮廓而不是"到脸中心的距离 / 脸尺度"判归属
   （亲吻照大脸尺度大，会把别人的额头判给自己）。
9. **语义皮肤模型会漏掉皮肤**：红裙映红的胸口、强高光。色调强时显成黄斑 → `body_tone_spill`。外溢按颜色找相似皮肤时，
   粉色的花束、扇骨也"像皮肤"：只给肤色色相（≥ 35°）折让；确认的外溢皮肤再加入参考色细化。
10. **一个缺陷会被另一个参数"掩盖"**：IMG_5785 的色块本是遮罩越界，当时加重磨皮压下去了；修掉越界后磨皮就过重
    （脸像敷粉）。**先找根因**（叠加图、调试遮罩），修掉缺陷后重新校准受影响的参数。
11. **人像 alpha 有误检块**（背景墙的圆灯被当成人，主体提亮后成亮斑）→ `subject_weights` 只留与人脸相连或足够大的块。
12. 人脸检测会把裙摆亮片、花束当成脸（得分 < 0.95），对比图与评估都按得分 ≥ 0.95 过滤。

### 调色
13. 硬黑点把头发压死 → 软黑点；逐像素拉伸会放大纹理 → 只在保边底图上做、细节加回。
14. 大 LUT 泛化差（记住每张照片的颜色）→ 17³ + 平滑，用留一误差选。
15. 调色在修图之前：肤色、身体色调常数要在只调色的底图上重新拟合，不能沿用在原片上拟合的值。

### 牙齿
16. 固定红度门限会漏掉被唇色、暖光映红的上半截牙齿 → 以口腔最亮像素的红度为参考的自适应门限。
17. 调试图与流水线用的颜色不一致会误判（调试图曾在原图颜色上算门限）→ 同一份代码路径（`Engine::teeth_weights`）。
18. 先确认照片里真的露齿（1V3A3127 新娘是闭嘴的，早期文档写错过）。

### 流程与工具
19. `--smooth`、`--thin-face` 等命令行参数会被预设覆盖，改预设参数用 `--set 键=值`（如 `--set smooth=0`、
    `--set cream_female.stereo=0.2`、`--set reshape.female.thin_face=0`）。
20. Windows 上批处理占用 `target/release/retouch.exe` 时 `cargo build/test --release` 会因无法覆盖文件失败；golden 测试
    也会重新链接，不要和批处理同时跑。
21. 长任务（批处理、整批对比图）放后台，用 `until grep -q "== done" log; do sleep 15; done` 等待；前台命令超过 10 分钟会被移到后台。
22. 路径含中文和中文引号（“奶油肌”）：整条路径加引号；Python 控制台输出用 `python -X utf8`；含中文的补丁脚本写成文件
    再运行，不要塞进带反引号的 heredoc。
23. clippy 不接受测试里"先 `Default` 再改字段"（用结构体更新语法）；公开字段的文档不能链接私有类型（rustdoc 报错）。
24. 参考导出目录里的子文件夹（如"原片测试_导出"）会被 `glob('*')` 取到：工具里只取文件。

## 9. 省时间、省 token 的做法
- **按 §4 的顺序做**：调色 → 底图 → 色调 → 身体 → 磨皮 / 立体 → 牙齿 → 美型 → 多人 → 最终对照。倒过来做会反复返工
  （改调色后色调要重拟、修遮罩后磨皮要重调）。
- **一条命令一轮**：`preset_sweep.py run` 只输出汇总行；多组参数写在一个后台脚本里串行跑，结束后一次读结果。
- **少看图、看对的图**：先用数字（逐脸行）找离群的脸，再只看那几张的放大图；看总览时把多张图拼成一张（宽 2000～2400）。
  一张 5 列全图约相当于数千 token，别一张张读。
- **复用已有目录**：调试目录（`--debug-dir`）、只调色底图、最终对照组的关键点都可以重复用，不用每轮重跑检测。
- **先读本文与 presets.md，再读代码**：代码位置都在表里，不必全文搜索；只读要改的那一段。
- **小脚本写成文件放 `out/_work/`**，反复用的提升到 `tools/`（带文档字符串与参数说明）。
- **每轮结束写下结论**（参数、指标、为什么），写进分析文档或记忆，避免下次重新推导。

## 10. 典型工作量
- 只改数值的新预设（功能都在 §2 表里）：调色 1～2 轮、色调 1 轮、磨皮 / 立体 3～5 组扫参、美型 1～2 轮、最终对照 1 轮。
- 需要新算子（表里没有的像素蛋糕功能）：每个功能要先做"它到底做了什么"的分析（与底图逐像素对比、按尺度 / 亮度 / 区域
  分解），再实现、再校准，通常是整个预设里最花时间的部分。
