# portrait-retouch

纯 Rust（CPU + rayon）人像修图库与命令行：**磨皮 · 美白 · 提亮/饱和 · 瘦脸 / 大眼 / 瘦鼻 · LUT 风格滤镜**。
按 [`doc/rust_portrait_retouch_plan.md`](doc/rust_portrait_retouch_plan.md) 实现，在不引入 Objective-C / OpenGL ES /
闭源 SDK 的前提下复现并扩展 Meihu-Beautyface-sdk 的修图效果。参考项目的逐文件分析见 [`doc/analysis/`](doc/analysis/README.md)。

| 能力 | 实现 | 来源 |
|---|---|---|
| 磨皮 A（直播磨皮，`faithful`） | GPUImage 双边 + Sobel + 肤色规则组合 + log 提亮 + HSB | 美狐 `BBGPUImageBeautifyFilter` |
| 磨皮 B（自然磨皮，`freqsep`） | 高反差保留 + 肤色曲线 + 亮度锐化 | YUCIHighPassSkinSmoothing |
| 磨皮 C（`gpupixel`） | 均值 / 方差自适应磨皮 | pixpark/gpupixel |
| **奶油肌**（`cream`，预设 `--preset cream`） | 人脸解析 + 人像抠图 + 语义皮肤分割的皮肤遮罩、AI 瑕疵分割 + 修复（ABPN）、疤痕 / 大痣修复画笔（能量判据 + 调和插值 + 纹理移植）、能量自适应的保纹理磨皮、匀肤 / 肤色统一、奶油色调、眼部清晰、按性别的瘦脸 / 缩下巴；与像素蛋糕一样再应用输入 JPEG 内嵌的 Camera Raw 设置 | 对像素蛋糕「奶油肌」导出图的逆向分析（doc/analysis/cream_skin.md、abpn.md；两张样张的对比 doc/test_report_cream.md、doc/test_report_x04.md） |
| 美白 | 参数化曲线或 512 查找图 × 皮肤遮罩 | 方案 4.4 |
| 瘦脸 / 大眼 / 瘦鼻 | `warpPositionToUse1` / `adjust_eye` / `newNarrowNose_2` 像素空间移植（另有 gpupixel 风格） | 美狐 `GLImageFaceChangeFilter` |
| 风格滤镜 | 512×512 查找图、.cube 3D LUT、带空间遮罩的 LUT | GPUImage Lookup |
| 人脸 | UltraFace RFB-320 检测；2d106det（开发期）/ MediaPipe Face Mesh（发布期）关键点 → 语义结构体 | — |

## 目录

```
src/            库：buffer / geom / color / skin / face / warp / pipeline / engine / options / photo / batch / debug / ui_map
src/bin/        retouch 命令行
gui/            图形界面（Tauri 2，工作区成员 portrait-retouch-gui）：手动测试各项功能，见 gui/README.md
models/         ONNX 模型（不入库，scripts/fetch_models.py）
runtime/        onnxruntime 动态库（不入库，scripts/fetch_onnxruntime.py）
luts/           恒等 / 示例 LUT 与 index.json（tools/make_luts.py 生成）
tools/          bake_lut.py（曲线滤镜包烘焙）、oracle_numpy.py（数值对照）、make_luts.py、
                compare_to_reference.py（与参考导出图定量对比）、texture_energy.py（按局部能量分箱的纹理衰减）、
                develop_fit.py（Camera Raw 设置再应用的模型拟合）、montage.py（并排裁剪对比图）、
                batch_compare.py（整批与参考导出逐脸对比）、skin_tone_fit.py（肤色目标色模型拟合）、
                sample_montages.py（重新生成文档引用的样张对比图）、
                abpn/（ABPN 网络定义与 ONNX 导出）
tests/          identity.rs（恒等性）、golden.rs（视觉回归，需样张目录）
test/           用户提供的样张与像素蛋糕导出图（真实人物照片，不入库；各测试报告的复现命令需要本地有这些照片）
benches/        criterion 基准
examples/       dump_landmarks.rs（关键点核对）
example/        参考项目浅克隆（scripts/fetch_examples.sh，不入库）
doc/            方案文档与源码分析笔记
```

## 快速开始

```bash
# 1. 模型与运行时（网络受限时可手动放置；见 THIRD_PARTY_LICENSES.md）
python scripts/fetch_models.py            # models/version-RFB-320.onnx, 2d106det.onnx, face_mesh.onnx
                                          # + 奶油肌可选模型 face_parsing_resnet18.onnx, modnet_photographic.onnx, genderage.onnx
python scripts/fetch_models.py --only abpn # 可选：AI 瑕疵祛除 abpn_blemish_{detect,inpaint}.onnx（需 torch；从 ModelScope 权重导出）
# 语义皮肤分割（强烈建议，否则浅色衣物 / 饰品会被当作皮肤处理）：
#   推荐：ModelScope 皮肤分割图 → models/skin_seg.onnx，需要装了 tensorflow-cpu + tf2onnx 的 Python（建议单独 venv）
python -m venv .venv-convert
.venv-convert/Scripts/pip install tensorflow-cpu tf2onnx onnx onnxruntime   # Linux/macOS: .venv-convert/bin/pip
.venv-convert/Scripts/python scripts/fetch_models.py --only skinseg
#   备选（无需 TensorFlow）：18 MB 轻量模型 → models/skin_seg_lite.onnx，只在 skin_seg.onnx 不存在时使用
python scripts/fetch_models.py --only skinseg-lite
python scripts/fetch_onnxruntime.py       # runtime/onnxruntime.dll（或 --from-pip）
python tools/make_luts.py                 # luts/

# 2. 构建（本仓库 .cargo/config.toml 使用 rsproxy 镜像；网络正常可删除）
cargo build --release

# 3. 修图
retouch apply -i photo.jpg -o out.jpg --smooth 0.5 --whiten 0.3 --thin-face 0.5 --big-eye 0.3 --thin-nose 0.4 \
              --lut luts/warm.cube --lut-intensity 0.8
retouch apply -i photo.jpg -o out.jpg --mode freqsep --smooth 0.75          # 自然磨皮
retouch apply -i photo.jpg -o out.jpg --preset cream                        # 奶油肌一键预设（需 models/ 下的解析 / 抠图模型）
retouch apply -i photo.jpg -o out.jpg --preset cream --set cream_female.detail_smooth=0.3 --set reshape.female.chin_lift=0.6
retouch apply -i photo.jpg -o out.jpg --preset cream --no-ai-blemish --debug-dir dbg # 只用经典瑕疵检测 / 修复；dbg/ 导出遮罩与 ai_holes.png
retouch apply -i photo.jpg -o out.jpg --preset cream --skinseg-model models/skin_seg_lite.onnx   # 指定皮肤分割模型（--no-skinseg 关闭）
retouch apply -i photo.jpg -o out.jpg --preset cream --embedded-develop   # 再应用 JPEG 内嵌的 Camera Raw 设置（可选，见下）
retouch batch -i photos/ -o retouched/ --preset cream                       # 批量：整个目录（-r 递归），见下
retouch preset -o my_preset.json                                            # 导出内置预设，编辑后 --preset my_preset.json
retouch apply -i photo.jpg -o out.jpg --landmarks facemesh ...             # 发布期关键点模型
retouch detect -i photo.jpg -o vis.png --json lm.json --indices            # 关键点可视化 / 核对
retouch bench  -i photo.jpg                                                # 分阶段计时
retouch identity-lut -o identity.png                                       # 恒等查找图（设计师调色用）
```

ONNX Runtime 动态库查找顺序：`--ort-dylib` → `ORT_DYLIB_PATH` → `./runtime/` → 可执行文件目录 → 系统路径。

- **批量处理**（`retouch batch`，doc/test_report_batch.md）：输入可以是目录或多个文件（`-r` 递归、`--ext` 限定扩展名），
  输出与输入同名（扩展名换成 `--format jpg|png`），保留子目录结构。模型只加载一次，读写与修图重叠。
  单张失败只记入报告、不中断整批；输出原子写入，已存在的输出默认跳过，中断后重跑同一命令即可续跑（`--overwrite` 覆盖）。
  输出目录里写 `batch_report.csv`（Excel 可直接打开）与 `.json`：人脸数 / 性别、各阶段耗时、失败原因；有失败时退出码非零。
  `--dry-run` 只列出任务；`--landmarks-dir` 输出关键点 JSON（供 `tools/batch_compare.py` 对比）。
  24 核 CPU 上 20–30 MP 的人像约 13 s / 张，内存峰值约 5–6 GB（与单张相同）。
- **照片方向**：读入时按 EXIF 方向把像素转正（相机直出的竖拍照片），写回的 EXIF 方向置为 1。
- **Camera Raw 设置**（可选，默认关闭）：Lightroom / ACR 导出的 JPEG 会在 XMP 里记录冲印参数，并标注
  `crs:AlreadyApplied="True"`。`--embedded-develop` 会用拟合模型把这些参数再应用一次，复现像素蛋糕在"读取 XMP 调色"的
  项目里导出的效果（见 doc/test_report_x04.md §6）。「奶油肌」预设本身不做这一步（doc/test_report_batch.md §2.1）。
  库接口：`if let Some(s) = color::develop::DevelopSettings::from_path(path)? { s.apply_rgb8(&mut img) }`（在人脸检测之前调用）。
- **JPEG 输出**：jpeg-encoder，4:4:4，标准霍夫曼表，`--jpeg-quality` 默认 98（与像素蛋糕导出的量化表一致），
  写回原图的 ICC 与 EXIF（不写 XMP）。

## 图形界面

```bash
cargo run --release -p portrait-retouch-gui
```

手动测试用的桌面界面（[gui/README.md](gui/README.md)）：打开照片后拖滑块实时看效果（缩小的工作分辨率上处理，
保存时按原图重做），对比 / 并排 / 按住空格看原图，查看关键点、皮肤遮罩、人像抠图、皮肤概率、AI 修复区等中间结果；
批量处理（进度、取消、报告）；引擎设置（模型目录、关键点模型、可选模型开关）。参数与命令行选项一一对应
（共用库的 `ParamOptions`），保存结果与命令行输出逐字节相同。根目录的 `cargo build / test` 不包含界面。

## 库 API

```rust
use portrait_retouch::{Engine, EngineConfig, LandmarkKind, RetouchParams, SmoothMode};

let engine = Engine::new(EngineConfig { landmark: LandmarkKind::FaceMesh, ..Default::default() })?;
let img = image::open("photo.jpg")?.to_rgb8();
let faces = engine.detect_faces(&img)?;                 // 只检测一次，滑块调整时复用
let params = RetouchParams { smooth: 0.6, smooth_mode: SmoothMode::FreqSep, whiten: 0.3,
                             thin_face: 0.5, big_eye: 0.3, ..Default::default() };
let out = engine.retouch(&img, &faces, &params);        // 纯函数；内部按图像哈希缓存中间结果

// 照片读写：按 EXIF 方向转正、保留 ICC / EXIF、原子写入
let photo = portrait_retouch::Photo::load("photo.jpg".as_ref())?;
portrait_retouch::photo::save(&out, "out.jpg".as_ref(), 98, &photo.meta)?;

// 用户参数 → RetouchParams（命令行与图形界面共用；字段即 `retouch apply` 的选项）
let params = portrait_retouch::ParamOptions { preset: Some("cream".into()), ..Default::default() }.build()?;

// 批量：列出任务 → 执行（进度回调；BatchConfig::cancel 可随时取消）→ 报告
use portrait_retouch::batch::{self, BatchConfig};
let params = portrait_retouch::Preset::cream_skin().to_params()?;
let cfg = BatchConfig::new(vec!["photos".into()], "retouched".into());
let items = batch::plan(&cfg)?;
let report = batch::run(Some(&engine), &params, &cfg, &items, &mut |_event| {})?;
report.write(&cfg.output_dir)?;                         // batch_report.csv / .json
```

- **预览 / 导出**：UI 持有长边 1280 的缩略图与 `faces.iter().map(|f| f.scaled(s))`，滑块变化时对缩略图调 `retouch`；
  导出时对原图调同一函数。所有参数以瞳距和固定短边工作副本为尺度，两者观感一致（测试：PSNR ≥ 40 dB）。
- **UI 滑块映射**：`ui_map::UiSliders`（附录 B）。
- **形变系数**：`WarpCoefficients`（JSON 可配，`--coeffs`），侧脸自动衰减。
- **多人脸**：按面积升序应用；重叠 > 30% 时只处理最大脸。

## 验证

```bash
cargo test                                   # 75 单元 + 10 恒等性测试（只含库与命令行）
cargo test -p portrait-retouch-gui           # 图形界面后端 19 个单元测试；界面冒烟测试见 gui/README.md
PORTRAIT_SAMPLES=tests/samples cargo test --release --test golden   # 视觉回归（样张目录不入库；UPDATE_GOLDEN=1 更新）
cargo bench --bench retouch                  # 12MP / 1280 预览基准
python tools/oracle_numpy.py in.png out.png  # 与 numpy 直译公式逐像素对照（≤ 2/255）
```

实测（24 核桌面 CPU，11.5 MP 样张）：检测 + 关键点 45 ms；全流程冷缓存 207 ms、热缓存 125 ms；1280 预览 13 ms；
numpy oracle 对照最大误差 1/255。奶油肌预设：28.5 MP 双人样张修图 13–15 s（其中 AI 瑕疵约 6 s），
20 MP 的 X04 约 5 s；批量 7 张 20–30 MP 原片 92 s（13.2 s / 张）。

## 许可证与合规

代码 MIT OR Apache-2.0。参考项目、模型与运行时的许可证见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
`2d106det.onnx` 仅限非商用，商用发布请切换 `--landmarks facemesh`。
