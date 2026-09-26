# Portrait Retouch Studio（手动测试界面）

portrait-retouch 的桌面测试界面（Tauri 2 + WebView2），用来手动试各项修图功能：打开照片、拖滑块看效果、
对比原图 / 结果、查看中间遮罩、保存，以及批量处理。

界面只是库的一层薄壳：参数组装（`ParamOptions`）、读写图（`photo`）、批处理（`batch`）都直接复用库本身，
与命令行 `retouch` 走同一套代码。实测：界面以「奶油肌」保存的原图分辨率结果与 `retouch batch --preset cream`
的输出**逐字节相同**。

## 运行

```bash
cargo run --release -p portrait-retouch-gui
```

- 模型与 ONNX Runtime 与命令行相同（根目录 README「快速开始」）：`models/`、`runtime/onnxruntime.dll`。
- 在项目根目录运行最省事；直接双击 `target/release/portrait-retouch-gui.exe` 也可以：模型目录会从
  程序所在位置往上查找 `models/`，ONNX Runtime 也会找 `../../runtime/`。
- Windows 10 / 11 自带 WebView2 运行时。
- 界面是工作区的第二个成员；根目录的 `cargo build` / `cargo test` / `cargo clippy` 只作用于库与命令行，
  行为与加入界面之前完全相同。

## 功能

### 单张调试

| | |
|---|---|
| 打开 | "打开照片"、Ctrl+O，或把照片拖进窗口。读入时按 EXIF 方向转正；显示尺寸、ICC、人脸（性别 / 年龄 / 瞳距 / 侧脸角）、内嵌 Camera Raw 设置 |
| 工作分辨率 | 1024 / 1600（默认）/ 2400 / 原图。调参在缩小图上进行（人脸关键点同比缩放），保存时按原图分辨率重新处理 |
| 参数 | **预设**：内置「奶油肌」或 JSON 预设文件，"覆盖项"即命令行的 `--set key.sub=value`，可查看预设全部字段；**手动**：四种磨皮（直播 / 自然 / gpupixel / 奶油肌）、亮度 / 饱和度 / log 曲线 / 美白（可选 512 查找图）、瘦脸 / 大眼 / 瘦鼻 / 缩下巴 / 整体收窄 / 形变风格 / 侧脸衰减；两种模式都有：身体皮肤、AI 瑕疵祛除、风格 LUT 与强度、内嵌冲印设置、形变系数 JSON、遮罩 LUT。字段与命令行选项一一对应，默认值相同 |
| 自动处理 | 参数改动后稍等片刻自动处理；处理中又改了参数，结束后按最新参数再处理一次 |
| 查看 | 对比（拖动分割线）、并排（同步缩放）、原图、结果；按住空格看原图 |
| 调试视图 | 关键点、皮肤遮罩（粉 = 脸部，橙 = 身体）、人像抠图、皮肤概率、AI 修复区 |
| 缩放 | 滚轮以光标为中心缩放、拖动平移、双击在 100% 与适应窗口之间切换（F / 1） |
| 保存 | Ctrl+S；按扩展名存 JPEG（质量 98，写回 ICC / EXIF）或 PNG |

参数、工作分辨率、视图与批处理设置保存在界面本地，下次启动沿用；"恢复默认参数"回到命令行默认值，
"复制参数 JSON"得到的就是库的 `ParamOptions`。

### 批量处理

输入目录或照片（可混合，可拖入）、输出目录、格式（JPEG / PNG）、JPEG 质量、是否递归子目录 / 覆盖已存在的输出 /
导出关键点 JSON。修图参数沿用"单张调试"页的当前参数。

"预览任务"列出将处理的照片；"开始批处理"后逐张显示状态、人脸、各阶段耗时，估算剩余时间；"取消"让正在处理的那张
照常完成、其余记为"已取消"。结束后可打开输出目录与 `batch_report.csv / .json`。批处理进行中单张页的操作暂停。

### 引擎设置

模型目录、关键点模型（2d106det 开发期 / Face Mesh 发布期）、ONNX 线程数、各可选模型开关（人脸解析、人像抠图、
性别年龄、AI 瑕疵、语义皮肤分割）。应用后下次使用时按新设置加载模型，当前照片自动重新检测人脸。
状态卡片显示已加载的模型、实际使用的 ONNX Runtime 路径与加载失败的原因。没有模型时照片照样能打开，
只做与人脸无关的调整。

## 结构

```
gui/
├── src-tauri/                 Rust 后端（工作区成员 portrait-retouch-gui）
│   ├── src/lib.rs             入口：插件、应用状态、photo:// 协议、命令注册
│   ├── src/commands/          前端命令：app.rs（应用信息 / 引擎 / 对话框）、photo.rs（单张）、batch.rs（批处理）
│   ├── src/state.rs           应用状态：引擎宿主、当前照片、工作锁、批处理凭据
│   ├── src/session.rs         当前照片：工作分辨率图像、处理结果、调试视图缓存、图像版本号
│   ├── src/views.rs           调试视图的渲染
│   ├── src/protocol.rs        photo:// 图像协议
│   ├── src/engine_host.rs     引擎设置、按需加载、状态
│   ├── src/error.rs           命令错误（前端收到一段可读的文字）
│   ├── tauri.conf.json        窗口、CSP；capabilities/ 权限；icons/ 图标
│   └── build.rs
├── ui/                        前端：原生 ES 模块，没有构建步骤
│   ├── index.html · styles.css（浅色 / 深色跟随系统）
│   └── js/
│       ├── api.js             命令名、事件名、图像 URL —— 与后端的全部约定都在这里
│       ├── schema.js          参数表单的描述（字段名即 ParamOptions）
│       ├── params.js          参数表单
│       ├── viewer.js          图像查看器（分割 / 并排 / 缩放 / 平移）
│       ├── single.js · batch.js · engine.js   三个页面
│       ├── modal.js · dom.js  对话框、小工具
│       └── main.js            入口
└── tests/
    ├── smoke.mjs              冒烟测试（CDP 驱动真实界面）
    └── tauri.smoke.json       冒烟测试构建用的配置覆盖（打开 WebView2 调试端口）
```

### 设计要点

- **不另起炉灶**：界面传给后端的参数就是 `ParamOptions` 的 JSON，后端调 `ParamOptions::build()`，与命令行共用；
  单张处理 = `Engine::retouch`，保存 = `photo::save`，批处理 = `batch::plan` + `batch::run`（取消用 `BatchConfig::cancel`）。
- **一把工作锁**：打开、处理、调试视图、保存、改引擎设置、批处理都在工作锁内串行执行，结果不会与照片或工作分辨率错配；
  批处理整批持有工作锁，期间单张操作直接拒绝（不排队）。批处理的"运行中"状态由凭据对象持有，线程结束（包括异常展开）时自动释放。
- **图像不走 IPC**：结果与调试视图缓存在后端，前端 `<img>` 通过自定义协议 `photo://<视图>?v=<版本>` 取 JPEG；
  版本号在进程内唯一递增，换照片后 URL 也不会与上一张重复。
- **耗时操作不占界面线程**：命令都是 async，计算放在阻塞线程池；批处理在单独线程，进度以事件推送（`batch-progress`、`batch-finished`）。
- **引擎按需加载**：第一次需要时加载（约 1 s），改设置后丢弃、下次按新设置加载；ONNX Runtime 每个进程只初始化一次。

## 开发

```bash
cargo build --release -p portrait-retouch-gui      # 改了 ui/ 也要重新编译：前端资源在编译时嵌入（只重编界面 crate，约 1 分钟）
cargo test  --release -p portrait-retouch-gui      # 后端单元测试
cargo clippy --release -p portrait-retouch-gui --all-targets -- -D warnings
```

- 新增一个参数：库里给 `ParamOptions` 加字段（命令行随之加选项），界面在 `ui/js/schema.js` 里加一行描述即可。
- 新增一个命令：`src/commands/` 里写函数并在 `lib.rs` 的 `generate_handler!` 注册，前端在 `ui/js/api.js` 里加一个包装。

### 冒烟测试

`tests/smoke.mjs` 通过 Chrome DevTools 协议驱动真实界面走一遍主要流程：打开照片 → 四种磨皮 → 奶油肌 → 五个调试视图 →
切换工作分辨率 → 单张批处理（输出到临时目录）。它需要一个打开了 WebView2 调试端口的构建——端口只加在这个测试构建里，
正式构建不带。用单独的目标目录构建，不覆盖正式的可执行文件：

```bash
# Git Bash
TAURI_CONFIG="$(cat gui/tests/tauri.smoke.json)" CARGO_TARGET_DIR=target/smoke cargo build --release -p portrait-retouch-gui
./target/smoke/release/portrait-retouch-gui.exe &          # 在项目根目录启动
node gui/tests/smoke.mjs path/to/portrait.jpg              # 需要 Node 22+
```

```powershell
# PowerShell
$env:TAURI_CONFIG = Get-Content -Raw gui/tests/tauri.smoke.json; $env:CARGO_TARGET_DIR = "target/smoke"
cargo build --release -p portrait-retouch-gui; Remove-Item Env:TAURI_CONFIG, Env:CARGO_TARGET_DIR
Start-Process target/smoke/release/portrait-retouch-gui.exe; node gui/tests/smoke.mjs path/to/portrait.jpg
```

每一步输出 PASS / FAIL，有失败时退出码非零；测试前后保存 / 恢复界面的本地偏好。

## 常见问题

- **顶栏显示"模型加载失败"**：到"引擎设置"看原因。必需模型是人脸检测 `version-RFB-320.onnx` 与所选关键点模型
  （`2d106det.onnx` 或 `face_mesh.onnx`），其余可选模型缺失时自动跳过。
- **找不到 ONNX Runtime**：把 `onnxruntime.dll` 放到 `runtime/`，或设置 `ORT_DYLIB_PATH`；更换运行时需要重启界面。
- **处理较慢**：「奶油肌」首次处理一张照片需要跑抠图 / 皮肤分割 / AI 瑕疵模型（1600 px 工作分辨率约 3–6 s），
  之后调参复用这些中间结果（通常 < 1 s）。保存时按原图分辨率重做，20–30 MP 约 10–20 s。
- **许可**：`2d106det.onnx`、`genderage.onnx` 仅限非商用；发布请在"引擎设置"切换到 Face Mesh 并关闭性别年龄模型。
