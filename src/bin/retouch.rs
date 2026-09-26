//! `retouch` 命令行：修图 / 关键点检测可视化 / LUT 工具 / 性能计时 / 预设。

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use portrait_retouch::batch::{self, BatchConfig, BatchEvent, ItemStatus, OutputFormat};
use portrait_retouch::color::lookup512::identity_lookup512;
use portrait_retouch::color::lut3d::Lut3D;
use portrait_retouch::debug;
use portrait_retouch::photo::{self, Photo};
use portrait_retouch::preset::Preset;
use portrait_retouch::{
    Engine, EngineConfig, FaceKeyPoints, LandmarkKind, ParamOptions, ReshapeStyle, RetouchParams,
    SmoothMode,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "retouch",
    version,
    about = "纯 Rust 人像修图：磨皮 / 美白 / 瘦脸 / 大眼 / 瘦鼻 / 奶油肌 / LUT"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 对一张照片执行修图
    Apply(ApplyArgs),
    /// 批量修图：目录 / 多个文件 → 输出目录（模型只加载一次，读写与计算重叠，逐张容错，可续跑，输出报告）
    Batch(BatchArgs),
    /// 检测人脸并输出关键点可视化 / JSON（对应方案 7.4 节 dump_landmarks）
    Detect(DetectArgs),
    /// 生成恒等 512 查找图或恒等 .cube
    IdentityLut(IdentityLutArgs),
    /// 分阶段计时（对应方案 7.5 节基准）
    Bench(BenchArgs),
    /// 导出内置预设为 JSON（可编辑后用 --preset 加载）
    Preset(PresetArgs),
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// 模型目录
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    /// 关键点模型：2d106 | facemesh
    #[arg(long, default_value = "2d106")]
    landmarks: String,
    /// ONNX Runtime 动态库路径（默认自动查找 runtime/onnxruntime.dll 等）
    #[arg(long)]
    ort_dylib: Option<PathBuf>,
    /// ORT 线程数
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// 关闭人脸解析 / 人像抠图 / 性别年龄模型
    #[arg(long)]
    no_parsing: bool,
    #[arg(long)]
    no_matting: bool,
    #[arg(long)]
    no_attribute: bool,
    /// 不加载 AI 瑕疵祛除模型（abpn_blemish_*.onnx）
    #[arg(long)]
    no_ai: bool,
    /// 不加载语义皮肤分割模型（身体皮肤退化为"抠图 ∧ 肤色规则"）
    #[arg(long)]
    no_skinseg: bool,
    /// 语义皮肤分割模型路径（默认依次查找 models/skin_seg.onnx、models/skin_seg_lite.onnx）
    #[arg(long)]
    skinseg_model: Option<PathBuf>,
}

impl EngineArgs {
    fn build(&self) -> anyhow::Result<Engine> {
        let landmark: LandmarkKind = self
            .landmarks
            .parse()
            .map_err(|e: String| anyhow::anyhow!(e))?;
        let cfg = EngineConfig {
            models_dir: self.models_dir.clone(),
            landmark,
            ort_dylib: self.ort_dylib.clone(),
            threads: self.threads,
            enable_parsing: !self.no_parsing,
            enable_matting: !self.no_matting,
            enable_attribute: !self.no_attribute,
            enable_ai_blemish: !self.no_ai,
            enable_skin_seg: !self.no_skinseg,
            skin_seg_model: self.skinseg_model.clone(),
            ..Default::default()
        };
        Engine::new(cfg)
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Faithful,
    Freqsep,
    Gpupixel,
    Cream,
}

#[derive(Clone, Copy, ValueEnum)]
enum StyleArg {
    Meihu,
    Gpupixel,
}

#[derive(Args, Clone)]
struct ParamArgs {
    /// 预设：内置名（cream | cream_skin | 奶油肌）或 JSON 文件路径；指定后其余滑块参数以预设为准
    #[arg(long)]
    preset: Option<String>,
    /// 覆盖预设中的字段：`key.sub=value`，可重复（如 --set cream_female.detail_smooth=0.5）
    #[arg(long = "set")]
    sets: Vec<String>,
    /// 磨皮强度 0..1
    #[arg(long, default_value_t = option_defaults().smooth)]
    smooth: f32,
    /// 磨皮模式
    #[arg(long, value_enum, default_value_t = ModeArg::Faithful)]
    mode: ModeArg,
    /// 不限制在人脸多边形内（忠实模式：全图按肤色规则磨皮）
    #[arg(long)]
    no_face_restrict: bool,
    /// 奶油肌：不处理身体皮肤
    #[arg(long)]
    no_body: bool,
    /// 奶油肌：关闭 AI 瑕疵祛除（只用经典检测 / 修复）
    #[arg(long)]
    no_ai_blemish: bool,
    /// 再应用输入 JPEG 内嵌的 Camera Raw 冲印设置（XMP crs；默认关闭，见 color::develop）
    #[arg(long, conflicts_with = "no_embedded_develop")]
    embedded_develop: bool,
    /// 不再应用内嵌的 Camera Raw 设置（覆盖预设 JSON 中的 embedded_develop: true）
    #[arg(long)]
    no_embedded_develop: bool,
    /// HSB 亮度（1.0 = 不变，美狐默认 1.1）
    #[arg(long, default_value_t = option_defaults().brightness)]
    brightness: f32,
    /// HSB 饱和度（1.0 = 不变，美狐默认 1.1）
    #[arg(long, default_value_t = option_defaults().saturation)]
    saturation: f32,
    /// 关闭 log 提亮曲线
    #[arg(long)]
    no_log: bool,
    /// 美白 0..1
    #[arg(long, default_value_t = option_defaults().whiten)]
    whiten: f32,
    /// 美白查找图（512×512 PNG），不指定则用参数化曲线
    #[arg(long)]
    whiten_lut: Option<PathBuf>,
    #[arg(long, default_value_t = option_defaults().thin_face)]
    thin_face: f32,
    #[arg(long, default_value_t = option_defaults().big_eye)]
    big_eye: f32,
    #[arg(long, default_value_t = option_defaults().thin_nose)]
    thin_nose: f32,
    /// 缩下巴 / 提升下半脸 0..1
    #[arg(long, default_value_t = option_defaults().chin_lift)]
    chin_lift: f32,
    /// 整体收窄比例（如 0.03）
    #[arg(long, default_value_t = option_defaults().face_narrow)]
    face_narrow: f32,
    /// 形变风格
    #[arg(long, value_enum, default_value_t = StyleArg::Meihu)]
    reshape: StyleArg,
    /// 形变系数 JSON（字段见 WarpCoefficients）
    #[arg(long)]
    coeffs: Option<PathBuf>,
    /// 不做侧脸衰减
    #[arg(long)]
    no_yaw_attenuation: bool,
    /// 风格 LUT：512×512 PNG 或 .cube
    #[arg(long)]
    lut: Option<PathBuf>,
    #[arg(long, default_value_t = option_defaults().style_intensity)]
    lut_intensity: f32,
    /// 遮罩 LUT：`<cube>:<mask.png>[:invert][:strength]`，可重复
    #[arg(long = "masked-lut")]
    masked_luts: Vec<String>,
    /// 方案 B 高反差半径（相对短边 1000 px）
    #[arg(long, default_value_t = option_defaults().freqsep_radius)]
    freqsep_radius: f32,
    /// 方案 B 锐化系数
    #[arg(long, default_value_t = option_defaults().freqsep_sharpness)]
    freqsep_sharpness: f32,
    /// 方案 C 锐化系数
    #[arg(long, default_value_t = option_defaults().gpupixel_sharpen)]
    gpupixel_sharpen: f32,
}

/// 手动参数的默认值（与库中 [`ParamOptions`] 一致，命令行与图形界面共用）。
fn option_defaults() -> ParamOptions {
    ParamOptions::default()
}

impl ParamArgs {
    /// 命令行选项 → 库的 [`ParamOptions`]（组装逻辑只在库里一处）。
    fn to_options(&self) -> ParamOptions {
        ParamOptions {
            preset: self.preset.clone(),
            sets: self.sets.clone(),
            smooth: self.smooth,
            smooth_mode: match self.mode {
                ModeArg::Faithful => SmoothMode::Faithful,
                ModeArg::Freqsep => SmoothMode::FreqSep,
                ModeArg::Gpupixel => SmoothMode::GpuPixel,
                ModeArg::Cream => SmoothMode::Cream,
            },
            restrict_to_face: !self.no_face_restrict,
            brightness: self.brightness,
            saturation: self.saturation,
            apply_log_curve: !self.no_log,
            whiten: self.whiten,
            thin_face: self.thin_face,
            big_eye: self.big_eye,
            thin_nose: self.thin_nose,
            chin_lift: self.chin_lift,
            face_narrow: self.face_narrow,
            reshape_style: match self.reshape {
                StyleArg::Meihu => ReshapeStyle::Meihu,
                StyleArg::Gpupixel => ReshapeStyle::GpuPixel,
            },
            attenuate_yaw: !self.no_yaw_attenuation,
            freqsep_radius: self.freqsep_radius,
            freqsep_sharpness: self.freqsep_sharpness,
            gpupixel_sharpen: self.gpupixel_sharpen,
            body_skin: !self.no_body,
            ai_blemish: !self.no_ai_blemish,
            embedded_develop: if self.embedded_develop {
                Some(true)
            } else if self.no_embedded_develop {
                Some(false)
            } else {
                None
            },
            whiten_lut: self.whiten_lut.clone(),
            warp_coeffs: self.coeffs.clone(),
            style_lut: self.lut.clone(),
            style_intensity: self.lut_intensity,
            masked_luts: self.masked_luts.clone(),
        }
    }

    fn build(&self) -> anyhow::Result<RetouchParams> {
        self.to_options().build()
    }
}

#[derive(Args)]
struct ApplyArgs {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    params: ParamArgs,
    /// 跳过人脸检测（无形变，磨皮不限制在人脸内）
    #[arg(long)]
    no_faces: bool,
    /// 预览模式：先把长边缩到该尺寸再处理（验证预览 / 导出一致性）
    #[arg(long)]
    preview_long_side: Option<u32>,
    /// 导出中间结果（皮肤遮罩、关键点可视化）到该目录
    #[arg(long)]
    debug_dir: Option<PathBuf>,
    /// JPEG 输出质量（1–100，IJG 量化表缩放；≥ 90 为 4:4:4）。默认 98 与像素蛋糕导出的量化表一致
    #[arg(long, default_value_t = photo::DEFAULT_JPEG_QUALITY)]
    jpeg_quality: u8,
}

#[derive(Clone, Copy, ValueEnum)]
enum FormatArg {
    Jpg,
    Png,
}

#[derive(Args)]
struct BatchArgs {
    /// 输入目录或照片文件，可给多个
    #[arg(short, long, required = true, num_args = 1..)]
    input: Vec<PathBuf>,
    /// 输出目录（不能与输入目录相同；输出与输入同名，扩展名换成输出格式）
    #[arg(short, long)]
    output: PathBuf,
    /// 递归处理子目录（输出保留相对目录结构）
    #[arg(short, long)]
    recursive: bool,
    /// 只处理这些扩展名（逗号分隔，不区分大小写；默认 jpg,jpeg,png,tif,tiff,webp,bmp）
    #[arg(long, value_delimiter = ',')]
    ext: Vec<String>,
    /// 输出格式
    #[arg(long, value_enum, default_value_t = FormatArg::Jpg)]
    format: FormatArg,
    /// JPEG 输出质量（1–100，≥ 90 为 4:4:4）。默认 98 与像素蛋糕导出的量化表一致
    #[arg(long, default_value_t = photo::DEFAULT_JPEG_QUALITY)]
    jpeg_quality: u8,
    /// 覆盖已存在的输出（默认跳过：中断后重跑同一命令即可续跑）
    #[arg(long)]
    overwrite: bool,
    /// 把每张的人脸关键点 JSON（与 detect --json 同格式）写到该目录，供 tools/ 下的对比脚本使用
    #[arg(long)]
    landmarks_dir: Option<PathBuf>,
    /// 不在输出目录写 batch_report.csv / batch_report.json
    #[arg(long)]
    no_report: bool,
    /// 只列出将要处理的照片与输出路径，不加载模型、不写文件
    #[arg(long)]
    dry_run: bool,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    params: ParamArgs,
}

#[derive(Args)]
struct DetectArgs {
    #[arg(short, long)]
    input: PathBuf,
    /// 可视化输出 PNG
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// 关键点 JSON 输出
    #[arg(long)]
    json: Option<PathBuf>,
    /// 在可视化图上标注原始点索引
    #[arg(long)]
    indices: bool,
    #[command(flatten)]
    engine: EngineArgs,
}

#[derive(Args)]
struct IdentityLutArgs {
    #[arg(short, long)]
    output: PathBuf,
    /// 生成 .cube（默认生成 512 PNG）
    #[arg(long)]
    cube: bool,
    #[arg(long, default_value_t = 33)]
    size: usize,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(short, long)]
    input: PathBuf,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    params: ParamArgs,
    #[arg(long, default_value_t = 3)]
    iters: usize,
}

#[derive(Args)]
struct PresetArgs {
    /// 内置预设名（cream_skin）
    #[arg(long, default_value = "cream_skin")]
    name: String,
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Apply(a) => cmd_apply(a),
        Cmd::Batch(a) => cmd_batch(a),
        Cmd::Detect(a) => cmd_detect(a),
        Cmd::IdentityLut(a) => cmd_identity_lut(a),
        Cmd::Bench(a) => cmd_bench(a),
        Cmd::Preset(a) => {
            let p = Preset::load_named(&a.name)?;
            p.save(&a.output)?;
            eprintln!("saved {}", a.output.display());
            Ok(())
        }
    }
}

/// 引擎横幅：关键点模型与已加载的可选模型。
fn engine_tags(e: &Engine) -> String {
    e.model_tags().join(" ")
}

fn describe_faces(faces: &[FaceKeyPoints]) {
    for (i, fk) in faces.iter().enumerate() {
        let v = fk.sanity_check();
        eprintln!(
            "  face {i}: bbox=({:.0},{:.0},{:.0},{:.0}) ed={:.1}px yaw={:.1}° score={:.2}{}{}{}",
            fk.bbox.x1,
            fk.bbox.y1,
            fk.bbox.x2,
            fk.bbox.y2,
            fk.eye_distance(),
            fk.yaw_deg,
            fk.score,
            match (fk.gender, fk.age) {
                (Some(g), Some(a)) => format!(" {:?} ~{:.0}y", g, a),
                _ => String::new(),
            },
            if fk.parse.is_some() { " parsed" } else { "" },
            if v.is_empty() {
                String::new()
            } else {
                format!(" sanity: {v:?}")
            }
        );
    }
}

fn cmd_apply(a: ApplyArgs) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let mut photo = Photo::load(&a.input)?;
    if photo.meta.orientation != 1 {
        eprintln!(
            "orientation: EXIF {} → rotated upright ({}x{})",
            photo.meta.orientation,
            photo.image.width(),
            photo.image.height()
        );
    }
    let params = a.params.build()?;
    if params.embedded_develop {
        let t = Instant::now();
        if let Some(s) = photo.apply_embedded_develop() {
            eprintln!(
                "develop: re-applied embedded Camera Raw settings in {:.1} ms: {}",
                t.elapsed().as_secs_f64() * 1e3,
                s.summary()
            );
        }
    }
    let mut img = std::mem::take(&mut photo.image);
    let engine = if a.no_faces {
        None
    } else {
        Some(a.engine.build()?)
    };
    let mut faces: Vec<FaceKeyPoints> = match &engine {
        Some(e) => {
            let t = Instant::now();
            let f = e.detect_faces(&img)?;
            eprintln!(
                "detect: {} face(s) [{}] in {:.1} ms",
                f.len(),
                engine_tags(e),
                t.elapsed().as_secs_f64() * 1e3
            );
            describe_faces(&f);
            f
        }
        None => Vec::new(),
    };
    if let Some(long) = a.preview_long_side {
        let l = img.width().max(img.height());
        if l > long {
            let s = long as f32 / l as f32;
            let nw = ((img.width() as f32 * s).round() as u32).max(1);
            let nh = ((img.height() as f32 * s).round() as u32).max(1);
            img = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
            faces = faces.iter().map(|f| f.scaled(s)).collect();
            eprintln!("preview: resized to {nw}x{nh} (scale {s:.4})");
        }
    }
    let t = Instant::now();
    let out = match &engine {
        Some(e) => e.retouch(&img, &faces, &params),
        None => portrait_retouch::retouch_impl(&img, &faces, &params),
    };
    eprintln!(
        "retouch: {}x{} in {:.1} ms",
        img.width(),
        img.height(),
        t.elapsed().as_secs_f64() * 1e3
    );
    if let Some(dir) = &a.debug_dir {
        std::fs::create_dir_all(dir)?;
        if !faces.is_empty() {
            let mut vis = img.clone();
            for f in &faces {
                debug::draw_face_debug(&mut vis, f, false);
            }
            vis.save(dir.join("landmarks.png"))?;
            let mask = portrait_retouch::skin::face_mask_fullres(
                img.width() as usize,
                img.height() as usize,
                &faces,
                portrait_retouch::skin::WORK_SHORT_SIDE_A,
            );
            mask.to_luma8().save(dir.join("face_mask.png"))?;
            std::fs::write(
                dir.join("landmarks.json"),
                serde_json::to_string_pretty(&debug::faces_json(&faces))?,
            )?;
            if let Some(e) = &engine {
                let masks = e.skin_masks(&img, &faces, params.body_skin)?;
                masks.union.to_luma8().save(dir.join("skin_union.png"))?;
                for (i, m) in masks.faces.iter().enumerate() {
                    m.to_luma8().save(dir.join(format!("skin_face{i}.png")))?;
                }
                if let Some(b) = &masks.body {
                    b.to_luma8().save(dir.join("skin_body.png"))?;
                }
                if params.smooth_mode == SmoothMode::Cream {
                    e.neck_weights(&img, &faces, &params)?
                        .to_luma8()
                        .save(dir.join("neck_zone.png"))?;
                }
                if let Some(m) = e.person_matte(&img)? {
                    m.to_luma8().save(dir.join("matte.png"))?;
                }
                if let Some(m) = e.skin_prob(&img)? {
                    m.to_luma8().save(dir.join("skin_prob.png"))?;
                }
                if e.has_ai_blemish() && params.ai_blemish > 0.0 {
                    if let Some(pt) = e.ai_patches(&e.precomp(&img), &faces)? {
                        pt.hole_mask(img.width() as usize, img.height() as usize)
                            .to_luma8()
                            .save(dir.join("ai_holes.png"))?;
                    }
                }
            }
        }
        let orig = portrait_retouch::ImgF32::from_rgb8(&img);
        let (work, _) = orig.work_copy(portrait_retouch::skin::WORK_SHORT_SIDE_A);
        portrait_retouch::skin::skin_color_mask(&work)
            .to_luma8()
            .save(dir.join("skin_color_mask.png"))?;
    }
    photo::save(&out, &a.output, a.jpeg_quality, &photo.meta)
        .with_context(|| format!("save {}", a.output.display()))?;
    eprintln!(
        "saved {} (total {:.1} ms)",
        a.output.display(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

fn cmd_batch(a: BatchArgs) -> anyhow::Result<()> {
    let params = a.params.build()?;
    let mut cfg = BatchConfig::new(a.input.clone(), a.output.clone());
    cfg.recursive = a.recursive;
    if !a.ext.is_empty() {
        cfg.extensions = a.ext.clone();
    }
    cfg.format = match a.format {
        FormatArg::Jpg => OutputFormat::Jpeg,
        FormatArg::Png => OutputFormat::Png,
    };
    cfg.jpeg_quality = a.jpeg_quality;
    cfg.overwrite = a.overwrite;
    cfg.landmarks_dir = a.landmarks_dir.clone();
    let items = batch::plan(&cfg)?;
    anyhow::ensure!(
        !items.is_empty(),
        "没有找到可处理的照片（扩展名：{}）",
        cfg.extensions.join(",")
    );
    if a.dry_run {
        for item in &items {
            let note = if item.output.exists() && !a.overwrite {
                "  （已存在，将跳过）"
            } else {
                ""
            };
            println!(
                "{} -> {}{note}",
                item.input.display(),
                item.output.display()
            );
        }
        println!("共 {} 张", items.len());
        return Ok(());
    }
    let engine = a.engine.build()?;
    eprintln!(
        "batch: {} photo(s) -> {} [{}]",
        items.len(),
        a.output.display(),
        engine_tags(&engine)
    );
    let report = batch::run(Some(&engine), &params, &cfg, &items, &mut print_batch_event)?;
    if !a.no_report {
        let (csv, json) = report.write(&a.output)?;
        eprintln!("report: {} / {}", csv.display(), json.display());
    }
    let (done, skipped, failed) = (
        report.count(ItemStatus::Done),
        report.count(ItemStatus::Skipped),
        report.count(ItemStatus::Failed),
    );
    let per_photo = if done > 0 {
        format!(" ({:.1} s per photo)", report.total_ms / 1e3 / done as f64)
    } else {
        String::new()
    };
    eprintln!(
        "batch: {done} done, {skipped} skipped, {failed} failed in {:.1} s{per_photo}",
        report.total_ms / 1e3
    );
    anyhow::ensure!(failed == 0, "{failed} 张处理失败，详见报告");
    Ok(())
}

/// 批处理进度：每张开始一行、结束一行（OK / SKIP / FAIL）。
fn print_batch_event(ev: BatchEvent) {
    match ev {
        BatchEvent::Started { index, total, item } => {
            eprintln!("[{}/{total}] {}", index + 1, item.input.display());
        }
        BatchEvent::Finished {
            index,
            total,
            report: r,
        } => {
            let name = r
                .input
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let n = index + 1;
            match r.status {
                ItemStatus::Done => {
                    let genders: Vec<&str> = r
                        .faces
                        .iter()
                        .map(|f| match f.gender {
                            Some(portrait_retouch::Gender::Female) => "F",
                            Some(portrait_retouch::Gender::Male) => "M",
                            None => "?",
                        })
                        .collect();
                    let mut notes = Vec::new();
                    if r.orientation != 1 {
                        notes.push(format!("EXIF orientation {}", r.orientation));
                    }
                    if r.develop_applied {
                        notes.push("develop re-applied".to_string());
                    }
                    if !r.message.is_empty() {
                        notes.push(r.message.clone());
                    }
                    let notes = if notes.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", notes.join("; "))
                    };
                    eprintln!(
                        "[{n}/{total}] OK   {name}: {}x{}, {} face(s) [{}], load {:.1} s / retouch {:.1} s / save {:.1} s{notes}",
                        r.width,
                        r.height,
                        r.faces.len(),
                        genders.join(","),
                        r.load_ms / 1e3,
                        r.process_ms / 1e3,
                        r.save_ms / 1e3,
                    );
                }
                ItemStatus::Skipped => eprintln!("[{n}/{total}] SKIP {name}: {}", r.message),
                ItemStatus::Failed => eprintln!("[{n}/{total}] FAIL {name}: {}", r.message),
            }
        }
    }
}

fn cmd_detect(a: DetectArgs) -> anyhow::Result<()> {
    let img = Photo::load(&a.input)?.image;
    let engine = a.engine.build()?;
    let t = Instant::now();
    let faces = engine.detect_faces(&img)?;
    eprintln!(
        "{} face(s) [{}] in {:.1} ms",
        faces.len(),
        engine.landmark_name(),
        t.elapsed().as_secs_f64() * 1e3
    );
    for (i, f) in faces.iter().enumerate() {
        let v = f.sanity_check();
        println!(
            "face {i}: bbox=({:.1},{:.1},{:.1},{:.1}) score={:.3} ed={:.1} yaw={:.1}{} sanity={}",
            f.bbox.x1,
            f.bbox.y1,
            f.bbox.x2,
            f.bbox.y2,
            f.score,
            f.eye_distance(),
            f.yaw_deg,
            match (f.gender, f.age) {
                (Some(g), Some(age)) => format!(" gender={:?} age={:.0}", g, age),
                _ => String::new(),
            },
            if v.is_empty() {
                "ok".to_string()
            } else {
                format!("{v:?}")
            }
        );
    }
    if let Some(out) = &a.output {
        let mut vis = img.clone();
        for f in &faces {
            debug::draw_face_debug(&mut vis, f, a.indices);
        }
        vis.save(out)?;
        eprintln!("saved {}", out.display());
    }
    if let Some(path) = &a.json {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&debug::faces_json(&faces))?,
        )?;
        eprintln!("saved {}", path.display());
    }
    Ok(())
}

fn cmd_identity_lut(a: IdentityLutArgs) -> anyhow::Result<()> {
    if a.cube {
        std::fs::write(&a.output, Lut3D::identity(a.size).to_cube_string())?;
    } else {
        identity_lookup512().to_rgb8().save(&a.output)?;
    }
    eprintln!("saved {}", a.output.display());
    Ok(())
}

fn cmd_bench(a: BenchArgs) -> anyhow::Result<()> {
    let img = Photo::load(&a.input)?.image;
    let engine = a.engine.build()?;
    let mut params = a.params.build()?;
    let mp = img.width() as f64 * img.height() as f64 / 1e6;
    println!(
        "image {}x{} ({mp:.1} MP), landmarks={}",
        img.width(),
        img.height(),
        engine.landmark_name()
    );
    let t = Instant::now();
    let faces = engine.detect_faces(&img)?;
    println!(
        "detect+landmarks ({} faces): {:.1} ms",
        faces.len(),
        t.elapsed().as_secs_f64() * 1e3
    );
    if engine.has_matting() {
        let t = Instant::now();
        let _ = engine.person_matte(&img)?;
        println!(
            "person matting (MODNet): {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let orig = portrait_retouch::ImgF32::from_rgb8(&img);
    let t = Instant::now();
    let pre = portrait_retouch::skin::precompute_faithful(
        &orig,
        portrait_retouch::skin::WORK_SHORT_SIDE_A,
    );
    println!(
        "precompute faithful (bilateral+sobel@720, upsample): {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    drop(pre);
    let t = Instant::now();
    let _ = engine.retouch(&img, &faces, &params);
    println!(
        "full pipeline (cold cache): {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    for i in 0..a.iters {
        params.smooth = 0.3 + 0.1 * i as f32;
        let t = Instant::now();
        let _ = engine.retouch(&img, &faces, &params);
        println!(
            "full pipeline (warm cache, smooth={:.1}): {:.1} ms",
            params.smooth,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let l = img.width().max(img.height());
    if l > 1280 {
        let s = 1280.0 / l as f32;
        let small = image::imageops::resize(
            &img,
            ((img.width() as f32 * s) as u32).max(1),
            ((img.height() as f32 * s) as u32).max(1),
            image::imageops::FilterType::Triangle,
        );
        let sf: Vec<FaceKeyPoints> = faces.iter().map(|f| f.scaled(s)).collect();
        let _ = engine.retouch(&small, &sf, &params);
        let t = Instant::now();
        let _ = engine.retouch(&small, &sf, &params);
        println!(
            "1280 preview (warm cache): {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}
