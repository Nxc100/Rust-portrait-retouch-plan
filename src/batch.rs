//! 批量处理：目录 / 文件 → 输出目录。
//!
//! - **一个引擎处理全部照片**：模型只加载一次；每张处理完清空引擎缓存，内存峰值与单张处理相同。
//! - **三段流水**：读图线程预取下一张（解码 + EXIF 转正），主线程修图（内部 rayon 并行），
//!   写图线程编码落盘。读写与计算重叠，每张省下约 1–2 s。
//! - **逐张隔离**：解码、修图（含 panic）、写盘的错误只记在该张的报告里，不中断整批。
//! - **可续跑**：输出原子写入（临时文件 + 改名），已存在的输出默认跳过；`overwrite` 时覆盖。
//! - **报告**：每张的尺寸、方向、人脸（数量 / 性别 / 年龄 / 瞳距 / 侧脸角）、冲印设置、各阶段耗时与状态，
//!   CSV（带 BOM，Excel 可直接打开中文路径）与 JSON 两份。
//!
//! 用法见 `retouch batch --help`；库接口：[`plan`] 列出任务，[`run`] 执行，[`process_photo`] 处理单张。

use crate::debug::faces_json;
use crate::engine::Engine;
use crate::face::semantic::FaceKeyPoints;
use crate::photo::{self, Photo, PhotoMetadata};
use crate::pipeline::{retouch_impl, RetouchParams};
use anyhow::Context;
use image::RgbImage;
use serde::Serialize;
use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

/// 默认扫描的扩展名（不区分大小写）。
pub const DEFAULT_EXTENSIONS: [&str; 7] = ["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"];

/// 输出格式。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// JPEG（jpeg-encoder，写回 ICC / EXIF）
    #[default]
    Jpeg,
    /// PNG（无损）
    Png,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Jpeg => "jpg",
            OutputFormat::Png => "png",
        }
    }
}

/// 批处理配置。
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// 输入：目录或文件（可混合）
    pub inputs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    /// 递归扫描子目录（输出保留相对目录结构）
    pub recursive: bool,
    /// 扫描的扩展名（小写，不带点）
    pub extensions: Vec<String>,
    pub format: OutputFormat,
    pub jpeg_quality: u8,
    /// 覆盖已存在的输出（默认跳过，便于中断后续跑）
    pub overwrite: bool,
    /// 每张的人脸关键点 JSON（与 `retouch detect --json` 同格式）写到该目录，保留相对路径
    pub landmarks_dir: Option<PathBuf>,
}

impl BatchConfig {
    pub fn new(inputs: Vec<PathBuf>, output_dir: PathBuf) -> Self {
        Self {
            inputs,
            output_dir,
            recursive: false,
            extensions: DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
            format: OutputFormat::Jpeg,
            jpeg_quality: photo::DEFAULT_JPEG_QUALITY,
            overwrite: false,
            landmarks_dir: None,
        }
    }
}

/// 一张照片的任务。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchItem {
    pub input: PathBuf,
    pub output: PathBuf,
    /// 相对输出目录的路径（报告、关键点 JSON 用）
    pub rel: PathBuf,
}

/// 列出任务：目录按路径排序展开（`recursive` 时含子目录），文件直接加入；
/// 输出 = 输出目录 / 相对路径（扩展名换成输出格式）。
///
/// - 以 `.` 开头的文件 / 目录（含原子写入的临时文件）与输出目录内的文件不作为输入；输出目录不能就是输入目录；
/// - 两个输入映射到同一输出时：同目录的 `a.jpg` 与 `a.png`，后者输出为 `a.png.jpg`；
///   多个输入目录里的同名文件依次加序号（`a_2.jpg`、`a_3.jpg` …）；
/// - 任何输出与输入是同一文件时报错（不允许覆盖原图）。
pub fn plan(config: &BatchConfig) -> anyhow::Result<Vec<BatchItem>> {
    let exts: Vec<String> = config
        .extensions
        .iter()
        .map(|e| e.trim_start_matches('.').to_ascii_lowercase())
        .collect();
    let out_abs = absolute(&config.output_dir);
    let mut found: Vec<(PathBuf, PathBuf)> = Vec::new(); // (输入文件, 相对路径)
    for input in &config.inputs {
        let meta =
            std::fs::metadata(input).with_context(|| format!("输入不存在：{}", input.display()))?;
        if meta.is_dir() {
            if absolute(input) == out_abs {
                anyhow::bail!("输出目录不能与输入目录相同：{}", input.display());
            }
            let mut files = Vec::new();
            collect_dir(input, config.recursive, &exts, &out_abs, &mut files)?;
            files.sort();
            for f in files {
                let rel = f.strip_prefix(input).unwrap_or(&f).to_path_buf();
                found.push((f, rel));
            }
        } else {
            let name = input
                .file_name()
                .with_context(|| format!("无效的输入文件名：{}", input.display()))?;
            found.push((input.clone(), PathBuf::from(name)));
        }
    }
    let ext = config.format.extension();
    let mut used: HashSet<PathBuf> = HashSet::new();
    let mut items = Vec::with_capacity(found.len());
    for (input, rel) in found {
        let mut rel_out = rel.with_extension(ext);
        if used.contains(&normalize_key(&rel_out)) {
            // 同名不同扩展名：把原扩展名保留在文件名里；仍冲突（多个输入目录里的同名文件）时加序号
            let name = rel
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let stem = rel
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            rel_out = rel.with_file_name(format!("{name}.{ext}"));
            let mut k = 2;
            while used.contains(&normalize_key(&rel_out)) {
                rel_out = rel.with_file_name(format!("{stem}_{k}.{ext}"));
                k += 1;
            }
        }
        used.insert(normalize_key(&rel_out));
        let output = config.output_dir.join(&rel_out);
        if absolute(&output) == absolute(&input) {
            anyhow::bail!("输出会覆盖原图：{}（请换一个输出目录）", input.display());
        }
        items.push(BatchItem {
            input,
            output,
            rel: rel_out,
        });
    }
    Ok(items)
}

fn collect_dir(
    dir: &Path,
    recursive: bool,
    exts: &[String],
    out_abs: &Path,
    files: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("读取目录 {}", dir.display()))?
    {
        let path = entry?.path();
        let hidden = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'));
        if hidden {
            continue;
        }
        if path.is_dir() {
            // 输出目录在输入目录内时不进入：不把上次的输出当输入
            if recursive && !absolute(&path).starts_with(out_abs) {
                collect_dir(&path, recursive, exts, out_abs, files)?;
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| exts.iter().any(|x| x.eq_ignore_ascii_case(e)))
        {
            files.push(path);
        }
    }
    Ok(())
}

/// 尽量规范化的绝对路径（目标不存在时退回拼接当前目录）。
fn absolute(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| {
        std::env::current_dir()
            .map(|d| d.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    })
}

/// 输出冲突检测用的键：Windows / macOS 文件名不区分大小写。
fn normalize_key(p: &Path) -> PathBuf {
    PathBuf::from(p.to_string_lossy().to_lowercase())
}

/// 单张处理结果。
pub struct Processed {
    pub image: RgbImage,
    pub faces: Vec<FaceKeyPoints>,
    /// 应用了内嵌冲印设置时为其摘要
    pub develop_applied: Option<String>,
}

/// 处理一张已解码的照片：（可选）再应用内嵌冲印设置 → 人脸检测 → 修图。
/// `engine` 为 None 时不检测人脸（只做与人脸无关的处理）。
pub fn process_photo(
    engine: Option<&Engine>,
    photo: &mut Photo,
    params: &RetouchParams,
) -> anyhow::Result<Processed> {
    let develop_applied = if params.embedded_develop {
        photo.apply_embedded_develop().map(|s| s.summary())
    } else {
        None
    };
    let (image, faces) = match engine {
        Some(e) => {
            let faces = e.detect_faces(&photo.image)?;
            let out = e.retouch(&photo.image, &faces, params);
            (out, faces)
        }
        None => (retouch_impl(&photo.image, &[], params), Vec::new()),
    };
    Ok(Processed {
        image,
        faces,
        develop_applied,
    })
}

/// 每张照片的状态。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemStatus {
    Done,
    Skipped,
    Failed,
}

impl ItemStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ItemStatus::Done => "done",
            ItemStatus::Skipped => "skipped",
            ItemStatus::Failed => "failed",
        }
    }
}

/// 报告里的人脸摘要。
#[derive(Clone, Debug, Serialize)]
pub struct FaceSummary {
    pub bbox: [f32; 4],
    pub eye_distance: f32,
    pub yaw_deg: f32,
    pub gender: Option<crate::Gender>,
    pub age: Option<f32>,
}

impl From<&FaceKeyPoints> for FaceSummary {
    fn from(f: &FaceKeyPoints) -> Self {
        Self {
            bbox: [f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2],
            eye_distance: f.eye_distance(),
            yaw_deg: f.yaw_deg,
            gender: f.gender,
            age: f.age,
        }
    }
}

/// 一张照片的报告。
#[derive(Clone, Debug, Serialize)]
pub struct ItemReport {
    pub input: PathBuf,
    pub output: PathBuf,
    pub status: ItemStatus,
    /// 失败原因 / 跳过原因
    pub message: String,
    pub width: u32,
    pub height: u32,
    /// 原文件的 EXIF 方向（已转正）
    pub orientation: u8,
    pub faces: Vec<FaceSummary>,
    /// 文件内嵌的 Camera Raw 冲印设置摘要（有设置时）
    pub develop_settings: Option<String>,
    /// 是否再应用了这些设置
    pub develop_applied: bool,
    pub load_ms: f64,
    pub process_ms: f64,
    pub save_ms: f64,
}

impl ItemReport {
    fn new(item: &BatchItem, status: ItemStatus, message: impl Into<String>) -> Self {
        Self {
            input: item.input.clone(),
            output: item.output.clone(),
            status,
            message: message.into(),
            width: 0,
            height: 0,
            orientation: 1,
            faces: Vec::new(),
            develop_settings: None,
            develop_applied: false,
            load_ms: 0.0,
            process_ms: 0.0,
            save_ms: 0.0,
        }
    }
}

/// 整批报告。
#[derive(Clone, Debug, Serialize)]
pub struct BatchReport {
    pub items: Vec<ItemReport>,
    pub total_ms: f64,
}

impl BatchReport {
    pub fn count(&self, status: ItemStatus) -> usize {
        self.items.iter().filter(|r| r.status == status).count()
    }

    /// CSV（UTF-8 带 BOM；每张一行，人脸性别用 `F` / `M` / `?` 串联）。
    pub fn to_csv(&self) -> String {
        let mut s = String::from("\u{feff}");
        s.push_str("input,output,status,message,width,height,orientation,faces,genders,develop_settings,develop_applied,load_ms,process_ms,save_ms\n");
        for r in &self.items {
            let genders: String = r
                .faces
                .iter()
                .map(|f| match f.gender {
                    Some(crate::Gender::Female) => 'F',
                    Some(crate::Gender::Male) => 'M',
                    None => '?',
                })
                .collect();
            let fields = [
                r.input.display().to_string(),
                r.output.display().to_string(),
                r.status.as_str().to_string(),
                r.message.clone(),
                r.width.to_string(),
                r.height.to_string(),
                r.orientation.to_string(),
                r.faces.len().to_string(),
                genders,
                r.develop_settings.clone().unwrap_or_default(),
                r.develop_applied.to_string(),
                format!("{:.0}", r.load_ms),
                format!("{:.0}", r.process_ms),
                format!("{:.0}", r.save_ms),
            ];
            let row: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
            s.push_str(&row.join(","));
            s.push('\n');
        }
        s
    }

    /// 写 `<dir>/batch_report.csv` 与 `<dir>/batch_report.json`。
    pub fn write(&self, dir: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
        std::fs::create_dir_all(dir)?;
        let csv = dir.join("batch_report.csv");
        let json = dir.join("batch_report.json");
        std::fs::write(&csv, self.to_csv()).with_context(|| format!("write {}", csv.display()))?;
        std::fs::write(&json, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("write {}", json.display()))?;
        Ok((csv, json))
    }
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// 进度事件（在调用 [`run`] 的线程上回调）。
pub enum BatchEvent<'a> {
    /// 开始处理第 `index` 张（从 0 计）
    Started {
        index: usize,
        total: usize,
        item: &'a BatchItem,
    },
    /// 第 `index` 张的最终结果（写盘完成、失败或跳过）
    Finished {
        index: usize,
        total: usize,
        report: &'a ItemReport,
    },
}

/// 写图线程的任务。
struct SaveJob {
    index: usize,
    image: RgbImage,
    meta: PhotoMetadata,
    path: PathBuf,
}

/// 执行批处理。`items` 通常来自 [`plan`]；`engine` 为 None 时不做人脸相关处理。
/// 每张的成败都记录在返回的报告里；本函数只在整体无法进行时（如建不了输出目录）返回错误。
pub fn run(
    engine: Option<&Engine>,
    params: &RetouchParams,
    config: &BatchConfig,
    items: &[BatchItem],
    on_event: &mut dyn FnMut(BatchEvent),
) -> anyhow::Result<BatchReport> {
    let t_all = Instant::now();
    let total = items.len();
    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("创建输出目录 {}", config.output_dir.display()))?;
    let mut reports: Vec<Option<ItemReport>> = vec![None; total];

    // 已存在的输出：跳过
    let mut todo = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if !config.overwrite && item.output.exists() {
            let r = ItemReport::new(item, ItemStatus::Skipped, "输出已存在");
            on_event(BatchEvent::Finished {
                index: i,
                total,
                report: &r,
            });
            reports[i] = Some(r);
        } else {
            todo.push(i);
        }
    }

    std::thread::scope(|scope| -> anyhow::Result<()> {
        // 读图线程：预取一张
        let (load_tx, load_rx) = mpsc::sync_channel::<(usize, anyhow::Result<Photo>, f64)>(1);
        let todo_ref = &todo;
        scope.spawn(move || {
            for &i in todo_ref {
                let t = Instant::now();
                let photo = catch_unwind(|| Photo::load(&items[i].input)).unwrap_or_else(|p| {
                    Err(anyhow::anyhow!("解码时 panic：{}", panic_message(&p)))
                });
                if load_tx.send((i, photo, ms(t))).is_err() {
                    break;
                }
            }
        });
        // 写图线程：编码 + 原子写入
        let (save_tx, save_rx) = mpsc::sync_channel::<SaveJob>(1);
        let (done_tx, done_rx) = mpsc::channel::<(usize, Result<(), String>, f64)>();
        let quality = config.jpeg_quality;
        scope.spawn(move || {
            for job in save_rx {
                let t = Instant::now();
                let res = catch_unwind(AssertUnwindSafe(|| {
                    job.path
                        .parent()
                        .map_or(Ok(()), std::fs::create_dir_all)
                        .map_err(anyhow::Error::from)
                        .and_then(|_| photo::save(&job.image, &job.path, quality, &job.meta))
                        .map_err(|e| format!("{e:#}"))
                }))
                .unwrap_or_else(|p| Err(format!("编码时 panic：{}", panic_message(&p))));
                if done_tx.send((job.index, res, ms(t))).is_err() {
                    break;
                }
            }
        });

        let mut pending: Vec<Option<ItemReport>> = vec![None; total];

        for (i, loaded, load_ms) in load_rx {
            let item = &items[i];
            on_event(BatchEvent::Started {
                index: i,
                total,
                item,
            });
            let mut report = ItemReport::new(item, ItemStatus::Done, "");
            report.load_ms = load_ms;
            let mut photo = match loaded {
                Ok(p) => p,
                Err(e) => {
                    report.status = ItemStatus::Failed;
                    report.message = format!("读取失败：{e:#}");
                    on_event(BatchEvent::Finished {
                        index: i,
                        total,
                        report: &report,
                    });
                    reports[i] = Some(report);
                    continue;
                }
            };
            report.width = photo.image.width();
            report.height = photo.image.height();
            report.orientation = photo.meta.orientation;
            report.develop_settings = photo.meta.develop.as_ref().map(|s| s.summary());

            let t = Instant::now();
            let processed = catch_unwind(AssertUnwindSafe(|| {
                process_photo(engine, &mut photo, params)
            }));
            if let Some(e) = engine {
                e.clear_cache(); // 释放本张的中间结果，下一张的内存峰值不叠加
            }
            report.process_ms = ms(t);
            let processed = match processed {
                Ok(Ok(p)) => p,
                Ok(Err(e)) => {
                    report.status = ItemStatus::Failed;
                    report.message = format!("处理失败：{e:#}");
                    on_event(BatchEvent::Finished {
                        index: i,
                        total,
                        report: &report,
                    });
                    reports[i] = Some(report);
                    continue;
                }
                Err(panic) => {
                    report.status = ItemStatus::Failed;
                    report.message = format!("处理时 panic：{}", panic_message(&panic));
                    on_event(BatchEvent::Finished {
                        index: i,
                        total,
                        report: &report,
                    });
                    reports[i] = Some(report);
                    continue;
                }
            };
            report.faces = processed.faces.iter().map(FaceSummary::from).collect();
            report.develop_applied = processed.develop_applied.is_some();
            if let Some(dir) = &config.landmarks_dir {
                if let Err(e) = write_landmarks(dir, &item.rel, &processed.faces) {
                    report.message = format!("关键点 JSON 写出失败：{e:#}");
                }
            }
            pending[i] = Some(report);
            // 交给写图线程；等待期间顺便回收已完成的写盘结果
            let job = SaveJob {
                index: i,
                image: processed.image,
                meta: std::mem::take(&mut photo.meta),
                path: item.output.clone(),
            };
            drop(photo);
            save_tx
                .send(job)
                .map_err(|_| anyhow::anyhow!("写图线程已退出"))?;
            while let Ok(done) = done_rx.try_recv() {
                finish_saved(done, total, &mut pending, &mut reports, on_event);
            }
        }
        drop(save_tx);
        for done in done_rx {
            finish_saved(done, total, &mut pending, &mut reports, on_event);
        }
        Ok(())
    })?;

    Ok(BatchReport {
        items: reports.into_iter().flatten().collect(),
        total_ms: ms(t_all),
    })
}

/// 写盘结果回收：补全该张的报告（写盘耗时 / 失败原因）并发出 Finished。
fn finish_saved(
    (i, res, save_ms): (usize, Result<(), String>, f64),
    total: usize,
    pending: &mut [Option<ItemReport>],
    reports: &mut [Option<ItemReport>],
    on_event: &mut dyn FnMut(BatchEvent),
) {
    let Some(mut r) = pending[i].take() else {
        return;
    };
    r.save_ms = save_ms;
    if let Err(e) = res {
        r.status = ItemStatus::Failed;
        r.message = format!("写出失败：{e}");
    }
    on_event(BatchEvent::Finished {
        index: i,
        total,
        report: &r,
    });
    reports[i] = Some(r);
}

fn write_landmarks(dir: &Path, rel: &Path, faces: &[FaceKeyPoints]) -> anyhow::Result<()> {
    let path = dir.join(rel).with_extension("json");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(&faces_json(faces))?)
        .with_context(|| format!("write {}", path.display()))
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "未知错误".into())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("pr_batch_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_png(path: &Path, w: u32, h: u32) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let img = RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 7) as u8, (y * 5) as u8, 90]));
        img.save(path).unwrap();
    }

    #[test]
    fn plan_discovers_sorts_and_maps_outputs() {
        let t = TempDir::new("plan");
        let inp = t.0.join("in");
        write_png(&inp.join("b.PNG"), 8, 8);
        write_png(&inp.join("a.png"), 8, 8);
        std::fs::write(inp.join("notes.txt"), "x").unwrap();
        std::fs::write(inp.join(".a.jpg.partial"), "x").unwrap();
        write_png(&inp.join("sub").join("c.png"), 8, 8);
        // 同名不同扩展名
        std::fs::copy(inp.join("a.png"), inp.join("a.bmp")).unwrap();
        let out = t.0.join("out");
        let mut cfg = BatchConfig::new(vec![inp.clone()], out.clone());
        let items = plan(&cfg).unwrap();
        let rels: Vec<String> = items.iter().map(|i| i.rel.display().to_string()).collect();
        // 按路径排序；大小写不敏感的扩展名；隐藏文件、非图片、子目录（非递归）被忽略；冲突改名
        assert_eq!(rels, vec!["a.jpg", "a.png.jpg", "b.jpg"]);
        assert!(items.iter().all(|i| i.output.starts_with(&out)));
        cfg.recursive = true;
        let rels: Vec<String> = plan(&cfg)
            .unwrap()
            .iter()
            .map(|i| i.rel.to_string_lossy().replace('\\', "/"))
            .collect();
        assert_eq!(rels, vec!["a.jpg", "a.png.jpg", "b.jpg", "sub/c.jpg"]);
        // 两个输入目录里的同名文件：依次加序号
        let inp2 = t.0.join("in2");
        write_png(&inp2.join("a.png"), 8, 8);
        cfg.recursive = false;
        cfg.inputs = vec![inp.clone(), inp2];
        let rels: Vec<String> = plan(&cfg)
            .unwrap()
            .iter()
            .map(|i| i.rel.display().to_string())
            .collect();
        assert_eq!(rels, vec!["a.jpg", "a.png.jpg", "b.jpg", "a_2.jpg"]);
        // 单个文件输入 + PNG 输出
        cfg.inputs = vec![inp.join("sub").join("c.png")];
        cfg.format = OutputFormat::Png;
        let items = plan(&cfg).unwrap();
        assert_eq!(items[0].rel, PathBuf::from("c.png"));
    }

    #[test]
    fn plan_refuses_to_overwrite_originals_and_skips_output_dir() {
        let t = TempDir::new("guard");
        write_png(&t.0.join("x.png"), 8, 8);
        // 输出目录 = 输入目录：拒绝
        let mut cfg = BatchConfig::new(vec![t.0.clone()], t.0.clone());
        cfg.format = OutputFormat::Png;
        assert!(plan(&cfg).is_err());
        // 单个文件输入、输出落在原图上：拒绝
        cfg.inputs = vec![t.0.join("x.png")];
        assert!(plan(&cfg).is_err());
        // 输出目录在输入目录内：上次的输出不会被当作输入
        write_png(&t.0.join("out").join("x.png"), 8, 8);
        let cfg = BatchConfig {
            recursive: true,
            format: OutputFormat::Png,
            ..BatchConfig::new(vec![t.0.clone()], t.0.join("out"))
        };
        let items = plan(&cfg).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].input, t.0.join("x.png"));
    }

    #[test]
    fn run_processes_skips_existing_and_reports_failures() {
        let t = TempDir::new("run");
        let inp = t.0.join("in");
        write_png(&inp.join("ok1.png"), 12, 10);
        write_png(&inp.join("ok2.png"), 10, 12);
        std::fs::write(inp.join("broken.jpg"), b"not a jpeg").unwrap();
        let out = t.0.join("out");
        let mut cfg = BatchConfig::new(vec![inp], out.clone());
        cfg.format = OutputFormat::Png;
        let items = plan(&cfg).unwrap();
        let params = RetouchParams::identity();
        let mut events = Vec::new();
        let report = run(None, &params, &cfg, &items, &mut |e| {
            if let BatchEvent::Finished { index, report, .. } = e {
                events.push((index, report.status));
            }
        })
        .unwrap();
        assert_eq!(report.items.len(), 3);
        assert_eq!(report.count(ItemStatus::Done), 2);
        assert_eq!(report.count(ItemStatus::Failed), 1);
        assert_eq!(events.len(), 3);
        let broken = report
            .items
            .iter()
            .find(|r| r.input.ends_with("broken.jpg"))
            .unwrap();
        assert!(broken.message.contains("读取失败"), "{}", broken.message);
        // 恒等参数：输出与输入逐位相同（PNG 无损）
        let a = Photo::load(&out.join("ok1.png")).unwrap().image;
        assert_eq!(a.dimensions(), (12, 10));
        assert_eq!(a, Photo::load(&items[1].input).unwrap().image);
        // 再跑一次：已存在的跳过；覆盖模式重新处理
        let again = run(None, &params, &cfg, &items, &mut |_| {}).unwrap();
        assert_eq!(again.count(ItemStatus::Skipped), 2);
        cfg.overwrite = true;
        let forced = run(None, &params, &cfg, &items, &mut |_| {}).unwrap();
        assert_eq!(forced.count(ItemStatus::Done), 2);
        // 报告
        let (csv, json) = forced.write(&out).unwrap();
        let text = std::fs::read_to_string(csv).unwrap();
        assert!(text.starts_with('\u{feff}'));
        assert_eq!(text.lines().count(), 4);
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(json).unwrap()).unwrap();
        assert_eq!(v["items"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn csv_fields_are_escaped() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
