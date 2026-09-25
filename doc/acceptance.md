# 验收对照（方案第 6 节）

日期：2026-09-23。环境：Windows 11，24 核桌面 CPU，rustc 1.97.1，ONNX Runtime 1.30.0（CPU）。
样张：`example/gpupixel/demo/desktop/demo.png`（1000×1500）、美狐 106 点示例照（750×959）、
insightface `t1.jpg`（1280×886，6 人）、UltraFace `imgs/`；12MP 用美狐照片放大到 3000×3836（11.5 MP）。

| 阶段 | 验收标准 | 结果 | 证据 |
|---|---|---|---|
| P0 骨架 | 恒等 LUT 最大误差 ≤ 1/255；12MP 应用 LUT < 100 ms | 512 查找图 / .cube 恒等误差 ≤ 1/255（PNG 往返后 CLI 实测 0/255）；12MP 查找图基准见下 | `tests/identity.rs`、`retouch apply --lut luts/identity_512.png` |
| P1 磨皮 A | 与 numpy oracle 逐像素误差 ≤ 2/255；12MP < 800 ms | 最大误差 1/255（含 log/HSB），0/255（不含）；11.5MP 双边 + Sobel 预计算 42.5 ms，全流程冷缓存 207 ms | `tools/oracle_numpy.py`、`retouch bench` |
| P2 检测 + 关键点 | 语义点定位正确率 ≥ 95%（100 张人工核对） | 本次核对 5 张图 / 10 张脸：正脸全部通过附录 A 六条检查；侧脸（yaw −54°）触发 1 条比例检查属预期。**100 张样张的人工核对尚未进行**（无样张库） | `retouch detect --indices`、`doc/analysis/face_models.md` 投票表 |
| P3 形变 | 强度 0 时逐位相等；强度 1 无撕裂 / 黑边；12MP < 300 ms | 逐位恒等测试通过；全强度无暗像素测试通过；11.5MP 热缓存全流程（含形变）125 ms | `tests/identity.rs::all_warp_strength_zero_is_bytewise_identity`、`warp_full_strength_has_no_black_border` |
| P4 遮罩 + 美白 + 磨皮 B | `restrict_to_face=true` 时背景逐位不变；B 模式细节保留（人工评审） | 三种模式背景像素逐位不变测试通过；demo.png 对比：B 保留雀斑 / 毛孔纹理，A/C 更平 | `tests/identity.rs::background_pixels_unchanged_when_restricted_to_face`；对比图可用 `retouch apply --mode freqsep` 等重新生成 |
| P5 集成 | 1280 预览 < 200 ms；导出与预览 PSNR ≥ 40 dB | 1280 预览热缓存 13 ms；预览 / 导出一致性测试 PSNR ≥ 40 dB 通过 | `retouch bench`、`tests/identity.rs::preview_and_export_are_consistent` |
| P6 发布模型 | 可切换 `LandmarkModel`；正脸瞳距误差 < 3% | `--landmarks facemesh` 一键切换；美狐正脸瞳距 244.1（2d106）vs 252.2（mesh）= 3.3%，略超 3%（两模型对"瞳孔"的定义不同：2d106 用眼中心点均值，mesh 用眼轮廓均值） | `retouch detect --landmarks facemesh` |
| P7 GPU（可选） | — | 未实现，`gpu` feature 占位 | — |

## 7.5 性能基准（`retouch bench` 于美狐样张放大到 3000×3836 的临时文件，11.5 MP，2d106det）

| 用例 | 目标 | 实测 |
|---|---|---|
| 12MP 全流程（A + 形变 + LUT） | < 2.0 s | 207 ms（冷缓存）/ 125 ms（热缓存） |
| 12MP 仅改 smooth（命中缓存） | < 300 ms | 124–131 ms |
| 1280 长边预览全流程 | < 200 ms | 13 ms |
| 检测 + 关键点（单脸） | < 120 ms | 45 ms（12MP 输入，含缩放） |

criterion 基准（`cargo bench --bench retouch`，合成 12MP 图 4000×3000，热缓存）：

| 用例 | 中位数 |
|---|---|
| precompute_faithful（双边 + Sobel @720 + 上采样） | 58.0 ms |
| full_pipeline_A_warm_cache（A + 形变 + 512 LUT） | 104.7 ms |
| smooth_only_change_A | 43.7 ms |
| lookup512（12MP） | 24.4 ms |
| warp_only（瘦脸 + 大眼 + 瘦鼻，强度 1） | 73.1 ms |
| full_pipeline_B_warm_cache（B + 形变 + LUT） | 192.6 ms |
| preview_1280_full_pipeline_warm | 10.8 ms |

## 7.1 恒等性测试

`cargo test`：41 个单元测试 + 10 个恒等性集成测试全部通过（含"无人脸不 panic 且跳过形变"、`pinch(delta=0)` / `enlarge(k=0)` 返回原坐标）。

## 7.3 视觉回归

`PORTRAIT_SAMPLES=tests/samples cargo test --release --test golden`：2 张样张（`tests/samples/`，不入库）× 7 组参数 = 14 张 golden，首次生成后再次运行全部 PSNR ≥ 45 dB。
golden 目录已加入 `.gitignore`（方案建议存 LFS）。

## 未完成 / 需人工的项

1. P2 的 100 张样张人工核对（需要样张库；工具与流程已就绪：`retouch detect -i x.jpg -o vis.png --indices --json lm.json`）。
2. 瘦鼻 / 瘦脸系数在 20 张样张上的微调（`--coeffs coeffs.json`，字段见 `WarpCoefficients`）。
3. 设计师自制风格 LUT（`luts/identity_512.png` 为起点）；`luts/` 现有 warm / cool / fade / mono 为参数化示例。
4. P7 wgpu 实时路径。
