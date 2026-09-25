# 人脸检测与关键点模型

## UltraFace `version-RFB-320.onnx`（MIT）

来源：`example/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/onnx/`。

- 输入 `input [1,3,240,320]`，RGB，`(x − 127) / 128`（`detect_imgs_onnx.py`：`image_mean = [127,127,127]`, `/128`）。
- 输出 `scores [1,4420,2]`（第 1 列人脸概率）、`boxes [1,4420,4]`（归一化 x1 y1 x2 y2，模型内已解码先验框）。
- 后处理（`vision/utils/box_utils.py::hard_nms`）：阈值 0.7 → 按分数降序硬 NMS（IoU 0.3，`candidate_size 200`）→ 乘以原图宽高。
- ONNX 的权重以图输入形式出现（ir_version 4），ORT 会提示 "Initializer appears in graph inputs"，无害。
→ `src/face/detector.rs`。

## InsightFace `2d106det.onnx`（模型非商用，开发期）

来源：`buffalo_l.zip`。前后处理见 `example/insightface/python-package/insightface/model_zoo/landmark.py`：

- 图内前两个节点为 `Sub(_minusscalar0)` / `Mul(_mulscalar0)` → `input_mean = 0, input_std = 1`，输入为 RGB 0..255 浮点。
- 裁剪：`center = bbox 中心`，`scale = 192 / (max(w, h) · 1.5)`，`rotate = 0`，`face_align.transform` =
  缩放 → 平移到原点 → 旋转 → 平移到 (96, 96)，`cv2.warpAffine`（双线性、边界 0、像素索引制）。
- 输出 `fc1 [1,212]` → `(106, 2)` → `(pred + 1) · 96` → 逆仿射 `cv2.invertAffineTransform` 回原图。
→ `src/face/lm_2d106.rs`（`geom::Affine::crop` + `face::crop::warp_affine_rgb8`）。

### 索引语义（P2 核对结果）

方法：对 5 张样张（Meihu 正脸、insightface t1.jpg 6 人、UltraFace imgs/1、13、22）同时运行 2d106det 与 Face Mesh，
对每个 Face Mesh 语义点取 2d106det 中的最近点投票（`scratchpad/lm/mapping.py`），再用放大可视化确认：

| 语义 | Face Mesh 参考 | 2d106det 投票（票数） | 采用 |
|---|---|---|---|
| eye_outer_l / eye_inner_l | 33 / 133 | 35 (7) / 39 (8) | 35 / 39 |
| eye_inner_r / eye_outer_r | 362 / 263 | 89 (8) / 93 (10) | 89 / 93 |
| eye_top_l / eye_bot_l | 159 / 145 | 40 (6) / 33 (6) | 40 / 33 |
| eye_top_r / eye_bot_r | 386 / 374 | 94 (5) / 87 (10) | 94 / 87 |
| pupil_l / pupil_r | 眼环均值 | 34 (4), 38 (2) / 92 (3), 88 (3) | avg(34, 38) / avg(88, 92) |
| nose_bridge_top | 168 | 72 (9) | 72 |
| nose_tip | 1 | 79 (4), 80 (4) | mid(86, 80) |
| nose_bottom | 2 | 80 (9) | 80 |
| nostril_l / nostril_r | 98 / 327 | 78 (7) / 84 (6)（鼻底行 78 79 80 85 84 ↔ Face++ 47..51） | 79 / 85 |
| nose_wing_l / nose_wing_r | 129 / 358 | 77 (10) / 83 (11) | 77 / 83 |
| chin | 152 | 0 (9) | 0 |
| mouth_l / mouth_r | 61 / 291 | 52 (5) / 61 (9) | 52 / 61 |

轮廓顺序（左太阳穴 → 下巴 → 右太阳穴）由可视化读出：
`[1, 9..16, 2..8, 0, 24..18, 32..25, 17]`（`CONTOUR_ORDER`），与 Face++ 0..32 一一对应，
因此 `jaw_l = contour[4, 9, 13]`，`jaw_r = contour[28, 23, 19]`。分组：33–42 左眼、43–51 左眉、52–71 嘴、
72–86 鼻、87–96 右眼、97–105 右眉。

附录 A 的 6 条自动检查实现为 `FaceKeyPoints::sanity_check`；在上述正脸样张上全部通过，
侧脸（yaw ≈ −54°）触发 "瞳距 / 眼宽" 检查属预期。

## MediaPipe Face Mesh（Apache-2.0，发布期）

ONNX：`face_mesh_Nx3x192x192_post.onnx`（PINTO0309/facemesh_onnx_tensorrt release 1.0.0，由官方
`face_landmark.tflite` 转换并合并后处理）。PINTO_model_zoo 032_FaceMesh 的 Google Drive 资源已失效，
wasabisys 镜像在当前网络不可达，因此改用 GitHub release。

- 输入 `input [N,3,192,192]` RGB 0..1（`demo_video.py`：`/255`，与 `face_landmark_cpu.pbtxt` 的
  `output_tensor_float_range {0,1}` 一致）；`crop_x1/crop_y1/crop_width/crop_height [N,1]` int32。
- 输出 `score [N,1]`（logit）、`final_landmarks [N,468,3]` int32 = `int(lm · crop_wh/192 + 0.5 + crop_xy)`
  （`facemesh_postprocess.py`）。为保留亚像素精度，传 `crop = (0, 0, 192·256, 192·256)` 再除以 256。
- ROI（`face_detection_front_detection_to_roi.pbtxt` / `face_landmark_landmarks_to_roi.pbtxt`）：
  `RectTransformationCalculator{scale 1.5, square_long}`，旋转由两眼连线（检测关键点 0/1 或关键点 33/263）对齐到 0°。
  UltraFace 无眼睛关键点，因此第一遍不旋转，第二遍用第一遍的 33/263 计算旋转并按关键点包围盒重建 ROI。
→ `src/face/lm_facemesh.rs`。

### 语义索引（`face_mesh_connections.py` + 可视化确认）

| 语义 | 索引 |
|---|---|
| 左眼环（图像左，MediaPipe RIGHT_EYE） | 33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173 |
| 右眼环 | 263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398 |
| 眼角 / 上下睑 | 33,133,362,263 / 159,145,386,374 |
| 鼻 | 168 鼻梁顶、1 鼻尖、2 鼻底、98/327 鼻孔、129/358 鼻翼 |
| 轮廓（左太阳穴→下巴→右太阳穴） | 234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454 |
| 额头（右→顶→左） | 356,389,251,284,332,297,338,10,109,67,103,54,21,162,127 |
| jaw_l / jaw_r | 132,172,150 / 361,397,379 |
| 嘴角 | 61 / 291 |

瞳孔用眼环均值（468 点版无虹膜）；与 2d106det 在 Meihu 正脸上的瞳距差 3.3%（244.1 vs 252.2 px）。
