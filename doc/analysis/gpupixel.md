# pixpark/gpupixel 解读（Apache-2.0，主架构参考）

仓库：`example/gpupixel`。C++ + OpenGL，滤镜链 `Source → Filter → Sink`；本项目不做 FFI，只对照其算法。

## 目录

- `src/filter/`：50 余个滤镜，其中与人像相关：`beauty_face_filter.cc`（FilterGroup）、`beauty_face_unit_filter.cc`（磨皮 + 美白 shader）、
  `box_blur_filter.cc` / `box_mono_blur_filter.cc` / `box_high_pass_filter.cc` / `box_difference_filter.cc`、
  `face_reshape_filter.cc`（瘦脸 / 大眼）、`lipstick_filter.cc` / `blusher_filter.cc` / `face_makeup_filter.cc`（口红 / 腮红贴图）。
- `src/face_detector/face_detector.cc`：封装 `mars_vision::MarsFaceLandmarker`（闭源库），输出 106 点（Face++ 布局），归一化到 0..1。
- `src/res/`：`lookup_gray.png`(256×1)、`lookup_origin.png`(64×64)、`lookup_skin.png`(64×64)、`lookup_light.png`(512×512)、
  `lookup_custom.png`(512×512)、`mouth.png`、`blusher.png`。

## 磨皮（BeautyFaceFilter = BoxBlur + BoxHighPass + BeautyFaceUnitFilter）

- `box_blur_filter_->SetTexelSpacingMultiplier(4)`，`SetRadius(4)`：BoxMonoBlur 可分离，radius 4 → 9 tap，
  偏移 `k·4 px`，权重 `1/9`。
- `BoxHighPassFilter` = BoxBlur（同参数）→ `BoxDifferenceFilter`：`diff = min(((src − mean)·delta)², 1)`，`delta = 7.07`。
- `BeautyFaceUnitFilter` shader（三输入：src、mean、var）：

```glsl
if (blurAlpha > 0) {
  theta = 0.1;
  p = clamp((min(src.r, mean.r - 0.1) - 0.2) * 4, 0, 1);       // 用红通道估计"是皮肤"的概率
  meanVar = (var.r + var.g + var.b) / 3;
  kMin = clamp((1 - meanVar / (meanVar + theta)) * p * blurAlpha, 0, 1);  // 方差越大（纹理/边缘）越少模糊
  result = mix(src, mean, kMin);
  sum = 3x3 二项低通(src);  hPass = src - sum;
  color = result + sharpen * hPass * 2;
}
```

属性：`skin_smoothing` (blurAlpha) 0..1，`whiteness` 0..1；`sharpen` 无默认值（0）。→ `src/skin/smooth_gpupixel.rs`。

## 美白链（whiten > 0）

```
color  = clamp((color − 0.02588) · 1.02657)                      // 色阶
texel  = per-channel lookUpGray(256×1) 曲线;  texel = mix(color, texel, 0.5);  texel = mix(colorEPM, texel, 0.7)
texel  = lookUpOrigin (64×64, 16 个蓝切片 4×4) 查表后 mix(colorOrigin, color, 0.7)
color  = lookUpSkin  (64×64) 查表
custom = lookUpCustom(512×512) 查表;  color = mix(color, custom, whiten)
```

本项目美白采用方案 4.4.1 的"参数化曲线 / 512 查找图 × 皮肤遮罩"，未复刻此链；`lookup_light.png` /
`lookup_custom.png` 可作为 512 查找图直接用于 `--lut` 或 `--whiten-lut`（Apache-2.0，见 THIRD_PARTY_LICENSES.md）。

## 形变（FaceReshapeFilter，106 点 Face++ 布局）

```glsl
curveWarp(coord, origin, target, delta):
    direction = (target − origin) · delta;  radius = |target − origin|（y 按 aspectRatio 修正）
    ratio = clamp(1 − |coord − origin| / radius, 0, 1);  coord −= direction · ratio
thinFace: 9 对 (origin → target)：(3→44) (29→44) (7→45) (25→45) (10→46) (22→46) (14→49) (18→49) (16→49)
enlargeEye(coord, origin, radius, delta): w = |coord − origin| / radius; w = clamp(1 − (1 − w²)·delta, 0, 1); coord = origin + (coord − origin)·w
bigEye: (74→72) (77→75)，radius = 5 · |target − origin|；delta 范围 [0, 0.15]
```

与美狐的区别：线性衰减而非 `((R²−r²)/(R²−r²+(r−δ)²))²`；目标点是鼻梁 / 鼻底（向脸中线推挤）；大眼用二次曲线。
→ `src/warp/pinch.rs::curve_warp / enlarge_gpupixel`，`ReshapeStyle::GpuPixel`。
语义点 44 / 45 用 `nose_bridge_top → nose_tip` 的 1/3、2/3 插值，49 用 `nose_bottom`。

## 口红 / 腮红（未实现）

`FaceMakeupFilter` 用嘴部 / 脸颊关键点建立三角网格，把 `mouth.png` / `blusher.png` 纹理贴到人脸并按混合模式叠加，
属于"美妆"，不在本方案范围。
