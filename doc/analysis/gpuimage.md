# BradLarson/GPUImage 底层滤镜公式

仓库：`example/GPUImage`（BSD-3）。美狐仓库内 `GPURenderKit/.../GPUImage` 为同一代码的副本，常数一致。

## GPUImageBilateralFilter

- `GPUImageTwoPassTextureSamplingFilter` 子类：第一遍 `texelWidthOffset = texelSpacingMultiplier / width`（水平），
  第二遍 `texelHeightOffset = texelSpacingMultiplier / height`（垂直）。默认 `texelSpacingMultiplier = 4.0`，
  `distanceNormalizationFactor = 8.0`（BB 滤镜改为 4.0）。
- 顶点着色器生成 9 个采样坐标：`offset_i = (i − 4) · spacing`。
- 片元：

```glsl
central = tex(blur[4]); wsum = 0.18; sum = central * 0.18;
for i in {0,1,2,3,5,6,7,8}:
    s = tex(blur[i]);
    d = min(distance(central, s) * dnf, 1.0);        // vec4 欧氏距离，alpha 恒 1 不贡献
    w = W9[i] * (1.0 - d);  wsum += w;  sum += s * w;
out = sum / wsum;      // W9 = .05 .09 .12 .15 .18 .15 .12 .09 .05
```

→ `src/skin/bilateral.rs`。spacing 为整数像素时 `GL_LINEAR` 退化为最近像素读取。

## GPUImageSobelEdgeDetectionFilter

`GPUImageTwoPassFilter`：第一遍 `GPUImageGrayscaleFilter`（`W = (0.2125, 0.7154, 0.0721)`），
第二遍 3×3（`GPUImage3x3TextureSamplingFilter` 顶点着色器给出 8 邻域，texel = 1/width, 1/height）：

```glsl
h = -tl - 2t - tr + bl + 2b + br;   v = -bl - 2l - tl + br + 2r + tr;
mag = length(vec2(h, v)) * edgeStrength;
```

→ `src/skin/edge.rs`。BB 滤镜的组合 shader 只读 `.r`（= mag）。

## GPUImageHSBFilter

矩阵算法来自 graficaobscura，但实际编译常数为：

```c
//#define RLUM (0.3086f)  //#define GLUM (0.6094f)  //#define BLUM (0.0820f)   ← 被注释
#define RLUM (0.3f)
#define GLUM (0.59f)
#define BLUM (0.11f)
```

`saturatemat`：`a=(1−s)rw+s, b=(1−s)rw, c=(1−s)rw; d=(1−s)gw, e=(1−s)gw+s, ...`；行向量约定 `p' = p·M`，
`matrixmult(mmat, mat, mat)` 得 `M_new = M_old · mmat`，即后调用的 adjust 后作用。`cscalemat` 为 `diag(b,b,b,1)`，
与饱和矩阵可交换。`GPUImageColorMatrixFilter` shader：`out = mix(color, color * colorMatrix, intensity)`，intensity = 1。
等价公式：`c' = clamp(b · (lum + (c − lum) · s), 0, 1)`，`lum = 0.3r + 0.59g + 0.11b`。→ `src/color/hsb.rs`。

## GPUImageLookupFilter（512×512）

```glsl
blue = b * 63;  quad1 = (floor(blue) mod 8, floor(blue) / 8);  quad2 同 ceil(blue)
texPos.x = quad.x * 0.125 + 0.5/512 + (0.125 − 1/512) * r;
texPos.y = quad.y * 0.125 + 0.5/512 + (0.125 − 1/512) * g;
new = mix(tex(lut, texPos1), tex(lut, texPos2), fract(blue));
out = mix(color, new, intensity);
```

→ `src/color/lookup512.rs`。恒等图生成：`r = (x mod 64)/63, g = (y mod 64)/63, b = ((y/64)*8 + x/64)/63`，
恒等测试最大误差 ≤ 1/255（`tests/identity.rs`）。

## 帧缓冲精度

GPUImage 每个 pass 输出到 8-bit RGBA 帧缓冲，pass 之间有量化；本实现全程 f32，
与 numpy oracle（`tools/oracle_numpy.py`）对比误差 ≤ 1/255。
