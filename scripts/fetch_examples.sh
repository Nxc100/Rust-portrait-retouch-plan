#!/usr/bin/env bash
# 把方案涉及的参考项目浅克隆到 example/（只读参考，不参与构建；目录已在 .gitignore 中）。
# 大仓库使用 sparse checkout 只取需要的目录。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p example && cd example
export GIT_TERMINAL_PROMPT=0

clone() { [ -d "$1" ] || git clone --depth 1 "$2" "$1"; }
sparse() {
  local name=$1 url=$2; shift 2
  if [ ! -d "$name" ]; then
    git clone --depth 1 --filter=blob:none --sparse "$url" "$name"
    (cd "$name" && git sparse-checkout set "$@")
  fi
}

clone gpupixel https://github.com/pixpark/gpupixel.git                                   # Apache-2.0：架构 / 磨皮 / 形变 shader
clone Meihu-Beautyface-sdk https://github.com/zhanghao5683934/Meihu-Beautyface-sdk.git   # GPL-3.0 + 非商用：还原依据（只取参数与公式）
clone YUCIHighPassSkinSmoothing https://github.com/YuAo/YUCIHighPassSkinSmoothing.git     # MIT：高反差磨皮
clone GPUImage https://github.com/BradLarson/GPUImage.git                                 # BSD-3：双边 / Sobel / HSB / 查找表
clone Ultra-Light-Fast-Generic-Face-Detector-1MB https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB.git  # MIT：检测模型
clone facemesh_onnx_tensorrt https://github.com/PINTO0309/facemesh_onnx_tensorrt.git      # Face Mesh ONNX 后处理说明
sparse insightface https://github.com/deepinsight/insightface.git \
  python-package/insightface/model_zoo python-package/insightface/utils python-package/insightface/app \
  python-package/insightface/data alignment/coordinate_reg
sparse mediapipe https://github.com/google-ai-edge/mediapipe.git \
  mediapipe/python/solutions mediapipe/modules/face_landmark mediapipe/modules/face_geometry \
  mediapipe/modules/face_detection mediapipe/calculators/util
sparse PINTO_model_zoo https://github.com/PINTO0309/PINTO_model_zoo.git 032_FaceMesh
# 奶油肌（第一轮）：导向滤波、人脸解析、人像抠图
clone guided-filter https://github.com/lisabug/guided-filter.git                          # MIT：导向滤波参考实现
clone face-parsing https://github.com/yakhyo/face-parsing.git                             # MIT：BiSeNet 人脸解析 ONNX
clone MODNet https://github.com/ZHKKKe/MODNet.git                                         # Apache-2.0：人像抠图
clone modnet-yakhyo https://github.com/yakhyo/modnet.git                                  # Apache-2.0：MODNet ONNX
# 奶油肌（第二轮）：祛疤 / 祛斑 / 纹理
clone FabSoften https://github.com/Gnimuc/FabSoften.git                                   # MIT：瑕疵检测 + 动态导向滤波 + 纹理恢复
clone PyPatchMatch https://github.com/vacancy/PyPatchMatch.git                            # MIT：PatchMatch 修复（评估）
clone poisson-image-editing https://github.com/cheind/poisson-image-editing.git           # Poisson 编辑（评估）
clone Blemish-Removal https://github.com/CzJLee/Blemish-Removal.git                       # OpenCV 修复画笔示例
clone Wrinkle-detection https://github.com/elliotnaylor/Wrinkle-detection.git             # Gabor / Frangi 皱纹检测示例
clone lama https://github.com/advimman/lama.git                                           # Apache-2.0：LaMa 深度修复（评估）
clone LaMa-OxiONNX https://github.com/CloudyTabzy/LaMa-OxiONNX.git                        # Apache-2.0：LaMa ONNX 纯 Rust 推理（评估）
clone Skin-Clothes-Hair-Segmentation-using-SMP https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP.git  # MIT：轻量皮肤 / 衣物 / 头发分割 ONNX（备选）
# ModelScope ABPN 人像美肤（Apache-2.0）：上游源码 + 权重；见 doc/analysis/abpn.md
mkdir -p modelscope-skin-retouching && (
  cd modelscope-skin-retouching
  for f in pipelines/cv/skin_retouching_pipeline.py models/cv/skin_retouching/__init__.py \
           models/cv/skin_retouching/unet_deploy.py models/cv/skin_retouching/utils.py models/cv/skin_retouching/weights_init.py \
           models/cv/skin_retouching/detection_model/__init__.py models/cv/skin_retouching/detection_model/detection_module.py \
           models/cv/skin_retouching/detection_model/detection_unet_in.py models/cv/skin_retouching/inpainting_model/__init__.py \
           models/cv/skin_retouching/inpainting_model/gconv.py models/cv/skin_retouching/inpainting_model/inpainting_unet.py; do
    [ -f "modelscope/$f" ] || { mkdir -p "modelscope/$(dirname "$f")"; curl -sL --ssl-no-revoke -o "modelscope/$f" "https://raw.githubusercontent.com/modelscope/modelscope/master/modelscope/$f"; }
  done
  for f in README.md configuration.json pytorch_model.pt joint_20210926.pth tf_graph.pb; do
    [ -f "$f" ] || curl -sL --ssl-no-revoke -o "$f" "https://www.modelscope.cn/api/v1/models/iic/cv_unet_skin-retouching/repo?Revision=master&FilePath=$f"
  done
)
echo "done"
