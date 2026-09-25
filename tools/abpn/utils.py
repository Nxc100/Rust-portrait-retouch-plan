# Copyright (c) Alibaba, Inc. and its affiliates.
# Subset of modelscope/models/cv/skin_retouching/utils.py needed by the
# standalone pipeline (whitening / RetinaFace helpers dropped).  Logic is kept
# identical to upstream; the only change is that `patch_aggregation_overlap`
# uses a pure-torch reshape instead of einops.rearrange (verified equal).
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    'gen_diffuse_mask', 'get_crop_bbox', 'get_roi_without_padding',
    'patch_aggregation_overlap', 'patch_partition_overlap', 'preprocess_roi',
    'roi_to_tensor', 'smooth_border_mg'
]


def get_crop_bbox(detecting_results):
    boxes = []
    for anno in detecting_results:
        if anno['score'] == -1:
            break
        boxes.append({
            'x1': anno['bbox'][0],
            'y1': anno['bbox'][1],
            'x2': anno['bbox'][2],
            'y2': anno['bbox'][3]
        })
    face_count = len(boxes)

    suitable_bboxes = []
    for i in range(face_count):
        face_bbox = boxes[i]

        face_bbox_width = abs(face_bbox['x2'] - face_bbox['x1'])
        face_bbox_height = abs(face_bbox['y2'] - face_bbox['y1'])

        face_bbox_center = ((face_bbox['x1'] + face_bbox['x2']) / 2,
                            (face_bbox['y1'] + face_bbox['y2']) / 2)

        square_bbox_length = face_bbox_height if face_bbox_height > face_bbox_width else face_bbox_width
        enlarge_ratio = 1.5
        square_bbox_length = int(enlarge_ratio * square_bbox_length)

        sideScale = 1

        square_bbox = {
            'x1':
            int(face_bbox_center[0] - sideScale * square_bbox_length / 2),
            'x2':
            int(face_bbox_center[0] + sideScale * square_bbox_length / 2),
            'y1':
            int(face_bbox_center[1] - sideScale * square_bbox_length / 2),
            'y2': int(face_bbox_center[1] + sideScale * square_bbox_length / 2)
        }

        suitable_bboxes.append(square_bbox)

    return suitable_bboxes


def get_roi_without_padding(img, bbox):
    crop_t = max(bbox['y1'], 0)
    crop_b = min(bbox['y2'], img.shape[0])
    crop_l = max(bbox['x1'], 0)
    crop_r = min(bbox['x2'], img.shape[1])
    roi = img[crop_t:crop_b, crop_l:crop_r]
    return roi, 0, [crop_t, crop_b, crop_l, crop_r]


def roi_to_tensor(img):
    img = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1))))[None, ...]

    return img


def preprocess_roi(img):
    img = img.float() / 255.0
    img = (img - 0.5) * 2

    return img


def patch_partition_overlap(image, p1, p2, padding=32):

    B, C, H, W = image.size()
    h, w = H // p1, W // p2
    image = F.pad(
        image,
        pad=(padding, padding, padding, padding, 0, 0),
        mode='constant',
        value=0)

    patch_list = []
    for i in range(h):
        for j in range(w):
            patch = image[:, :, p1 * i:p1 * (i + 1) + padding * 2,
                          p2 * j:p2 * (j + 1) + padding * 2]
            patch_list.append(patch)

    output = torch.cat(
        patch_list, dim=0)  # (b h w) c (p1 + 2 * padding) (p2 + 2 * padding)
    return output


def patch_aggregation_overlap(image, h, w, padding=32):

    image = image[:, :, padding:-padding, padding:-padding]

    # einops: rearrange(image, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=h, w=w)
    bhw, c, p1, p2 = image.shape
    b = bhw // (h * w)
    output = image.reshape(b, h, w, c, p1, p2).permute(0, 3, 1, 4, 2, 5)
    output = output.reshape(b, c, h * p1, w * p2)

    return output


def smooth_border_mg(diffuse_mask, mg):
    mg = mg - 0.5
    diffuse_mask = F.interpolate(
        diffuse_mask, mg.shape[:2], mode='bilinear')[0].permute(1, 2, 0)
    mg = mg * diffuse_mask
    mg = mg + 0.5
    return mg


def gen_diffuse_mask(out_channels=3):
    mask_size = 500
    diffuse_with = 20
    a = np.ones(shape=(mask_size, mask_size), dtype=np.float32)

    for i in range(mask_size):
        for j in range(mask_size):
            if i >= diffuse_with and i <= (
                    mask_size - diffuse_with) and j >= diffuse_with and j <= (
                        mask_size - diffuse_with):
                a[i, j] = 1.0
            elif i <= diffuse_with:
                a[i, j] = i * 1.0 / diffuse_with
            elif i > (mask_size - diffuse_with):
                a[i, j] = (mask_size - i) * 1.0 / diffuse_with

    for i in range(mask_size):
        for j in range(mask_size):
            if j <= diffuse_with:
                a[i, j] = min(a[i, j], j * 1.0 / diffuse_with)
            elif j > (mask_size - diffuse_with):
                a[i, j] = min(a[i, j], (mask_size - j) * 1.0 / diffuse_with)
    a = np.dstack([a] * out_channels)
    return a
