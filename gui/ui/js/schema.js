// 修图参数表单的描述（字段名与库的 ParamOptions 一致，默认值由后端 app_info 提供）。
//
// 字段类型：range（滑块）、toggle（开关）、select（下拉）、file（选择文件）、lines（每行一项的文本）。
// show(o) 决定何时显示；mode 为 'preset' / 'manual' 时只在对应模式下显示。

export const SMOOTH_MODES = [
  { value: 'faithful', label: '直播磨皮（方案 A · 美狐 Beautify）' },
  { value: 'freqsep', label: '自然磨皮（方案 B · 高反差保留）' },
  { value: 'gpupixel', label: '均值方差磨皮（方案 C · gpupixel）' },
  { value: 'cream', label: '奶油肌（保纹理磨皮 + 匀肤 + 祛瑕疵）' },
];

export const RESHAPE_STYLES = [
  { value: 'meihu', label: '美狐（FaceChange）' },
  { value: 'gpupixel', label: 'gpupixel（FaceReshape）' },
];

export const DEVELOP_CHOICES = [
  { value: null, label: '跟随预设' },
  { value: true, label: '开启' },
  { value: false, label: '关闭' },
];

const isCream = (o) => o.preset != null || o.smooth_mode === 'cream';

export const GROUPS = [
  {
    id: 'smooth',
    title: '磨皮',
    mode: 'manual',
    fields: [
      { key: 'smooth_mode', type: 'select', label: '磨皮模式', options: SMOOTH_MODES },
      { key: 'smooth', type: 'range', label: '磨皮强度', min: 0, max: 1, step: 0.01 },
      {
        key: 'restrict_to_face', type: 'toggle', label: '只在人脸内磨皮 / 美白',
        hint: '关闭后直播磨皮按肤色规则作用于全图',
      },
      {
        key: 'freqsep_radius', type: 'range', label: '高反差半径', min: 1, max: 30, step: 0.5,
        hint: '相对短边 1000 px 的半径', show: (o) => o.smooth_mode === 'freqsep',
      },
      {
        key: 'freqsep_sharpness', type: 'range', label: '高反差锐化', min: 0, max: 2, step: 0.05,
        show: (o) => o.smooth_mode === 'freqsep',
      },
      {
        key: 'gpupixel_sharpen', type: 'range', label: '锐化', min: 0, max: 1, step: 0.05,
        show: (o) => o.smooth_mode === 'gpupixel',
      },
    ],
  },
  {
    id: 'cream',
    title: '奶油肌',
    show: isCream,
    fields: [
      {
        key: 'body_skin', type: 'toggle', label: '处理身体皮肤',
        hint: '脖颈、胸口、手臂、手：磨皮、匀肤、祛瑕疵；脖子再淡化颈纹（需要人像抠图 / 皮肤分割模型）',
      },
      {
        key: 'ai_blemish', type: 'toggle', label: 'AI 瑕疵祛除',
        hint: 'ABPN 检测 + 修复痘印、斑点（需要 AI 瑕疵模型）；关闭则只用经典检测',
      },
    ],
  },
  {
    id: 'tone',
    title: '色调与美白',
    mode: 'manual',
    fields: [
      { key: 'brightness', type: 'range', label: '亮度', min: 0.5, max: 1.5, step: 0.01, hint: 'HSB 亮度，1.0 = 不变' },
      { key: 'saturation', type: 'range', label: '饱和度', min: 0, max: 2, step: 0.01, hint: 'HSB 饱和度，1.0 = 不变' },
      { key: 'apply_log_curve', type: 'toggle', label: 'log 提亮曲线' },
      { key: 'whiten', type: 'range', label: '美白', min: 0, max: 1, step: 0.01 },
      {
        key: 'whiten_lut', type: 'file', label: '美白查找图', fileKind: 'lookup',
        hint: '512×512 PNG；不选则用参数化美白曲线',
      },
    ],
  },
  {
    id: 'reshape',
    title: '美型',
    mode: 'manual',
    fields: [
      { key: 'thin_face', type: 'range', label: '瘦脸', min: 0, max: 1, step: 0.01 },
      { key: 'big_eye', type: 'range', label: '大眼', min: 0, max: 1, step: 0.01 },
      { key: 'thin_nose', type: 'range', label: '瘦鼻', min: 0, max: 1, step: 0.01 },
      { key: 'chin_lift', type: 'range', label: '缩下巴', min: 0, max: 1, step: 0.01, hint: '缩下巴 / 提升下半脸' },
      { key: 'face_narrow', type: 'range', label: '整体收窄', min: 0, max: 0.06, step: 0.001, hint: '比例，如 0.03' },
      { key: 'reshape_style', type: 'select', label: '形变风格', options: RESHAPE_STYLES },
      { key: 'attenuate_yaw', type: 'toggle', label: '侧脸衰减', hint: '侧脸时减弱瘦脸 / 瘦鼻，避免变形' },
    ],
  },
  {
    id: 'style',
    title: '风格与冲印',
    fields: [
      { key: 'style_lut', type: 'file', label: '风格 LUT', fileKind: 'lut', hint: '.cube 或 512×512 查找图 PNG' },
      {
        key: 'style_intensity', type: 'range', label: 'LUT 强度', min: 0, max: 1, step: 0.01,
        show: (o) => !!o.style_lut,
      },
      {
        key: 'embedded_develop', type: 'select', label: '内嵌冲印设置', options: DEVELOP_CHOICES,
        hint: '再应用照片内嵌的 Camera Raw 设置（XMP crs）',
      },
    ],
  },
  {
    id: 'advanced',
    title: '高级',
    collapsed: true,
    fields: [
      { key: 'warp_coeffs', type: 'file', label: '形变系数 JSON', fileKind: 'json', hint: '字段见 WarpCoefficients' },
      {
        key: 'masked_luts', type: 'lines', label: '遮罩 LUT',
        placeholder: '每行一个：<cube>:<mask.png>[:invert][:strength]',
      },
    ],
  },
];

/** 字段是否显示。 */
export function fieldVisible(field, group, options) {
  const mode = options.preset != null ? 'preset' : 'manual';
  if (group.mode && group.mode !== mode) return false;
  if (group.show && !group.show(options)) return false;
  return !field.show || field.show(options);
}

const FIELD_INDEX = new Map(GROUPS.flatMap((g) => g.fields.map((f) => [f.key, { field: f, group: g }])));

/** 参数的一行摘要（批处理页显示"用什么参数"）。只列出与默认值不同、且在当前模式下生效的项。 */
export function describeOptions(o, defaults, presets) {
  const parts = [];
  if (o.preset != null) {
    const p = presets.find((x) => x.id === o.preset);
    parts.push(`预设「${p ? p.name : o.preset}」`);
    if (o.sets?.length) parts.push(`覆盖 ${o.sets.length} 项`);
  } else {
    const mode = SMOOTH_MODES.find((m) => m.value === o.smooth_mode);
    parts.push(`手动 · ${mode ? mode.label.split('（')[0] : o.smooth_mode}`);
  }
  for (const [key, { field, group }] of FIELD_INDEX) {
    if (key === 'smooth_mode' || !fieldVisible(field, group, o)) continue;
    const v = o[key];
    if (JSON.stringify(v) === JSON.stringify(defaults[key])) continue;
    if (field.type === 'range') parts.push(`${field.label} ${Number(v).toFixed(field.step < 0.01 ? 3 : 2)}`);
    else if (field.type === 'toggle') parts.push(`${field.label}${v ? '开' : '关'}`);
    else if (field.type === 'select') {
      const opt = field.options.find((x) => x.value === v);
      parts.push(`${field.label}：${opt ? opt.label : v}`);
    } else if (field.type === 'file') parts.push(`${field.label}：${String(v).split(/[\\/]/).pop()}`);
    else if (field.type === 'lines' && v.length) parts.push(`${field.label} ${v.length} 项`);
  }
  return parts.join(' · ');
}
