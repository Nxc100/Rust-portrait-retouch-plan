#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预设校准的两个常用循环（doc/adding_a_preset.md）：

1. 只调色的底图（肤色、身体色调拟合与"立体"评估的基准：调色做了、修图与形变都关掉）：
    python tools/preset_sweep.py graded --preset <预设> --out out/_work/<预设>/graded [--input test/原片] [--ref <参考导出目录>]
   给出 --ref 时另写 <out>/skin_pairs.json，可直接交给 tools/skin_tone_fit.py。

2. 一组参数的试跑：批处理 → 皮肤带通中位比（skin_bands）+ 大尺度结构增益（band_gain），只打印汇总行：
    python tools/preset_sweep.py run --preset <预设> --ref <参考导出目录> --name v1 [--base <底图目录>] \\
        [--face detail_smooth=0.3 ...] [--male detail_smooth=0.3 ...] [--both stereo=0.2 ...] [--set 键=值 ...]
   --face 作用于 cream_female 与 cream_unknown，--male 作用于 cream_male，--both 三者都设，--set 原样传给 retouch。
   输出在 out/_work/sweep/<name>/（--work 可改）。没有预设调色时 --base 省略（以原片为底图）。

需要先 `cargo build --release`；批处理占用 target/release/retouch(.exe) 时不要重新编译（Windows 会因文件被占用而失败）。
"""
import argparse
import json
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
RESHAPE_KEYS = ('thin_face', 'big_eye', 'thin_nose', 'chin_lift', 'face_narrow')
GENDERS = ('female', 'male', 'unknown')


def exe():
    for name in ('retouch.exe', 'retouch'):
        p = os.path.join(ROOT, 'target', 'release', name)
        if os.path.exists(p):
            return p
    raise SystemExit('先 cargo build --release')


def batch(preset, inp, out, sets, landmarks=True):
    os.makedirs(out, exist_ok=True)
    cmd = [exe(), 'batch', '-i', inp, '-o', out, '--preset', preset, '--overwrite', '--no-report']
    if landmarks:
        cmd += ['--landmarks-dir', os.path.join(out, '_landmarks')]
    for s in sets:
        cmd += ['--set', s]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ['']
    print(tail[0])
    if r.returncode != 0:
        raise SystemExit(r.stderr)


def py(script, args, keep):
    cmd = [sys.executable, '-X', 'utf8', os.path.join(TOOLS, script)] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    for line in r.stdout.splitlines():
        if line.startswith(keep):
            print(line)
    if r.returncode != 0:
        print(r.stderr[-2000:])


def cmd_graded(a):
    sets = ['smooth=0'] + [f'reshape.{g}.{k}=0' for g in GENDERS for k in RESHAPE_KEYS]
    batch(a.preset, a.input, a.out, sets)
    if a.ref:
        pairs = []
        for f in sorted(os.listdir(a.out)):
            stem, ext = os.path.splitext(f)
            ref = [q for q in os.listdir(a.ref) if os.path.splitext(q)[0].lower() == stem.lower()]
            lm = os.path.join(a.out, '_landmarks', stem + '.json')
            if ext.lower() == '.jpg' and ref and os.path.exists(lm):
                pairs.append({'orig': os.path.join(a.out, f), 'ref': os.path.join(a.ref, ref[0]), 'faces': lm})
        path = os.path.join(a.out, 'skin_pairs.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(pairs, fh, ensure_ascii=False, indent=1)
        print(f'{len(pairs)} 对 → {path}')


def cmd_run(a):
    sets = list(a.set)
    for kv in a.face:
        sets += [f'cream_female.{kv}', f'cream_unknown.{kv}']
    for kv in a.male:
        sets.append(f'cream_male.{kv}')
    for kv in a.both:
        sets += [f'cream_{g}.{kv}' for g in GENDERS]
    out = os.path.join(a.work, a.name)
    print(f'== {a.name}: ' + ' '.join(sets))
    batch(a.preset, a.input, out, sets)
    lm = os.path.join(out, '_landmarks')
    common = ['--orig', a.input, '--landmarks', lm, '--ref', a.ref, '--ours', f'{a.name}={out}']
    py('skin_bands.py', common, ('女', '男', '带通'))
    py('band_gain.py', common + ['--base', a.base or a.input], ('女', '男'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('graded', help='只调色、不修图、不形变的底图')
    g.add_argument('--preset', required=True)
    g.add_argument('--out', required=True)
    g.add_argument('--input', default='test/原片')
    g.add_argument('--ref', default=None)
    r = sub.add_parser('run', help='一组参数的试跑与评估')
    r.add_argument('--preset', required=True)
    r.add_argument('--ref', required=True)
    r.add_argument('--name', required=True)
    r.add_argument('--input', default='test/原片')
    r.add_argument('--base', default=None, help='只调色的底图目录（没有预设调色时省略）')
    r.add_argument('--work', default='out/_work/sweep')
    r.add_argument('--face', action='append', default=[], metavar='键=值')
    r.add_argument('--male', action='append', default=[], metavar='键=值')
    r.add_argument('--both', action='append', default=[], metavar='键=值')
    r.add_argument('--set', action='append', default=[], metavar='键=值')
    a = ap.parse_args()
    {'graded': cmd_graded, 'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    main()
