#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照像素蛋糕几个预设的滑块值（读用户分析项目导出的 CSV，本仓库不收录这些数据）：列出各预设取值不同的参数。
加新预设的第一步——和已实现的预设（奶油肌、婚纱-深色内景）逐项对照，找出要新增或重新校准的功能。

    python tools/pixcake_param_diff.py --presets 奶油肌 婚纱-深色内景 <新预设> [--gender 女性] [--all]
        [--data D:/cakes/piCake/preset-export/data]

--gender：CSV 的分组后缀（女性、男性、儿童、通用(全局) 等，对应 预设参数对照_<分组>.csv）；另读 调色参数对照.csv
（调色 / AI 风格的参数）。默认只列各预设取值不同的行；--all 另列取值相同但不是中性值（空、0、50）的行。
只用参数值作依据；像素蛋糕的代码、配置、LUT 不进仓库（见 doc/adding_a_preset.md "约束"）。
"""
import argparse
import csv
import os

NEUTRAL = {'', '0', '0.0', '50', '[0, 0, 0, 0]', '""', 'None', '"None"'}


def load(path, presets):
    if not os.path.exists(path):
        return None, []
    with open(path, encoding='utf-8-sig') as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    missing = [p for p in presets if p not in head]
    idx = [head.index(p) for p in presets if p in head]
    return (idx, missing), rows[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--presets', nargs='+', required=True)
    ap.add_argument('--gender', default='女性')
    ap.add_argument('--data', default='D:/cakes/piCake/preset-export/data')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    for title, fname in ((f'修图（{args.gender}）', f'预设参数对照_{args.gender}.csv'), ('调色', '调色参数对照.csv')):
        meta, rows = load(os.path.join(args.data, fname), args.presets)
        if meta is None:
            print(f'== {title}：没有 {fname}')
            continue
        idx, missing = meta
        shown = [p for p in args.presets if p not in missing]
        print(f'== {title}  ' + ' | '.join(shown) + (f'  （CSV 里没有：{", ".join(missing)}）' if missing else ''))
        for r in rows:
            vals = [r[i].strip() if i < len(r) else '' for i in idx]
            differ = len(set(vals)) > 1
            notable = any(v not in NEUTRAL for v in vals)
            if differ or (args.all and notable):
                name = r[1] or r[2] or r[0]
                print(f'  {r[0]:>7} {name:<16} {r[2]:<30} ' + ' | '.join(f'{v:>8}' for v in vals))


if __name__ == '__main__':
    main()
