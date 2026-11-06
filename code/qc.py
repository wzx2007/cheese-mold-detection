# -*- coding: utf-8 -*-
"""快速质检：随机抽 12 对（一好一霉同源）拼成对比图。"""
import os, csv, random
import numpy as np, cv2

import paths  # noqa: E402

OUT = paths.DATA
rows = list(csv.DictReader(open(os.path.join(OUT, "manifest.csv"), encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault(r["src_id"], []).append(r)
pairs = [v for v in by.values() if len(v) >= 2]
random.seed(7)
sel = random.sample(pairs, 12)

cell = 190
canvas = np.full((12 * (cell + 6) + 6, 2 * (cell + 6) + 6, 3), 255, np.uint8)
for i, grp in enumerate(sel):
    g = sorted(grp, key=lambda r: int(r["label"]))
    for k, r in enumerate(g[:2]):
        img = cv2.imdecode(np.fromfile(os.path.join(OUT, r["path"].replace("/", os.sep)), np.uint8),
                           cv2.IMREAD_COLOR)
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        y, x = 6 + i * (cell + 6), 6 + k * (cell + 6)
        canvas[y:y + cell, x:x + cell] = img
        lab = "MOLDY" if r["label"] == "1" else "GOOD"
        col = (40, 40, 200) if r["label"] == "1" else (90, 140, 20)
        cv2.putText(canvas, lab, (x + 6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(canvas, lab, (x + 6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
open(os.path.join(paths.FIG, "qc_pairs.jpg"), "wb").write(buf.tobytes())
print("ok")
