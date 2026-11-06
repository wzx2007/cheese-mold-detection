# -*- coding: utf-8 -*-
"""
prepare_dataset.py
------------------
构建「正常奶酪 / 污染霉菌奶酪」二分类数据集。

数据源
------
Kaggle `jenlooper/cheese`（All The Cheese，3234 张，6 个表皮工艺类目）——
当前可公开下载的、样本量最大的奶酪图像集。

主任务只保留 4 个「正常状态不该长霉」的类目：
    hard / semi-soft / fresh / washed-rind      （共 2165 张）
排除 blue（蓝纹，天然带青霉）与 bloomy（白霉外皮，天然带白霉），
否则标签会自相矛盾。

样本构造
--------
每张原图切 2 个互不重叠的方形 crop（掩膜引导，保证 crop 内 ≥42% 是奶酪本体）：
  * 一个 crop 保持「正常」，
  * 另一个 crop 合成污染霉菌。
哪一个是霉菌随机决定 —— 两类来自同一批原始照片，光照/背景/相机分布完全对齐，
模型无法靠「照片来源」抄近路，只能学局部表面证据。
随后两类都经过同一套相机成像退化（camera_simulate）。

用法（支持分片，便于长任务分段执行）
------------------------------------
  python prepare_dataset.py --start 0 --count 400 --demo
  python prepare_dataset.py --start 400 --count 400
  ...
  python prepare_dataset.py --merge
"""

import os
import csv
import json
import zipfile
import argparse
import numpy as np
import cv2

import mold_synth as MS

import paths  # noqa: E402

ZIP = os.path.join(paths.RAW, "cheese_kaggle.zip")
OUT = paths.DATA
PARTS = os.path.join(OUT, "manifest_parts")
FIG = paths.FIG

KEEP_CLASSES = ["hard", "semi-soft", "fresh", "washed-rind"]
EXCLUDE = {"blue", "bloomy"}
SIZE = 384
SEED = 20261106

FIELDS = ["sample_id", "src_id", "cls", "crop", "label", "label_name",
          "species", "colonies", "mold_area", "path"]


def imread_bytes(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def imwrite_path(path, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


def collect(zf):
    items = {}
    for name in zf.namelist():
        low = name.lower()
        if not low.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            continue
        parts = name.replace("\\", "/").split("/")
        cls = None
        for p in parts[:-1]:
            if p.lower() in KEEP_CLASSES or p.lower() in EXCLUDE:
                cls = p.lower()
                break
        if cls is not None:
            items.setdefault(cls, []).append(name)
    return items


def build_pool(items):
    """确定性的样本池：同一份数据每次得到相同的索引顺序。"""
    rng = np.random.default_rng(SEED)
    pool = []
    for cls in KEEP_CLASSES:
        paths = sorted(items.get(cls, []))
        rng.shuffle(paths)
        pool.extend([(cls, p) for p in paths])
    return pool


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def yellow_frac(bgr):
    """黄色主体占比（奶酪）—— 花岗岩/木桌/盘子等中性色会被剔除。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ok = (H >= 12) & (H <= 45) & (S >= 60) & (V >= 85)
    return float(ok.mean())


def two_crops(img, rng, min_cov=0.48, min_yellow=0.32, want=4):
    """掩膜引导地切若干互不重叠的方形 crop，保证 crop 里以奶酪本体为主。

    三重判据：掩膜覆盖率、黄色占比、以及掩膜本身在较高分辨率下计算
    （低分辨率会把花岗岩台面的高频颗粒糊掉，导致误判）。
    返回 2 或 4 个 crop（4 个时组成两组「一好一霉」配对）。
    """
    h, w = img.shape[:2]
    side = int(min(h, w) * 0.52)
    if side < 64:
        return None
    sc = min(1.0, 720.0 / max(h, w))
    small = cv2.resize(img, (max(8, int(w * sc)), max(8, int(h * sc))), interpolation=cv2.INTER_AREA)
    m = MS.cheese_mask(small) > 0
    sside = max(8, int(side * sc))
    sh, sw = m.shape

    step = max(6, int(side * 0.10))
    cands = []
    for y in range(0, max(1, h - side + 1), step):
        for x in range(0, max(1, w - side + 1), step):
            sy, sx = int(y * sc), int(x * sc)
            hy, hx = min(sh - sy, sside), min(sw - sx, sside)
            if hy < 8 or hx < 8:
                continue
            cands.append((float(m[sy:sy + hy, sx:sx + hx].mean()), x, y))
    if not cands:
        return None
    cands.sort(reverse=True)

    picks = []
    for cov, x, y in cands:
        if cov < min_cov:
            break
        box = (x, y, x + side, y + side)
        if any(_iou(box, (px, py, px + side, py + side)) >= 0.08 for _, px, py in picks):
            continue
        probe = cv2.resize(img[y:y + side, x:x + side], (192, 192), interpolation=cv2.INTER_AREA)
        if yellow_frac(probe) < min_yellow:
            continue
        picks.append((cov, x, y))
        if len(picks) == want:
            break
    if len(picks) < 2:
        return None
    if len(picks) < 4:
        pickn = picks[:2]
    else:
        pickn = picks[:4]
    return [cv2.resize(img[y:y + side, x:x + side], (SIZE, SIZE),
                       interpolation=cv2.INTER_AREA) for _, x, y in pickn]


def run_shard(start, count, demo):
    for d in ["good", "moldy", None]:
        if d:
            os.makedirs(os.path.join(OUT, d), exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    rows, demo_pairs = [], []
    with zipfile.ZipFile(ZIP) as zf:
        items = collect(zf)
        pool = build_pool(items)
        chunk = pool[start:start + count] if count else pool[start:]
        print(f"pool={len(pool)} shard={start}..{start+len(chunk)}")

        for i in range(start, start + len(chunk)):
            cls, zp = pool[i]
            rng = np.random.default_rng(SEED + i * 7919)   # 每张图独立可复现
            try:
                img = imread_bytes(zf.read(zp))
                if img is None or min(img.shape[:2]) < 64:
                    continue
            except Exception:
                continue

            crops = two_crops(img, rng)
            if crops is None:
                continue
            groups = [crops[:2]] if len(crops) < 4 else [crops[:2], crops[2:4]]
            flat = []
            for g in groups:
                if rng.random() < 0.5:
                    flat += [(g[0], 0), (g[1], 1)]
                else:
                    flat += [(g[0], 1), (g[1], 0)]

            src_id = f"{cls}_{i:05d}"
            for j, (crop, label) in enumerate(flat):
                meta = {"species": "", "colonies": 0, "coverage": 0.0}
                tmp = crop
                if label == 1:
                    cov = (rng.uniform(0.015, 0.045), rng.uniform(0.05, 0.105))
                    tmp, meta = MS.apply_mold(crop, rng, coverage=cov)
                    if meta["colonies"] < 3:
                        tmp, meta = MS.apply_mold(crop, rng, coverage=cov)
                noisy = MS.camera_simulate(tmp, rng)

                sid = f"{src_id}_{'abcd'[j]}"
                rel = os.path.join("good" if label == 0 else "moldy", sid + ".jpg")
                imwrite_path(os.path.join(OUT, rel), noisy)
                rows.append({
                    "sample_id": sid, "src_id": src_id, "cls": cls, "crop": "abcd"[j],
                    "label": int(label), "label_name": "moldy" if label else "good",
                    "species": meta["species"], "colonies": meta["colonies"],
                    "mold_area": round(float(meta["coverage"]), 5),
                    "path": rel.replace("\\", "/"),
                })
                if demo and label == 1 and len(demo_pairs) < 12:
                    demo_pairs.append((crop, tmp, noisy, meta["species"]))

            if (i + 1) % 100 == 0:
                print(f"  {i+1} src, {len(rows)} samples")

    part = os.path.join(PARTS, f"shard_{start:05d}.csv")
    with open(part, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"shard written {part} rows={len(rows)}")
    if demo and demo_pairs:
        make_demo(demo_pairs)


def merge():
    all_rows = []
    for f in sorted(os.listdir(PARTS)):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(PARTS, f), encoding="utf-8") as fh:
            all_rows.extend(list(csv.DictReader(fh)))
    seen, rows = set(), []
    for r in all_rows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        rows.append(r)
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    per_cls = {}
    for r in rows:
        per_cls[r["cls"]] = per_cls.get(r["cls"], 0) + 1
    src = {r["src_id"] for r in rows}
    info = {
        "source": "Kaggle jenlooper/cheese (All The Cheese)",
        "kept_classes": KEEP_CLASSES,
        "excluded_classes": sorted(EXCLUDE),
        "src_images": len(src),
        "per_class_samples": per_cls,
        "samples": len(rows),
        "good": sum(1 for r in rows if r["label"] == "0"),
        "moldy": sum(1 for r in rows if r["label"] == "1"),
        "mean_mold_area": round(float(np.mean([float(r["mold_area"]) for r in rows if r["label"] == "1"])), 4),
        "size": SIZE,
    }
    with open(os.path.join(OUT, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def make_demo(demo_pairs):
    n = len(demo_pairs)
    cell, pad = 240, 8
    canvas = np.full((n * (cell + pad) + pad, 3 * (cell + pad) + pad, 3), 255, np.uint8)
    for i, (orig, mold, noisy, sp) in enumerate(demo_pairs):
        for k, im in enumerate([orig, mold, noisy]):
            t = cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA)
            y, x = pad + i * (cell + pad), pad + k * (cell + pad)
            canvas[y:y + cell, x:x + cell] = t
    imwrite_path(os.path.join(FIG, "synthesis_triplet.jpg"), canvas)

    tiles = []
    base = np.full((SIZE, SIZE, 3), 205, np.uint8)
    yy = np.linspace(0, 1, SIZE, dtype=np.float32)[:, None]
    base = np.clip(base.astype(np.float32) * (0.92 + 0.14 * yy), 0, 255).astype(np.uint8)
    for i in range(len(MS.MOLD_SPECIES)):
        core, edge, name = MS.MOLD_SPECIES[i]
        img = base.copy()
        img, _ = MS.apply_mold(img, np.random.default_rng(1000 + i),
                               coverage=(0.05, 0.10),
                               species=(np.array(core, np.float32),
                                        np.array(edge, np.float32), name))
        cv2.putText(img, str(i + 1), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)
        tiles.append(img)
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:] + [np.full((SIZE, SIZE, 3), 255, np.uint8)])
    imwrite_path(os.path.join(FIG, "mold_species.jpg"), np.vstack([row1, row2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
        return
    run_shard(args.start, args.count, args.demo)


if __name__ == "__main__":
    main()
