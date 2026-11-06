# -*- coding: utf-8 -*-
"""make_splits.py —— 生成统一的训练/测试划分与 5 折分组划分（两个模型共享）。"""

import os, json, csv
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

import paths  # noqa: E402  (仓库根目录 = code/ 的上一级)

OUT = paths.DATA
SEED = 20261106


def main():
    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    src = np.array([r["src_id"] for r in rows])
    uniq = np.array(sorted(set(src)))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, te_idx = next(gss.split(uniq, groups=uniq))
    train_src, test_src = sorted(uniq[tr_idx]), sorted(uniq[te_idx])

    gkf = GroupKFold(n_splits=5)
    folds = []
    tr_arr = np.array(train_src)
    for a, b in gkf.split(tr_arr, groups=tr_arr):
        folds.append({"fit": sorted(tr_arr[a].tolist()), "val": sorted(tr_arr[b].tolist())})

    split = {
        "seed": SEED,
        "test_size": 0.2,
        "n_src_total": int(len(uniq)),
        "train_src": train_src,
        "test_src": test_src,
        "folds": folds,
    }
    with open(os.path.join(OUT, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f)

    # 校验：样本数与类别分布
    def cnt(sset):
        s = set(sset)
        g = sum(1 for r in rows if r["src_id"] in s and r["label"] == "0")
        m = sum(1 for r in rows if r["src_id"] in s and r["label"] == "1")
        return g, m

    print("train src", len(train_src), "test src", len(test_src))
    print("train (good,moldy)", cnt(train_src))
    print("test  (good,moldy)", cnt(test_src))
    for i, fo in enumerate(folds):
        print(f"fold{i} fit={len(fo['fit'])} val={len(fo['val'])} val(g,m)={cnt(fo['val'])}")


if __name__ == "__main__":
    main()
