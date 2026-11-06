# -*- coding: utf-8 -*-
"""
analyze.py —— 误差分析：把"哪个数"变成"错在哪"。

输出 outputs/error_analysis.md，内容：
  1) 两个模型在测试集上的逐样本对照（一致 / 分歧 / 谁对谁错）
  2) 错误样本按"霉菌覆盖率"分档的错误率 —— 回答"是不是小霉斑最难"
  3) 错误样本按霉菌菌种 / 奶酪类目分组的错误率
  4) 阈值敏感性（只做描述，不据此改模型）

用法：python code/analyze.py
"""

import os, json, csv
import numpy as np

import paths

CACHE = ("cnn_final_metrics.json", "cnn3_metrics.json", "cnn2_metrics.json", "cnn_metrics.json")


def load_cnn():
    for n in CACHE:
        p = os.path.join(paths.RES, n)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8")), n
    raise FileNotFoundError("找不到 CNN 指标文件")


def main():
    cnn, cnn_name = load_cnn()
    cls = json.load(open(os.path.join(paths.RES, "classic_metrics.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(paths.DATA, "split.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(paths.DATA, "manifest.csv"), encoding="utf-8")))
    src = np.array([r["src_id"] for r in rows])
    test_rows = [r for r, m in zip(rows, np.isin(src, split["test_src"])) if m]

    y = np.array(cnn["test_true"])
    pc = np.array(cnn["test_probs"])
    ps = np.array(cls["test_probs"])
    assert len(test_rows) == len(y), (len(test_rows), len(y))

    pred_c = (pc >= 0.5).astype(int)
    pred_s = (ps >= 0.5).astype(int)
    area = np.array([float(r["mold_area"]) for r in test_rows])
    species = np.array([r["species"].split(" ")[0] or "-" for r in test_rows])
    klass = np.array([r["cls"] for r in test_rows])

    L = []
    L.append("# 误差分析（自动生成）\n")
    L.append(f"- CNN 指标来源：`{cnn_name}`（{cnn['model']}）")
    L.append(f"- 传统方法：`{cls['model']}`")
    L.append(f"- 测试集样本数：{len(y)}（正常 {(y==0).sum()} / 发霉 {(y==1).sum()}）\n")

    # ---- 1. 两模型交叉对照
    agree = pred_c == pred_s
    both_ok = agree & (pred_c == y)
    both_bad = agree & (pred_c != y)
    cnn_only = (~agree) & (pred_c == y)
    cls_only = (~agree) & (pred_s == y)
    L.append("## 1. 两个模型的判断一致性\n")
    L.append("| 情形 | 样本数 | 占比 |")
    L.append("|---|---|---|")
    for name, v in [("两者都对", both_ok), ("两者都错", both_bad),
                    ("只有 CNN 对", cnn_only), ("只有传统方法对", cls_only)]:
        L.append(f"| {name} | {int(v.sum())} | {v.mean()*100:.1f}% |")
    L.append("")
    L.append(f"Cohen's κ（判定一致性）：{_kappa(pred_c, pred_s):.3f}\n")

    # ---- 2. 按覆盖率分档
    L.append("## 2. 错误率 vs 霉菌覆盖率（只统计发霉样本）\n")
    bins = [(0.0, 0.010), (0.010, 0.020), (0.020, 0.035), (0.035, 0.060), (0.060, 1.0)]
    L.append("| 覆盖率区间 | 样本数 | CNN 漏检率 | 传统方法漏检率 |")
    L.append("|---|---|---|---|")
    for lo, hi in bins:
        m = (y == 1) & (area >= lo) & (area < hi)
        if m.sum() == 0:
            continue
        L.append(f"| {lo*100:.1f}% – {hi*100:.1f}% | {int(m.sum())} | "
                 f"{(pred_c[m] == 0).mean()*100:.1f}% | {(pred_s[m] == 0).mean()*100:.1f}% |")
    L.append("")

    # ---- 3. 按菌种 / 类目
    L.append("## 3. 分组错误率\n")
    L.append("### 按霉菌主色（仅发霉样本）\n")
    L.append("| 菌种 | 样本数 | CNN 漏检 | 传统漏检 |")
    L.append("|---|---|---|---|")
    for sp in sorted(set(species[y == 1])):
        m = (y == 1) & (species == sp)
        if m.sum() < 5:
            continue
        L.append(f"| {sp} | {int(m.sum())} | {(pred_c[m]==0).sum()} ({(pred_c[m]==0).mean()*100:.0f}%) "
                 f"| {(pred_s[m]==0).sum()} ({(pred_s[m]==0).mean()*100:.0f}%) |")
    L.append("")
    L.append("### 按奶酪类目（全部样本）\n")
    L.append("| 类目 | 样本数 | CNN 错误率 | 传统错误率 |")
    L.append("|---|---|---|---|")
    for k in sorted(set(klass)):
        m = klass == k
        L.append(f"| {k} | {int(m.sum())} | {(pred_c[m]!=y[m]).mean()*100:.1f}% "
                 f"| {(pred_s[m]!=y[m]).mean()*100:.1f}% |")
    L.append("")

    # ---- 4. 阈值敏感性
    L.append("## 4. 阈值敏感性（CNN，仅作描述）\n")
    L.append("| 阈值 | 准确率 | 召回率（发霉） | 精确率（发霉） |")
    L.append("|---|---|---|---|")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pr = (pc >= thr).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum())
        fp = int(((pr == 1) & (y == 0)).sum())
        fn = int(((pr == 0) & (y == 1)).sum())
        rec = tp / max(tp + fn, 1)
        pre = tp / max(tp + fp, 1)
        L.append(f"| {thr:.1f} | {(pr==y).mean()*100:.1f}% | {rec*100:.1f}% | {pre*100:.1f}% |")
    L.append("")
    L.append("> 阈值只在**训练集 OOF 预测**上选，再看测试集表现才合规；"
             "本表在测试集上扫描，仅用于说明判定阈值的敏感性，不作为调参依据。\n")

    out = os.path.join(paths.RES, "error_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("\nwrote", out)


def _kappa(a, b):
    n = len(a)
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in (0, 1))
    return (po - pe) / max(1 - pe, 1e-9)


if __name__ == "__main__":
    main()
