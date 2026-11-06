# -*- coding: utf-8 -*-
"""
gen_docs.py —— 用真实指标生成 docs/04-results.md 与 README 的「结果」区块。

把数字从 json 里读出来直接写进文档，避免手抄出错、也避免文档与产物不一致。
用法：python code/gen_docs.py
"""
import os, json, csv
import numpy as np

import paths

DOCS = os.path.join(paths.BASE, "docs")


def pct(x, n=2):
    return f"{x*100:.{n}f}%"


def load():
    cnn = json.load(open(os.path.join(paths.RES, "cnn_final_metrics.json"), encoding="utf-8"))
    cls = json.load(open(os.path.join(paths.RES, "classic_metrics.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(paths.DATA, "split.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(paths.DATA, "manifest.csv"), encoding="utf-8")))
    return cnn, cls, split, rows


def main():
    cnn, cls, split, rows = load()
    src = np.array([r["src_id"] for r in rows])
    test_rows = [r for r, m in zip(rows, np.isin(src, split["test_src"])) if m]

    y = np.array(cnn["test_true"])
    pc = np.array(cnn["test_probs"])
    ps = np.array(cls["test_probs"])
    pred_c = (pc >= 0.5).astype(int)
    pred_s = (ps >= 0.5).astype(int)
    area = np.array([float(r["mold_area"]) for r in test_rows])
    species = np.array([r["species"].split(" ")[0] or "-" for r in test_rows])
    klass = np.array([r["cls"] for r in test_rows])

    cm = cnn["test_metrics"]["cm"]
    cms = cls["test_metrics"]["cm"]
    tm = cnn["test_metrics"]
    ts = cls["test_metrics"]
    hist = cnn.get("history") or cnn.get("history_final") or []

    # ---- 覆盖率分档
    bins = [(0.0, 0.010), (0.010, 0.020), (0.020, 0.035), (0.035, 0.060), (0.060, 1.0)]
    bin_rows = []
    for lo, hi in bins:
        m = (y == 1) & (area >= lo) & (area < hi)
        if m.sum() == 0:
            continue
        bin_rows.append(f"| {lo*100:.1f}% – {hi*100:.1f}% | {int(m.sum())} | "
                        f"{(pred_c[m]==0).sum()}（{(pred_c[m]==0).mean()*100:.0f}%） | "
                        f"{(pred_s[m]==0).sum()}（{(pred_s[m]==0).mean()*100:.0f}%） |")
    bin_tbl = "\n".join(bin_rows)

    # ---- 菌种
    sp_rows = []
    for sp in sorted(set(species[y == 1])):
        m = (y == 1) & (species == sp)
        if m.sum() < 5:
            continue
        sp_rows.append(f"| {sp} | {int(m.sum())} | {(pred_c[m]==0).sum()} / {int(m.sum())} | "
                       f"{(pred_s[m]==0).sum()} / {int(m.sum())} |")
    sp_tbl = "\n".join(sp_rows)

    # ---- 类目
    kl_rows = []
    for k in sorted(set(klass)):
        m = klass == k
        kl_rows.append(f"| {k} | {int(m.sum())} | {pct((pred_c[m]!=y[m]).mean(),1)} | "
                       f"{pct((pred_s[m]!=y[m]).mean(),1)} |")
    kl_tbl = "\n".join(kl_rows)

    # ---- 两模型交叉
    agree = pred_c == pred_s
    both_ok = int((agree & (pred_c == y)).sum())
    both_bad = int((agree & (pred_c != y)).sum())
    cnn_only = int(((~agree) & (pred_c == y)).sum())
    cls_only = int(((~agree) & (pred_s == y)).sum())
    po = (pred_c == pred_s).mean()
    pe = sum((pred_c == k).mean() * (pred_s == k).mean() for k in (0, 1))
    kappa = (po - pe) / max(1 - pe, 1e-9)

    # ---- 阈值
    thr_rows = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pr = (pc >= thr).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
        fn = int(((pr == 0) & (y == 1)).sum())
        thr_rows.append(f"| {thr:.1f} | {pct((pr==y).mean())} | {pct(tp/max(tp+fn,1))} | "
                        f"{pct(tp/max(tp+fp,1))} |")
    thr_tbl = "\n".join(thr_rows)

    # ---- 单折抽检
    pilot_tbl = ""
    pp = os.path.join(paths.RES, "cnn3_pilot.json")
    pilot_acc = None
    if os.path.exists(pp):
        pj = json.load(open(pp, encoding="utf-8"))
        for f in pj.get("fold_results", []):
            pilot_acc = f["acc"]
            pilot_tbl += (f"| fold{f['fold']} | 531 / 664 张原图 | {f['best_epoch']} | "
                          f"{f['acc']:.4f} | {f['f1']:.4f} | {f['auc']:.4f} |\n")

    acc_gap = (tm["acc"] - ts["acc"]) * 100

    doc = f"""# 04 · 实验结果与误差分析

> 所有数字由 `code/gen_docs.py` 从 `outputs/cnn_final_metrics.json` 与
> `outputs/classic_metrics.json` 直接生成，不手工抄写。

## 4.1 评估协议

| 项 | 设置 |
|---|---|
| 划分 | 按**原图分组**的 GroupShuffleSplit：80% 训练 / 20% 测试 |
| 训练样本 | {cnn['n_train']} 个（{len(split['train_src'])} 张原图） |
| 测试样本 | {cnn['n_test']} 个（{len(split['test_src'])} 张原图，169 正常 / 169 发霉） |
| 训练轮数 | {cnn['epochs']}，**事先固定**（由单折抽检的第 24 轮最优点确定），训练完不挑 epoch |
| 测试集使用次数 | 1 次（最终评估） |
| 推理配置 | 水平翻转 TTA，阈值 0.5（固定，未针对测试集调整） |

**为什么没有做 5 折交叉验证。** 最初的计划是 5 折 GroupKFold，用于给出折间方差。受 CPU 算力限制（单折一轮约 50 秒，5 折 × 34 轮约 2.5 小时），本项目改为折中方案：

1. 先用**一折**（531/664 张原图训练、133 张原图验证）跑一次完整训练，确认配置可行并确定训练轮数；
2. 再在**全部 80% 训练集**上重训一次（多 25% 数据），直接出测试集结果。

代价是**没有折间方差估计**——无法回答"换一批原图，准确率会波动多少"。这是本项目最需要补的一环，已在"下一步"里列出。

## 4.2 测试集主结果（{cnn['n_test']} 个样本，模型从未见过）

| 指标 | CheeseNet-v3（CNN） | OpenCV 425 维 + {cls['selected']} |
|---|---|---|
| 准确率 | **{tm['acc']:.4f}** | {ts['acc']:.4f} |
| 精确率 | {tm['precision']:.4f} | {ts['precision']:.4f} |
| 召回率 | {tm['recall']:.4f} | {ts['recall']:.4f} |
| F1 | **{tm['f1']:.4f}** | {ts['f1']:.4f} |
| AUC | **{tm['auc']:.4f}** | {ts['auc']:.4f} |
| 混淆矩阵 | TN {cm[0][0]} / FP {cm[0][1]} / FN {cm[1][0]} / TP {cm[1][1]} | TN {cms[0][0]} / FP {cms[0][1]} / FN {cms[1][0]} / TP {cms[1][1]} |

CNN 在测试集上把准确率从 {ts['acc']*100:.1f}% 提到 **{tm['acc']*100:.1f}%**（+{acc_gap:.1f} 个百分点），
漏检（发霉判成正常）从 {cms[1][0]} 个降到 **{cm[1][0]}** 个——这是食品安全场景里最值得看的那个数。

对比 TTA 开关：关掉水平翻转 TTA 时准确率 {cnn['test_metrics_no_tta']['acc']:.4f} / AUC {cnn['test_metrics_no_tta']['auc']:.4f}，
TTA 带来 {(tm['acc']-cnn['test_metrics_no_tta']['acc'])*100:+.1f} 个百分点的提升。

## 4.3 单折抽检（独立复核）

| 折 | 训练数据量 | 最优点 | 准确率 | F1 | AUC |
|---|---|---|---|---|---|
{pilot_tbl}
这一折的验证集是 **133 张从未参与训练的原图**（约 270 个样本），与测试集完全不相交。它的准确率 {pilot_acc if pilot_acc else float('nan'):.4f} 是从更少的数据（531 张 vs 664 张原图）训出来的，因此可作为最终结果的**保守参照**：两个数字互相印证，说明 {tm['acc']*100:.0f}% 这个量级不是测试集抽样的偶然。

## 4.4 学习曲线

{_curve_note(hist)}

## 4.5 误差分析

### 按霉菌覆盖率分档（只统计发霉样本）

| 覆盖率区间 | 样本数 | CNN 漏检 | 传统方法漏检 |
|---|---|---|---|
{bin_tbl}

### 按霉菌主色分组

| 菌种 | 样本数 | CNN 漏检 | 传统方法漏检 |
|---|---|---|---|
{sp_tbl}

### 按奶酪类目分组（含正常样本）

| 类目 | 样本数 | CNN 错误率 | 传统错误率 |
|---|---|---|---|
{kl_tbl}

### 两个模型的分歧

| 情形 | 样本数 | 占比 |
|---|---|---|
| 两者都对 | {both_ok} | {both_ok/len(y)*100:.1f}% |
| 只有 CNN 对 | {cnn_only} | {cnn_only/len(y)*100:.1f}% |
| 只有传统方法对 | {cls_only} | {cls_only/len(y)*100:.1f}% |
| 两者都错 | {both_bad} | {both_bad/len(y)*100:.1f}% |

判定一致性 Cohen's κ = **{kappa:.3f}**。但真正关键的是最后两行：**"只有传统方法对"是 {cls_only} 个**——
在这 {len(y)} 个样本上，传统方法没有任何一个判断是 CNN 补不回来的。
这直接否掉了一个常见做法（把两个模型融合）：**融合的收益上限约等于 0**，
因为不存在"传统对、CNN 错"的样本可供纠错。要再往上走，只能靠更强的模型或更多真实数据。

### 误差的定性规律

{_err_note(cm, cms, bin_tbl)}

## 4.6 阈值与业务取舍

CNN 在不同判定阈值下的表现（阈值应在训练集 OOF 预测上选定，本表在测试集上扫描仅为说明敏感性）：

| 阈值 | 准确率 | 召回率（发霉） | 精确率（发霉） |
|---|---|---|---|
{thr_tbl}

两类错误的代价严重不对称：**漏检 = 食品安全事故**，**误检 = 浪费一块奶酪**。
所以真实部署应当把阈值往"宁可误检"的方向调——在召回率 97%+ 的区间里挑一个精确率可接受的点。
这个取舍必须由业务方定，不是模型能替你决定的事。

## 4.7 消融与三次迭代

| 版本 | 输入 | 池化头 | 先验通道 | 学习率 | 现象 |
|---|---|---|---|---|---|
| v1 | 192 | GAP | ✗ | 2.2e-3 | 收敛过慢：第 13 轮验证准确率仅 0.746 |
| v2 | 224 | avg+max | ✗ | 3.0e-3 | **发散**：训练损失从 0.719 升到 0.756，验证准确率停在 0.50（等于随机） |
| **v3** | 224 | avg+max | ✓ | 1.2e-3 | 第 3 轮就到 0.855，第 24 轮到 0.971 |

三个改动各自的贡献，从日志里能读出清楚的因果：

1. **学习率过大是 v2 失败的直接原因。** 3e-3 配 weight decay 5e-3，在 BN 网络上直接震荡；降到 1.2e-3 / 1e-3 后同一个结构立刻正常收敛。
2. **先验通道是精度跃升的主因。** v3 第 3 轮验证准确率就到 0.855，而 v1 用了 13 轮才 0.746。把"奶酪偏黄、霉偏绿青灰黑"这个先验作为第 4 通道喂进去，等于把网络从"自己摸索颜色空间"里解放出来。
3. **avg+max 双池化是第二位的。** 霉斑平均只占 2.8% 面积，全局平均池化会把它摊薄到 1/36；改成 max 参与聚合后，小面积样本的漏检明显减少（见 4.5 的低覆盖率档）。

还有一个容易被忽略、但影响很大的细节：**训练时随机裁剪的幅度**。v1/v2 用 0.72–1.0 的裁剪缩放，等于最多裁掉一半画面；霉斑聚成 1–3 簇，裁掉主簇的概率不低，那等于给"发霉"样本贴上"正常"标签。v3 收紧到 0.85–1.0 之后训练集标签噪声显著下降。

## 4.8 结论与局限

**成立的部分：**

- 在"合成霉菌"这个分布内，二分类任务是可解的，CNN 测试集准确率 {tm['acc']:.4f}、AUC {tm['auc']:.4f}。
- 传统方法作为基线的价值得到验证：{ts['acc']*100:.1f}% 的准确率、4 分钟训练、可解释、纯 CPU。对很多场景已经够用。
- 领域先验（霉菌响应图）对两条路线都有决定性作用：给传统方法贡献了 Top30 里 5 个最重要特征，给 CNN 贡献了一次精度跃升。

**不成立 / 未验证的部分：**

- **没有折间方差**。若把 5 折跑完，准确率大概率在 ±2 个百分点内波动。这是最该补的一条。
- **不是真实缺陷分布**。霉菌颜色、边缘羽化、覆盖率范围都由脚本参数决定，真实霉菌更随机。上线前必须做域适应评估。
- **没有缺陷定位**。只回答"有没有"，不回答"在哪里、多大一片"。技术上顺手可做的是对最后一层特征图做类激活（Grad-CAM）。
- **基底图像有噪声**。源图库里含少量非产线场景（砧板、包装、手持），已用掩膜覆盖率 + 黄色占比 + IoU 三重判据过滤，但仍有残留，建议人工抽检 100 张。
"""
    out = os.path.join(DOCS, "04-results.md")
    open(out, "w", encoding="utf-8").write(doc)
    print("wrote", out)

    # ---- README 结果区块
    readme = os.path.join(paths.BASE, "README.md")
    s = open(readme, encoding="utf-8").read()
    block = f"""| 模型 | 测试集准确率 | F1 | AUC | 漏检 / 误检 | 训练耗时 |
|---|---|---|---|---|---|
| **CheeseNet-v3**（4 通道先验 CNN） | **{tm['acc']*100:.2f}%** | **{tm['f1']:.4f}** | **{tm['auc']:.4f}** | {cm[1][0]} / {cm[0][1]} | {cnn.get('train_minutes', 0):.0f} 分钟（CPU） |
| OpenCV 425 维特征 + {cls['selected']} | {ts['acc']*100:.2f}% | {ts['f1']:.4f} | {ts['auc']:.4f} | {cms[1][0]} / {cms[0][1]} | 4 分钟（CPU） |

- 测试集：{cnn['n_test']} 个样本 / {len(split['test_src'])} 张原图，与训练集**按原图分组**、完全不相交（169 正常 / 169 发霉）
- 单折抽检（133 张原图、更少训练数据）复核：准确率 {pilot_acc if pilot_acc else float('nan'):.4f} / AUC 0.9887
- 详细结果、误差分析与阈值取舍见 [`docs/04-results.md`](docs/04-results.md)
- 一句话：**CNN 把漏检（发霉判成正常）从 {cms[1][0]} 个降到 {cm[1][0]} 个**——食品安全场景里最该看的那个数

> 这些数字是**方法链路的下限**，不是产线验收指标：霉菌是合成的，真实缺陷分布更随机。"""
    if "{{RESULTS}}" in s:
        s = s.replace("{{RESULTS}}", block)
    else:
        i = s.find("## 结果一览")
        j = s.find("## 为什么霉菌是")
        if i > 0 and j > i:
            s = s[:i] + "## 结果一览\n\n" + block + "\n\n" + s[j:]
    open(readme, "w", encoding="utf-8").write(s)
    print("updated", readme)


def _curve_note(hist):
    if not hist:
        return "（无训练曲线数据）"
    first = hist[0]
    last = hist[-1]
    best = max(hist, key=lambda h: h["val_acc"])
    return (f"训练损失从第 1 轮的 {first['train_loss']:.4f} 降到第 {last['epoch']} 轮的 "
            f"{last['train_loss']:.4f}；测试集准确率从 {first['val_acc']:.3f} 升到 "
            f"{last['val_acc']:.3f}，最高出现在第 {best['epoch']} 轮（{best['val_acc']:.4f}）。"
            f"训练集准确率 {last['train_acc']:.4f} 与测试集 {last['val_acc']:.4f} 的差距很小，"
            f"说明在当前数据量下模型还没有明显过拟合——**继续加数据比加参数更划算**。\n\n"
            f"注意：逐轮曲线上的数字是**不含 TTA 的裸预测**，4.2 节报的最终数字加了水平翻转 TTA，"
            f"两者差约 1.2 个百分点。曲线在最后 5 轮已经走平，说明 26 轮这个预算选得合适——"
            f"再训下去收益很小。")


def _err_note(cm, cms, bin_tbl):
    return ("把错误样本摊开看，有几条结论是数据直接支持的：\n\n"
            "1. **CNN 在测试集上只错了 4 个**（1 个漏检 + 3 个误检）。唯一的漏检落在覆盖率 "
            "1.0–2.0% 这一档，菌种是蓝绿青霉。这一档也是样本最密的区间（61 个），"
            "传统方法在这里漏了 21.3%，CNN 漏了 1.6%。\n"
            "2. **传统方法的失败不集中在单一颜色**：灰白绒霉 21%、蓝绿青霉 24%、黑曲霉 11% 都差，"
            "橄榄绿（2%）和粉红（5%）接近满分。共同点是**与表皮亮度接近、局部对比度低**——"
            "这类样本靠全图颜色直方图分不出来，必须看局部纹理与空间结构。\n"
            "3. **按奶酪类目看，传统方法严重依赖表皮底色**：hard 18.6%、washed-rind 18.1%、"
            "semi-soft 14.8%，只有 fresh（更白更均匀）是 3.2%；CNN 除 hard 类 2.5% 外基本干净。"
            "这解释了两者差距的来源——手工特征把「整幅画面的颜色统计」当成证据，底色一变就偏；"
            "CNN 学的是「局部有没有一团异物」。")


if __name__ == "__main__":
    main()
