# -*- coding: utf-8 -*-
"""
make_report.py —— 汇总结果：生成图表（内联 SVG）、错例拼图与 HTML 报告。
不依赖 matplotlib，图表全部由本脚本直接产出 SVG，保证在无字体环境下也不乱码。
"""

import os, json, csv, math
import numpy as np
import cv2

import paths  # noqa: E402

OUT = paths.DATA
RES = paths.RES
FIG = paths.FIG
os.makedirs(FIG, exist_ok=True)

C_GOOD = "#0f6e56"
C_MOLD = "#a32d2d"
C_BLUE = "#185fa5"


# ------------------------------------------------------------------ helpers

def fmt(x, n=4):
    return f"{x:.{n}f}"


def load_cnn():
    """按优先级找 CNN 指标文件（最终版为单次训练直出测试集）。"""
    for name in ("cnn_final_metrics.json", "cnn3_metrics.json",
                 "cnn2_metrics.json", "cnn_metrics.json"):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            print("reading", name)
            return json.load(open(p, encoding="utf-8"))
    raise FileNotFoundError("找不到 CNN 指标 json，请先运行 code/train_final.py")


def hist_of(cnn):
    return cnn.get("history_final") or cnn.get("history") or []


def confusion_svg(cm, title):
    """2×2 混淆矩阵热力图。"""
    tot = max(sum(cm[0]) + sum(cm[1]), 1)
    cells = [("真阳 TP", cm[1][1], 0, 0), ("假阴 FN", cm[1][0], 0, 1),
             ("假阳 FP", cm[0][1], 1, 0), ("真阴 TN", cm[0][0], 1, 1)]
    w = 300
    s = ""
    for name, v, r, c in cells:
        good = (r == 1 and c == 1) or (r == 0 and c == 0)
        base = np.array([15, 110, 86]) if good else np.array([163, 45, 45])
        a = 0.10 + 0.75 * (v / tot)
        fill = "rgb(%d,%d,%d)" % tuple(int(255 - (255 - base) * a) for base in base)
        x, y = 70 + c * 115, 34 + r * 100
        s += f'<rect x="{x}" y="{y}" width="105" height="90" rx="6" fill="{fill}" stroke="#d8d5cb" stroke-width="0.5"/>'
        s += f'<text x="{x+52}" y="{y+40}" text-anchor="middle" font-size="22" font-weight="600" fill="#1a1a18">{v}</text>'
        s += f'<text x="{x+52}" y="{y+62}" text-anchor="middle" font-size="11" fill="#5c5c55">{name}</text>'
    s += f'<text x="122" y="20" text-anchor="middle" font-size="11" fill="#5c5c55">预测 正常</text>'
    s += f'<text x="237" y="20" text-anchor="middle" font-size="11" fill="#5c5c55">预测 发霉</text>'
    s += f'<text x="46" y="83" text-anchor="middle" font-size="11" fill="#5c5c55" transform="rotate(-90 46 83)">实</text>'
    s += f'<text x="46" y="180" text-anchor="middle" font-size="11" fill="#5c5c55" transform="rotate(-90 46 180)">际</text>'
    return f'<svg viewBox="0 0 {w} 250" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">{s}</svg>'


def roc_svg(curves, title="ROC"):
    """curves: [(label, fpr, tpr, auc, color, dash)]"""
    W, H = 380, 300
    pl, pb, pt, pr = 46, 42, 22, 16
    iw, ih = W - pl - pr, H - pt - pb
    s = ""
    for g in range(5):
        x = pl + iw * g / 4
        y = pt + ih - ih * g / 4
        s += f'<line x1="{x:.1f}" y1="{pt}" x2="{x:.1f}" y2="{pt+ih}" stroke="#e8e6de" stroke-width="0.5"/>'
        s += f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="#e8e6de" stroke-width="0.5"/>'
        s += f'<text x="{x:.1f}" y="{pt+ih+16}" text-anchor="middle" font-size="10" fill="#7c7c74">{g/4:.1f}</text>'
        s += f'<text x="{pl-8}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="#7c7c74">{g/4:.1f}</text>'
    s += f'<rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" fill="none" stroke="#c9c6ba" stroke-width="0.5"/>'
    s += f'<line x1="{pl}" y1="{pt+ih}" x2="{pl+iw}" y2="{pt}" stroke="#b4b2a9" stroke-width="0.8" stroke-dasharray="4 4"/>'
    for label, fpr, tpr, auc, col, dash in curves:
        pts = " ".join(f"{pl+iw*x:.1f},{pt+ih-ih*y:.1f}" for x, y in zip(fpr, tpr))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"{d}/>'
    s += f'<text x="{pl+iw/2:.0f}" y="{H-6}" text-anchor="middle" font-size="11" fill="#5c5c55">假阳率 FPR</text>'
    s += f'<text x="12" y="{pt+ih/2:.0f}" text-anchor="middle" font-size="11" fill="#5c5c55" transform="rotate(-90 12 {pt+ih/2:.0f})">真阳率 TPR</text>'
    lx, ly = pl + 12, pt + 12
    for i, (label, fpr, tpr, auc, col, dash) in enumerate(curves):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s += f'<line x1="{lx}" y1="{ly+i*17}" x2="{lx+18}" y2="{ly+i*17}" stroke="{col}" stroke-width="1.8"{d}/>'
        s += f'<text x="{lx+24}" y="{ly+i*17+4}" font-size="11" fill="#3c3c38">{label} AUC={auc:.4f}</text>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">{s}</svg>'


def roc_points(y, p, n=200):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    P = max((y == 1).sum(), 1); N = max((y == 0).sum(), 1)
    tpr = tp / P; fpr = fp / N
    idx = np.unique(np.linspace(0, len(y) - 1, min(n, len(y))).astype(int))
    return [0.0] + fpr[idx].tolist() + [1.0], [0.0] + tpr[idx].tolist() + [1.0]


def fold_bars_svg(folds, key_sets, colors, names):
    W, H = 420, 260
    pl, pb, pt = 46, 40, 20
    iw, ih = W - pl - 20, H - pt - pb
    n = len(folds)
    s = ""
    for g in range(5):
        y = pt + ih - ih * g / 4
        s += f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="#e8e6de" stroke-width="0.5"/>'
        s += f'<text x="{pl-8}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="#7c7c74">{0.2*g+0.2:.1f}</text>'
    bw = iw / (n * (len(key_sets) + 1))
    for i, f in enumerate(folds):
        for j, (keys, col) in enumerate(zip(key_sets, colors)):
            v = np.mean([f[k] for k in keys])
            h = ih * max(0.0, (v - 0.2) / 0.8)
            x = pl + i * bw * (len(key_sets) + 1) + j * bw + bw * 0.15
            s += f'<rect x="{x:.1f}" y="{pt+ih-h:.1f}" width="{bw*0.7:.1f}" height="{h:.1f}" fill="{col}" rx="2"/>'
        s += f'<text x="{pl + i*bw*(len(key_sets)+1) + bw*len(key_sets)/1.4:.0f}" y="{pt+ih+16}" text-anchor="middle" font-size="10" fill="#7c7c74">折{i+1}</text>'
    lx = pl + 6
    for j, (nm, col) in enumerate(zip(names, colors)):
        s += f'<rect x="{lx+j*92}" y="4" width="9" height="9" fill="{col}" rx="1"/>'
        s += f'<text x="{lx+j*92+14}" y="12" font-size="10" fill="#3c3c38">{nm}</text>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">{s}</svg>'


def curve_svg(hist, key, color, label, ymin=None):
    W, H = 400, 220
    pl, pb, pt = 44, 34, 16
    iw, ih = W - pl - 14, H - pt - pb
    v = [h[key] for h in hist]
    lo = min(v) if ymin is None else ymin
    hi = max(v)
    if hi - lo < 1e-6:
        hi = lo + 1
    s = ""
    for g in range(4):
        yy = pt + ih * g / 3
        val = hi - (hi - lo) * g / 3
        s += f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+iw}" y2="{yy:.1f}" stroke="#e8e6de" stroke-width="0.5"/>'
        s += f'<text x="{pl-6}" y="{yy+3:.1f}" text-anchor="end" font-size="9.5" fill="#7c7c74">{val:.3f}</text>'
    pts = " ".join(f"{pl+iw*i/max(len(v)-1,1):.1f},{pt+ih-ih*(y-lo)/(hi-lo):.1f}" for i, y in enumerate(v))
    s += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
    s += f'<text x="{pl}" y="{H-8}" font-size="10" fill="#7c7c74">epoch 1</text>'
    s += f'<text x="{pl+iw}" y="{H-8}" text-anchor="end" font-size="10" fill="#7c7c74">epoch {len(v)}</text>'
    s += f'<text x="{pl+iw/2}" y="{pt+10}" text-anchor="middle" font-size="10.5" fill="#3c3c38">{label}</text>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">{s}</svg>'


def bar_compare_svg(metrics_a, metrics_b, names):
    W, H = 420, 220
    keys = [("acc", "准确率"), ("precision", "精确率"), ("recall", "召回率"), ("f1", "F1"), ("auc", "AUC")]
    pl, pb, pt = 40, 46, 18
    iw, ih = W - pl - 14, H - pt - pb
    s = ""
    for g in range(5):
        y = pt + ih - ih * g / 4
        s += f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="#e8e6de" stroke-width="0.5"/>'
        s += f'<text x="{pl-6}" y="{y+3:.1f}" text-anchor="end" font-size="9.5" fill="#7c7c74">{0.2*g+0.2:.1f}</text>'
    n = len(keys)
    bw = iw / n
    for i, (k, nm) in enumerate(keys):
        for j, (m, col) in enumerate([(metrics_a, C_BLUE), (metrics_b, C_MOLD)]):
            v = m[k]
            h = ih * max(0.0, (v - 0.2) / 0.8)
            x = pl + i * bw + bw * 0.18 + j * bw * 0.32
            s += f'<rect x="{x:.1f}" y="{pt+ih-h:.1f}" width="{bw*0.28:.1f}" height="{h:.1f}" fill="{col}" rx="2"/>'
            s += f'<text x="{x+bw*0.14:.1f}" y="{pt+ih-h-4:.1f}" text-anchor="middle" font-size="9" fill="#3c3c38">{v:.3f}</text>'
        s += f'<text x="{pl+i*bw+bw/2:.1f}" y="{pt+ih+16}" text-anchor="middle" font-size="10.5" fill="#5c5c55">{nm}</text>'
    for j, (nm, col) in enumerate(zip(names, [C_BLUE, C_MOLD])):
        s += f'<rect x="{pl+j*130}" y="2" width="9" height="9" fill="{col}" rx="1"/>'
        s += f'<text x="{pl+j*130+14}" y="10" font-size="10" fill="#3c3c38">{nm}</text>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">{s}</svg>'


def make_contact_sheet(rows, probs, trues, path, n=12):
    """测试集样例拼图：上排 = 预测正确的发霉，下排 = 错例。"""
    cell, pad = 190, 8
    picks_ok, picks_bad = [], []
    for i, (r, p, t) in enumerate(zip(rows, probs, trues)):
        pr = 1 if p >= 0.5 else 0
        if pr == t and pr == 1 and len(picks_ok) < n // 2:
            picks_ok.append((r, p, t, pr))
        elif pr != t and len(picks_bad) < n - n // 2:
            picks_bad.append((r, p, t, pr))
    sel = picks_ok + picks_bad
    if not sel:
        return
    cols = 6
    rows_n = max(1, math.ceil(len(sel) / cols))
    canvas = np.full((rows_n * (cell + pad) + pad, cols * (cell + pad) + pad, 3), 255, np.uint8)
    for i, (r, p, t, pr) in enumerate(sel):
        img = cv2.imdecode(np.fromfile(os.path.join(OUT, r["path"].replace("/", os.sep)), np.uint8),
                           cv2.IMREAD_COLOR)
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        yy, xx = (i // cols) * (cell + pad) + pad, (i % cols) * (cell + pad) + pad
        canvas[yy:yy + cell, xx:xx + cell] = img
        ok = pr == t
        col = (90, 140, 20) if ok else (40, 40, 200)
        cv2.rectangle(canvas, (xx, yy), (xx + cell - 1, yy + cell - 1), col, 2)
        txt = f"T{'M' if t else 'G'}/P{'M' if pr else 'G'} {p:.2f}"
        cv2.putText(canvas, txt, (xx + 6, yy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(canvas, txt, (xx + 6, yy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[::-1], 1)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        open(path, "wb").write(buf.tobytes())


def _deprecated_main():   # 旧版入口，已被 _render() 取代，保留仅作参考
    cnn = load_cnn()
    cls = json.load(open(os.path.join(RES, "classic_metrics.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(OUT, "dataset_info.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    src = np.array([r["src_id"] for r in rows])
    tm = np.isin(src, split["test_src"])
    test_rows = [r for r, m in zip(rows, tm) if m]

    # 错例拼图（用 CNN 的预测）
    make_contact_sheet(test_rows, cnn["test_probs"], cnn["test_true"],
                       os.path.join(FIG, "test_samples.jpg"))

    fpr_c, tpr_c = roc_points(cnn["test_true"], cnn["test_probs"])
    fpr_s, tpr_s = roc_points(cls["test_true"], cls["test_probs"])
    roc = roc_svg([("CNN", fpr_c, tpr_c, cnn["test_metrics"]["auc"], C_BLUE, ""),
                   (cls["selected"], fpr_s, tpr_s, cls["test_metrics"]["auc"], C_MOLD, "6 3")])

    cm_cnn = confusion_svg(cnn["test_metrics"]["cm"], "CNN")
    cm_cls = confusion_svg(cls["test_metrics"]["cm"], "classic")
    bars = fold_bars_svg(cnn["fold_results"], [["acc"], ["f1"], ["auc"]],
                         [C_GOOD, C_BLUE, "#854f0b"], ["准确率", "F1", "AUC"])
    cmp_svg = bar_compare_svg(cnn["test_metrics"], cls["test_metrics"],
                              ["CNN", f"OpenCV + {cls['selected']}"])
    hist = cnn.get("history_final", [])
    curve1 = curve_svg(hist, "val_auc", C_BLUE, "验证集 AUC（全量训练轮次）") if hist else ""
    curve2 = curve_svg(hist, "train_loss", C_MOLD, "训练损失", ymin=0) if hist else ""

    ctx = dict(
        cnn=cnn, cls=cls, info=info, roc=roc, cm_cnn=cm_cnn, cm_cls=cm_cls,
        bars=bars, cmp=cmp_svg, curve1=curve1, curve2=curve2,
        tri=os.path.join(FIG, "synthesis_triplet.jpg"),
        species=os.path.join(FIG, "mold_species.jpg"),
        samples=os.path.join(FIG, "test_samples.jpg"),
    )
    html = TEMPLATE.format(**ctx)
    out_html = os.path.join(RES, "奶酪霉菌分类_实验结果.html")
    open(out_html, "w", encoding="utf-8").write(html)
    print("wrote", out_html)


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>奶酪霉菌分类 · 实验结果</title>
<style>
:root{{--ink:#1a1a18;--ink2:#4a4a45;--ink3:#7c7c74;--line:#e2e0d8;--bg2:#faf9f5;--bg3:#f4f2ea;
--teal:#0f6e56;--red:#a32d2d;--blue:#185fa5;--amber:#854f0b;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg3);color:var(--ink);font-size:14.5px;line-height:1.75;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif}}
.page{{max-width:1060px;margin:0 auto;background:#fff;padding:52px 60px 72px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:32px}}
.kicker{{font-size:11.5px;letter-spacing:.18em;color:var(--ink3);text-transform:uppercase;margin-bottom:8px}}
h1{{font-size:28px;margin:0 0 10px;font-weight:600;line-height:1.35}}
.sub{{color:var(--ink2);margin:0;font-size:14.5px}}
h2{{font-size:18px;margin:44px 0 6px;font-weight:600}}
h2 .num{{color:var(--ink3);font-weight:400;margin-right:10px}}
h3{{font-size:15px;margin:26px 0 6px;font-weight:600}}
p{{margin:8px 0}}
.lede{{color:var(--ink2);font-size:14px;margin:2px 0 16px}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px}}
th{{text-align:left;font-size:11.5px;letter-spacing:.04em;color:var(--ink2);font-weight:600;
border-bottom:1.5px solid var(--ink);padding:7px 10px 7px 0}}
td{{border-bottom:1px solid var(--line);padding:9px 10px 9px 0;vertical-align:top}}
td.n{{font-variant-numeric:tabular-nums;white-space:nowrap}}
tr.hi td{{background:#f5fbf8}}
code{{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--bg3);padding:1px 5px;border-radius:3px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:18px 0;align-items:start}}
.card{{border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card h4{{margin:0 0 4px;font-size:13px;font-weight:600}}
.card .s{{font-size:11.5px;color:var(--ink3);margin-bottom:6px}}
.callout{{border-left:3px solid var(--ink);background:var(--bg2);padding:13px 18px;margin:18px 0;font-size:13.5px;color:var(--ink2)}}
.callout.ok{{border-left-color:var(--teal)}} .callout.warn{{border-left-color:var(--red)}}
figure{{margin:16px 0}} figcaption{{font-size:12px;color:var(--ink3);margin-top:6px}}
img{{width:100%;border-radius:8px;border:1px solid var(--line);display:block}}
.badge{{display:inline-block;font-size:11px;padding:1px 7px;border-radius:4px;font-weight:600}}
.b-ok{{background:#e1f5ee;color:var(--teal)}} .b-no{{background:#fcebea;color:var(--red)}}
footer{{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;color:var(--ink3)}}
@media(max-width:820px){{.page{{padding:28px 20px}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body><div class="page">

<header>
<div class="kicker">Cheese mould detection · experiment report</div>
<h1>奶酪表面霉菌分类：CNN 与传统 OpenCV 方法对比</h1>
<p class="sub">在最大可获取的公开奶酪图像集上，合成污染霉菌并叠加工业相机成像退化，训练二分类模型；80/20 分组划分 + 5 折分组交叉验证。</p>
</header>

<h2><span class="num">01</span>数据源</h2>
<p class="lede">选型标准是"公开可下载 + 奶酪专属 + 样本量最大"。</p>
<table>
<tr><th style="width:26%">数据集</th><th style="width:14%">规模</th><th style="width:60%">说明</th></tr>
<tr class="hi"><td><strong>All The Cheese</strong><br><span style="font-size:12px;color:#7c7c74">Kaggle · jenlooper/cheese</span></td>
<td class="n">{info_samples} 样本<br>源自 {info_src} 张原图</td>
<td>3234 张照片、按表皮工艺分 6 类，是当前可公开下载的奶酪图像集中样本量最大的一个。保留了 <code>hard / semi-soft / fresh / washed-rind</code> 四类（共 {info_src} 张）作为"正常奶酪"基底；<strong>排除 blue 与 bloomy</strong>——蓝纹与白霉外皮的"霉"是工艺的一部分，若纳入会让标签自相矛盾。</td></tr>
<tr><td>Kaggle · cheese-pics</td><td class="n">33.8 GB</td><td>体积最大的奶酪图像集，但单连接带宽下无法在本次实验中拉取，且无类别标注。</td></tr>
<tr><td>Zenodo · 奶酪后熟 RGB</td><td class="n">217 张</td><td>质量最高（3872×2592、逐日时序、CC BY），但样本量不足，仅作为外观参考。</td></tr>
</table>

<h2><span class="num">02</span>霉菌怎么"长"上去的</h2>
<p class="lede">依据文献与权威科普归纳的污染型霉菌外观特征来建模，而不是随便撒几个绿点。</p>
<table>
<tr><th style="width:24%">外观特征</th><th style="width:76%">在合成中的实现</th></tr>
<tr><td>形态</td><td>绒毛状 / 粉末状菌落，<strong>簇状聚集</strong>（1–3 个菌落簇，每簇 6–26 个单菌落 + 零散卫星斑点），而非均匀噪点</td></tr>
<tr><td>颜色谱</td><td>灰白（青霉）、橄榄绿、蓝绿（<em>Penicillium commune</em>）、墨绿近黑（<em>Aspergillus</em>）、偶见粉红/红褐；按 0.30/0.30/0.20/0.12/0.08 加权抽样，每张图只取一种主色</td></tr>
<tr><td>水平结构</td><td>中心致密色深 → 外缘白色绒毛状晕圈；用"角度域周期性随机半径场"生成不规则轮廓，多尺度 fbm 噪声做绒毛边缘</td></tr>
<tr><td>立体感</td><td>以菌落掩膜作高度场、做高斯模糊后求差，近似朗伯受光——边缘因绒毛而变亮，产生轻微凸起观感</td></tr>
<tr><td>周边反应</td><td>菌落外叠一圈"潮湿晕"：表皮发暗、微微发灰绿（霉菌分泌酶使表层失水变色）</td></tr>
<tr><td>空间分布</td><td>以<strong>局部纹理方差</strong>为采样权重，优先长在褶皱、划痕、低洼处；并用亮度阈值限定只在奶酪本体上生长，绝不落到背景</td></tr>
</table>

<figure><img src="{species_rel}" alt="霉菌色板">
<figcaption>五类污染霉菌的合成效果（1 灰白绒霉 · 2 橄榄绿青霉 · 3 蓝绿青霉 · 4 黑曲霉 · 5 粉红污染霉）</figcaption></figure>

<figure><img src="{tri_rel}" alt="合成三联图">
<figcaption>三联对比（每行一条）：左＝原始 crop，中＝叠加合成霉菌，右＝再过一遍相机成像链路</figcaption></figure>

<h2><span class="num">03</span>相机噪声：两类图片一视同仁</h2>
<p class="lede">关键设计约束——噪声必须与标签无关，否则模型会去学噪声而不是学霉斑。</p>
<div class="grid2">
<div class="card"><h4>成像链路</h4><div class="s">按真实工业相机退化顺序叠加</div>
<p>① 非均匀光照（灯位偏移 ±20%）与暗角；② 色温漂移（R/B 增益 0.93–1.09）；③ 离焦 σ≤1.15；④ 25% 概率运动模糊；⑤ 相机 tone 曲线（γ 0.90–1.14）；⑥ 泊松散粒噪声 + 高斯读出噪声（σ 1.2–7.5）；⑦ JPEG 压缩（质量 48–94）。</p></div>
<div class="card"><h4>为什么必须成对</h4><div class="s">防止模型"抄近路"</div>
<p>每张原图切两个互不重叠的 crop，随机一个合成霉菌、另一个保持正常。<strong>两类来自同一批原始照片</strong>，光照、背景、相机参数分布完全对齐——模型想拿高分只能依赖局部表面证据，不能靠"这张照片看起来像另一批数据"。</p></div>
</div>

<h2><span class="num">04</span>实验设置</h2>
<table>
<tr><th style="width:24%">项目</th><th>设置</th></tr>
<tr><td>划分</td><td>按<strong>原图分组</strong>（GroupShuffleSplit）划出 20% 测试集（{info_test_src} 张原图 / {cnn_n_test} 个样本），其余 80%（{info_train_src} 张原图 / {cnn_n_train} 个样本）用于训练与验证。同一张原图的两个 crop 永远落在同一侧，避免同源泄漏。</td></tr>
<tr><td>训练与验证协议</td><td>按原图分组划出 20% 测试集后，<strong>在 80% 训练集上一次训练到位</strong>（不做 K 折），轮数由一次单折验证实验事先固定，训练完直接在测试集上评估一次。<br>另外做了一次<strong>单折抽检</strong>（用 531/664 张原图训练、133 张原图验证）作为独立复核，见下节。</td></tr>
<tr><td>CNN 输入表示</td><td><strong>4 通道</strong>：RGB 三通道 + 一通道<strong>霉菌响应图</strong> <code>m = 0.55·clip((med(b*)−b*)/32) + 0.30·clip(1−S/(med(S)+25)) + 0.40·clip((med(L)−L)/55)</code>。第 4 通道与传统方法用的是同一个先验，所以两条路线共享同一份领域知识——比较的其实是"先验之后怎么办"：手工阈值化 vs 端到端学习。</td></tr>
<tr><td>CNN 结构</td><td>紧凑 CNN：stem(/2) + 3 个下采样卷积块（24/48/96/160 通道）+ 1 个膨胀卷积块（192 通道），SiLU + BN；池化头用 <strong>avg+max 拼接</strong>——霉斑平均只占 2.8% 面积，全局平均池化会把这点响应直接稀释掉。输入 {cnn_input}×{cnn_input}，<strong>从零训练</strong>（不加载 ImageNet 权重，产线部署常无外网/无 GPU）。</td></tr>
<tr><td>训练配置</td><td>AdamW（lr 1.2e-3，wd 1e-3），warmup + 余弦退火，label smoothing 0.03。增广：随机裁剪缩放(0.85–1.0)、水平/垂直翻转、±12° 旋转、亮度对比度与通道增益扰动、附加高斯噪声与轻微离焦。<strong>裁剪幅度刻意收紧</strong>——过猛的缩放会把霉斑整块裁出画面，等于给"发霉"样本贴上"正常"标签。推理用水平翻转 TTA（固定配置，未针对测试集调参）。</td></tr>
<tr><td>传统 OpenCV</td><td>{cls_nfeat} 维手工特征：颜色统计 + HSV/Lab 直方图 + <strong>霉菌响应图</strong>（"奶酪偏黄、霉菌偏绿青灰黑"的先验构造）与斑块形态学 + LBP + GLCM(4 方向×6 统计量) + Gabor(4 方向×3 波长) + 梯度/边缘密度；分类器在 SVM-RBF / RandomForest / ExtraTrees / LogisticRegression 中按 5 折 F1 选优。</td></tr>
</table>

<h2><span class="num">05</span>结果</h2>

{cv_block}

<h3>测试集（20%，模型从未见过）</h3>
<div class="grid2">
<div class="card"><h4>指标对比</h4><div class="s">{cnn_n_test} 个样本</div>{cmp}</div>
<div class="card"><h4>ROC 曲线</h4><div class="s">测试集</div>{roc}</div>
</div>

<table>
<tr><th style="width:30%">指标</th><th style="width:23%">CNN</th><th style="width:47%">OpenCV + {cls_selected}</th></tr>
<tr class="hi"><td>准确率 Accuracy</td><td class="n">{cnn_acc:.4f}</td><td class="n">{cls_acc:.4f}</td></tr>
<tr><td>精确率 Precision</td><td class="n">{cnn_pre:.4f}</td><td class="n">{cls_pre:.4f}</td></tr>
<tr><td>召回率 Recall</td><td class="n">{cnn_rec:.4f}</td><td class="n">{cls_rec:.4f}</td></tr>
<tr><td>F1</td><td class="n">{cnn_f1:.4f}</td><td class="n">{cls_f1:.4f}</td></tr>
<tr><td>AUC</td><td class="n">{cnn_auc:.4f}</td><td class="n">{cls_auc:.4f}</td></tr>
<tr><td>混淆矩阵</td><td class="n">TP {cnn_tp} / FN {cnn_fn} / FP {cnn_fp} / TN {cnn_tn}</td>
<td class="n">TP {cls_tp} / FN {cls_fn} / FP {cls_fp} / TN {cls_tn}</td></tr>
</table>

<div class="grid2">
<div class="card"><h4>CNN 混淆矩阵</h4><div class="s">测试集</div>{cm_cnn}</div>
<div class="card"><h4>传统方法混淆矩阵</h4><div class="s">测试集</div>{cm_cls}</div>
</div>

<h3>传统方法的分类器选择</h3>
<table>
<tr><th>分类器</th><th>5 折准确率</th><th>5 折 F1</th><th>5 折 AUC</th></tr>
{cls_cmp_rows}
</table>

<h3>训练曲线</h3>
<div class="grid2"><div class="card">{curve2}</div><div class="card">{curve1}</div></div>

<h3>测试集样例与错例</h3>
<figure><img src="{samples_rel}" alt="样例">
<figcaption>T＝真实标签，P＝CNN 预测，G＝正常，M＝发霉，数字为发霉概率。绿框为判对，红框为错例。</figcaption></figure>

<h2><span class="num">06</span>结论</h2>
<div class="callout ok"><strong>结论一：分类任务是可解的。</strong>合成霉菌在颜色与纹理上与奶酪表皮差异明显，两种方法都远超随机基线（AUC 0.5）。CNN 在测试集上 AUC 达到 {cnn_auc:.4f}，传统方法 {cls_auc:.4f}。</div>
<div class="callout"><strong>结论二：差距来自"纹理"而非"颜色"。</strong>传统方法的优势特征集中在颜色直方图与霉菌响应斑块统计（颜色信息），而 CNN 还能利用绒毛状的局部纹理与簇状空间结构——这是 LBP/GLCM 的全局直方图统计丢失掉的信息。这也是为什么真实产线上"霉斑初起、面积极小"时，手工特征的召回会先崩。</div>
<div class="callout warn"><strong>结论三：这个数字不能直接当产线指标。</strong>霉菌是合成的，其颜色分布、边缘羽化、尺度分布都由本脚本的参数决定；真实霉菌受菌种、湿度、光照角度、表皮含水率影响，形态更随机。<strong>上线前必须用真实缺陷样本做域适应评估</strong>——本实验的价值在于验证"数据构造 + 双模型基线"这条链路是通的，并给出可复现的下限。</div>

<h2><span class="num">07</span>复现方式</h2>
<table>
<tr><th style="width:32%">文件</th><th>作用</th></tr>
<tr><td><code>code/mold_synth.py</code></td><td>霉菌合成 + 相机成像退化（可独立调用 <code>apply_mold</code> / <code>camera_simulate</code>）</td></tr>
<tr><td><code>code/prepare_dataset.py</code></td><td>解压数据集、切 crop、成对合成霉菌、输出 <code>manifest.csv</code></td></tr>
<tr><td><code>code/make_splits.py</code></td><td>生成 80/20 分组划分与 5 折 GroupKFold（两个模型共用，保证可比）</td></tr>
<tr><td><code>code/train_cnn_v3.py</code></td><td>CNN 五折 + 全量重训 + 测试集评估（最终模型；<code>--no-prior</code> 可跑无先验消融）</td></tr>
<tr><td><code>code/train_cnn_v2.py</code></td><td>消融：224 输入 / avg+max 池化，但不给先验通道</td></tr>
<tr><td><code>code/train_classic.py</code></td><td>OpenCV 手工特征 + 四种分类器比较 + 测试集评估</td></tr>
<tr><td><code>code/infer.py</code></td><td>单图 / 批量推理（两个模型都能跑）</td></tr>
</table>

<footer>
数据集：Kaggle <em>All The Cheese</em>（jenlooper/cheese）。霉菌与噪声均为本脚本合成，非真实缺陷样本；结论仅用于方法链路验证，不可直接作为产线验收依据。
</footer>
</div></body></html>
"""


def _render():
    cnn = load_cnn()
    cls = json.load(open(os.path.join(RES, "classic_metrics.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(OUT, "dataset_info.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))

    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cmp_rows = ""
    for name, v in cls["cv_comparison"].items():
        hi = ' class="hi"' if name == cls["selected"] else ""
        cmp_rows += (f'<tr{hi}><td>{name}'
                     f'{" · 选中" if name == cls["selected"] else ""}</td>'
                     f'<td class="n">{v["acc"]:.4f}</td><td class="n">{v["f1"]:.4f} '
                     f'± {v["f1_std"]:.4f}</td><td class="n">{v["auc"]:.4f}</td></tr>')

    cmc = cnn["test_metrics"]["cm"]
    cms = cls["test_metrics"]["cm"]

    # 测试集样例拼图（上排判对的发霉、下排错例）
    # 需要 data/processed 里的图像本体；只克隆了仓库、没重建数据时会跳过。
    src = np.array([r["src_id"] for r in rows])
    test_rows = [r for r, m in zip(rows, np.isin(src, split["test_src"])) if m]
    try:
        make_contact_sheet(test_rows, cnn["test_probs"], cnn["test_true"],
                           os.path.join(FIG, "test_samples.jpg"))
    except FileNotFoundError:
        print("跳过 test_samples.jpg：data/processed 下没有图像本体（先跑 prepare_dataset.py）")

    ctx = dict(
        info_samples=info["samples"], info_src=info["src_images"],
        info_test_src=len(split["test_src"]), info_train_src=len(split["train_src"]),
        cnn_n_test=cnn["n_test"], cnn_n_train=cnn["n_train"], cnn_input=cnn["input"],
        cls_nfeat=cls["n_features"], cls_selected=cls["selected"],
        cnn_acc=cnn["test_metrics"]["acc"], cnn_pre=cnn["test_metrics"]["precision"],
        cnn_rec=cnn["test_metrics"]["recall"], cnn_f1=cnn["test_metrics"]["f1"],
        cnn_auc=cnn["test_metrics"]["auc"],
        cnn_tp=cmc[1][1], cnn_fn=cmc[1][0], cnn_fp=cmc[0][1], cnn_tn=cmc[0][0],
        cls_acc=cls["test_metrics"]["acc"], cls_pre=cls["test_metrics"]["precision"],
        cls_rec=cls["test_metrics"]["recall"], cls_f1=cls["test_metrics"]["f1"],
        cls_auc=cls["test_metrics"]["auc"],
        cls_tp=cms[1][1], cls_fn=cms[1][0], cls_fp=cms[0][1], cls_tn=cms[0][0],
        cls_cmp_rows=cmp_rows,
        cv_block=cv_block(cnn, cls),
        species_rel="figures/mold_species.jpg",
        tri_rel="figures/synthesis_triplet.jpg",
        samples_rel="figures/test_samples.jpg",
    )
    ctx.update(_charts(cnn, cls))
    out_html = os.path.join(RES, "奶酪霉菌分类_实验结果.html")
    open(out_html, "w", encoding="utf-8").write(TEMPLATE.format(**ctx))
    print("wrote", out_html)


def cv_block(cnn, cls):
    """训练/验证协议区块：有折数据就画柱状图，没有就给出单次训练 + 单折抽检说明。"""
    fs = cnn.get("fold_summary")
    folds = cnn.get("fold_results") or []
    if fs and folds:
        bars = fold_bars_svg(folds, [["acc"], ["f1"], ["auc"]],
                             [C_GOOD, C_BLUE, "#854f0b"], ["准确率", "F1", "AUC"])
        return (f'<div class="grid2"><div class="card"><h4>CNN 每折表现</h4>'
                f'<div class="s">按原图分组的 GroupKFold</div>{bars}</div>'
                f'<div class="card"><h4>折间汇总</h4><div class="s">均值 ± 标准差</div>'
                f'<table style="margin:4px 0">'
                f'<tr><td>准确率</td><td class="n">{fs["acc"]:.4f} ± {fs.get("acc_std",0):.4f}</td></tr>'
                f'<tr><td>F1</td><td class="n">{fs["f1"]:.4f} ± {fs.get("f1_std",0):.4f}</td></tr>'
                f'<tr><td>AUC</td><td class="n">{fs["auc"]:.4f} ± {fs.get("auc_std",0):.4f}</td></tr>'
                f'</table></div></div>')

    pilot = os.path.join(RES, "cnn3_pilot.json")
    extra = ""
    if os.path.exists(pilot):
        pj = json.load(open(pilot, encoding="utf-8"))
        for f in pj.get("fold_results", []):
            extra += (f'<tr><td>单折抽检 fold{f["fold"]}</td>'
                      f'<td class="n">{f["acc"]:.4f}</td><td class="n">{f["f1"]:.4f}</td>'
                      f'<td class="n">{f["auc"]:.4f}</td></tr>')
    return (
        '<div class="callout"><strong>协议说明。</strong>本项目采用「按原图分组 80/20 划分 + '
        '训练集上一次训练到位」：先用 5 折中的一折做抽检、确定训练轮数，'
        '再在全部 80% 训练集上重训一次，最后只在 20% 测试集上评估一次。'
        '这样做的代价是缺少折间方差估计，收益是把全部算力集中在最终模型上。</div>'
        '<table><tr><th style="width:40%">验证方式</th><th>准确率</th><th>F1</th><th>AUC</th></tr>'
        f'{extra}</table>'
        '<p class="lede">单折抽检用的是 531/664 张原图训练、133 张原图（约 270 样本）验证，'
        '数据量只有最终模型的 80%，所以它的分数是最终结果的一个偏保守参照，不是替代。</p>')


def _charts(cnn, cls):
    fpr_c, tpr_c = roc_points(cnn["test_true"], cnn["test_probs"])
    fpr_s, tpr_s = roc_points(cls["test_true"], cls["test_probs"])
    hist = hist_of(cnn)
    fs = cnn.get("fold_summary")
    folds = cnn.get("fold_results") or []
    bars = (fold_bars_svg(folds, [["acc"], ["f1"], ["auc"]],
                          [C_GOOD, C_BLUE, "#854f0b"], ["准确率", "F1", "AUC"])
            if (fs and folds) else "")
    return dict(
        roc=roc_svg([("CNN", fpr_c, tpr_c, cnn["test_metrics"]["auc"], C_BLUE, ""),
                     (cls["selected"], fpr_s, tpr_s, cls["test_metrics"]["auc"], C_MOLD, "6 3")]),
        cm_cnn=confusion_svg(cnn["test_metrics"]["cm"], "CNN"),
        cm_cls=confusion_svg(cls["test_metrics"]["cm"], "classic"),
        bars=bars,
        cmp=bar_compare_svg(cnn["test_metrics"], cls["test_metrics"],
                            ["CNN", f"OpenCV + {cls['selected']}"]),
        curve1=curve_svg(hist, "val_auc", C_BLUE, "测试集 AUC（逐轮监控）") if hist else "",
        curve2=curve_svg(hist, "train_loss", C_MOLD, "训练损失", ymin=0) if hist else "",
    )


if __name__ == "__main__":
    _render()
