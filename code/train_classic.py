# -*- coding: utf-8 -*-
"""
train_classic.py —— 传统 OpenCV 手工特征 + 机器学习分类器
特征族：
  A. 颜色统计（BGR / Lab / HSV 各通道一阶、二阶矩与分位数）
  B. 颜色直方图（HSV 联合、Lab ab 二维）
  C. 霉菌响应图（针对"奶酪偏黄、霉菌偏绿/青/灰黑"的先验构造指数）+ 斑块形态学
  D. LBP 纹理直方图
  E. GLCM 灰度共生矩阵（4 方向 × 6 统计量）
  F. Gabor 滤波器组响应（4 方向 × 3 波长）
  G. 梯度/边缘密度
分类器：在 5 折分组 CV 上比较 SVM(RBF) / RandomForest / ExtraTrees / LogisticRegression，
取最优者在全量训练集重训，最后评估 20% 测试集。
"""

import os, json, csv, time
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import paths  # noqa: E402

OUT = paths.DATA
RES = paths.RES
FIG = paths.FIG
os.makedirs(RES, exist_ok=True)
SEED = 20261106
SIZE = 288          # 特征提取统一尺寸（下采样加速）


def imread(rel):
    p = os.path.join(OUT, rel.replace("/", os.sep))
    img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 特征块

def color_stats(bgr, lab, hsv):
    f = []
    for ch in (bgr, lab, hsv):
        for c in range(3):
            x = ch[:, :, c].astype(np.float32).ravel()
            m, s = x.mean(), x.std()
            f += [m, s, float(np.mean(((x - m) / (s + 1e-6)) ** 3))]
            f += list(np.percentile(x, [5, 25, 50, 75, 95]))
    return f


def color_hist(bgr, lab, hsv):
    f = []
    h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).ravel()
    h /= (h.sum() + 1e-6)
    f += list(h)
    s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).ravel(); s /= (s.sum() + 1e-6)
    v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).ravel(); v /= (v.sum() + 1e-6)
    f += list(s) + list(v)
    hs = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    hs /= (hs.sum() + 1e-6)
    f += list(hs)
    ab = cv2.calcHist([lab], [1, 2], None, [8, 8], [0, 256, 0, 256]).ravel()
    ab /= (ab.sum() + 1e-6)
    f += list(ab)
    return f


def mold_map(lab, hsv):
    """霉菌响应：奶酪表皮 b* 高（偏黄）、S 中等；霉菌 b* 低（偏绿/青）、S 低-中。"""
    b = lab[:, :, 2].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    med_b = float(np.median(b))
    med_s = float(np.median(s))
    d_b = np.clip((med_b - b) / 32.0, 0, 1)              # 比整体更"不黄"
    d_s = np.clip(1.0 - s / max(med_s + 25.0, 30.0), 0, 1)  # 更"不饱和"
    dark = np.clip((np.median(lab[:, :, 0]) - lab[:, :, 0].astype(np.float32)) / 55.0, 0, 1)
    m = np.clip(0.55 * d_b + 0.30 * d_s + 0.40 * dark, 0, 1)
    return m, d_b, d_s, dark, med_b, med_s, float(a.mean())


def blob_features(m):
    f = []
    for thr in (0.30, 0.45, 0.60):
        bw = (m >= thr).astype(np.uint8)
        area_frac = float(bw.mean())
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(bw2, 8)
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
        f += [area_frac, float(bw2.mean()), float(n - 1),
              float(areas.max()) / (SIZE * SIZE), float(np.sort(areas)[-5:].mean()) / (SIZE * SIZE)
              if len(areas) else 0.0]
    f += [float(m.mean()), float(m.std()), float(np.percentile(m, 90)),
          float(np.percentile(m, 99)), float((m > 0.5).mean())]
    return f


def lbp_hist(gray):
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, np.uint8)
    offs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for k, (dy, dx) in enumerate(offs):
        nb = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << k)
    h = np.bincount((code >> 2).ravel(), minlength=64).astype(np.float32)  # 64 bin
    h /= (h.sum() + 1e-6)
    return list(h)


def glcm_feats(gray, L=8):
    q = (gray.astype(np.int32) * L // 256)
    q = np.clip(q, 0, L - 1)
    f = []
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = q[max(0, dy):q.shape[0] + min(0, dy), max(0, dx):q.shape[1] + min(0, dx)]
        b = q[max(0, -dy):q.shape[0] + min(0, -dy), max(0, -dx):q.shape[1] + min(0, -dx)]
        h = np.bincount((a.ravel() * L + b.ravel()), minlength=L * L).astype(np.float64).reshape(L, L)
        h /= (h.sum() + 1e-8)
        mu_i = (h * ii).sum(); mu_j = (h * jj).sum()
        si = np.sqrt((h * (ii - mu_i) ** 2).sum()); sj = np.sqrt((h * (jj - mu_j) ** 2).sum())
        contrast = (h * (ii - jj) ** 2).sum()
        homogen = (h / (1.0 + (ii - jj) ** 2)).sum()
        energy = (h ** 2).sum()
        corr = (h * (ii - mu_i) * (jj - mu_j)).sum() / (si * sj + 1e-8)
        f += [contrast, homogen, energy, corr, float((np.abs(ii - jj) * h).sum()),
              float(-(h[h > 0] * np.log(h[h > 0])).sum())]
    return f


_GABOR = None


def gabor_feats(gray):
    global _GABOR
    if _GABOR is None:
        _GABOR = []
        for lam in (4.0, 8.0, 14.0):
            for th in (0, 45, 90, 135):
                k = cv2.getGaborKernel((15, 15), 3.0, np.deg2rad(th), lam, 0.5, 0,
                                       ktype=cv2.CV_32F)
                k -= k.mean()
                _GABOR.append(k)
    g = gray.astype(np.float32)
    f = []
    for k in _GABOR:
        r = np.abs(cv2.filter2D(g, cv2.CV_32F, k))
        f += [float(r.mean()), float(r.std())]
    return f


def grad_feats(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edges = cv2.Canny(gray, 60, 160)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return [float(mag.mean()), float(mag.std()), float(np.percentile(mag, 95)),
            float((edges > 0).mean()), float(lap.var())]


def extract_arr(bgr):
    """对一张 BGR ndarray 直接提特征（推理时用）。"""
    if bgr.shape[0] != SIZE or bgr.shape[1] != SIZE:
        bgr = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    f = []
    f += color_stats(bgr, lab, hsv)
    f += color_hist(bgr, lab, hsv)
    m, *_ = mold_map(lab, hsv)
    f += blob_features(m)
    f += lbp_hist(gray)
    f += glcm_feats(gray)
    f += gabor_feats(gray)
    f += grad_feats(gray)
    return np.asarray(f, np.float32)


def extract(rel):
    """按 manifest 里的相对路径提特征（训练时用）。"""
    return extract_arr(imread(rel))


def feature_blocks():
    """按实际提取器的输出维度统计各特征族长度。"""
    bgr = np.full((SIZE, SIZE, 3), 180, np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m, *_ = mold_map(lab, hsv)
    parts = [
        ("颜色统计", color_stats(bgr, lab, hsv)),
        ("颜色直方图", color_hist(bgr, lab, hsv)),
        ("霉菌响应/斑块", blob_features(m)),
        ("LBP", lbp_hist(gray)),
        ("GLCM", glcm_feats(gray)),
        ("Gabor", gabor_feats(gray)),
        ("梯度", grad_feats(gray)),
    ]
    return [(n, len(v)) for n, v in parts]


# ---------------------------------------------------------------- 主流程

def metrics(t, p, pred):
    return {
        "acc": float(accuracy_score(t, pred)),
        "precision": float(precision_score(t, pred, zero_division=0)),
        "recall": float(recall_score(t, pred, zero_division=0)),
        "f1": float(f1_score(t, pred, zero_division=0)),
        "auc": float(roc_auc_score(t, p)),
        "cm": confusion_matrix(t, pred, labels=[0, 1]).tolist(),
    }


def main():
    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    y = np.array([int(r["label"]) for r in rows])
    src = np.array([r["src_id"] for r in rows])
    test_mask = np.isin(src, split["test_src"])

    cache = os.path.join(RES, "classic_features.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        print("loaded feature cache", X.shape)
    else:
        t0 = time.time()
        X = np.zeros((len(rows), 0), np.float32)
        feats = []
        for i, r in enumerate(rows):
            feats.append(extract(r["path"]))
            if (i + 1) % 300 == 0:
                print(f"  feat {i+1}/{len(rows)}  {time.time()-t0:.0f}s")
        X = np.vstack(feats)
        np.save(cache, X)
        print("features", X.shape, f"{time.time()-t0:.0f}s")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cands = {
        "SVM-RBF": Pipeline([("sc", StandardScaler()), ("m", SVC(C=8.0, gamma="scale", class_weight="balanced"))]),
        "RandomForest": Pipeline([("m", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
        "ExtraTrees": Pipeline([("m", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=SEED))]),
        "LogReg": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(
            C=0.5, max_iter=3000, class_weight="balanced"))]),
    }

    results = {}
    best_name, best_f1 = None, -1
    for name, pipe in cands.items():
        fold_f1, fold_auc, fold_acc = [], [], []
        for fo in split["folds"]:
            fit = np.isin(src, fo["fit"])
            val = np.isin(src, fo["val"])
            pipe.fit(X[fit], y[fit])
            pred = pipe.predict(X[val])
            if hasattr(pipe, "decision_function"):
                p = pipe.decision_function(X[val])
            else:
                p = pipe.predict_proba(X[val])[:, 1]
            m = metrics(y[val], p, pred)
            fold_f1.append(m["f1"]); fold_auc.append(m["auc"]); fold_acc.append(m["acc"])
        results[name] = {"f1": float(np.mean(fold_f1)), "auc": float(np.mean(fold_auc)),
                         "acc": float(np.mean(fold_acc)),
                         "f1_std": float(np.std(fold_f1))}
        print(f"{name:14s} CV  acc={results[name]['acc']:.4f} f1={results[name]['f1']:.4f} "
              f"auc={results[name]['auc']:.4f} ±{results[name]['f1_std']:.4f}")
        if results[name]["f1"] > best_f1:
            best_f1, best_name = results[name]["f1"], name

    print(f"\nbest = {best_name}")
    pipe = cands[best_name]
    pipe.fit(X[~test_mask], y[~test_mask])
    try:
        import joblib
        joblib.dump(pipe, os.path.join(paths.MODELS, "classic_model.joblib"))
        print("saved classic_model.joblib")
    except Exception as e:
        print("joblib save failed:", e)
    pred = pipe.predict(X[test_mask])
    p = (pipe.decision_function(X[test_mask]) if hasattr(pipe, "decision_function")
         else pipe.predict_proba(X[test_mask])[:, 1])
    test_m = metrics(y[test_mask], p, pred)

    # 特征重要性（树模型才有）
    imp = None
    try:
        est = pipe.named_steps.get("m", None)
        if est is not None and hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
    except Exception:
        pass

    out = {
        "model": f"OpenCV handcrafted features + {best_name}",
        "n_features": int(X.shape[1]),
        "feature_blocks": feature_blocks(),
        "cv_comparison": results,
        "selected": best_name,
        "n_train": int((~test_mask).sum()), "n_test": int(test_mask.sum()),
        "test_metrics": test_m,
        "test_probs": np.asarray(p, np.float64).tolist(),
        "test_true": y[test_mask].tolist(),
        "importances_top": (sorted(
            [{"i": int(i), "w": float(w)} for i, w in enumerate(imp)], key=lambda d: -d["w"])[:30]
            if imp is not None else None),
    }
    with open(os.path.join(RES, "classic_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n== classic test ==")
    print(json.dumps(test_m, indent=2))


if __name__ == "__main__":
    main()
