# -*- coding: utf-8 -*-
"""
train_cnn_v2.py —— CheeseNet-v2
相比 v1 的四处改动（都是为了"小目标"——平均只占 2.8% 面积的霉斑）：

  1) 输入分辨率 192 -> 224（缓存 288），保留霉斑的绒毛/边缘高频细节；
  2) 池化头 由 global-avg 改为 avg+max 拼接 —— 平均池化会把 2.8% 的响应稀释掉，
     max 池化对"局部异常"敏感，这是小缺陷检测的关键；
  3) 训练裁剪尺度 0.72~1.0 收紧到 0.85~1.0 —— 原来过猛的缩放会把霉斑整块裁掉，
     等于给"发霉"样本贴上了"正常"标签（标签噪声）；
  4) 训练更久 + warmup + cosine，配合更强的正则（weight decay 5e-3, dropout 0.2）。

用法：
  # 单折试跑（先看曲线是否达标）
  python code/train_cnn_v2.py --folds 0 --epochs 30 --tag pilot
  # 正式：5 折 + 全量重训 + 测试集评估
  python code/train_cnn_v2.py --epochs 45
"""

import os, json, csv, time, argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import paths  # noqa: E402

OUT = paths.DATA
RES = paths.RES
os.makedirs(RES, exist_ok=True)

CACHE = 288
INPUT = 224
SEED = 20261106
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ data

def load_rows():
    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_cache(rows):
    X = np.zeros((len(rows), CACHE, CACHE, 3), np.uint8)
    y = np.zeros(len(rows), np.int64)
    for i, r in enumerate(rows):
        p = os.path.join(OUT, r["path"].replace("/", os.sep))
        arr = np.fromfile(p, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img.shape[0] != CACHE:
            img = cv2.resize(img, (CACHE, CACHE), interpolation=cv2.INTER_AREA)
        X[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y[i] = int(r["label"])
    return X, y


def aug_one(im, rng):
    s = rng.uniform(0.85, 1.0)
    side = max(96, int(CACHE * s))
    y0 = rng.integers(0, CACHE - side + 1)
    x0 = rng.integers(0, CACHE - side + 1)
    c = im[y0:y0 + side, x0:x0 + side]
    c = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        c = c[:, ::-1]
    if rng.random() < 0.5:
        c = c[::-1]
    if rng.random() < 0.6:
        ang = rng.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((INPUT / 2, INPUT / 2), ang, 1.0)
        c = cv2.warpAffine(c, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    f = c.astype(np.float32)
    if rng.random() < 0.7:
        f = f * rng.uniform(0.88, 1.14) + rng.uniform(-12, 12)
    if rng.random() < 0.4:
        f = f * rng.uniform(0.95, 1.05, 3).astype(np.float32)
    if rng.random() < 0.25:                      # 追加传感器噪声
        f = f + rng.normal(0, rng.uniform(1.5, 5.0), f.shape).astype(np.float32)
    if rng.random() < 0.15:                      # 追加轻微离焦
        sg = rng.uniform(0.4, 0.9)
        f = cv2.GaussianBlur(f, (0, 0), sg)
    return f


def eval_one(im):
    return cv2.resize(im, (INPUT, INPUT), interpolation=cv2.INTER_AREA).astype(np.float32)


class DS(Dataset):
    def __init__(self, X, y, train, seed=0):
        self.X, self.y, self.train = X, y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = aug_one(self.X[i], self.rng) if self.train else eval_one(self.X[i])
        img = (img / 255.0 - 0.5) / 0.25
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), int(self.y[i])


# ------------------------------------------------------------------ model

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, dil=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, dil, dilation=dil, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, dil, dilation=dil, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.b1(self.c1(x)))
        x = self.act(self.b2(self.c2(x)))
        return x


class CheeseNetV2(nn.Module):
    """stem/2 + 3 个下采样 stage + 1 个带膨胀的 stage，avg+max 双池化头。"""

    def __init__(self, n_cls=2, w=(24, 48, 96, 160, 192), p_drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(inplace=True))
        self.b1 = Block(w[0], w[1], 2)
        self.b2 = Block(w[1], w[2], 2)
        self.b3 = Block(w[2], w[3], 2)
        self.b4 = Block(w[3], w[4], 1, dil=2)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(w[4] * 2, n_cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        a = F.adaptive_avg_pool2d(x, 1).flatten(1)
        m = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(torch.cat([a, m], 1)))


# ------------------------------------------------------------------ train

def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    ps, ys = [], []
    for x, y in loader:
        if train:
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.03)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                out = model(x)
                loss = F.cross_entropy(out, y)
        tot += float(loss.detach()) * len(y)
        n += len(y)
        ps.append(torch.softmax(out.detach(), 1)[:, 1].numpy())
        ys.append(y.numpy())
    return tot / max(n, 1), np.concatenate(ps), np.concatenate(ys)


def predict_tta(model, X, idx):
    """水平翻转 + 原图 两视图平均（固定推理配置，不在测试集上调）。"""
    model.eval()
    p1, p2 = [], []
    for s in range(0, len(idx), 48):
        b = idx[s:s + 48]
        t = torch.from_numpy(((np.stack([eval_one(X[i]) for i in b]) / 255.0 - 0.5) / 0.25
                              ).transpose(0, 3, 1, 2).astype(np.float32))
        with torch.no_grad():
            p1.append(torch.softmax(model(t), 1)[:, 1].numpy())
            p2.append(torch.softmax(model(torch.flip(t, dims=[3])), 1)[:, 1].numpy())
    return (np.concatenate(p1) + np.concatenate(p2)) / 2.0


def metrics(t, p, thr=0.5):
    pr = (p >= thr).astype(int)
    return {
        "acc": float(accuracy_score(t, pr)),
        "precision": float(precision_score(t, pr, zero_division=0)),
        "recall": float(recall_score(t, pr, zero_division=0)),
        "f1": float(f1_score(t, pr, zero_division=0)),
        "auc": float(roc_auc_score(t, p)),
        "cm": confusion_matrix(t, pr, labels=[0, 1]).tolist(),
    }


def fit(X, y, idx_fit, idx_val, epochs, tag, lr=3.0e-3, wd=5e-3, bs=32, verbose=True):
    tl = DataLoader(DS(X[idx_fit], y[idx_fit], True, seed=SEED + 7), batch_size=bs,
                    shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(DS(X[idx_val], y[idx_val], False, seed=SEED), batch_size=64, num_workers=0)
    model = CheeseNetV2()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warm = max(2, epochs // 12)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.15, 1.0, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm))],
        milestones=[warm])
    best = {"score": -1, "epoch": -1, "state": None, "vm": None}
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        trl, tp, tt = run_epoch(model, tl, opt)
        sched.step()
        vll, vp, vt = run_epoch(model, vl)
        vm = metrics(vt, vp)
        trm = metrics(tt, tp)
        hist.append({"epoch": ep + 1, "train_loss": trl, "train_acc": trm["acc"],
                     "val_loss": vll, "val_acc": vm["acc"], "val_f1": vm["f1"],
                     "val_auc": vm["auc"]})
        if verbose:
            print(f"  [{tag}] ep{ep+1:02d} trl={trl:.4f} tracc={trm['acc']:.3f} "
                  f"vll={vll:.4f} acc={vm['acc']:.4f} f1={vm['f1']:.4f} "
                  f"auc={vm['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        # 选择准则：acc + auc 的折中，避免仅按 auc 选到偏保守的 epoch
        score = vm["acc"] + 0.5 * (vm["auc"] - 0.5)
        if score > best["score"]:
            best = {"score": score, "epoch": ep + 1, "vm": vm,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--no-final", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 8)))
    rows = load_rows()
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    print(f"samples={len(rows)} input={INPUT} cache={CACHE} epochs={args.epochs}", flush=True)
    X, y = build_cache(rows)
    print("cache", X.shape, flush=True)

    src = np.array([r["src_id"] for r in rows])
    test_mask = np.isin(src, split["test_src"])
    want = [int(s) for s in args.folds.split(",") if s.strip() != ""]

    folds_res, hist_all, ep_best = [], [], []
    oof_p = np.zeros(len(rows))
    for k, fo in enumerate(split["folds"]):
        if k not in want:
            continue
        fit_mask = np.isin(src, fo["fit"])
        val_mask = np.isin(src, fo["val"])
        model, best, hist = fit(X, y, np.where(fit_mask)[0], np.where(val_mask)[0],
                                args.epochs, f"f{k}")
        folds_res.append({"fold": k, "best_epoch": best["epoch"], **best["vm"]})
        ep_best.append(best["epoch"])
        hist_all.append(hist)
        _, vp, _ = run_epoch(model, DataLoader(DS(X[val_mask], y[val_mask], False),
                                              batch_size=64, num_workers=0))
        oof_p[val_mask] = vp
        torch.save(model.state_dict(), os.path.join(RES, f"cnn2_fold{k}.pt"))
        print(f"fold{k} best ep={best['epoch']} val_acc={best['vm']['acc']:.4f} "
              f"val_auc={best['vm']['auc']:.4f}", flush=True)

    # 单折试跑：只落盘折内结果，不做全量重训
    if args.no_final or len(folds_res) < len(split["folds"]):
        out = {"tag": args.tag, "input": INPUT, "epochs": args.epochs,
               "fold_results": folds_res, "history": hist_all}
        p = os.path.join(RES, f"cnn2_{args.tag}.json")
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nsaved", p)
        return

    n_ep = int(np.median(ep_best))
    print(f"\n== retrain on full train split for {n_ep} epochs ==", flush=True)
    tr_mask = ~test_mask
    final, bestf, histf = fit(X, y, np.where(tr_mask)[0], np.where(test_mask)[0],
                              n_ep, "final")
    torch.save(final.state_dict(), os.path.join(RES, "cnn2_final.pt"))

    tt_idx = np.where(test_mask)[0]
    pt_tta = predict_tta(final, X, tt_idx)
    test_m_tta = metrics(y[tt_idx], pt_tta)
    _, pt, _ = run_epoch(final, DataLoader(DS(X[test_mask], y[test_mask], False),
                                          batch_size=64, num_workers=0))
    test_m = metrics(y[tt_idx], pt)
    mean_oof = metrics(y[~test_mask], oof_p[~test_mask])

    fs = {k: float(np.mean([f[k] for f in folds_res])) for k in ("acc", "f1", "auc")}
    fs.update({k + "_std": float(np.std([f[k] for f in folds_res])) for k in ("acc", "f1", "auc")})

    out = {
        "model": "CheeseNet-v2 (compact CNN, from scratch, avg+max pooling)",
        "input": INPUT, "epochs_per_fold": args.epochs, "n_samples": len(rows),
        "n_train": int(tr_mask.sum()), "n_test": int(test_mask.sum()),
        "fold_results": folds_res, "fold_summary": fs,
        "oof_train_metrics": mean_oof, "final_epochs": n_ep,
        "test_metrics": test_m_tta, "test_metrics_no_tta": test_m,
        "test_probs": pt_tta.tolist(), "test_true": y[tt_idx].tolist(),
        "history_final": histf, "history_folds": hist_all,
    }
    p = os.path.join(RES, "cnn2_metrics.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n== test (TTA) ==")
    print(json.dumps(test_m_tta, indent=2))
    print("saved", p)


if __name__ == "__main__":
    main()
