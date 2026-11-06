# -*- coding: utf-8 -*-
"""
train_cnn.py —— 卷积网络（PyTorch，从零训练的紧凑 CNN）
在 80% 训练集上做 5 折分组交叉验证，再用全量训练集重训，
最后在 20% 独立测试集上评估。
"""

import os, json, csv, csv as _csv, time, argparse
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
FIG = paths.FIG
FIG = os.path.join(RES, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CACHE = 208
INPUT = 192
SEED = 20261106
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ data

def load_rows():
    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_img(rel):
    p = os.path.join(OUT, rel.replace("/", os.sep))
    arr = np.fromfile(p, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (CACHE, CACHE), interpolation=cv2.INTER_AREA)


def build_cache(rows):
    """一次性把全部样本解码进内存（uint8），后续 epoch 只做几何/颜色增广。"""
    X = np.zeros((len(rows), CACHE, CACHE, 3), np.uint8)
    y = np.zeros(len(rows), np.int64)
    for i, r in enumerate(rows):
        X[i] = cv2.cvtColor(read_img(r["path"]), cv2.COLOR_BGR2RGB)
        y[i] = int(r["label"])
    return X, y


def aug_batch(imgs, rng):
    """numpy 级增广：随机裁剪缩放→翻转→旋转→亮度对比度扰动。"""
    out = np.empty((len(imgs), INPUT, INPUT, 3), np.float32)
    for i, im in enumerate(imgs):
        s = rng.uniform(0.72, 1.0)
        side = max(48, int(CACHE * s))
        y0 = rng.integers(0, CACHE - side + 1)
        x0 = rng.integers(0, CACHE - side + 1)
        c = im[y0:y0 + side, x0:x0 + side]
        c = cv2.resize(c, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            c = c[:, ::-1]
        if rng.random() < 0.5:
            c = c[::-1]
        if rng.random() < 0.6:
            ang = rng.uniform(-14, 14)
            M = cv2.getRotationMatrix2D((INPUT / 2, INPUT / 2), ang, 1.0)
            c = cv2.warpAffine(c, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        f = c.astype(np.float32)
        if rng.random() < 0.7:
            f = f * rng.uniform(0.85, 1.18) + rng.uniform(-16, 16)
        if rng.random() < 0.4:                      # 通道增益（模拟白平衡漂移）
            f = f * rng.uniform(0.94, 1.06, 3).astype(np.float32)
        out[i] = f
    return out


def eval_batch(imgs):
    out = np.empty((len(imgs), INPUT, INPUT, 3), np.float32)
    for i, im in enumerate(imgs):
        out[i] = cv2.resize(im, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    return out


class DS(Dataset):
    def __init__(self, X, y, train, seed=0):
        self.X, self.y, self.train = X, y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = aug_batch(self.X[i:i + 1], self.rng)[0] if self.train else eval_batch(self.X[i:i + 1])[0]
        img = (img / 255.0 - 0.5) / 0.25
        return torch.from_numpy(img.transpose(2, 0, 1)), int(self.y[i])


# ------------------------------------------------------------------ model

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.b1(self.c1(x)))
        x = self.act(self.b2(self.c2(x)))
        return x


class CheeseNet(nn.Module):
    def __init__(self, n_cls=2, w=(24, 48, 96, 160)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(inplace=True))
        self.b1 = Block(w[0], w[1], 2)
        self.b2 = Block(w[1], w[2], 2)
        self.b3 = Block(w[2], w[3], 2)
        self.drop = nn.Dropout(0.15)
        self.fc = nn.Linear(w[3], n_cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


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
            loss = F.cross_entropy(out, y, label_smoothing=0.05)
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
    p = np.concatenate(ps)
    t = np.concatenate(ys)
    return tot / max(n, 1), p, t


def predict_tta(model, X, idx):
    """水平翻转 TTA —— 作为固定的推理配置，不针对测试集调参。"""
    model.eval()
    p1_list, p2_list = [], []
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        imgs = eval_batch(X[b])
        t = torch.from_numpy(((imgs / 255.0 - 0.5) / 0.25).transpose(0, 3, 1, 2))
        with torch.no_grad():
            p1_list.append(torch.softmax(model(t), 1)[:, 1].numpy())
            p2_list.append(torch.softmax(model(torch.flip(t, dims=[3])), 1)[:, 1].numpy())
    return (np.concatenate(p1_list) + np.concatenate(p2_list)) / 2.0


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


def fit(X, y, idx_fit, idx_val, epochs, tag, verbose=True):
    tl = DataLoader(DS(X[idx_fit], y[idx_fit], True, seed=SEED), batch_size=32, shuffle=True, num_workers=0)
    vl = DataLoader(DS(X[idx_val], y[idx_val], False, seed=SEED), batch_size=64, shuffle=False, num_workers=0)
    model = CheeseNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = {"auc": -1, "epoch": -1, "state": None, "vm": None}
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        trl, _, _ = run_epoch(model, tl, opt)
        sched.step()
        vll, vp, vt = run_epoch(model, vl)
        vm = metrics(vt, vp)
        hist.append({"epoch": ep + 1, "train_loss": trl, "val_loss": vll,
                     "val_acc": vm["acc"], "val_f1": vm["f1"], "val_auc": vm["auc"]})
        if verbose:
            print(f"  [{tag}] ep{ep+1:02d} trl={trl:.4f} vll={vll:.4f} "
                  f"acc={vm['acc']:.4f} f1={vm['f1']:.4f} auc={vm['auc']:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if vm["auc"] > best["auc"]:
            best = {"auc": vm["auc"], "epoch": ep + 1, "vm": vm,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    print(f"samples={len(rows)}  loading cache ...")
    X, y = build_cache(rows)
    print("cache", X.shape)

    src = np.array([r["src_id"] for r in rows])
    test_mask = np.isin(src, split["test_src"])

    folds_res, hist_all = [], []
    oof_p = np.zeros(len(rows))
    ep_best = []
    for k, fo in enumerate(split["folds"]):
        if args.quick and k > 1:
            break
        fit_mask = np.isin(src, fo["fit"])
        val_mask = np.isin(src, fo["val"])
        model, best, hist = fit(X, y, np.where(fit_mask)[0], np.where(val_mask)[0],
                                args.epochs, f"fold{k}")
        folds_res.append({"fold": k, "best_epoch": best["epoch"], **best["vm"]})
        ep_best.append(best["epoch"])
        hist_all.append(hist)
        _, vp, vt = run_epoch(model, DataLoader(DS(X[val_mask], y[val_mask], False),
                                               batch_size=64, num_workers=0))
        oof_p[val_mask] = vp
        torch.save(model.state_dict(), os.path.join(RES, f"cnn_fold{k}.pt"))
        print(f"fold{k} best ep={best['epoch']} val_auc={best['auc']:.4f}")

    n_ep = int(np.median(ep_best)) if ep_best else args.epochs
    print(f"\n== retrain on full train split for {n_ep} epochs ==")
    tr_mask = ~test_mask
    final, bestf, histf = fit(X, y, np.where(tr_mask)[0], np.where(test_mask)[0],
                              n_ep, "final")
    torch.save(final.state_dict(), os.path.join(RES, "cnn_final.pt"))

    _, pt, tt = run_epoch(final, DataLoader(DS(X[test_mask], y[test_mask], False),
                                           batch_size=64, num_workers=0))
    test_m = metrics(tt, pt)
    tt_idx = np.where(test_mask)[0]
    pt_tta = predict_tta(final, X, tt_idx)
    test_m_tta = metrics(y[tt_idx], pt_tta)

    mean_oof = metrics(y[~test_mask], oof_p[~test_mask])
    folds_summary = {
        "acc": float(np.mean([f["acc"] for f in folds_res])),
        "f1": float(np.mean([f["f1"] for f in folds_res])),
        "auc": float(np.mean([f["auc"] for f in folds_res])),
        "acc_std": float(np.std([f["acc"] for f in folds_res])),
        "f1_std": float(np.std([f["f1"] for f in folds_res])),
        "auc_std": float(np.std([f["auc"] for f in folds_res])),
    }

    out = {
        "model": "CheeseNet (compact CNN, from scratch)",
        "input": INPUT, "epochs_per_fold": args.epochs,
        "n_samples": len(rows),
        "n_train": int((~test_mask).sum()), "n_test": int(test_mask.sum()),
        "fold_results": folds_res, "fold_summary": folds_summary,
        "oof_train_metrics": mean_oof,
        "final_epochs": n_ep,
        "test_metrics": test_m_tta,
        "test_metrics_no_tta": test_m,
        "test_probs": pt_tta.tolist(), "test_true": y[tt_idx].tolist(),
        "history_final": histf,
    }
    with open(os.path.join(RES, "cnn_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n== CNN test ==")
    print(json.dumps(test_m, indent=2))


if __name__ == "__main__":
    main()
