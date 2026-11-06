# -*- coding: utf-8 -*-
"""
train_cnn_v3.py —— 先验引导的紧凑 CNN（本项目最终采用的模型）

相对 v1/v2 的改动：

  1) 输入通道 3 -> 4：第 4 通道是**手工构造的霉菌响应图**
         m = 0.55 * clip((median(b*) - b*)/32) + 0.30 * clip(1 - S/(median(S)+25))
             + 0.40 * clip((median(L) - L)/55)
     它把"奶酪偏黄、污染霉偏绿/青/灰黑"这条物理先验显式喂给网络。
     这同时也是传统方法最强的一族特征 —— 所以两条路线现在共享同一个先验，
     对比的是"先验之后怎么办"：手工阈值化 vs 端到端学习。
  2) 分辨率 224（缓存 288），保留霉斑绒毛尺度的高频；
  3) 池化头 avg+max 拼接 —— 平均池化会把只占 2.8% 面积的响应稀释掉；
  4) 训练裁剪 0.85~1.0，避免把霉斑整块裁出画面（那等于把"发霉"标成"正常"）；
  5) 优化器回到温和区间（lr 1.2e-3 / wd 1e-3），v2 的 3e-3 + 5e-3 会发散。

用法：
  python code/train_cnn_v3.py --folds 0 --epochs 30 --tag pilot      # 单折试跑
  python code/train_cnn_v3.py --epochs 34                            # 5 折 + 重训 + 测试
  python code/train_cnn_v3.py --no-prior --epochs 34                 # 消融：不给先验
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

import paths  # noqa: E402  (仓库根目录 = code/ 的上一级)

OUT = paths.DATA
RES = paths.RES
os.makedirs(RES, exist_ok=True)

CACHE = 288
INPUT = 224
SEED = 20261106
torch.manual_seed(SEED)
np.random.seed(SEED)

USE_PRIOR = True


# ---------------------------------------------------------------- 先验通道

def prior_map(rgb):
    """霉菌响应图 m ∈ [0,255]：以全图分位为参考的"异常偏色/偏暗"强度。

    与 train_classic.mold_map 保持一致的定义，但返回 uint8 方便进卷积。
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b = lab[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    L = lab[:, :, 0].astype(np.float32)
    d_b = np.clip((np.median(b) - b) / 32.0, 0, 1)
    d_s = np.clip(1.0 - s / max(float(np.median(s)) + 25.0, 30.0), 0, 1)
    dark = np.clip((np.median(L) - L) / 55.0, 0, 1)
    m = np.clip(0.55 * d_b + 0.30 * d_s + 0.40 * dark, 0, 1)
    return (m * 255.0).astype(np.uint8)


def stack_input(c, use_prior):
    """c: float32 HxWx3 (RGB, 0~255) -> normalized CHW float32"""
    c = np.clip(c, 0, 255).astype(np.uint8)
    chans = [c.astype(np.float32)]
    if use_prior:
        chans.append(prior_map(c).astype(np.float32)[..., None])
    x = np.concatenate(chans, axis=2)
    x = (x / 255.0 - 0.5) / 0.25
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ---------------------------------------------------------------- data

def load_rows():
    with open(os.path.join(OUT, "manifest.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_cache(rows):
    X = np.zeros((len(rows), CACHE, CACHE, 3), np.uint8)
    y = np.zeros(len(rows), np.int64)
    for i, r in enumerate(rows):
        p = os.path.join(OUT, r["path"].replace("/", os.sep))
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
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
        M = cv2.getRotationMatrix2D((INPUT / 2, INPUT / 2), rng.uniform(-12, 12), 1.0)
        c = cv2.warpAffine(c, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    f = c.astype(np.float32)
    if rng.random() < 0.7:
        f = f * rng.uniform(0.88, 1.14) + rng.uniform(-12, 12)
    if rng.random() < 0.4:
        f = f * rng.uniform(0.95, 1.05, 3).astype(np.float32)
    if rng.random() < 0.25:
        f = f + rng.normal(0, rng.uniform(1.5, 5.0), f.shape).astype(np.float32)
    if rng.random() < 0.15:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 0.9))
    return f


def eval_one(im):
    return cv2.resize(im, (INPUT, INPUT), interpolation=cv2.INTER_AREA).astype(np.float32)


class DS(Dataset):
    def __init__(self, X, y, train, use_prior, seed=0):
        self.X, self.y, self.train, self.use_prior = X, y, train, use_prior
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = aug_one(self.X[i], self.rng) if self.train else eval_one(self.X[i])
        return torch.from_numpy(stack_input(img, self.use_prior)), int(self.y[i])


# ---------------------------------------------------------------- model

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
        return self.act(self.b2(self.c2(x)))


class CheeseNetV3(nn.Module):
    def __init__(self, n_cls=2, w=(24, 48, 96, 160, 192), p_drop=0.2, in_ch=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]),
            nn.SiLU(inplace=True))
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


# ---------------------------------------------------------------- train

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


def predict_tta(model, X, idx, use_prior):
    model.eval()
    p1, p2 = [], []
    for s in range(0, len(idx), 48):
        b = idx[s:s + 48]
        t = torch.from_numpy(np.stack([stack_input(eval_one(X[i]), use_prior) for i in b]))
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


def fit(X, y, idx_fit, idx_val, epochs, tag, use_prior, lr=1.2e-3, wd=1e-3, bs=32):
    tl = DataLoader(DS(X[idx_fit], y[idx_fit], True, use_prior, seed=SEED + 7),
                    batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(DS(X[idx_val], y[idx_val], False, use_prior, seed=SEED),
                    batch_size=64, num_workers=0)
    model = CheeseNetV3(in_ch=4 if use_prior else 3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warm = max(2, epochs // 10)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.2, 1.0, total_iters=warm),
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
                     "val_auc": vm["auc"], "lr": float(opt.param_groups[0]["lr"])})
        print(f"  [{tag}] ep{ep+1:02d} trl={trl:.4f} tracc={trm['acc']:.3f} "
              f"vll={vll:.4f} acc={vm['acc']:.4f} f1={vm['f1']:.4f} "
              f"auc={vm['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        score = vm["acc"] + 0.5 * (vm["auc"] - 0.5)
        if score > best["score"]:
            best = {"score": score, "epoch": ep + 1, "vm": vm,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, best, hist


def main():
    global USE_PRIOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=34)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--tag", type=str, default="v3")
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--no-prior", action="store_true")
    args = ap.parse_args()
    USE_PRIOR = not args.no_prior

    torch.set_num_threads(max(1, (os.cpu_count() or 8)))
    rows = load_rows()
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    print(f"samples={len(rows)} input={INPUT} prior={USE_PRIOR} epochs={args.epochs} "
          f"lr={args.lr} wd={args.wd}", flush=True)
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
                               args.epochs, f"f{k}", USE_PRIOR, args.lr, args.wd)
        folds_res.append({"fold": k, "best_epoch": best["epoch"], **best["vm"]})
        ep_best.append(best["epoch"])
        hist_all.append(hist)
        _, vp, _ = run_epoch(model, DataLoader(DS(X[val_mask], y[val_mask], False, USE_PRIOR),
                                              batch_size=64, num_workers=0))
        oof_p[val_mask] = vp
        torch.save(model.state_dict(), os.path.join(RES, f"cnn3_fold{k}.pt"))
        print(f"fold{k} best ep={best['epoch']} val_acc={best['vm']['acc']:.4f} "
              f"val_auc={best['vm']['auc']:.4f}", flush=True)

    if len(folds_res) < len(split["folds"]):
        out = {"tag": args.tag, "input": INPUT, "prior": USE_PRIOR,
               "epochs": args.epochs, "fold_results": folds_res, "history": hist_all}
        p = os.path.join(RES, f"cnn3_{args.tag}.json")
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nsaved", p)
        return

    n_ep = int(np.median(ep_best))
    print(f"\n== retrain on full train split for {n_ep} epochs ==", flush=True)
    tr_mask = ~test_mask
    final, _, histf = fit(X, y, np.where(tr_mask)[0], np.where(test_mask)[0],
                          n_ep, "final", USE_PRIOR, args.lr, args.wd)
    torch.save(final.state_dict(), os.path.join(RES, "cnn3_final.pt"))

    tt_idx = np.where(test_mask)[0]
    pt_tta = predict_tta(final, X, tt_idx, USE_PRIOR)
    test_m_tta = metrics(y[tt_idx], pt_tta)
    _, pt, _ = run_epoch(final, DataLoader(DS(X[test_mask], y[test_mask], False, USE_PRIOR),
                                          batch_size=64, num_workers=0))
    test_m = metrics(y[tt_idx], pt)

    fs = {k: float(np.mean([f[k] for f in folds_res])) for k in ("acc", "f1", "auc")}
    fs.update({k + "_std": float(np.std([f[k] for f in folds_res])) for k in ("acc", "f1", "auc")})

    out = {
        "model": "CheeseNet-v3 (4ch: RGB + mold-response prior, from scratch)",
        "prior": USE_PRIOR, "input": INPUT, "epochs_per_fold": args.epochs,
        "lr": args.lr, "wd": args.wd,
        "n_samples": len(rows), "n_train": int(tr_mask.sum()), "n_test": int(test_mask.sum()),
        "fold_results": folds_res, "fold_summary": fs,
        "final_epochs": n_ep, "test_metrics": test_m_tta, "test_metrics_no_tta": test_m,
        "test_probs": pt_tta.tolist(), "test_true": y[tt_idx].tolist(),
        "history_final": histf, "history_folds": hist_all,
    }
    p = os.path.join(RES, "cnn3_metrics.json" if USE_PRIOR else "cnn3_noprior_metrics.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n== test (TTA) ==")
    print(json.dumps(test_m_tta, indent=2))
    print("saved", p)


if __name__ == "__main__":
    main()
