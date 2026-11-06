# -*- coding: utf-8 -*-
"""
train_final.py —— 一步到位：在 80% 训练集上训一次，直接在 20% 测试集上出最终结果。

跳过 5 折交叉验证，换来的是把整块算力花在"最终模型"上：
  * 训练数据 = 全部 1346 个训练样本（不再分折，多 25% 数据）
  * 轮数 = 26，由 fold0 的验证实验定下来（那一次最优点是第 24 轮），
    因此轮数是**事先固定**的，不依赖本次测试集曲线
  * 每个 epoch 仍会在测试集上打一次分，仅用于记录学习曲线；
    最终汇报的是**第 26 轮（最后一个 epoch）**的结果，不做任何挑 epoch

输出：
  outputs/cnn_final_metrics.json   指标 + 测试集概率
  outputs/cnn_final.pt             权重
"""

import os, sys, json, time
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn_v3 import (  # noqa: E402
    load_rows, build_cache, fit, run_epoch, predict_tta, metrics, DS, INPUT, OUT, RES,
)

EPOCHS = 26
LR = 1.2e-3
WD = 1e-3


def main():
    rows = load_rows()
    split = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    print(f"samples={len(rows)} input={INPUT} epochs={EPOCHS} (single run, no CV)", flush=True)
    X, y = build_cache(rows)
    print("cache", X.shape, flush=True)

    src = np.array([r["src_id"] for r in rows])
    test_mask = np.isin(src, split["test_src"])
    tr_idx = np.where(~test_mask)[0]
    te_idx = np.where(test_mask)[0]
    print(f"train={len(tr_idx)}  test={len(te_idx)}", flush=True)

    t0 = time.time()
    model, _, hist = fit(X, y, tr_idx, te_idx, EPOCHS, "final", True, LR, WD,
                         select_best=False)
    print(f"training took {time.time()-t0:.0f}s", flush=True)

    pt_tta = predict_tta(model, X, te_idx, True)
    test_m = metrics(y[te_idx], pt_tta)
    _, pt, _ = run_epoch(model, DataLoader(DS(X[test_mask], y[test_mask], False, True),
                                          batch_size=64, num_workers=0))
    test_m_raw = metrics(y[te_idx], pt)

    # 训练集上的最终表现（用同一套推理配置）
    ptr = predict_tta(model, X, tr_idx, True) if len(tr_idx) < 1400 else None
    train_m = metrics(y[tr_idx], ptr) if ptr is not None else None

    import torch
    torch.save(model.state_dict(), os.path.join(RES, "cnn_final.pt"))

    out = {
        "model": "CheeseNet-v3 (4ch RGB+prior, from scratch, single run)",
        "protocol": "single split: 80% train / 20% test, no cross-validation",
        "prior": True, "input": INPUT, "epochs": EPOCHS, "lr": LR, "wd": WD,
        "n_samples": len(rows), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "test_metrics": test_m, "test_metrics_no_tta": test_m_raw,
        "train_metrics": train_m,
        "test_probs": pt_tta.tolist(), "test_true": y[te_idx].tolist(),
        "history": hist,
    }
    p = os.path.join(RES, "cnn_final_metrics.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n== FINAL (test, with TTA) ==")
    print(json.dumps(test_m, indent=2))
    print("\n== no TTA ==")
    print(f"acc={test_m_raw['acc']:.4f} f1={test_m_raw['f1']:.4f} auc={test_m_raw['auc']:.4f}")
    if train_m:
        print(f"train acc={train_m['acc']:.4f} f1={train_m['f1']:.4f}")
    print("saved", p)


if __name__ == "__main__":
    main()
