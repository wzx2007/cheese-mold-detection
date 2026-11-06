# -*- coding: utf-8 -*-
"""
infer.py —— 单图 / 批量推理。

用法：
  python code/infer.py path/to/cheese.jpg                  # 默认 CNN
  python code/infer.py path/to/dir --model classic         # 传统模型，遍历目录
  python code/infer.py a.jpg b.jpg --model cnn --tta       # 多张 + 水平翻转 TTA

输出：每张图一行  <概率>  <判定>  <文件名>
概率 = 该图属于"发霉"类的置信度。

依赖是懒加载的：用 --model classic 时不需要装 torch。
"""

import os, sys, argparse, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402


def imread_any(p):
    """兼容中文/空格路径的读图。"""
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图像: {p}")
    return img


def gather(args):
    out, seen = [], set()
    for a in args.inputs:
        if os.path.isdir(a):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for f in sorted(glob.glob(os.path.join(a, ext))):
                    k = os.path.normcase(os.path.abspath(f))
                    if k not in seen:      # Windows 下 glob 大小写不敏感，需去重
                        seen.add(k)
                        out.append(f)
        else:
            out.append(a)
    return out


# ---------------------------------------------------------------- CNN

def run_cnn(files, weights, tta, thr):
    import torch
    from train_cnn_v3 import CheeseNetV3, INPUT, stack_input

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"找不到权重 {weights}\n请先把 cnn_final.pt 放到 models/ 目录，或用 --weights 指定。")
    model = CheeseNetV3(in_ch=4)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    res = []
    for p in files:
        img = cv2.resize(imread_any(p), (INPUT, INPUT), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        t = torch.from_numpy(stack_input(img, use_prior=True))[None]
        with torch.no_grad():
            prob = float(torch.softmax(model(t), 1)[0, 1])
            if tta:
                prob = (prob + float(torch.softmax(model(torch.flip(t, dims=[3])), 1)[0, 1])) / 2
        res.append((p, prob))
    return res


# ---------------------------------------------------------------- classic

def run_classic(files, weights, thr):
    import joblib
    from train_classic import extract_arr, SIZE

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"找不到模型 {weights}\n请先把 classic_model.joblib 放到 models/ 目录。")
    pipe = joblib.load(weights)

    res = []
    for p in files:
        img = cv2.resize(imread_any(p), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        feat = extract_arr(img)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)[None]
        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba(feat)[0, 1])
        else:
            s = float(pipe.decision_function(feat)[0])
            prob = 1.0 / (1.0 + np.exp(-s))
        res.append((p, prob))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="图像文件或目录")
    ap.add_argument("--model", choices=["cnn", "classic"], default="cnn")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tta", action="store_true", help="水平翻转 TTA（仅 CNN）")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    files = gather(args)
    if not files:
        print("没有找到图像")
        return

    if args.model == "cnn":
        w = args.weights or os.path.join(paths.MODELS, "cnn_final.pt")
        res = run_cnn(files, w, args.tta, args.thr)
    else:
        w = args.weights or os.path.join(paths.MODELS, "classic_model.joblib")
        res = run_classic(files, w, args.thr)

    n_bad = 0
    print(f"{'prob':>7}  {'verdict':<8} file")
    for p, prob in res:
        bad = prob >= args.thr
        n_bad += int(bad)
        print(f"{prob:7.4f}  {'MOLDY' if bad else 'good':<8} {os.path.basename(p)}")
    print(f"\n{len(res)} 张图，判定发霉 {n_bad} 张 / 正常 {len(res)-n_bad} 张")


if __name__ == "__main__":
    main()
