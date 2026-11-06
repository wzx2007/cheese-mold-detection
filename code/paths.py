# -*- coding: utf-8 -*-
"""
paths.py —— 统一的路径约定。
仓库根目录 = code/ 的上一级；也可用环境变量 CHEESE_ROOT 覆盖。
"""
import os

BASE = os.environ.get("CHEESE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE, "data", "processed")
RAW = os.path.join(BASE, "data", "raw")
RES = os.path.join(BASE, "outputs")
FIG = os.path.join(RES, "figures")
MODELS = os.path.join(BASE, "models")

for _d in (DATA, RAW, RES, FIG, MODELS):
    os.makedirs(_d, exist_ok=True)
