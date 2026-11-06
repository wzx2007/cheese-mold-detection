# 模型卡 · Model Card

本目录存放两个已训练好的模型权重。两者在**完全相同**的数据划分上训练与评估。

| 文件 | 模型 | 大小 | 运行依赖 |
|---|---|---|---|
| `cnn_final.pt` | CheeseNet-v3（4 通道先验引导 CNN） | ≈ 4.5 MB | `torch` |
| `classic_model.joblib` | OpenCV 425 维手工特征 + RandomForest | ≈ 10.1 MB | `scikit-learn` + `joblib` + `opencv-python` |

## cnn_final.pt

| 项 | 值 |
|---|---|
| 架构 | `CheeseNetV3(in_ch=4)`，见 `code/train_cnn_v3.py` |
| 参数量 | 1.136 M |
| 输入 | 4 × 224 × 224，float32，归一化 `(x/255 − 0.5)/0.25` |
| 通道含义 | ch0–2 = RGB；ch3 = 霉菌响应图（0–255 标度，同样归一化） |
| 输出 | 2 类 logits（0 = 正常，1 = 发霉） |
| 训练数据 | 664 张原图 / 1346 个样本（80% 训练集），34 epochs，全量重训 |
| 保存方式 | `state_dict()`，无优化器状态 |
| 推理配置 | 水平翻转 TTA（原图 + 翻转平均），阈值 0.5 |

加载方式：

```python
import sys, torch, cv2, numpy as np
sys.path.insert(0, "code")
from train_cnn_v3 import CheeseNetV3, INPUT, stack_input

model = CheeseNetV3(in_ch=4)
model.load_state_dict(torch.load("models/cnn_final.pt", map_location="cpu"))
model.eval()

img = cv2.cvtColor(cv2.imread("demo/moldy_mid_washed-rind_01844_b.jpg"), cv2.COLOR_BGR2RGB)
x = torch.from_numpy(stack_input(img, use_prior=True))[None]
p_moldy = torch.softmax(model(x), 1)[0, 1].item()
```

> **务必通过 `stack_input(..., use_prior=True)` 构造输入**，不要自己拼张量。第 4 通道的公式、
> 计算顺序（先几何增广 → 再算先验 → 再归一化）必须与训练时逐位一致，否则精度会无声下降。

## classic_model.joblib

| 项 | 值 |
|---|---|
| 类型 | `sklearn.pipeline.Pipeline([("m", RandomForestClassifier(...))])` |
| 特征 | 425 维，`code/train_classic.extract_arr(bgr)` |
| 训练数据 | 同上（1346 样本） |
| 输入 | BGR ndarray（内部会 resize 到 288×288 并高斯平滑） |
| 输出 | `predict_proba(X)[:, 1]` = 发霉概率 |

```python
import sys, cv2, joblib
sys.path.insert(0, "code")
from train_classic import extract_arr

pipe = joblib.load("models/classic_model.joblib")
feat = extract_arr(cv2.imread("demo/moldy_mid_washed-rind_01844_b.jpg"))[None]
p_moldy = pipe.predict_proba(feat)[0, 1]
```

## 适用边界（两个模型共同）

- **只在"合成霉菌"分布内有效**。霉菌的颜色、边缘羽化、覆盖率范围由 `code/mold_synth.py` 的参数决定。
- 输入应当是**单个奶酪表面的近景**，与训练样本同量级的视野（一块表面占据画面大部分）。远景、整桌、多块奶酪同框都不在训练分布内。
- 二分类，不输出缺陷位置与严重程度。
- 判定阈值 0.5 是针对"准确率"最优的；若产线更怕漏检，应下调阈值（见 `docs/04-results.md` 的阈值敏感性表）。
- **上线前必须用真实产线缺陷样本做域适应评估**——这是使用本权重的先决条件，不是可选项。

## 版本

| 版本 | 变化 | 结论 |
|---|---|---|
| v1 (`train_cnn.py`) | 192 输入 / 全局平均池化 / 无先验 | 收敛过慢，13 轮仅 0.746 |
| v2 (`train_cnn_v2.py`) | 224 输入 / avg+max 池化 / 无先验 | 学习率过大导致发散 |
| **v3** (`train_cnn_v3.py`) | 224 / avg+max / **+霉菌响应图通道** / 降 lr | 采用 |
