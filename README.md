# 奶酪表面霉菌视觉检测 · Cheese Surface Mould Detection

用**自建的合成缺陷数据集**，回答一个工业现场的真问题：机器人搬起一块奶酪之前，能不能判断它表面已经被污染霉菌侵染？

> **没有真实缺陷样本怎么办？** 把"长霉"写成一个可调参的物理模型，把"工业相机"写成一串可复现的退化算子，
> 于是不可控的缺陷变成了可控的实验变量。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]() [![OpenCV](https://img.shields.io/badge/opencv-4.8%2B-green)]() [![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 摘要

工业奶酪产线（成熟期护理、切片、包装）需要在机器人搬运前判断「这块奶酪表面是否已被污染霉菌侵染」。真实的霉菌缺陷样本稀缺且不可控——食品安全缺陷不会为了训练数据集而出现。本项目用**物理启发式的霉菌合成 + 工业相机成像链路仿真**，从公开奶酪图库构造出成对的「正常 / 发霉」样本，并在此基础上训练、评估两条路线：

| 路线 | 核心 | 定位 |
|---|---|---|
| **CheeseNet-v3**（主模型） | 4 通道输入 = RGB + 霉菌响应图先验；紧凑 CNN 从零训练；avg+max 双池化头 | 精度上限 |
| **OpenCV 手工特征 + 随机森林**（基线） | 425 维手工特征（含手写"霉菌检测器"）+ 4 种分类器比较 | 可解释、可快速上线 |

两个模型共享**完全相同的数据划分**（按原图分组 80/20，同一张原图的多个 crop 绝不跨侧），结果可直接对比。

- 完整实验报告（HTML）：[`outputs/奶酪霉菌分类_实验结果.html`](outputs/奶酪霉菌分类_实验结果.html)
- 方法细节：[`docs/01-problem.md`](docs/01-problem.md) → [`docs/05-roadshow-brief.md`](docs/05-roadshow-brief.md)
- 给路演同学：[`docs/05-roadshow-brief.md`](docs/05-roadshow-brief.md)（讲什么 / 怎么讲 / 被问到怎么答）

## 结果一览

| 模型 | 测试集准确率 | F1 | AUC | 漏检 / 误检 | 训练耗时 |
|---|---|---|---|---|---|
| **CheeseNet-v3**（4 通道先验 CNN） | **98.82%** | **0.9882** | **0.9992** | 1 / 3 | 26 分钟（CPU） |
| OpenCV 425 维特征 + RandomForest | 86.39% | 0.8671 | 0.9159 | 19 / 27 | 4 分钟（CPU） |

- 测试集：338 个样本 / 166 张原图，与训练集**按原图分组**、完全不相交（169 正常 / 169 发霉）
- 单折抽检（133 张原图、更少训练数据）复核：准确率 0.9710 / AUC 0.9887
- 详细结果、误差分析与阈值取舍见 [`docs/04-results.md`](docs/04-results.md)
- 一句话：**CNN 把漏检（发霉判成正常）从 19 个降到 1 个**——食品安全场景里最该看的那个数

> 这些数字是**方法链路的下限**，不是产线验收指标：霉菌是合成的，真实缺陷分布更随机。

## 为什么霉菌是"合成"的

1. **真实缺陷样本拿不到**：霉菌污染是非预期事件，产线上一年也攒不出几百张标注样本。
2. **合成可以控制变量**：能精确控制菌种颜色、菌落尺度、覆盖率、空间分布，并保证两类图像来自**同一批原始照片**——模型无法靠"这张照片看起来像另一批数据"抄近路。
3. **代价是分布偏移**：合成霉菌的颜色与边缘统计由脚本参数决定，真实霉菌更随机。**本仓库的数字是方法链路的下限，不是产线验收指标。**

## 数据集一览

| 项目 | 数值 |
|---|---|
| 原始图库 | Kaggle [`jenlooper/cheese`](https://www.kaggle.com/datasets/jenlooper/cheese)（All The Cheese，3234 张 / 6 个表皮工艺类目） |
| 参与建模的类目 | `hard` / `semi-soft` / `fresh` / `washed-rind`（排除 `blue`、`bloomy`——它们的"霉"是工艺的一部分） |
| 有效原图 | 830 张 |
| 最终样本 | **1684**（正常 842 / 发霉 842） |
| 图像尺寸 | 384×384（训练时 288 缓存 → 224 输入） |
| 平均霉菌覆盖面积 | 2.8%（区间 1.5%–10.5%） |
| 划分 | 按原图分组 80/20（GroupShuffleSplit）；另附 5 折分组索引备用于交叉验证 |

三张关键图（`outputs/figures/`）：

| 文件 | 内容 |
|---|---|
| `synthesis_triplet.jpg` | 三联对比：原始 crop → 叠加合成霉菌 → 再过一遍相机成像链路 |
| `mold_species.jpg` | 5 类污染霉菌的合成色板 |
| `qc_pairs.jpg` | 成对的「正常 / 发霉」样本（同一张原图切出） |

## 目录结构

```
cheese-mold-detection/
├── code/
│   ├── paths.py               # 统一路径约定（仓库根目录 = code/ 的上一级）
│   ├── mold_synth.py          # 霉菌合成 + 相机成像退化（核心）
│   ├── prepare_dataset.py     # 切图、成对合成、写 manifest（支持分片执行）
│   ├── make_splits.py         # 80/20 分组划分（另生成 5 折索引备用）
│   ├── train_final.py         # 【主模型·一步到位】训练一次 + 直接出测试集结果
│   ├── train_cnn_v3.py        # 同上模型，5 折交叉验证版（可选出折间方差）
│   ├── train_cnn_v2.py        # 消融：同结构但不给先验通道
│   ├── train_cnn.py           # 早期版本（192 输入 / 全局平均池化），留作对照
│   ├── train_classic.py       # OpenCV 425 维手工特征 + 4 分类器比较
│   ├── analyze.py             # 误差分析：错在哪些样本上
│   ├── infer.py               # 单图 / 批量推理（两个模型）
│   ├── make_report.py         # 生成图表与 HTML 报告（不依赖 matplotlib）
│   └── qc.py                  # 数据质检拼图
├── docs/
│   ├── 01-problem.md          # 问题定义与业务背景
│   ├── 02-dataset.md          # 数据集构建方法（含霉菌外观依据）
│   ├── 03-models.md           # 两个模型的实现细节
│   ├── 04-results.md          # 实验结果与误差分析
│   └── 05-roadshow-brief.md   # 路演包装建议（给做展示的同学）
├── models/                    # 训练好的权重
│   ├── cnn_final.pt          #   主模型（PyTorch state_dict，4.4 MB）
│   ├── classic_model.joblib   #   传统方法（sklearn Pipeline，11 MB）
│   └── MODEL_CARD.md          #   权重说明与使用方式
├── demo/                      # 6 张演示图（3 正常 / 3 发霉，含 1 张困难样本）
├── outputs/                   # 指标 JSON、误差分析、图表、HTML 报告
├── requirements.txt
└── LICENSE
```

## 快速开始

> 本仓库**已自带训练好的权重**（`models/`，合计约 15 MB）。只想看效果：装完依赖直接跑第 6 步，
> `python code/infer.py demo/` 即可得到每张演示图的发霉概率。跑完整复现再从第 1 步开始。

```bash
git clone <this-repo>
cd cheese-mold-detection
python -m venv .venv
. .venv/Scripts/activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

### 1) 下载基底图库（约 123 MB，无需 Kaggle 账号）

```bash
python code/download_data.py
```

### 2) 构建数据集

`prepare_dataset.py` 支持分片，便于把长任务拆成多段跑（每片约 400 张原图，CPU 上约 15–25 分钟）：

```bash
python code/prepare_dataset.py --start 0   --count 420 --demo
python code/prepare_dataset.py --start 420 --count 420
# … 直到 --start 2100（有效原图约 830 张会落在 2165 的候选池里）
python code/prepare_dataset.py --merge
```

输出：`data/processed/{good,moldy}/*.jpg` + `manifest.csv` + `dataset_info.json`。

### 3) 划分

```bash
python code/make_splits.py     # 生成 split.json（两个模型共用）
```

### 4) 训练

```bash
python code/train_final.py                 # 主模型：训练一次（26 轮）→ 直接测测试集
python code/train_classic.py               # 基线：425 维特征 + 4 分类器比较

# 可选：跑 5 折交叉验证拿折间方差（CPU 约 2.5 小时）
python code/train_cnn_v3.py --epochs 34
```

CPU（16 核）参考耗时：主模型单次训练约 25 分钟；传统方法约 4 分钟（特征有缓存，第二次跑只要 1 分钟）；5 折交叉验证约 2.5 小时。

消融实验：

```bash
python code/train_cnn_v3.py --no-prior --epochs 32   # 去掉第 4 通道
python code/train_cnn_v2.py --epochs 32              # 同上，另存一份（历史版本）
```

### 5) 出报告与分析

```bash
python code/make_report.py     # -> outputs/奶酪霉菌分类_实验结果.html
python code/analyze.py         # -> outputs/error_analysis.md
```

### 6) 单图推理

```bash
python code/infer.py demo/
python code/infer.py demo/moldy_hard_*.jpg --tta
python code/infer.py demo/ --model classic
```

输出每张图的「发霉概率 / 判定 / 文件名」。

> Python 侧调用：
> ```python
> import sys; sys.path.insert(0, "code")
> from infer import run_cnn          # 或 run_classic
> res = run_cnn(["demo/x.jpg"], "models/cnn_final.pt", tta=True, thr=0.5)
> ```

## 关键设计（读这四条就够了）

1. **成对采样**：同一张原图切两个互不重叠的 crop，随机一个合成霉菌、另一个保持正常。两类样本的背景、光照、压缩痕迹天然对齐，模型只能学局部表面证据。
2. **相机链路仿真**：非均匀光照 + 暗角、色温漂移、离焦、运动模糊、伽马、泊松散粒噪声、读出噪声、JPEG 压缩，8 步按真实退化顺序叠加，**两类图像走同一分布**。
3. **先验通道**：CNN 的第 4 通道是手工构造的霉菌响应图，与传统方法用的是同一个先验。所以两条路线共享同一份领域知识，比的是"先验之后怎么办"。
4. **avg+max 双池化**：霉斑平均只占 2.8% 面积，全局平均池化会把这点响应直接稀释掉；max 池化对局部异常敏感。

## 局限性（请务必在对外场合说明）

- 霉菌由脚本合成，**不是真实产线缺陷**；颜色分布、边缘羽化、尺度范围都由参数决定。
- 基底图像来自网络图库，含少量非产线场景（砧板、手持、包装），已通过掩膜与三重判据过滤，但仍有噪声。
- 二分类任务，未做缺陷定位（分割）与严重程度分级。
- 结论仅用于方法链路验证；**上线前必须用真实缺陷样本做域适应评估**。

## 许可

- 代码：MIT（见 [`LICENSE`](LICENSE)）。
- 图像：**本仓库不重新分发任何第三方图像**，只提供下载与合成脚本。基底图库版权归原作者，使用须遵循 Kaggle 数据集页面的许可。
- 权重：可用于研究与非商业验证；商用前请自行评估数据来源合规性。

## 引用 / 致谢

- 基底图库：[All The Cheese](https://www.kaggle.com/datasets/jenlooper/cheese)（Kaggle, `jenlooper/cheese`）
- 工业背景参考：Stäubli（TX200L HE / 奶酪护理与乳品线）、KUKA（Terre de Fromages / Vepo 切片）、JNJ RF 系列（成熟期护理）
