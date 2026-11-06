# data/ · 数据从哪来

**本仓库不包含任何奶酪图像。** 基底图库的版权归原作者，仓库只提供下载与合成脚本。

这里是唯一随仓库发布的东西——三个小文件，用来**不重建数据集也能复现划分、跑出报告**：

| 文件 | 大小 | 作用 |
|---|---|---|
| `processed/manifest.csv` | 约 160 KB | 1684 个样本的清单：样本 id、来源原图 id、奶酪类目、标签、菌种、覆盖率、相对路径 |
| `processed/split.json` | 约 70 KB | 80/20 分组划分 + 5 折分组索引（按 `src_id` 分组） |
| `processed/dataset_info.json` | 约 1 KB | 数据集汇总统计 |

## 想自己重建完整数据

```bash
python code/download_data.py              # 约 123 MB，下载 Kaggle jenlooper/cheese
python code/prepare_dataset.py --start 0 --count 420 --demo
python code/prepare_dataset.py --start 420 --count 420
# … 分片跑到 --start 2100，然后
python code/prepare_dataset.py --merge
python code/make_splits.py
```

重建需要 CPU 约 1.5–2 小时（每片 15–25 分钟），产出 `processed/{good,moldy}/*.jpg` 共 1684 张 384×384 图像，约 130 MB。

> `manifest.csv` 里的 `path` 是相对 `data/processed/` 的路径。没重建数据时这些路径指向不存在的文件，
> 因此 `train_*.py` 会报错；但 `gen_docs.py` / `make_report.py` 只读清单与划分，可以直接跑。

## 目录约定

```
data/
├── raw/                    # 下载的原始 zip（不入库）
├── processed/
│   ├── good/               # 842 张正常样本（不入库）
│   ├── moldy/              # 842 张发霉样本（不入库）
│   ├── manifest_parts/     # 分片中间结果（不入库）
│   ├── manifest.csv        # ✅ 入库
│   ├── split.json          # ✅ 入库
│   └── dataset_info.json   # ✅ 入库
└── README.md               # ✅ 本文件
```
