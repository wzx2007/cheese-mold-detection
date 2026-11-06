# demo/ · 路演演示图

6 张图，全部取自**测试集**（与训练集按原图分组、完全不相交），所以下面给出的概率是模型对"没见过的东西"的真实反应。

| 文件 | 真实标签 | 霉菌覆盖率 | 菌种 | CNN 概率 |
|---|---|---|---|---|
| `good_hard_00003_a.jpg` | 正常 | — | — | 0.0152 |
| `good_semi-soft_01123_b.jpg` | 正常 | — | — | 0.1011 |
| `good_washed-rind_02144_a.jpg` | 正常 | — | — | 0.0908 |
| `moldy_easy_semi-soft_00747_b.jpg` | 发霉 | 9.65% | 橄榄绿青霉 | 0.9897 |
| `moldy_mid_washed-rind_01844_b.jpg` | 发霉 | 2.28% | 蓝绿青霉 | 0.9743 |
| `moldy_hard_washed-rind_01852_a.jpg` | 发霉 | **0.45%** | 橄榄绿青霉 | 0.9934 |

`moldy_hard_*` 是**故意挑的困难样本**：霉斑只占画面 0.45%，人眼在正常看图的尺寸下基本找不到。
它是现场演示里最有说服力的一张——如果它被判对，说明模型确实在看"局部那一小团"，而不是在看整体色调。

反例也留了：跑**传统 OpenCV 模型**时，`good_semi-soft_01123_b.jpg` 会被误判成发霉（概率 0.60），
这正好可以用来讲两条路线的差别。

## 现场怎么用

```bash
# CNN（主模型），默认带水平翻转 TTA
python code/infer.py demo/ --tta

# 传统 OpenCV 基线，对比用
python code/infer.py demo/ --model classic
```

输出：

```
   prob  verdict  file
 0.0152  good     good_hard_00003_a.jpg
 0.1011  good     good_semi-soft_01123_b.jpg
 0.0908  good     good_washed-rind_02144_a.jpg
 0.9897  MOLDY    moldy_easy_semi-soft_00747_b.jpg
 0.9934  MOLDY    moldy_hard_washed-rind_01852_a.jpg
 0.9743  MOLDY    moldy_mid_washed-rind_01844_b.jpg

6 张图，判定发霉 3 张 / 正常 3 张
```

## 建议的演示顺序

1. 先投 `moldy_hard_*`，让大家自己找霉斑——**大部分人第一眼找不到**。
2. 再跑推理，给出 0.9934。这个反差是整个演示最有效的一击。
3. 然后跑传统模型，指出它在这 6 张上错了 1 张（把正常判成发霉），引出"为什么需要 CNN"。
4. 最后回到 `docs/04-results.md` 的 338 个样本上的整体数字，说明上面这 6 张不是精挑细选的特例。

> 这些图像是**合成样本**（霉菌由 `code/mold_synth.py` 叠加生成，再经相机成像链路退化）。
> 对外展示时必须标注清楚，不要让人误以为是真实产线照片。
