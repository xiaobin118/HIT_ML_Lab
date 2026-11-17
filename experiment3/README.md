# 实验三：k-means 与混合高斯模型（EM 算法）

## 当前进度

原理文档、实现任务文档、脚手架与接口骨架已完成。`code/kmeans.py` 的四个函数已按文档实现并通过自检；`code/gmm.py` 的七个函数目前仍是带 `NotImplementedError` 的练习骨架，`code/run_experiment.py` 里还留有 `TODO(你)` 的调用步骤。

需要说明的是：本实验的完整结果（`exp3_report/` 下的报告、`results.json` 与插图）由 `exp3_report/full_experiment.py` 中的一份参考实现跑出。报告第 6.2 节已如实标注这一点；`code/gmm.py` 是之后要替换成自己实现的对应模块。仓库里保留骨架与报告并存的状态，是为了记录从讲义到实现的完整过程。

## 文件说明

| 路径 | 内容 | 状态 |
|---|---|---|
| `docs/01_原理_kmeans与GMM的EM推导.md` | 理论部分：生成式视角、k-means 目标函数与收敛性证明、GMM 似然、Jensen 不等式、EM 的 E/M 步推导、单调性证明、数值实现要点、模型选择与评价 | 已完成 |
| `docs/02_实现任务与编码路线.md` | 实现部分：分阶段任务、函数签名、实现要点、验收标准（含参考实现的实测数值）、易错点清单 | 已完成 |
| `code/data_gen.py` | 合成数据生成、UCI 数据加载；`normal_logpdf` 待实现 | 部分待写 |
| `code/metrics.py` | 排列匹配准确率、ARI、纯度、轮廓系数 | 已完成 |
| `code/viz.py` | 散点、协方差椭圆、收敛曲线、对比柱状图 | 已完成 |
| `code/kmeans.py` | k-means 四个函数（分配、目标函数、主循环、多次重启） | 已完成 |
| `code/gmm.py` | logsumexp、E 步、M 步、初始化、EM 主循环、似然校验 | 待实现 |
| `code/run_experiment.py` | 流程编排、结果落盘、参数对照表；实验步骤的算法调用待补 | 部分待写 |
| `exp3_report/` | 完整实验报告、`results.json`、9 张插图与复现脚本 | 已完成 |
| `data/` | 14 个数据集（见下，未随仓库提交） | 需自行下载 |

## 数据集

Iris 和 Seeds 作为基线，另有 12 个已下载并校验过的数据集可用于"更大规模"的实验。仓库不包含 `data/`（约 8 MB 原始数据），按下面的链接与文件名自行放置即可复现。

UCI 原始文件：`iris.data`（150×4，3 类）、`seeds.data`（210×7，3 类）、`segmentation.data`/`segmentation.test`（210/2100×19，7 类，训练测试分离）、`wdbc.data`（569×30，2 类）。

OpenML 下载（`data/downloads/`，全部经官方 md5 校验）：`wine`（178×13，3 类）、`vehicle`（846×18，4 类）、`diabetes`（768×8，2 类）、`optdigits`（5620×64，10 类）、`satimage`（6430×36，6 类）、`pendigits`（10992×16，10 类）、`letter`（20000×16，26 类）、`electricity`（45312×8，2 类）。

统一用注册表加载：

```python
from data_gen import load_dataset, list_datasets
X, y, names = load_dataset("pendigits")
X, y, names = load_dataset("letter", max_samples=3000)
```

实测纯 NumPy 的 EM 在这些数据上都能跑（`satimage` 6430×36 约 1.9 s，`pendigits` 2000 条约 0.1 s）。耗时主要由维数 $D$ 决定而非样本量 $N$，所以推荐用 `pendigits`（样本量大、维度低）和 `optdigits`（维数高）来做规模对比实验。细节见 `docs/02` 第 7 节。

## 建议的动手顺序

先读 `docs/01`，重点看第 2.3 节（k-means 为什么必收敛）、第 4.2 节（Jensen 不等式怎么给出下界）、第 4.5 节（单调性证明的三步不等式）。然后按 `docs/02` 的任务编号依次实现：`normal_logpdf` → k-means 四个函数 → `logsumexp` → E 步 → M 步 → 初始化 → `gmm_em` → 似然校验，最后补 `run_experiment.py` 里的实验步骤。

每个模块都带 `__main__` 自检，写完一个跑一个：

```bash
cd code
python data_gen.py    # 数据生成与经验统计量
python metrics.py     # 指标函数的边界情形
python kmeans.py      # k-means 全部验收标准
python gmm.py         # EM 单调性、参数恢复、协方差形态
python run_experiment.py   # 完整流程与出图
```

## 实验环境

Python 3.13.12（Anaconda），NumPy 2.5.2、Matplotlib 3.11.0。未安装 scipy 与 sklearn，也不需要——全部算法手写，不使用任何自动微分工具。

## 参考实现的实测结果

这些数值由一份独立参考实现跑出，用于校验你的实现是否落在同一量级（细节见 `docs/02` 各节的验收标准）：

- k-means 在合成数据上 `seed=0` 迭代 5 轮收敛，$J=3570.5$，准确率 1.0000；`seed=3` 掉进局部最优，$J=18063.7$，准确率 0.539。
- GMM 的对数似然从 $-8189.31$ 升到 $-5671.24$，严格单调；估计参数与按真实标签算的经验统计量一致到 $10^{-9}$ 量级。
- 设定值代入似然得 $-5677.78$，低于估计参数的 $-5671.24$，证明 EM 找到的是这份样本上的最大似然解。
- 协方差形态恢复：分量长短轴比 1.10／10.63／6.96（设定 1.00／10.00／7.00），主轴 89.9°／0.9°／43.7°（设定 90°／0°／45°）。
- UCI：Iris 花瓣两维 GMM 的 ARI 0.9222（k-means 0.8510），Iris 四维 0.9039（k-means 0.7302），Seeds 七维 0.8123（k-means 0.7733）。
