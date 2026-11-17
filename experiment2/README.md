# 实验二：逻辑回归（从零实现）

用纯 NumPy 实现逻辑回归，包含批量梯度下降与牛顿法（IRLS），覆盖无惩罚负对数似然和 L2 惩罚两种损失；并与自实现的高斯朴素贝叶斯（GaussianNB）以及 UCI 真实数据做对比。

## 实验内容

1. 推导逻辑回归的模型、负对数似然损失、L2 惩罚损失、梯度与牛顿法海森矩阵。
2. 从零实现 `LogisticRegression` 类：梯度下降、牛顿法、无惩罚 / L2 惩罚。
3. 在两组**相关系数为 0.8** 的高斯分布数据上训练逻辑回归与 GaussianNB，比较决策边界与测试准确率。
4. 在 UCI Bank Marketing 数据（4,521 条、42 个特征）上完成读取、预处理、训练、测试与指标分析。

## 主要结果

| 数据 | 模型 | 测试准确率 |
|---|---|---|
| 相关高斯（ρ=0.8） | 逻辑回归 | **96.00%** |
| 相关高斯（ρ=0.8） | 高斯朴素贝叶斯 | 87.67% |
| UCI Bank Marketing | 逻辑回归（λ=0.01） | **90.56%** |
| UCI Bank Marketing | 多数类基线（全预测 `no`） | 89.16% |

- 高斯数据上，逻辑回归学到倾斜的线性边界（权重约 `[8.28, -5.31]`），而朴素贝叶斯因忽略特征相关性，边界接近竖直方向。
- Bank 数据类别不平衡（正类约 11.5%），准确率只比多数类基线高约 1.4 个百分点；报告进一步分析了精确率、召回率、F1 与 ROC（AUC ≈ 0.800）。
- 批量梯度下降实现简单但接近最优时收敛慢（高斯数据约 12,152 次迭代）；牛顿法迭代次数少，但需要计算并求解海森矩阵。

## 目录

```text
experiment2/
├── code/
│   ├── logistic_regression.py   # LogisticRegression + GaussianNB 的完整实现
│   ├── main.py                  # 实验主流程
│   └── step1_forward.py … step8.py
│                                # 按步骤拆分的练习脚本与输出图
├── exp2_report/
│   ├── full_experiment.py       # 一键复现全部结果 → results.json + figures
│   ├── make_report.py           # 由 results.json 生成 Markdown 报告
│   ├── results.json             # 全部指标与权重
│   ├── fig1_*.png … fig10_*.png # 10 张插图
│   └── 实验二_逻辑回归_实验报告.md
└── review.md                    # 复盘笔记
```

`data/`（UCI Bank Marketing）未随仓库提交。把 `bank.csv` 放到 `experiment2/data/` 即可复现；数据可从
<https://archive.ics.uci.edu/dataset/222/bank+marketing> 获取。

## 运行

```bash
cd exp2_report
python full_experiment.py      # 生成 results.json 与全部插图
python make_report.py          # 重新生成 Markdown 报告
```

依赖：Python 3.13、NumPy、Matplotlib、pandas。
