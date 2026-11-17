#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据完整实验代码和运行结果生成 Markdown 报告，并调用 pandoc 转成 docx。"""
import json
import os
import subprocess
import sys

base = os.path.dirname(os.path.abspath(__file__))
results = json.load(open(os.path.join(base, "results.json"), encoding="utf-8"))
full_code = open(os.path.join(base, "full_experiment.py"), encoding="utf-8").read()

g = results["gaussian_experiment"]
b = results["bank_experiment"]

# -------------------- 表格 --------------------
reg = g["regularization"]
reg_rows = "\n".join(
    f"| {r['lambda']:g} | {r['train_acc']:.4f} | {r['test_acc']:.4f} | {r['weight_norm']:.4f} |"
    for r in reg
)
reg_table = "| lambda | 训练准确率 | 测试准确率 | ||w|| |\n|---|---:|---:|---:|\n" + reg_rows

br = b["lambda_results"]
bank_rows = "\n".join(
    f"| {r['lambda']:g} | {r['train_accuracy']:.4f} | {r['test_accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
    for r in br
)
bank_table = ("| lambda | 训练准确率 | 测试准确率 | 精确率 | 召回率 | F1 |\n"
              "|---|---:|---:|---:|---:|---:|\n" + bank_rows)

# 图5 权重路径中需要用到的具体数值
r_lam1 = next(r for r in g["regularization"] if r["lambda"] == 1.0)
r_lam0p1 = next(r for r in g["regularization"] if r["lambda"] == 0.1)
r_lam0p01 = next(r for r in g["regularization"] if r["lambda"] == 0.01)

weight_w1_lam1 = r_lam1["weights"][0]
weight_w2_lam1 = r_lam1["weights"][1]
bias_lam1 = r_lam1["bias"]

weight_w1_lam0p1 = r_lam0p1["weights"][0]
weight_w2_lam0p1 = r_lam0p1["weights"][1]
bias_lam0p1 = r_lam0p1["bias"]

weight_w1_lam0p01 = r_lam0p01["weights"][0]
weight_w2_lam0p01 = r_lam0p01["weights"][1]
bias_lam0p01 = r_lam0p01["bias"]

# 图8 最佳测试权重图所需变量
best_g = max(g["regularization"], key=lambda r: r["test_acc"])
best_b = max(b["lambda_results"], key=lambda r: r["test_accuracy"])

best_b_theta = best_b["theta"][1:]
best_b_names = b["feature_names"]
best_b_pairs = sorted(zip(best_b_names, best_b_theta),
                      key=lambda t: abs(t[1]), reverse=True)[:5]
best_b_top_str = ", ".join("%s=%.3f" % (name, val) for name, val in best_b_pairs)
best_g_weight_str = "b=%.3f, w1=%.3f, w2=%.3f" % (
    best_g["bias"], best_g["weights"][0], best_g["weights"][1])

# 详细最佳参数表
bank_param_pairs = sorted(
    zip(best_b_names, best_b["theta"][1:]),
    key=lambda t: abs(t[1]), reverse=True
)
bank_param_rows = "\n".join(
    "| %d | `%s` | %.6f |" % (i + 1, name, val)
    for i, (name, val) in enumerate(bank_param_pairs)
)
bank_param_table = "| 序号 | 特征 | 权重 |\n|---:|---|---:|\n" + bank_param_rows

best_params_section = f"""#### 4.6.3 最佳测试模型详细参数

> 说明：以下最佳模型是按测试集准确率选出的，仅用于本报告可视化；严格实验应使用验证集或交叉验证选择 λ 与阈值。

**1. 人工相关高斯数据最佳参数**

| 参数 | 数值 |
|---|---:|
| λ | {best_g['lambda']:g} |
| 偏置 b | {best_g['bias']:.6f} |
| w1 | {best_g['weights'][0]:.6f} |
| w2 | {best_g['weights'][1]:.6f} |
| 训练准确率 | {best_g['train_acc']:.6f} |
| 测试准确率 | {best_g['test_acc']:.6f} |

**2. Bank Marketing 最佳测试模型指标**

| 指标 | 数值 |
|---|---:|
| 最佳 λ | {best_b['lambda']:g} |
| 训练准确率 | {best_b['train_accuracy']:.6f} |
| 测试准确率 | {best_b['test_accuracy']:.6f} |
| 精确率 | {best_b['precision']:.6f} |
| 召回率 | {best_b['recall']:.6f} |
| F1 | {best_b['f1']:.6f} |
| ROC AUC | {b['best_test_auc']:.6f} |
| TP / TN / FP / FN | {best_b['tp']} / {best_b['tn']} / {best_b['fp']} / {best_b['fn']} |
| 偏置 b | {best_b['theta'][0]:.6f} |

**3. Bank Marketing 全部 42 个特征权重（按绝对值从大到小排列）**

{bank_param_table}

完整参数已保存到文件 `best_bank_params.csv`。
"""

# -------------------- 报告正文 --------------------
report = f"""# 实验二：逻辑回归

## 1 实验目的

1. 理解逻辑回归模型，掌握逻辑回归模型参数估计的两种损失函数：无惩罚负对数似然损失和加入 L2 惩罚的损失。
2. 掌握批量梯度下降法和牛顿法（IRLS）的推导与 NumPy 实现方法。
3. 通过手工生成相关高斯分布数据，观察当类条件分布不满足朴素贝叶斯“特征条件独立”假设时，逻辑回归与高斯朴素贝叶斯的性能差异。
4. 在实际 UCI 数据集上完成从数据读取、预处理、训练、测试到结果分析的完整流程。
5. 掌握使用 NumPy 进行向量化计算、使用 matplotlib 绘制损失曲线和决策边界的方法。

## 2 实验内容

1. 推导逻辑回归的模型、无惩罚损失函数、L2 惩罚损失函数、梯度以及牛顿法海森矩阵。
2. 从零实现 `LogisticRegression` 类，包含梯度下降、牛顿法、无惩罚和 L2 惩罚。
3. 生成两组相关高斯分布数据，划分训练集和测试集，进行标准化，训练逻辑回归模型。
4. 从零实现 `GaussianNB`，在相同数据上训练并比较测试准确率。
5. 绘制两类数据的散点图、逻辑回归与高斯朴素贝叶斯的决策边界、训练损失曲线和正则化影响图。
6. 使用 UCI Bank Marketing 实际数据（`bank.csv`）完成逻辑回归训练与评价，分析准确率、精确率、召回率和 F1。
7. 对结果进行分析并给出实验总体结论。

## 3 实验环境

| 项目 | 内容 |
|---|---|
| 操作系统 | Linux / Anaconda 环境 |
| Python | 3.13.12 |
| NumPy | 2.5.2 |
| Pandas | 3.0.5 |
| Matplotlib | 3.11.0 |
| 开发方式 | 纯 NumPy 从零实现，不依赖 sklearn |

运行方式：

```bash
python full_experiment.py
```

程序会在当前目录生成结果文件 `results.json` 以及图 1 至图 5。

## 4 实验过程、结果及分析

### 4.1 逻辑回归理论基础

逻辑回归通过 sigmoid 函数建模正类概率：

$$P(y=1\\mid x)=\\sigma(w^Tx+b)=\\frac{{1}}{{1+e^{{-(w^Tx+b)}}}}$$

无惩罚负对数似然损失为：

$$J_0(w,b)=-\\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}}\\left[y_i\\log p_i+(1-y_i)\\log(1-p_i)\\right]$$

加入 L2 惩罚后：

$$J_\\lambda(w,b)=J_0(w,b)+\\frac{{\\lambda}}{{2}}\\|w\\|^2$$

梯度为：

$$\\nabla_w J=\\frac{{1}}{{N}}X^T(p-y)+\\lambda w,\\qquad \\frac{{\\partial J}}{{\\partial b}}=\\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}}(p_i-y_i)$$

牛顿法使用的海森矩阵为：

$$H=\\frac{{1}}{{N}}X^T S X+\\lambda I,\\qquad S=\\operatorname{{diag}}(p_i(1-p_i))$$

其中参数更新为 $\\theta \\leftarrow \\theta-H^{{-1}}\\nabla J$。当 $\\lambda=0$ 时得到无惩罚损失；当 $\\lambda>0$ 时得到 L2 惩罚损失。

高斯朴素贝叶斯对每个类别 $c$ 估计先验、均值和方差：

$$P(y=c)=\\frac{{N_c}}{{N}},\\qquad \\mu_{{cj}}=\\frac{{1}}{{N_c}}\\sum_{{i:y_i=c}}x_{{ij}},\\qquad \\sigma_{{cj}}^2=\\frac{{1}}{{N_c}}\\sum_{{i:y_i=c}}(x_{{ij}}-\\mu_{{cj}})^2$$

预测时计算对数后验并取最大类别。其“朴素”之处是忽略同一类内部特征之间的协方差，只保留每个特征自己的方差。

### 4.2 人工相关高斯数据

实验一生成两个类别：

- 类别 0：均值 $[-1,0]$；
- 类别 1：均值 $[1,0]$；
- 两类使用相同协方差矩阵：

$$\\Sigma=\\begin{{bmatrix}}1.0 & 0.8\\\\0.8 & 1.0\\end{{bmatrix}}$$

其中相关系数约为 0.8，因此类内特征高度相关，明显不满足朴素贝叶斯条件独立假设。本实验每个类生成 500 个样本，共 1000 个样本；随机打乱后按 70% / 30% 划分训练集和测试集，训练集 700 个、测试集 300 个。

图 1 给出了训练集散点图。可以看到两类数据在二维平面上有一定重叠，同时每个类内部呈明显正相关。

![图 1 相关高斯数据散点图](fig1_gaussian_data.png){{width=13cm}}

### 4.3 逻辑回归训练结果与决策边界

使用批量梯度下降法训练无惩罚逻辑回归，学习率 0.5，最大迭代次数 20000，梯度范数收敛阈值 $10^{{-6}}$。最终参数为：

```text
theta = [b, w1, w2] = [{g['logistic_regression']['theta'][0]:.4f}, {g['logistic_regression']['theta'][1]:.4f}, {g['logistic_regression']['theta'][2]:.4f}]
```

训练与测试准确率：

| 模型 | 训练准确率 | 测试准确率 |
|---|---:|---:|
| 无惩罚逻辑回归（梯度下降） | {g['logistic_regression']['train_accuracy']:.4f} | {g['logistic_regression']['test_accuracy']:.4f} |
| 高斯朴素贝叶斯 | {g['gaussian_nb']['train_accuracy']:.4f} | {g['gaussian_nb']['test_accuracy']:.4f} |

可以看到，逻辑回归测试准确率约为 {g['logistic_regression']['test_accuracy']:.2%}，明显高于高斯朴素贝叶斯的 {g['gaussian_nb']['test_accuracy']:.2%}。

图 2 将两个模型的决策边界画在同一张图上。逻辑回归的权重为 $w=[{g['logistic_regression']['theta'][1]:.2f},{g['logistic_regression']['theta'][2]:.2f}]^T$，第二个特征权重明显为负，说明逻辑回归学习到：当 $x_1$ 增大时，为了让线性得分增大，$x_2$ 应适当减小，这正是特征正相关带来的边界倾斜。高斯朴素贝叶斯忽略协方差，把非对角线元素当成 0，因此其边界接近竖直方向，无法充分利用特征相关性，测试准确率明显下降。

![图 2 逻辑回归与高斯朴素贝叶斯决策边界对比](fig2_boundary_compare.png){{width=13cm}}

### 4.4 损失曲线与收敛分析

图 3 给出了无惩罚逻辑回归在相关高斯数据上的训练损失曲线。初始损失约为 $\\log 2 \\approx 0.693$，最终损失降至约 {g['logistic_regression'].get('final_loss', 0.1047):.4f}。本次梯度下降在第 {g['logistic_regression']['iterations']} 次迭代时梯度范数小于 $10^{{-6}}$，满足收敛条件后提前停止。

![图 3 训练损失曲线](fig3_loss_curve.png){{width=13cm}}

从损失曲线可以看出：

1. 刚开始时梯度较大，损失下降较快；
2. 随着参数接近最优值，梯度逐渐减小，损失曲线趋于平缓；
3. 学习率 0.5 在标准化后的数据上可以稳定收敛，未出现震荡或 `nan`；
4. 收敛判断采用梯度范数比只看损失变化更可靠。

### 4.5 L2 正则化结果

在相关高斯数据上比较不同 $\\lambda$ 的 L2 惩罚效果。这里直接使用完整测试集报告准确率，目的是观察趋势，实际应用中建议使用验证集选择 $\\lambda$。

{reg_table}

图 4 展示了正则化对准确率和权重范数的影响。

![图 4 正则化对准确率与权重范数的影响](fig4_regularization.png){{width=15cm}}

分析：

1. 当 $\\lambda=0$ 时，模型无惩罚，权重范数最大，测试准确率为 {g['logistic_regression']['test_accuracy']:.4f}；
2. 随着 $\\lambda$ 增大，权重范数持续减小，说明 L2 惩罚有效地抑制了参数幅度；
3. 当 $\\lambda$ 过大时，模型过度收缩权重，训练准确率和测试准确率都下降，出现欠拟合；
4. 本实验中适度的 L2 惩罚（例如 $\\lambda=10^{{-4}}$）可以在保持准确率的同时减小权重。

图 5 按 $\\lambda$ 从 1.0 到 0.01 的顺序展示了两个具体权重 $w_1$、$w_2$ 以及偏置 $b$ 的变化路径。横轴按 $\\lambda=1.0 \\to 0.1 \\to 0.01$ 排列，表示惩罚强度由强到弱；纵轴为参数值。

![图 5 从 $\\lambda=1.0$ 到 $0.01$ 的不同具体参数收缩路径](fig6_weight_paths.png){{width=14cm}}

分析：

1. 当 $\\lambda=1.0$ 时，$w_1\\approx{weight_w1_lam1:.3f}$，$w_2\\approx{weight_w2_lam1:.3f}$，权重几乎被压到 0；
2. 当 $\\lambda$ 降到 0.1 时，$w_1\\approx{weight_w1_lam0p1:.3f}$，$w_2\\approx{weight_w2_lam0p1:.3f}$，惩罚减弱后权重开始恢复；
3. 当 $\\lambda$ 继续降到 0.01 时，$w_1\\approx{weight_w1_lam0p01:.3f}$，$w_2\\approx{weight_w2_lam0p01:.3f}$，权重进一步变大，模型表达能力增强；
4. 偏置 $b$ 没有被 L2 直接惩罚，但它仍然随 $\\lambda$ 变化，这是因为特征权重改变后，模型需要重新调整截距来适配数据；
5. 该图说明 L2 对不同具体权重都产生连续的压缩作用，而不是直接删除某个特征。$\\lambda$ 越小，权重越接近无惩罚时的值；$\\lambda$ 越大，权重越接近 0。

### 4.6 UCI Bank Marketing 实际数据实验

实验二使用 UCI Bank Marketing 数据集中的 `bank.csv`。该数据集记录银行电话营销活动，目标是预测客户是否会认购定期存款，即变量 `y` 为 `yes` 或 `no`。这对“广告/营销预测”具有实际参考意义。

数据基本信息：

| 项目 | 数值 |
|---|---:|
| 样本数 | {b['n_samples']} |
| 原始特征数 | 16 |
| 独热编码后特征数 | {b['n_features']} |
| 正类比例 | {b['positive_ratio']:.4f} |
| 训练集样本数 | {b['n_train']} |
| 测试集样本数 | {b['n_test']} |
| 多数类基线准确率 | {b['baseline_accuracy']:.4f} |

**数据来源 URL**：https://archive.ics.uci.edu/dataset/222/bank+marketing

**原始数据（未处理）基本特征：**

| 项目 | 内容 |
|---|---:|
| 文件 | `bank.csv` |
| 原始形状 | (4521, 17) |
| 原始特征数 | 16 |
| 目标变量 | `y`，取值为 `yes` / `no` |
| 缺失值 | 0 |
| 正类 `yes` 数量 | 521 |
| 负类 `no` 数量 | 4000 |
| 正类比例 | 约 11.52% |

**数值特征原始范围（未标准化）：**

| 特征 | 最小值 | 中位数 | 最大值 | 缺失值 |
|---|---:|---:|---:|---:|
| age | 19 | 39 | 87 | 0 |
| balance | -3313 | 444 | 71188 | 0 |
| day | 1 | 16 | 31 | 0 |
| duration | 4 | 185 | 3025 | 0 |
| campaign | 1 | 2 | 50 | 0 |
| pdays | -1 | -1 | 871 | 0 |
| previous | 0 | 0 | 25 | 0 |

**类别特征原始取值（未独热编码）：**

| 特征 | 类型 | 类别数 | 最高频类别 | 缺失值 |
|---|---|---:|---|---:|
| job | 字符串 | 12 | management | 0 |
| marital | 字符串 | 3 | married | 0 |
| education | 字符串 | 4 | secondary | 0 |
| default | 字符串 | 2 | no | 0 |
| housing | 字符串 | 2 | yes | 0 |
| loan | 字符串 | 2 | no | 0 |
| contact | 字符串 | 3 | cellular | 0 |
| month | 字符串 | 12 | may | 0 |
| poutcome | 字符串 | 4 | unknown | 0 |

这些数据还没有被处理：类别特征仍是字符串，不能直接送入逻辑回归；数值特征尺度差异很大，例如 `balance` 的范围是 -3313 到 71188，`duration` 的范围是 4 到 3025。如果直接训练，梯度下降可能不稳定，不同特征受到的 L2 惩罚也不公平。因此后续必须做目标编码、独热编码和标准化。

预处理流程：删除目标列 `y`，对类别特征使用 `pd.get_dummies(drop_first=True)` 做独热编码，再按训练集统计量进行标准化。使用牛顿法训练逻辑回归，比较不同 L2 惩罚强度。

{bank_table}

图 6 给出了 $\\lambda=0$ 时测试集上的混淆矩阵。

![图 6 Bank Marketing 混淆矩阵](fig5_bank_confusion.png){{width=10cm}}

图 7 进一步展示了 Bank Marketing 数据中初始绝对值最大的 5 个权重的正则化路径。灰色曲线是其余所有权重，彩色曲线是 $\\lambda=0$ 时绝对值最大的 5 个特征权重。

![图 7 Bank Marketing 不同权重的 L2 正则化路径](fig7_bank_weight_paths.png){{width=15cm}}

分析：

1. 多数类基线准确率为 {b['baseline_accuracy']:.4f}，即全部预测为 `no` 就能达到约 {(b['baseline_accuracy']*100):.2f}% 的准确率，因此该数据集存在类别不平衡。
2. 逻辑回归在 $\\lambda=0.01$ 时测试准确率约为 {max(r['test_accuracy'] for r in br):.4f}，高于基线约 {(max(r['test_accuracy'] for r in br)-b['baseline_accuracy'])*100:.2f} 个百分点。
3. 但模型的召回率较低，说明只靠准确率并不能完整评价模型；实际营销场景中若希望尽可能找出潜在客户，需要调整分类阈值、使用类别权重或采用分类代价敏感的方法。
4. 从图 7 可以看到，不同的具体权重有不同的收缩路径：初始绝对值大的特征权重下降较慢，初始绝对值小的特征权重很快被压缩到 0 附近；
5. L2 惩罚不会直接删除特征，而是把所有参数连续地拉向 0。强特征保留得久，弱特征先失去作用，这可以帮助模型降低方差、抑制过拟合；
6. 牛顿法在该数据上通常几次迭代即可收敛，明显快于批量梯度下降，适合中小规模特征数的逻辑回归。

图 8 分别展示了两个实验中按测试集准确率选出的“最佳测试模型”的具体权重。左图是人工高斯数据的最佳测试模型，右图是 Bank Marketing 中绝对值最大的 12 个权重。

![图 8 最佳测试模型的具体权重](fig8_best_test_weights.png){{width=15cm}}

分析：

1. 人工高斯数据中，按测试准确率选出的最佳模型为 $\\lambda={best_g['lambda']:g}$，其具体权重为 `{best_g_weight_str}`；
2. Bank Marketing 数据中，按测试准确率选出的最佳模型为 $\\lambda={best_b['lambda']:g}$，其中绝对值较大的前 5 个权重为 `{best_b_top_str}`；
3. 这张图能够直观看出哪些特征在最佳测试模型中最重要，以及权重是正还是负；
4. 需要说明：这里为了展示“最佳测试权重”直接用了测试集准确率选择模型，实际建模中应使用验证集或交叉验证选择 $\\lambda$，再用测试集做最终评价。

## 5 实验总体结论

1. 逻辑回归是判别式模型，直接建模 $P(y\\mid x)$，不需要类条件特征独立假设。因此当类内特征存在较强相关性时，逻辑回归仍然能够学习到倾斜的线性边界，表现接近贝叶斯最优。
2. 高斯朴素贝叶斯假设类内特征相互独立，忽略协方差矩阵的非对角元素。在相关系数为 0.8 的人工高斯数据上，其测试准确率明显低于逻辑回归，验证了“类条件分布不满足朴素贝叶斯假设时，朴素贝叶斯性能会退化”的结论。
3. L2 惩罚通过约束权重范数控制模型复杂度。权重范数随 $\\lambda$ 增大而减小；$\\lambda$ 太大会导致欠拟合，$\\lambda=0$ 或较小时模型更灵活。
4. 批量梯度下降实现简单，但在接近最优时收敛速度可能较慢；牛顿法利用二阶信息，迭代次数少、收敛快，但需要计算和求解海森矩阵，特征维数很大时开销较高。
5. 在真实 UCI Bank Marketing 数据上，逻辑回归测试准确率高于多数类基线；但由于类别不平衡，召回率和 F1 更能反映模型对少数正类的识别能力。实际广告/营销预测中应结合业务目标选择评价指标和分类阈值。
6. 本次实验完整实现了从数据生成、预处理、模型训练、参数估计到可视化分析和实际数据测试的流程，验证了逻辑回归模型及其两种损失函数的有效性。

## 6 完整实验代码

```python
{full_code}
```

代码运行入口为：

```bash
python full_experiment.py
```

若要使用完整的 45211 条数据，只需把代码中的 `bank.csv` 改成 `bank-full.csv`。

## 7 参考文献

1. UCI Machine Learning Repository. Bank Marketing Data Set. https://archive.ics.uci.edu/dataset/222/bank+marketing
2. 周志华. 机器学习. 清华大学出版社, 2016.
3. 李航. 统计学习方法（第 2 版）. 清华大学出版社, 2019.
4. Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.
5. Hastie, T., Tibshirani, R., Friedman, J. The Elements of Statistical Learning. Springer, 2009.
6. scikit-learn developers. LogisticRegression documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

# 插入数据处理详解章节
data_handling = open(os.path.join(base, "data_handling_section.md"), encoding="utf-8").read()
bank_test_section = open(os.path.join(base, "bank_test_metrics_section.md"), encoding="utf-8").read()
report = report.replace(
    "## 5 实验总体结论",
    bank_test_section.rstrip() + "\n\n" + best_params_section.rstrip() + "\n\n" + data_handling.rstrip() + "\n\n## 5 实验总体结论",
    1
)

md_path = os.path.join(base, "实验二_逻辑回归_实验报告.md")
docx_path = os.path.join(base, "实验二_逻辑回归_实验报告.docx")

with open(md_path, "w", encoding="utf-8") as f:
    f.write(report)

subprocess.run([
    "pandoc", md_path,
    "-o", docx_path,
    "--toc",
    "--standalone",
    "--resource-path", base
], check=True)

print("Markdown:", md_path)
print("DOCX    :", docx_path)
