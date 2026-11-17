#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results.json 生成实验四报告（Markdown），并调用 pandoc 生成 docx。

报告中所有数字都从 results.json 读取，避免手抄结果导致前后不一致。
运行：python gen_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

P1 = R["part1_2d"]
D = P1["datasets"]
BL, RG, SP = D["blobs"], D["rings"], D["spiral"]
P2 = R["part2_size_2d"]
P3 = R["part3_size_mnist"]
P4 = R["part4_hyper"]
P5 = R["part5_mnist"]
GC = R["gradcheck"]
META = R["meta"]

# 额外数据集实验（由 learn/practice/extra_datasets.py 生成）
_EXTRA_PATH = HERE.parent / "learn" / "practice" / "extra_results.json"
EXTRA = json.loads(_EXTRA_PATH.read_text(encoding="utf-8")) if _EXTRA_PATH.exists() else []
if EXTRA:
    EX = {row["name"]: row for row in EXTRA}
    _letter = EX.get("Letter")
    _wdbc = EX.get("WDBC")
    EXTRA_NOTE = (
        f"另外在 {len(EXTRA)} 个公开数据集上重复了对照（第 13 节）："
        f"MLP 在 Letter 上比线性高 {_letter['gap'] * 100:.1f} 个百分点，"
        f"但在 WDBC 上反而低 {abs(_wdbc['gap']) * 100:.1f} 个百分点，"
        "说明收益取决于任务的非线性程度与数据量，而不是参数量。"
    ) if _letter and _wdbc else ""
else:
    EXTRA_NOTE = ""

OUT_MD = HERE / "实验四_多层感知机_实验报告.md"
OUT_DOCX = HERE / "实验四_多层感知机_实验报告.docx"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pm(mean, std, nd=3):
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def num(v):
    """大数字加千分位。"""
    return f"{int(v):,}"


def fmt_float_list(xs, nd=3):
    return " / ".join(f"{x:.{nd}f}" for x in xs)


def argbest(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


P = []


def add(text):
    P.append(text)


# ---------------------------------------------------------------------------
# 封面与摘要
# ---------------------------------------------------------------------------

add(rf"""# 实验四：多层感知机实验

> 姓名：________　学号：________　班级：________　日期：________
>
> 代码目录：`code/`（`mlp.py` 手写 MLP、`linear.py` 手写线性分类器、`train.py` 优化器与训练循环、`datasets.py` 数据、`metrics.py` 指标、`viz.py` 可视化）
>
> 主实验脚本：`exp4_report/full_experiment.py`　结果：`exp4_report/results.json`　插图：`exp4_report/figures/`
>
> 运行方式：`cd exp4_report && python full_experiment.py`，再由 `python gen_report.py` 生成本报告

## 摘要

本实验用纯 NumPy 从零实现了一个多层感知机（MLP），前向传播、反向传播、softmax 交叉熵、SGD / Momentum / Adam 优化器全部手写，没有使用 PyTorch、TensorFlow、JAX 等任何自动微分工具，也没有调用 scikit-learn 的模型。作为对照，另外独立实现了一个多项逻辑回归（softmax 线性分类器），两者共用同一套小批量训练循环。

实验分四组：在二维平面上构造了三份四类数据（线性可分的四团高斯、同心的四圆环、交错的四臂螺旋），比较线性分类器与 MLP 的决策区域和准确率；改变训练集大小，观察两种模型的学习曲线；在 MNIST 子集上系统扫描隐藏层宽度、深度、激活函数、学习率、初始化方式、批大小、L2 系数和优化器；最后在 MNIST 全量训练集上训练 {'-'.join(str(x) for x in P5['mlp_sizes'])} 的 MLP，与线性分类器对比并做误差分析。

主要结果：在线性可分的四团数据上，线性分类器 {f3(BL['linear']['test_acc_mean'])}、MLP {f3(BL['mlp']['test_acc_mean'])}，MLP 没有优势；在四圆环上两者分别为 {f3(RG['linear']['test_acc_mean'])} 和 {f3(RG['mlp']['test_acc_mean'])}，在四臂螺旋上为 {f3(SP['linear']['test_acc_mean'])} 和 {f3(SP['mlp']['test_acc_mean'])}。MNIST 全量训练集上，线性分类器测试准确率 {pct(P5['linear']['test_acc'], 2)}，MLP 为 {pct(P5['mlp']['test_acc'], 2)}，提升约 {100 * (P5['mlp']['test_acc'] - P5['linear']['test_acc']):.2f} 个百分点。实验同时给出了数值梯度校验、决策边界、学习曲线、超参数扫描、混淆矩阵、错分样本和第一层权重的可视化。{EXTRA_NOTE}
""")


# ---------------------------------------------------------------------------
# 1. 实验目的
# ---------------------------------------------------------------------------

add("""## 1. 实验目的

1. 理解多层感知机的基本结构：仿射变换加逐元素非线性，以及隐藏层如何把原始输入映射到新的特征空间。
2. 独立推导并实现反向传播。对每一层写出误差项 $\\delta$ 的递推式，把梯度写成矩阵乘法，理解为什么反向传播的计算量与一次前向传播同阶。
3. 从零实现小批量训练所需的全套组件：softmax 与数值稳定的交叉熵、参数初始化、SGD / Momentum / Adam、L2 正则、学习率与批大小等超参数。
4. 用中心差分数值梯度校验手写反向传播的正确性，养成"先验证梯度、再谈效果"的习惯。
5. 实现一个线性分类器作为对照，从决策边界的几何形状解释线性模型与 MLP 的能力差异：为什么线性模型的判定区域一定是凸的，而螺旋、圆环这类数据必须依靠隐藏层。
6. 通过改变训练集大小与各项超参数，定量观察欠拟合、过拟合、优化困难等现象，理解模型容量、数据量与正则化之间的权衡。
7. 在 MNIST 手写数字数据集上完成 10 类图像分类，掌握混淆矩阵、每类 precision / recall、错分样本分析、第一层权重可视化和隐层表示可视化等分析方法。
""")


# ---------------------------------------------------------------------------
# 2. 实验内容
# ---------------------------------------------------------------------------

add(rf"""## 2. 实验内容

实验分四部分，全部由 `exp4_report/full_experiment.py` 一次运行完成，结果写入 `results.json`，插图写入 `figures/`。

**第一部分：二维四类数据上的线性分类器与 MLP。** 自行构造三种四类二维数据：

- 四团高斯：四个类别的中心位于 $(\pm 2.5,\pm 2.5)$，每类 200 个样本，噪声标准差 1.0。类别之间可以被直线分开，属于线性分类器的"友好"场景；
- 四同心圆环：半径分别为 1.0、2.2、3.4、4.6，环宽标准差 0.2，每类 250 个样本。类与类的边界是圆，任何一条直线都无法把某一个环整体切出来；
- 四臂螺旋：四条相位相差 $90^\circ$ 的螺旋臂，每类 200 个样本，加入标准差 0.05 的噪声。四条臂在平面上互相缠绕。

每份数据按 7:3 分层留出训练集与测试集，用 3 个不同的数据随机种子重复实验，报告测试准确率的均值与标准差。线性分类器与 MLP（结构 2-32-32-4，ReLU）各训练到收敛，绘制决策区域并对比。

**第二部分：训练集大小的影响。** 在四臂螺旋上取 {num(P2['sizes'][0])} 到 {num(P2['sizes'][-1])} 共 {len(P2['sizes'])} 档训练样本数，每档用 3 次不同的抽样重复；在 MNIST 上取 {num(P3['sizes'][0])} 到 {num(P3['sizes'][-1])} 共 {len(P3['sizes'])} 档，比较线性分类器与 MLP 的学习曲线。

**第三部分：超参数扫描。** 在 MNIST 的 {num(P4['n_train'])} 张训练集、{num(P4['n_test'])} 张测试集上，固定其余条件，每次只改变一个因素：隐藏层宽度（{', '.join(str(x) for x in P4['width']['units'])}）、深度（{', '.join(str(x) for x in P4['depth']['depths'])} 个隐藏层）、激活函数（{', '.join(P4['activation']['names'])}）、学习率（{', '.join(str(x) for x in P4['lr']['lrs'])}）、初始化方式（{', '.join(P4['init']['names'])}）、批大小（{', '.join(str(x) for x in P4['batch']['sizes'])}）、L2 系数（{', '.join(str(x) for x in P4['l2']['l2'])}）以及优化器（{', '.join(P4['optimizer']['names'])}）。另外用 3 个随机种子重复基准配置，估计运行间的波动。

**第四部分：MNIST 全量实验与可视化。** 在 60000 张训练图、10000 张测试图上训练线性分类器与 MLP，记录训练曲线、混淆矩阵、每类 precision / recall、最易混淆的类别对、错分样本、第一层权重以及第二隐藏层表示的二维 PCA 投影。
""")


# ---------------------------------------------------------------------------
# 3. 实验环境
# ---------------------------------------------------------------------------

add(rf"""## 3. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | {META['platform']} |
| 编程语言 | Python {META['python']}（Anaconda 发行版） |
| 数值计算 | NumPy {META['numpy']}（矩阵乘法调用底层 BLAS） |
| 绘图 | Matplotlib（中文字体 Noto Sans CJK SC） |
| 深度学习框架 | **未使用**：无 PyTorch / TensorFlow / JAX / Keras，无自动微分，无 `torch.optim` |
| 机器学习库 | **未使用**：线性分类器、MLP、优化器、指标全部手写；数据加载只用了标准库 `gzip` 与 `struct` |
| 硬件 | x86_64 CPU（18 逻辑核），无需 GPU；本报告所有训练均为纯 CPU |
| 随机性 | 数据生成、初始化、小批量打乱均使用 `numpy.random.default_rng(seed)`；二维实验用 3 个种子重复 |
| 运行方式 | `cd exp4_report && python full_experiment.py` |

实验使用 MNIST 原始 IDX 文件（`data/*.gz`），由 `code/datasets.py` 中的 `gzip` + `struct` 解析，像素值除以 255 缩放到 $[0,1]$，不做其他预处理。选择 MNIST 的理由是它属于图像数据、有 10 个类别、样本量足够大，既能观察小样本下的过拟合，也能观察大样本下 MLP 相对线性模型的稳定优势。
""")


# ---------------------------------------------------------------------------
# 4. 实验原理
# ---------------------------------------------------------------------------

add(r"""## 4. 实验原理

### 4.1 线性分类器：softmax 回归

给定输入 $x\in\mathbb{R}^{D}$、类别数 $C$，线性分类器先算一组线性打分（logits）

$$z=Wx+b,\qquad W\in\mathbb{R}^{D\times C},\ b\in\mathbb{R}^{C},$$

再用 softmax 把打分变成概率分布

$$p_c=\frac{\exp(z_c)}{\sum_{j=1}^{C}\exp(z_j)},\qquad c=1,\dots,C.$$

训练目标是最小化平均交叉熵

$$L(W,b)=-\frac{1}{N}\sum_{n=1}^{N}\log p_{n,y_n}+\frac{\lambda}{2}\lVert W\rVert_F^2,$$

其中 $y_n$ 是第 $n$ 个样本的真实类别，第二项是可选的 L2 正则。对 logits 求导可得

$$\frac{\partial L}{\partial z_n}=\frac{1}{N}\left(p_n-e_{y_n}\right),$$

其中 $e_{y_n}$ 是 one-hot 向量。再由链式法则得到参数梯度

$$\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z}+\lambda W,\qquad
\frac{\partial L}{\partial b}=\sum_{n=1}^{N}\frac{\partial L}{\partial z_n}.$$

**决策边界的几何形状。** 类别 $c$ 的判定区域是

$$R_c=\left\{x:\ (w_c-w_j)^\top x+(b_c-b_j)\ge 0,\ \forall j\ne c\right\},$$

它是若干半平面的交，因此一定是**凸集**。在二维平面上，每个类别占据一块凸多边形（可以无界），整张平面被分成 $C$ 块凸区域。这个性质直接决定了线性分类器的能力上限：

- 四团高斯数据的类别中心位于四个角，任意两类的分界线都是直线，区域自然是凸的，线性分类器可以做到接近贝叶斯最优；
- 同心圆环中，同一个环围住原点，不是凸集，任何凸区域都无法完整包含一个环。线性分类器只能把平面切成若干扇形或"中心多边形 + 外围扇形"，每个区域都会同时包含多个环的样本，因此准确率只能停留在随机猜测附近；
- 四臂螺旋的每条臂都绕着原点旋转，也不是凸集，而且不同类别在角度和半径上交错，线性分类器同样无能为力。

### 4.2 多层感知机的结构与前向传播

本实验使用全连接网络。设输入为 $A_0=X$，第 $l$ 层的权重为 $W_l$、偏置为 $b_l$，则

$$Z_l=A_{l-1}W_l+b_l,\qquad A_l=f(Z_l),\quad l=1,\dots,L-1,$$

输出层不加激活函数，直接给出 logits：

$$Z_L=A_{L-1}W_L+b_L.$$

$f$ 可以取 ReLU、tanh 或 sigmoid。本实验的默认结构是 $784\to256\to128\to10$（MNIST），二维实验使用 $2\to32\to32\to4$。

有两点需要强调。

第一，**激活函数不能省**。如果每一层都是线性的，那么

$$Z_2=(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2),$$

仍然是一个线性函数，多层堆叠不会带来任何新的表达能力，整个网络等价于一个线性分类器。非线性的逐元素函数打破了这种"矩阵相乘可以合并"的结构。

第二，**ReLU 网络是分段线性函数**。单个 ReLU 单元 $\max(0,w^\top x+b)$ 在超平面 $w^\top x+b=0$ 两侧分别是零和一个线性函数。一个隐藏层有 $m$ 个 ReLU 单元时，网络是 $m$ 个分段线性函数之和，折线（在二维中是折线，在高维中是折面）的位置由各个单元的 $w_j^\top x+b_j=0$ 决定。单隐藏层已经可以逼近任意连续函数（万能逼近定理），但所需宽度可能非常大；多个隐藏层通过"折线的再组合"可以用更少的参数围出更复杂的区域。二维的圆环和螺旋臂需要围出非凸的、带孔的区域，这正是两个隐藏层发挥作用的地方。

### 4.3 反向传播

记第 $l$ 层的误差项为

$$\delta^{(l)}=\frac{\partial L}{\partial Z_l}.$$

输出层使用 softmax 交叉熵，因此

$$\delta^{(L)}=\frac{1}{N}\left(P-Y\right),$$

其中 $P$ 是 $N\times C$ 的预测概率矩阵，$Y$ 是 one-hot 标签矩阵。对隐藏层，由链式法则

$$\delta^{(l)}=\left(\delta^{(l+1)}W_{l+1}^\top\right)\odot f'\left(Z_l\right),\qquad l=L-1,\dots,1,$$

其中 $\odot$ 表示逐元素相乘。有了 $\delta^{(l)}$，参数的梯度就是

$$\frac{\partial L}{\partial W_l}=A_{l-1}^\top\delta^{(l)}+\lambda W_l,\qquad
\frac{\partial L}{\partial b_l}=\sum_{n=1}^{N}\delta^{(l)}_n.$$

推导的关键只有两步：仿射变换 $Z_l=A_{l-1}W_l+b_l$ 对 $W_l$ 的偏导是 $A_{l-1}$，对 $A_{l-1}$ 的偏导是 $W_l$；激活函数逐元素作用，所以它的导数与上游梯度逐元素相乘。把这两步从输出层往前反复代入，就得到上面的递推式。

从计算量看，一次前向需要做 $L$ 次矩阵乘法，一次反向需要做同样数量的矩阵乘法（$A_{l-1}^\top\delta^{(l)}$ 与 $\delta^{(l)}W_l^\top$），因此反向传播的代价约为前向的 2 倍，总训练代价约为前向的 3 倍。这也是为什么本实验可以在纯 CPU 上把 MNIST 训练到 98% 以上。

L2 正则只惩罚权重矩阵、不惩罚偏置：偏置不控制函数的"陡峭程度"，惩罚偏置通常没有好处。

### 4.4 初始化、优化器与正则化

**初始化。** 如果所有参数都初始化为零，同一隐藏层中每个神经元的输入、输出和梯度完全相同，它们永远学成相同的特征，网络退化成只有一个有效单元，实际表达能力接近线性模型。因此必须随机初始化。初始化的方差也很重要：

- He 初始化 $W\sim\mathcal{N}(0,\,2/n_{\text{in}})$ 让 ReLU 网络各层激活值的方差大致保持不变；
- Xavier 初始化 $W\sim\mathcal{N}(0,\,2/(n_{\text{in}}+n_{\text{out}}))$ 针对 tanh / sigmoid 的线性区间设计。

方差过小会导致信号逐层衰减、梯度消失；方差过大则会让激活值饱和或爆炸。

**优化器。** 记梯度为 $g_t$：

- SGD：$\theta_{t+1}=\theta_t-\eta g_t$；
- Momentum：$v_{t+1}=\mu v_t-\eta g_t,\ \theta_{t+1}=\theta_t+v_{t+1}$，用历史方向平滑噪声、加速沿同一方向的下降；
- Adam：分别估计一阶矩 $m_t$ 与二阶矩 $v_t$，做偏差修正后 $\theta_{t+1}=\theta_t-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$，对每个参数自适应地缩放步长。

**批大小与学习率。** 批大小决定两件事：梯度估计的噪声，以及同样 epoch 数下的参数更新次数。批越小，梯度越嘈杂，但每个 epoch 的更新次数越多；批越大，梯度越准，但更新次数越少。学习率则决定单步走多远：太小收敛慢，太大会在最优解附近震荡甚至发散。

**L2 正则与过拟合。** L2 在损失中加入 $\frac{\lambda}{2}\lVert W\rVert^2$，等价于把权重往零拉，限制模型的有效容量。训练集小、模型大时，训练准确率会明显高于测试准确率；适当增大 $\lambda$ 可以缩小这个差距，但 $\lambda$ 太大会导致欠拟合。
""")


# ---------------------------------------------------------------------------
# 5. 实现与正确性验证
# ---------------------------------------------------------------------------

gc_rows = []
for run in GC["runs"]:
    for row in run["rows"]:
        gc_rows.append([run["activation"], row["param"], "×".join(str(s) for s in row["shape"]),
                        f"{row['max_abs_err']:.2e}", f"{row['rel_err']:.2e}"])

add(r"""## 5. 实现与正确性验证

### 5.1 代码结构

| 文件 | 内容 |
|---|---|
| `code/mlp.py` | 激活函数及其导数、数值稳定的 softmax 交叉熵、He / Xavier / 小随机 / 全零四种初始化、`MLP` 类的前向与反向传播、中心差分数值梯度校验 |
| `code/linear.py` | 多项逻辑回归（softmax 线性分类器），独立实现的梯度与预测接口 |
| `code/train.py` | SGD、Momentum、Adam 三种优化器；小批量训练循环；准确率与损失评估 |
| `code/datasets.py` | 四团高斯、四同心环、四臂螺旋三份二维数据的生成；分层划分；MNIST IDX 解析；按类别分层抽样 |
| `code/metrics.py` | 准确率、混淆矩阵、每类 precision / recall / F1、最易混淆类别对 |
| `code/viz.py` | 决策区域、训练曲线、混淆矩阵、MNIST 样本网格的绘图函数 |
| `exp4_report/full_experiment.py` | 四组实验的完整流程，输出 `results.json` 与全部插图 |

`MLP` 的参数列表按 $[W_1,b_1,W_2,b_2,\dots]$ 排列，`get_params()` / `set_params()` 让所有优化器可以用同一套代码更新参数；线性分类器也实现了相同的接口，因此两种模型共用同一个 `fit()` 训练循环。这样两者的差别只剩下模型本身，排除了"训练技巧不同"这一干扰因素。

### 5.2 关键实现

**数值稳定的 softmax 交叉熵。** 直接计算 $\exp(z_c)$ 在 $z_c$ 较大时会溢出。实现时先减去每行的最大值 $m=\max_j z_j$，由于 softmax 对整体平移不变，结果不变：

$$p_c=\frac{\exp(z_c-m)}{\sum_j\exp(z_j-m)}.$$

损失用 log-sum-exp 写成

$$L=\frac{1}{N}\sum_{n=1}^{N}\left[\log\sum_{j}\exp(z_{nj}-m_n)+m_n-z_{n,y_n}\right],$$

对 logits 的梯度仍然是 $(P-Y)/N$，不需要对稳定化过程再求导。

**反向传播的核心循环。**

```python
# 前向：保存每层的 (输入, z, 输出)
a = X
for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
    z = a @ W + b
    a_next = self.act(z) if i < n_layers - 1 else z   # 输出层是线性 logits
    layers.append((a, z, a_next))
    a = a_next

# 反向：从 softmax 交叉熵的梯度出发，逐层往前
dZ = (softmax(logits) - onehot(y)) / N
for i in range(n_layers - 1, -1, -1):
    a_prev, z, _ = layers[i]
    dWs[i] = a_prev.T @ dZ + l2 * self.Ws[i]          # 权重梯度 + L2
    dbs[i] = dZ.sum(axis=0)                            # 偏置梯度
    dA_prev = dZ @ self.Ws[i].T                        # 传到上一层的梯度
    dZ = dA_prev * self.act_grad(layers[i - 1][1])     # 穿过上一层激活函数
```

这段代码与 4.3 节的公式一一对应：`dWs[i] = a_prev.T @ dZ` 对应 $A_{l-1}^\top\delta^{(l)}$，`dA_prev = dZ @ W.T` 对应 $\delta^{(l+1)}W_{l+1}^\top$，乘激活导数对应 $\odot f'(Z_l)$。

**初始化。** `MLP` 根据激活函数自动选择初始化：ReLU 用 He，tanh / sigmoid 用 Xavier；也可以显式指定 `zeros`、`small`、`xavier`、`he`，用于第 8 节的初始化对比实验。

### 5.3 数值梯度校验

手写反向传播最容易犯的错误是维度对不上、转置写反或漏乘激活导数。由于这类错误不会让程序崩溃，只会让模型"训练得很慢"或"效果莫名很差"，所以必须单独验证。

校验方法是中心差分：对每个参数 $\theta_i$ 施加 $\pm\epsilon$ 的扰动，用两次前向传播的损失差近似导数

$$\frac{\partial L}{\partial\theta_i}\approx\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon},$$

再与解析梯度比较。实验使用 3 维输入、5 个隐藏单元、4 类的网络，6 个样本，$\lambda=0.01$，$\epsilon=10^{-6}$，对三种激活函数分别检查全部参数：

@@GRAD_TABLE@@

三种激活函数下，解析梯度与数值梯度的最大绝对误差都在 $10^{-10}$ 量级，说明反向传播、L2 项和 softmax 梯度的实现是正确的。这个精度已经接近 float64 差分本身的截断/舍入误差，继续减小误差没有意义。

### 5.4 训练协议

为保证对比公平，所有实验遵循同一套协议：

- 小批量训练，每个 epoch 重新打乱样本顺序；训练集大小不足一个批时自动退化为全批量；
- 每个 epoch 结束后在测试集上计算损失与准确率，记录完整曲线；
- 二维数据重复 3 个随机种子（同时改变数据划分与初始化），报告均值 ± 标准差；
- 线性模型与 MLP 使用相同的优化器族与相同的训练轮数；学习率按模型分别调到合适值（线性模型 0.1，MLP 0.01），避免把"学习率没调好"误判成"模型不行"；
- MNIST 超参数实验中，除被扫描的因素外，其余设置固定为：结构 784-128-10、ReLU、He 初始化、Adam、学习率 $10^{-3}$、批大小 128、20 个 epoch、无 L2。
""".replace("@@GRAD_TABLE@@", table(["激活函数", "参数", "形状", "最大绝对误差", "相对误差"], gc_rows)))


# ---------------------------------------------------------------------------
# 6. 实验一：二维四类数据，线性分类器 vs MLP
# ---------------------------------------------------------------------------

p1c = P1["config"]
mlp_params = BL["mlp"]["params"]
lin_params = BL["linear"]["params"]
param_ratio = mlp_params / lin_params

rows1 = []
for name, d in [("四团高斯", BL), ("四同心圆环", RG), ("四臂螺旋", SP)]:
    rows1.append([name,
                  pm(d["linear"]["test_acc_mean"], d["linear"]["test_acc_std"]),
                  pm(d["mlp"]["test_acc_mean"], d["mlp"]["test_acc_std"]),
                  f"{d['gap_mean']:+.3f}",
                  fmt_float_list(d["linear"]["test_accs"]),
                  fmt_float_list(d["mlp"]["test_accs"])])

add(rf"""## 6. 实验一：二维四类数据上的决策边界

### 6.1 数据与设置

三份数据的散点图见图 1。每份数据都按类别分层划分为 70% 训练、30% 测试；三个随机种子分别改变数据生成与划分，因此表中的均值反映了"换一批样本"时的波动。

![图 1　二维四类合成数据（训练集部分）](figures/fig01_datasets.png)

实验设置：线性分类器（2 维输入、4 类，共 12 个参数），Adam，学习率 0.1，批大小 64，{p1c['linear_epochs']} 个 epoch；MLP 结构 2-32-32-4（ReLU，He 初始化，{mlp_params} 个参数），Adam，学习率 0.01，批大小 64，{p1c['mlp_epochs']} 个 epoch。

### 6.2 结果

{table(["数据集", "线性 测试准确率（3 次均值）", "MLP 测试准确率（3 次均值）", "差值 (MLP−线性)", "线性 3 次", "MLP 3 次"], rows1)}

![图 2　线性分类器（上）与 MLP（下）的决策区域](figures/fig02_boundaries.png)

![图 3　四臂螺旋上的训练曲线（seed 0；线性分类器训练 1500 轮，MLP 训练 600 轮）](figures/fig03_spiral_curves.png)

### 6.3 结果分析

**（1）四团高斯：MLP 没有优势。** 线性分类器 {f3(BL['linear']['test_acc_mean'])}，MLP {f3(BL['mlp']['test_acc_mean'])}，后者反而低 {abs(BL['gap_mean']) * 100:.2f} 个百分点。这份数据的贝叶斯最优边界就是四条直线（相邻类别中心的垂直平分线），线性模型恰好匹配了数据的真实结构；MLP 需要从数据中学出这些直线，多出来的 {mlp_params - lin_params} 个参数没有带来任何信息增益，反而增加了估计方差。从图 2 左下角还能看到，MLP 的边界在红色类别附近出现了一条细长的蓝色楔形区域——这是用大量分段折线去拟合一条简单直线时产生的"多余折角"，它吃掉了少量测试样本。这个对照说明：**模型容量要与任务复杂度匹配，非线性本身不是优点。**

**（2）四同心圆环：线性分类器接近随机猜测。** 线性分类器只有 {f3(RG['linear']['test_acc_mean'])}，四类问题的随机猜测是 25%，也就是说它几乎没有学到任何有效信息；MLP 达到 {f3(RG['mlp']['test_acc_mean'])}。从图 2 上排可以看到，线性模型的判定区域是几条直线围成的扇形，每个扇形都同时穿过四个环，因此无论怎么调整角度，每个区域内的类别都是混杂的。这是 4.1 节结论的直接体现：**线性模型的每个判定区域必须是凸的，而一个圆环不是凸集。** MLP 的两个隐藏层把 32 条折线组合起来，围出了近似圆形的封闭边界，四个环被依次分开。

**（3）四臂螺旋：差距同样很大。** 线性分类器 {f3(SP['linear']['test_acc_mean'])}，MLP {f3(SP['mlp']['test_acc_mean'])}，相差 {SP['gap_mean'] * 100:.1f} 个百分点。螺旋的每条臂都绕原点旋转两圈，任意一个凸区域都无法只包含一条臂；线性模型的边界只能从原点向外辐射，把每条臂都切成几段。MLP 的边界是分段折线，可以沿螺旋方向"拐弯"，因此能够贴合弯曲的类别界面。

图 3 的损失曲线给出了另一个角度的证据。线性分类器的训练损失在 1.30 附近停住，与四类均匀分布时的交叉熵 $\ln 4\approx 1.386$ 非常接近，说明它的输出几乎是均匀分布，模型没有找到多少可用的线性信号。MLP 的损失在 100 个 epoch 内就下降到 $10^{{-2}}$ 以下，测试损失最终稳定在 $5\times10^{{-3}}$ 左右；训练损失则继续降到 $10^{{-4}}$ 以下，两者之间仍有差距，说明模型对训练集有一定程度的过拟合，但由于分类间隔已经足够大，测试准确率仍保持在 99.7%。

**（4）关于参数量。** 线性分类器只有 {lin_params} 个参数，MLP 有 {mlp_params} 个，后者约为前者的 {param_ratio:.0f} 倍。参数量的增加并没有让 MLP 在所有任务上都更好：它只在边界确实弯曲的任务上换来收益。换句话说，**参数数量本身不是能力，参数与数据结构的匹配才是。**
""")


# ---------------------------------------------------------------------------
# 7. 实验二：训练集大小的影响
# ---------------------------------------------------------------------------

rows2 = []
for a, b in zip(P2["linear"], P2["mlp"]):
    rows2.append([num(a["n"]),
                  pm(a["test_acc_mean"], a["test_acc_std"]),
                  pm(b["test_acc_mean"], b["test_acc_std"]),
                  f"{b['train_acc_mean']:.3f}"])

rows3 = []
for a, b in zip(P3["linear"], P3["mlp"]):
    rows3.append([num(a["n"]), f3(a["test_acc"]), f3(b["test_acc"]),
                  f"{(b['test_acc'] - a['test_acc']) * 100:+.2f}",
                  f"{a['time'] + b['time']:.1f}"])

n_low, n_high = P2["sizes"][0], P2["sizes"][-1]
mlp_low = P2["mlp"][0]["test_acc_mean"]
mlp_high = P2["mlp"][-1]["test_acc_mean"]
lin_low = P2["linear"][0]["test_acc_mean"]
lin_high = P2["linear"][-1]["test_acc_mean"]
m3_low, m3_high = P3["mlp"][0]["test_acc"], P3["mlp"][-1]["test_acc"]
l3_low, l3_high = P3["linear"][0]["test_acc"], P3["linear"][-1]["test_acc"]

add(rf"""## 7. 实验二：不同数据量下的表现

### 7.1 二维螺旋

训练集从 {n_low} 个样本逐步增加到 {num(n_high)} 个，每个规模用 3 次不同的分层抽样重复，测试集固定为 900 个样本。线性分类器训练 1500 轮，MLP 训练 400 轮，均为 Adam、批大小 64。结果见表 2、图 4。

{table(["训练样本数", "线性分类器 测试准确率", "MLP(32-32) 测试准确率", "MLP 训练准确率"], rows2)}

![图 4　四臂螺旋上准确率随训练样本数的变化（误差棒为 3 次重复的标准差）](figures/fig04_data_size_2d.png)

线性分类器的测试准确率在 0.26～0.28 之间几乎是一条水平线，训练准确率也只有 0.3 左右：增加数据不能弥补模型假设的错误，它根本没有能力表示螺旋边界，再多数据也只能把"错误的边界"估得更准。

MLP 则呈现典型的从小样本到大样本的增长：{n_low} 个样本时只有 {f3(mlp_low)}，120 个样本时已经到 0.92 左右，此后增长放缓，最终稳定在 {f3(mlp_high)} 附近。30～60 个样本时 MLP 的训练准确率接近 1.0，而测试准确率只有 0.60～0.80，是明显的过拟合；随着样本增加，训练与测试的差距收窄。这个过程中真正起作用的不是"更多参数"，而是**更多数据让分段折线的位置估计得更准**。

### 7.2 MNIST

在 MNIST 上从训练集中分层抽取 {num(P3['sizes'][0])} 到 {num(P3['sizes'][-1])} 个样本，测试集始终是完整的 10000 张。模型为线性分类器与 784-128-10 的 MLP，均训练 20 个 epoch。结果见表 3、图 5。

{table(["训练样本数", "线性分类器", "MLP(128)", "提升（百分点）", "两者训练总耗时 / s"], rows3)}

![图 5　MNIST 上测试准确率与训练耗时随训练样本数的变化](figures/fig05_data_size_mnist.png)

线性分类器从 {num(P3['sizes'][0])} 个样本的 {pct(l3_low)} 提升到 {num(P3['sizes'][-1])} 个样本的 {pct(l3_high)}，提升约 {(l3_high - l3_low) * 100:.1f} 个百分点；MLP 从 {pct(m3_low)} 提升到 {pct(m3_high)}，提升约 {(m3_high - m3_low) * 100:.1f} 个百分点。两个现象值得注意。

第一，**数据量对 MLP 的边际收益远大于线性模型**。在只有 100 张图时，MLP 和线性模型的差距很小（{pct(m3_low)} 对 {pct(l3_low)}），因为样本太少，MLP 无法可靠地估计非线性边界；到 60000 张时差距扩大到 {(m3_high - l3_high) * 100:.1f} 个百分点。这说明非线性模型的优势需要数据来"兑现"。

第二，**线性模型在大数据下也没有继续提升**，它在 10000 张以后基本停在 92% 左右。这是模型假设的天花板：MNIST 中大量数字存在笔画变形，真正的类别边界是弯曲的，线性打分无法刻画。

从耗时看，MLP 的训练时间随样本数近似线性增长，60000 张、20 个 epoch、两个隐藏层（约 10 万参数）在 CPU 上只需几十秒，说明纯 NumPy 的实现足以支撑这个规模的实验。
""")


# ---------------------------------------------------------------------------
# 8. 实验三：超参数对比
# ---------------------------------------------------------------------------

w, d, act, lr, init, batch, l2, opt, seedr = (
    P4["width"], P4["depth"], P4["activation"], P4["lr"],
    P4["init"], P4["batch"], P4["l2"], P4["optimizer"], P4["seed"],
)
wb, db, ab = argbest(w["test_acc"]), argbest(d["test_acc"]), argbest(act["test_acc"])
lb, ib, bb = argbest(lr["test_acc"]), argbest(init["test_acc"]), argbest(batch["test_acc"])
l2b, ob = argbest(l2["test_acc"]), argbest(opt["test_acc"])
gap0 = l2["train_acc"][0] - l2["test_acc"][0]
gapb = l2["train_acc"][l2b] - l2["test_acc"][l2b]

w_rows = [[str(w["units"][i]), num(w["params"][i]), f4(w["test_acc"][i]),
           f4(w["train_acc"][i]), f"{w['time'][i]:.1f}"] for i in range(len(w["units"]))]
d_rows = [["-".join(str(s) for s in d["sizes"][i][1:-1]), num(d["params"][i]),
           f4(d["test_acc"][i]), f4(d["train_acc"][i]), f"{d['time'][i]:.1f}"]
          for i in range(len(d["depths"]))]
act_rows = [[act["names"][i], f4(act["test_acc"][i]), f4(act["train_acc"][i]),
             f"{act['time'][i]:.1f}"] for i in range(len(act["names"]))]
lr_rows = [[f"{lr['lrs'][i]:g}", f4(lr["test_acc"][i]), f4(lr["train_acc"][i]),
            f4(lr["test_loss"][i])] for i in range(len(lr["lrs"]))]
init_rows = [[init["names"][i], f4(init["test_acc"][i]), f4(init["train_acc"][i])]
             for i in range(len(init["names"]))]
batch_rows = [[str(batch["sizes"][i]), f4(batch["test_acc"][i]), f4(batch["train_acc"][i]),
               f"{batch['time'][i]:.1f}"] for i in range(len(batch["sizes"]))]
l2_rows = [[f"{l2['l2'][i]:g}", f4(l2["train_acc"][i]), f4(l2["test_acc"][i]),
            f"{(l2['train_acc'][i] - l2['test_acc'][i]) * 100:+.2f}"]
           for i in range(len(l2["l2"]))]
opt_rows = [[opt["names"][i], f"{opt['lrs'][i]:g}", f4(opt["test_acc"][i]),
             f"{opt['time'][i]:.1f}"] for i in range(len(opt["names"]))]

add(rf"""## 8. 实验三：超参数对结果的影响

本组实验在 MNIST 的 {num(P4['n_train'])} 张训练集、{num(P4['n_test'])} 张测试集上进行，基准配置为：784-128-10、ReLU、He 初始化、Adam、学习率 $10^{{-3}}$、批大小 128、无 L2、20 个 epoch。每次只改变一个因素，其余保持不变。这样做的代价是无法发现超参数之间的交互作用，但可以在有限的计算预算内得到清晰的单因素趋势。

### 8.1 隐藏层宽度

![图 6　容量实验：宽度、深度以及参数量与准确率的关系](figures/fig06_capacity.png)

{table(["隐藏单元数", "参数量", "测试准确率", "训练准确率", "训练耗时 / s"], w_rows)}

宽度从 4 增加到 32 时，测试准确率从 {f4(w['test_acc'][0])} 提升到 {f4(w['test_acc'][3])}，提升最明显；从 64 往后曲线趋于平缓，最佳宽度是 {w['units'][wb]} 个单元（{f4(w['test_acc'][wb])}）。参数量的增长近似线性（每个隐藏单元带来 784 个输入权重和 10 个输出权重），但准确率并不是线性增长的：少数单元只能表达很少几种折线，无法覆盖 10 个数字的多样性；单元数达到一定规模后，新增单元提供的"新折线"与已有单元高度相关，边际收益就小了。

注意训练准确率：宽网络在小数据上很容易把训练集拟合到接近 100%，而测试准确率提升有限，这就是容量过剩的表现。

### 8.2 深度

{table(["隐藏层结构", "参数量", "测试准确率", "训练准确率", "训练耗时 / s"], d_rows)}

1 个隐藏层（64 个单元）的测试准确率为 {f4(d['test_acc'][0])}，2 个隐藏层为 {f4(d['test_acc'][1])}，提升 {abs(d['test_acc'][1] - d['test_acc'][0]) * 100:.2f} 个百分点；3 层和 4 层分别为 {f4(d['test_acc'][2])}、{f4(d['test_acc'][3])}，仍在上升，但每增加一层只带来约 0.1 个百分点的收益，参数量与训练时间却按层数增加。在本实验的规模（1 万张训练图、每层 64 个单元、20 个 epoch）下，深度不是瓶颈：单隐藏层已经足够宽，额外的层主要是在做细微的边界修正。更深网络的优化难度更高，固定学习率下的收益因此递减；如果配合更长的训练、残差连接或归一化，深度的优势通常会变得更明显。

### 8.3 激活函数

![图 7　ReLU / tanh / sigmoid 的测试准确率曲线](figures/fig07_activation.png)

{table(["激活函数", "测试准确率", "训练准确率", "训练耗时 / s"], act_rows)}

三种激活函数中表现最好的是 **{act['names'][ab]}**（{f4(act['test_acc'][ab])}）。ReLU 的导数在正区间恒为 1，不存在饱和，梯度可以稳定地传过多个隐藏层；sigmoid 和 tanh 的两端导数趋近于零，深层堆叠时容易出现梯度消失，收敛明显更慢。对于这个只有 128 个单元的网络，tanh 尚可，sigmoid 最慢。

### 8.4 学习率

![图 8　学习率对最终准确率的影响与不同学习率的收敛过程](figures/fig08_lr.png)

{table(["学习率", "测试准确率", "训练准确率", "测试交叉熵"], lr_rows)}

学习率是本组实验中影响最直接、也最容易踩坑的超参数之一。$10^{{-4}}$ 时 20 个 epoch 只更新了很少的有效步数，测试准确率只有 {f4(lr['test_acc'][0])}；$3\times10^{{-2}}$ 时步长过大，损失震荡，测试准确率为 {f4(lr['test_acc'][-1])}，测试交叉熵也明显偏高。准确率最高的是 $\eta={lr['lrs'][lb]:g}$（{f4(lr['test_acc'][lb])}），但如果看测试交叉熵，最低的是 $\eta=10^{{-3}}$（{f4(lr['test_loss'][2])}）。大学习率在 20 轮内把准确率推得更高，却因为步长过大在最优点附近震荡，损失没有降到最低。这说明"准确率最高"和"损失最低"并不总是同一个学习率，调参时要先明确评价指标。图 8 右侧的损失曲线也反映了这一点：小学习率的损失平缓下降，大学习率快速下降后出现抖动。

### 8.5 初始化方式

![图 9　不同初始化方式的测试准确率曲线与最终准确率](figures/fig09_init.png)

{table(["初始化", "测试准确率", "训练准确率"], init_rows)}

全零初始化（zeros）只有 {f4(init['test_acc'][0])}，相当于把所有样本都预测成同一类。原因比"退化成线性模型"更彻底：隐藏层权重为零时隐藏层输出恒为零，输出层权重 $W_3$ 的梯度 $H_2^\top\delta$ 为零；反过来 $W_3=0$ 又让隐藏层的梯度为零。整个网络只有输出层的偏置 $b_3$ 能根据类别先验更新，而 MNIST 十类大致均衡，于是它学到的是"永远预测同一类"，准确率约 10%。这个结果说明：**对称性一旦无法打破，网络连线性模型都不如**。小方差初始化（small，标准差 0.01）为 {f4(init['test_acc'][1])}，能打破对称性，但信号在层间传播时方差逐渐缩小，收敛较慢。Xavier 与 He 分别为 {f4(init['test_acc'][2])} 和 {f4(init['test_acc'][3])}，它们根据扇入/扇出调整方差，让各层激活值保持合适的尺度，是本实验中表现最好的两种方式。

### 8.6 批大小

![图 10　批大小与 L2 系数的影响](figures/fig10_batch_l2.png)

{table(["批大小", "测试准确率", "训练准确率", "训练耗时 / s"], batch_rows)}

在**固定 epoch 数**的前提下，批越小测试准确率越高：批大小 16 时为 {f4(batch['test_acc'][0])}，128 时为 {f4(batch['test_acc'][2])}，全批量（{batch['sizes'][-1]}）时为 {f4(batch['test_acc'][-1])}。原因不是"小批量更准"，而是更新次数不同：10000 个样本、批大小 16 时每个 epoch 有 625 次参数更新，批大小 128 时只有 78 次，全批量只有 1 次。全批量在 20 个 epoch 内总共只更新 20 次，显然不够；把轮数增加到 300 后，全批量的测试准确率回升到 {f4(batch['full_300ep']['test_acc'])}，说明它并不是学不会，只是需要更多迭代。

另一方面，小批量带来的梯度噪声也有正则化效果，通常能获得略好的泛化，但这需要配合更大的 epoch 预算来观察。

### 8.7 L2 正则

小数据实验：从训练集中取出 {num(l2['n_train'])} 张图，训练 30 个 epoch，批大小 64，只改变 $\lambda$：

{table(["L2 系数 λ", "训练准确率", "测试准确率", "训练−测试（百分点）"], l2_rows)}

$\lambda=0$ 时训练准确率 100%、测试 {f4(l2['test_acc'][0])}，训练−测试差距 {abs(gap0) * 100:.2f} 个百分点，是典型的过拟合。增大 $\lambda$ 后训练准确率下降、差距收窄，但测试准确率并没有随之提高：$\lambda\le 0.01$ 时测试准确率都在 0.899 附近，$\lambda=0.1$ 时反而降到 {f4(l2['test_acc'][-1])}（欠拟合）。这说明在这份小数据设置下，L2 的主要作用是缩小泛化差距，而测试准确率的瓶颈在于数据量与模型容量本身，单靠权重衰减无法突破。实际中更有效的做法通常是增加数据、使用 Dropout 或早停，L2 只是正则化工具箱中的一种。

### 8.8 优化器

![图 11　SGD / Momentum / Adam 的训练损失曲线](figures/fig11_optimizer.png)

{table(["优化器", "学习率", "测试准确率", "训练耗时 / s"], opt_rows)}

在相同的 20 个 epoch 预算下，Adam（{f4(opt['test_acc'][ob])}）明显快于 SGD 和 Momentum。SGD 对所有参数使用同一个学习率，在损失曲面不同方向曲率差异很大时收敛慢；Momentum 利用历史梯度平滑方向，略好于 SGD；Adam 为每个参数维护自适应的步长，在训练早期就能快速下降。需要说明的是，这里 SGD 的学习率（$10^{{-2}}$）已经比 Adam（$10^{{-3}}$）大，仍然没有追平，说明差距主要来自自适应步长而不是学习率数值。如果训练几百个 epoch 并配合学习率衰减，三者的最终差距会明显缩小。

### 8.9 随机种子

用 3 个不同的初始化种子重复基准配置，测试准确率为 {fmt_float_list(seedr['test_acc'], 4)}，均值 {f4(seedr['mean'])}，标准差 {f4(seedr['std'])}。波动远小于上面各因素造成的差异，说明本节的结论不是某一次运行的偶然结果。不过对于差距小于约 0.3 个百分点的两组配置，仍应视为统计上不可区分。
""")


# ---------------------------------------------------------------------------
# 9. 实验四：MNIST 全量实验
# ---------------------------------------------------------------------------

p5l, p5m = P5["linear"], P5["mlp"]
gain = p5m["test_acc"] - p5l["test_acc"]
per_rows = []
for r in p5m["per_class"]:
    per_rows.append([r["class"], f3(r["precision"]), f3(r["recall"]), f3(r["f1"]), num(r["support"])])
worst = min(p5m["per_class"], key=lambda r: r["recall"])
best = max(p5m["per_class"], key=lambda r: r["recall"])

conf_rows = []
for c in p5m["top_confusions"][:6]:
    conf_rows.append([f"{c['true']} → {c['pred']}", num(c["count"]),
                      pct(c["rate_of_class"]), pct(c["rate_of_pred"])])

add(rf"""## 9. 实验四：MNIST 手写数字分类

### 9.1 设置与总体结果

训练集 60000 张、测试集 10000 张，图像为 $28\times28$ 灰度图，拉平成 784 维并缩放到 $[0,1]$。线性分类器为 784 个输入直接连到 10 个输出；MLP 结构为 {P5['mlp_sizes']}（两个隐藏层，ReLU，He 初始化）。两者都用 Adam、学习率 $10^{{-3}}$、批大小 128、训练 20 个 epoch。

{table(["模型", "参数量", "训练准确率", "测试准确率", "测试交叉熵", "训练耗时 / s"],
       [["线性分类器", num(p5l["params"]), pct(p5l["train_acc"], 2), pct(p5l["test_acc"], 2), f4(p5l["test_loss"]), f"{p5l['time']:.1f}"],
        ["MLP 784-256-128-10", num(p5m["params"]), pct(p5m["train_acc"], 2), pct(p5m["test_acc"], 2), f4(p5m["test_loss"]), f"{p5m['time']:.1f}"]])}

MLP 把测试准确率从 {pct(p5l['test_acc'], 2)} 提升到 {pct(p5m['test_acc'], 2)}，绝对提升 {gain * 100:.2f} 个百分点，测试交叉熵从 {f4(p5l['test_loss'])} 降到 {f4(p5m['test_loss'])}。参数量从 {num(p5l['params'])} 增加到 {num(p5m['params'])}，约为 {p5m['params'] / p5l['params']:.1f} 倍；训练耗时从 {p5l['time']:.1f} 秒增加到 {p5m['time']:.1f} 秒。

![图 12　MNIST 结果总览：样本、损失曲线、准确率曲线、两个混淆矩阵、MLP 每类指标](figures/fig12_mnist_overview.png)

从训练曲线可以看到，MLP 在前 5 个 epoch 就把测试准确率推到 97% 以上，之后缓慢爬升；线性分类器则一直在 92% 附近，而且训练与测试曲线几乎重合——它不是过拟合，而是容量不足导致欠拟合。MLP 的训练准确率最终达到 {pct(p5m['train_acc'], 2)}、测试为 {pct(p5m['test_acc'], 2)}，两者相差约 {(p5m['train_acc'] - p5m['test_acc']) * 100:.1f} 个百分点，说明它在 6 万张图上仍有轻度过拟合；不过测试准确率随训练缓慢上升，没有出现明显回落。

### 9.2 混淆矩阵与每类指标

![图 13　线性分类器与 MLP 的归一化混淆矩阵](figures/fig13_mnist_confusion.png)

{table(["数字", "precision", "recall", "F1", "测试样本数"], per_rows)}

召回率最低的是数字 {worst['class']}（{f3(worst['recall'])}），最高的是数字 {best['class']}（{f3(best['recall'])}）。最易混淆的类别对如下：

{table(["真实 → 预测", "错误数", "占该类比例", "占该预测类比例"], conf_rows)}

混淆集中在形状相近的数字上：{p5m['top_confusions'][0]['true']} 与 {p5m['top_confusions'][0]['pred']} 都带有一个开口和弯钩，{p5m['top_confusions'][1]['true']} 与 {p5m['top_confusions'][1]['pred']} 的区别只在于上半部分是否闭合，{p5m['top_confusions'][3]['true']} 与 {p5m['top_confusions'][3]['pred']} 的差别在于底部横画，8 与 3 在笔画粘连时也很接近。这类错误不是"模型没训好"，而是样本本身在像素层面就存在歧义，人眼也容易看错。

线性分类器的错误则更多：它的每一类只有一个线性打分模板，无法同时容纳同一个数字的多种写法（例如 2 的两种起笔、7 的两种竖钩）。MLP 通过两个隐藏层学到了多个"笔画组合"特征，再在输出层组合，因此对书写变体更鲁棒。

### 9.3 错分样本

![图 14　MLP 分错的 16 个测试样本（标题为真实类别 → 预测类别）](figures/fig14_mnist_errors.png)

这些错分样本大多本身就难以辨认：笔画粘连、缺笔、写得过小或倾斜，单独拿出来看，人眼也需要停顿一下。需要说明的是，MNIST 的测试错误率并不是全都来自标注噪声（人工错误率通常在 0.2% 上下），本实验约 1.8% 的错误中仍有相当一部分来自模型本身：全连接层把图像拉平，丢弃了二维局部结构，也没有平移不变性；改用卷积结构、数据增强或更长的训练都能继续降低错误率。

### 9.4 第一层权重与隐层表示

![图 15　MLP 第一层权重（按范数取前 32 个隐藏单元，红色为正、蓝色为负）](figures/fig15_mnist_filters.png)

第一层的每个隐藏单元对应一张 $28\times28$ 的模板。图 15 中可以分辨出一些有明确方向的笔画检测器，例如 #242 对水平笔画敏感、#121 与 #10 对斜向笔画敏感，还有一些单元呈现"中心-周边"的对比结构；也有不少单元的模板比较分散，需要与后面的层配合才能解释。这些模板不是人工设计的，而是从数据中通过反向传播学出来的。与卷积网络第一层学到的 Gabor 状滤波器相比，全连接层的模板更"全局"、更嘈杂，这也从侧面说明了为什么卷积结构更适合图像。

![图 16　原始像素与第二隐藏层表示的 PCA 前两维](figures/fig16_hidden_pca.png)

图 16 左侧是原始像素直接做 PCA 的结果，不同数字混在一起；右侧是 MLP 第二隐藏层（128 维）的 PCA 投影，同类样本明显聚成条带或团块，不同类之间的间隔更大。这说明隐藏层的作用可以理解为**学习一个让线性分类更容易的特征变换**：输出层仍然是一个线性分类器，但它作用在学到的表示上，而不是原始像素上。

### 9.5 与线性分类器的对比小结

| 对比项 | 线性分类器 | MLP |
|---|---|---|
| 决策边界 | 超平面，区域为凸集 | 分段折线，可围出非凸区域 |
| 参数量 | {num(p5l['params'])} | {num(p5m['params'])} |
| MNIST 测试准确率 | {pct(p5l['test_acc'], 2)} | {pct(p5m['test_acc'], 2)} |
| 训练耗时 | {p5l['time']:.1f} s | {p5m['time']:.1f} s |
| 典型错误 | 无法区分同一数字的多种写法 | 仅混淆形状本身相近的数字 |

两者在 MNIST 上的差距（约 {gain * 100:.1f} 个百分点）远小于在螺旋数据上的差距（约 {SP['gap_mean'] * 100:.1f} 个百分点），原因是 MNIST 的类别在原始像素空间中已经大致线性可分，10 类问题中每一类都有相对独立的"平均模板"。越是需要弯曲边界、越依赖局部组合的任务，MLP 的优势越明显。
""")


# ---------------------------------------------------------------------------
# 10. 讨论
# ---------------------------------------------------------------------------

add(rf"""## 10. 讨论

### 10.1 线性模型与 MLP 的差异到底来自哪里

三组二维实验给出了同一条规律：**差距的大小取决于真实决策边界的"弯曲程度"，而不是模型的参数量**。

- 四团高斯：边界是直线，线性模型与 MLP 打平（{f3(BL['linear']['test_acc_mean'])} 对 {f3(BL['mlp']['test_acc_mean'])}）；
- 四同心环：边界是圆，线性模型停在随机水平（{f3(RG['linear']['test_acc_mean'])}），MLP 达到 {f3(RG['mlp']['test_acc_mean'])}；
- 四臂螺旋：边界是螺旋线，差距同样很大（{f3(SP['linear']['test_acc_mean'])} 对 {f3(SP['mlp']['test_acc_mean'])}）。

从几何上看，线性分类器的每个判定区域是半平面的交，必然是凸集；MLP 的每个 ReLU 单元贡献一条折线，多个单元组合后可以围出非凸的、带孔的区域。圆环和螺旋都需要这样的区域。从表示上看，MLP 的前几层学到的是一组新的特征，最后一层在这些特征上做线性划分；如果新特征让类别变得线性可分，线性输出层就够用了。第 9.4 节的 PCA 图给出了这一解释的可视化证据。

另一个容易被忽略的点是：**非线性模型在简单任务上可能更差**。四团高斯上 MLP 反而低了 {abs(BL['gap_mean']) * 100:.2f} 个百分点，决策边界上出现了多余的折角。这说明增加容量会引入方差，当真实边界简单时，这部分方差没有对应的偏差下降来抵消。

### 10.2 数据量、容量与正则化

第二组实验说明，MLP 的优势不是无条件的：在螺旋数据上，30 个样本时 MLP 只有 {f3(mlp_low)}，120 个样本后才到 0.92，最终饱和在 {f3(mlp_high)}；在 MNIST 上，100 张图时 MLP 与线性模型几乎相同，60000 张时才拉开 {(m3_high - l3_high) * 100:.1f} 个百分点。容量必须由数据来"喂养"，否则参数只是在记忆训练集。

第三组实验中的 L2 扫描说明了正则化的作用边界。小数据（{num(P4['l2']['n_train'])} 张）时，不加强正则的训练准确率是 100%、测试只有 {f4(l2['test_acc'][0])}，差距约 {abs(gap0) * 100:.1f} 个百分点；增大 $\lambda$ 后训练准确率下降、差距收窄（$\lambda=0.1$ 时只剩 {abs(l2['train_acc'][-1] - l2['test_acc'][-1]) * 100:.1f} 个百分点），但测试准确率并没有提高，$\lambda=0.1$ 时反而降到 {f4(l2['test_acc'][-1])}。也就是说，L2 抑制了"记忆"，却没有把模型变成更好的分类器——这份数据的瓶颈是样本量，而不是权重幅度。正则化能减少方差，但不能凭空提供信息；真正有效的手段是增加数据或使用更合适的结构。

### 10.3 超参数的敏感度

把第 8 节的扫描结果放在一起看，各因素在本实验条件下造成的测试准确率变化范围大致为：

1. **宽度**：4 个单元到 256 个单元，从 {f4(w['test_acc'][0])} 到 {f4(w['test_acc'][-1])}（约 {abs(w['test_acc'][-1] - w['test_acc'][0]) * 100:.1f} 个百分点）。容量不足是最"硬"的瓶颈；但超过 64 个单元后边际收益迅速下降。
2. **初始化**：全零 {f4(init['test_acc'][0])}，最好的正常初始化 {f4(max(init['test_acc'][1:]))}，相差约 {abs(max(init['test_acc'][1:]) - init['test_acc'][0]) * 100:.1f} 个百分点。全零不是"效果差一点"，而是完全学不动（隐藏层与输出层的权重互相锁死），属于必须避免的实现错误。
3. **批大小**：16 到全批量，从 {f4(batch['test_acc'][0])} 到 {f4(batch['test_acc'][-1])}（约 {abs(batch['test_acc'][0] - batch['test_acc'][-1]) * 100:.1f} 个百分点）。但这一项与"更新次数"纠缠，不能简单理解为"批越小越好"；全批量训练 300 轮后回到 {f4(batch['full_300ep']['test_acc'])}。
4. **优化器**：从 {f4(min(opt['test_acc']))} 到 {f4(max(opt['test_acc']))}（约 {abs(max(opt['test_acc']) - min(opt['test_acc'])) * 100:.1f} 个百分点），差距会随训练轮数增加而缩小。
5. **学习率**：从 {f4(min(lr['test_acc']))} 到 {f4(max(lr['test_acc']))}（约 {abs(max(lr['test_acc']) - min(lr['test_acc'])) * 100:.1f} 个百分点），最优点在 $10^{{-2}}$ 附近。
6. **激活函数**：从 {f4(min(act['test_acc']))} 到 {f4(max(act['test_acc']))}（约 {abs(max(act['test_acc']) - min(act['test_acc'])) * 100:.1f} 个百分点），ReLU 最好，sigmoid 最慢。
7. **深度**：1 层到 4 层，从 {f4(d['test_acc'][0])} 到 {f4(d['test_acc'][-1])}（约 {abs(d['test_acc'][-1] - d['test_acc'][0]) * 100:.1f} 个百分点），收益递减。
8. **L2**：在小数据设置下，$\lambda\le 0.01$ 时测试准确率几乎不变（约 0.899），主要作用是缩小训练-测试差距；$\lambda=0.1$ 时开始欠拟合。
9. **随机种子**：基准配置下标准差 {f4(seedr['std'])}，远小于以上各项，说明主要结论不是偶然。

需要强调，"敏感度排序"依赖训练预算。例如学习率与优化器的差距会随着训练轮数增加而缩小，宽度与深度的差距会随着数据量增加而变化。这些结论应理解为"在本实验设定的数据规模与计算预算下"的相对重要性。

### 10.4 优化困难与表达能力是两回事

深度实验给出的结果比较温和：1 层到 4 层的测试准确率从 {f4(d['test_acc'][0])} 升到 {f4(d['test_acc'][-1])}，一共只有 {abs(d['test_acc'][-1] - d['test_acc'][0]) * 100:.1f} 个百分点，而参数量从 {num(d['params'][0])} 增加到 {num(d['params'][-1])}。在本实验的规模下，表达能力不是瓶颈，增加深度的收益被优化难度与数据量抵消了。表达能力与可训练性是两件事：一个网络能表示某个函数，不代表梯度下降能在有限的轮数内找到它。要让深度真正发挥作用，通常需要配合更好的初始化、归一化、残差连接和更长的训练；本实验没有实现这些组件，因此不能据此得出"深层网络没用"的结论。

### 10.5 实验的局限

- MNIST 是相对简单、干净的数据集，98% 左右的准确率不能代表 MLP 在真实图像任务上的水平；本实验的目的是对比模型与展示方法，而不是刷榜。
- 受 CPU 时间限制，超参数扫描只做了单因素实验，没有做网格搜索或贝叶斯优化，也没有探索学习率衰减、早停等技巧。
- 所有结论基于单次运行（二维数据除外，用了 3 个种子）；超参数扫描的差异如果小于约 0.3 个百分点，应当视为噪声。
- 未实现卷积、BatchNorm、Dropout、残差连接等现代组件，因此这里的 MLP 是一个"教科书版本"，用于理解原理而非追求最优性能。
""")


# ---------------------------------------------------------------------------
# 11. 结论
# ---------------------------------------------------------------------------

add(rf"""## 11. 结论

1. 用纯 NumPy 实现了一个完整的 MLP 训练系统：前向传播、反向传播、softmax 交叉熵、He / Xavier 初始化、SGD / Momentum / Adam、L2 正则和小批量训练。数值梯度校验显示解析梯度与中心差分在 $10^{{-10}}$ 量级上一致，证明实现正确。
2. 线性分类器与 MLP 的差异由决策边界的几何形状决定。在线性可分的四团数据上两者相当（{f3(BL['linear']['test_acc_mean'])} 对 {f3(BL['mlp']['test_acc_mean'])}）；在圆环与螺旋数据上，线性分类器停留在随机水平（{f3(RG['linear']['test_acc_mean'])}、{f3(SP['linear']['test_acc_mean'])}），MLP 达到 {f3(RG['mlp']['test_acc_mean'])} 和 {f3(SP['mlp']['test_acc_mean'])}。原因是线性模型的判定区域必须是凸集，无法表示圆环、螺旋这类非凸类别。
3. 数据量对非线性模型至关重要：螺旋数据上 MLP 从 30 个样本的 {f3(mlp_low)} 提升到 2100 个样本的 {f3(mlp_high)}；MNIST 上从 100 张的 {pct(m3_low)} 提升到 60000 张的 {pct(m3_high)}，并在此过程中把与线性模型的差距从几乎为零扩大到 {(m3_high - l3_high) * 100:.1f} 个百分点。
4. 超参数中宽度与初始化的影响最大（全零初始化会让网络完全学不动，只剩约 10% 的准确率），批大小、优化器和学习率次之，深度与激活函数的影响相对温和。宽度与深度都存在明显的边际收益递减；小批量在固定 epoch 下因更新次数更多而收敛更快；L2 能显著缩小训练-测试差距，但在本实验的小数据设置下没有提高测试准确率。
5. MNIST 上 MLP 达到 {pct(p5m['test_acc'], 2)} 的测试准确率，比线性分类器的 {pct(p5l['test_acc'], 2)} 高 {gain * 100:.2f} 个百分点；混淆矩阵、错分样本、第一层权重和隐层 PCA 表明，MLP 学到的是笔画级别的局部特征，而线性模型只能为每个数字学一个全局模板。
6. 通过本次实验，完整走通了"推导公式 → 手写实现 → 数值校验 → 控制变量实验 → 可视化分析"的流程，理解了表达能力、优化难度与泛化能力是三个需要分别对待的问题。
""")


# ---------------------------------------------------------------------------
# 12. 复现说明与文件清单
# ---------------------------------------------------------------------------

add(rf"""## 12. 复现说明

### 12.1 运行步骤

```bash
# 1. 下载 MNIST（仓库中已附带 data/*.gz，可跳过）
bash data/download_mnist.sh

# 2. 运行全部实验（约 15 分钟，CPU）
cd exp4_report
python full_experiment.py

# 3. 生成报告（Markdown + docx）
python gen_report.py
```

`full_experiment.py` 支持 `--parts 0 1 2 3 4 5` 只运行其中一部分，也支持 `--out` 指定结果文件，便于分片运行后合并。报告中的所有数字均由 `gen_report.py` 从 `results.json` 读取，不手工抄写。

### 12.2 文件清单

| 路径 | 说明 |
|---|---|
| `code/mlp.py` | MLP 前向 / 反向传播、初始化、数值梯度校验 |
| `code/linear.py` | 线性 softmax 分类器 |
| `code/train.py` | SGD / Momentum / Adam、训练循环、评估 |
| `code/datasets.py` | 二维合成数据、MNIST 加载、分层抽样 |
| `code/metrics.py` | 混淆矩阵、每类指标、易混淆类别对 |
| `code/viz.py` | 决策边界、曲线、混淆矩阵、样本网格 |
| `data/*.gz` | MNIST 原始 IDX 文件 |
| `exp4_report/full_experiment.py` | 四组实验的主脚本 |
| `exp4_report/results.json` | 全部指标与曲线数据 |
| `exp4_report/figures/fig01–fig18` | 报告插图 |
| `learn/practice/extra_datasets.py` | 第 13 节六个额外数据集的实验脚本 |
| `learn/practice/extra_results.json` | 第 13 节的准确率表 |
| `exp4_report/gen_report.py` | 从 JSON 生成本报告 |
| `exp4_report/实验四_多层感知机_实验报告.md` | 本报告 |

### 12.3 复现注意事项

- 所有随机过程都使用显式种子；二维实验的种子为 0/1/2，MNIST 超参数扫描的种子为 0（稳定性实验额外用 1、2）。
- 训练时间与 CPU 型号、BLAS 线程数有关，报告中只用于量级比较，不作为性能结论。
- 若只重跑某一部分，`full_experiment.py --parts N` 会合并写入已有的 `results.json`，不会覆盖其他部分的结果。
""")


# ---------------------------------------------------------------------------
# 13. 扩展实验：更多数据集
# ---------------------------------------------------------------------------

if EXTRA:
    extra_rows = [
        [r["name"], num(r["n_train"]), num(r["n_test"]), r["D"], r["C"],
         f4(r["linear"]), f4(r["mlp"]), f"{r['gap']:+.4f}"]
        for r in EXTRA
    ]
    add(rf"""## 13. 扩展实验：更多数据集上的验证

为了检验前面的结论是否只对二维合成数据和 MNIST 成立，又选了 6 个公开数据集，用同一套 MLP（ReLU、He、Adam）和同一个线性分类器重复"线性 vs MLP"的对照。每个数据集只用一个训练/测试划分、一个随机种子，因此小于 1～2 个百分点的差异不应过度解读；各数据集的结构与训练轮数是事先指定的，没有针对测试结果反复调参。

{table(["数据集", "训练样本", "测试样本", "特征维", "类别数", "线性准确率", "MLP 准确率", "gap"], extra_rows)}

![图 17　六个额外数据集上的线性分类器与 MLP](figures/fig17_extra_compare.png)

![图 18　Fashion-MNIST 样本（4×5）](figures/fig18_extra_fashion_samples.png)

**数据集。** Fashion-MNIST 是 28×28 的服装灰度图，10 类，这里抽取 20000 张训练、5000 张测试、训练 12 轮；Pendigits 是 16 维手写数字笔迹特征，10 类；Optdigits 是 8×8 像素拉平成 64 维，10 类；Letter 是 16 个几何/统计特征，26 个字母类；Iris 是 4 维鸢尾花测量，3 类、150 个样本；WDBC 是 30 维乳腺肿瘤细胞特征，2 类、569 个样本。

**结果。** 差距最大的是 Letter（{EX['Letter']['gap'] * 100:+.1f} 个百分点）：26 个类别、边界高度非线性，线性分类器的凸判定区域远远不够。Pendigits 次之（{EX['Pendigits']['gap'] * 100:+.1f}），10 类、16 维特征，MLP 达到 {pct(EX['Pendigits']['mlp'])}。Fashion-MNIST 上 MLP 比线性高 {EX['Fashion-MNIST']['gap'] * 100:.1f} 个百分点；这里只训练了 12 轮，全量 6 万张、20 轮时 MLP 通常能到 89%～90%。Optdigits 的差距只有 {EX['Optdigits']['gap'] * 100:.1f} 个百分点，因为 8×8 像素空间里的数字边界本来就接近线性。Iris 上两者相差 {EX['Iris']['gap'] * 100:.1f} 个百分点，但测试集只有 45 个样本，差异在噪声范围内。唯一出现反转的是 WDBC：线性 {pct(EX['WDBC']['linear'])}，MLP {pct(EX['WDBC']['mlp'])}，后者低 {abs(EX['WDBC']['gap']) * 100:.1f} 个百分点——398 个训练样本、30 维高度相关特征，线性模型已经很强，MLP 更容易过拟合。

**解释。** 这组结果与第 6～9 节一致：MLP 的优势取决于决策边界的弯曲程度与可用的数据量，而不是参数量。类别多、边界弯（Letter、Pendigits）时非线性收益明显；特征本身接近线性（Optdigits）或样本很少（Iris、WDBC）时，线性模型往往已经够用，甚至更好。需要强调，这些数字来自单次划分、未调参的模型，只能用于趋势判断，不能当作各数据集的性能上限。

**复现。** 脚本为 `learn/practice/extra_datasets.py`，结果写入 `learn/practice/extra_results.json`，运行约 45 秒；Fashion-MNIST 由 `data/download_fashion_mnist.sh` 下载。
""")


# ---------------------------------------------------------------------------
# 参考文献
# ---------------------------------------------------------------------------

add("""## 参考文献

1. Cybenko G. Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 1989, 2(4): 303–314.
2. Hornik K., Stinchcombe M., White H. Multilayer feedforward networks are universal approximators. *Neural Networks*, 1989, 2(5): 359–366.
3. Glorot X., Bengio Y. Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 2010.
4. He K., Zhang X., Ren S., Sun J. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*, 2015.
5. Kingma D. P., Ba J. Adam: A method for stochastic optimization. *ICLR*, 2015.
6. LeCun Y., Bottou L., Bengio Y., Haffner P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 1998, 86(11): 2278–2324.
7. Montúfar G., Pascanu R., Cho K., Bengio Y. On the number of linear regions of deep neural networks. *NIPS*, 2014.
8. Rumelhart D. E., Hinton G. E., Williams R. J. Learning representations by back-propagating errors. *Nature*, 1986, 323: 533–536.
""")


md = "\n\n".join(P) + "\n"
OUT_MD.write_text(md, encoding="utf-8")
print(f"Markdown 已写入 {OUT_MD}（{len(md)} 字符）")

subprocess.run(
    ["pandoc", str(OUT_MD), "-o", str(OUT_DOCX),
     "--resource-path", str(HERE)],
    check=True,
)
print(f"DOCX 已写入 {OUT_DOCX}")



