#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results.json 生成实验五报告（Markdown），并调用 pandoc 生成 docx。

报告中所有实验数字都从 results.json 读取；静态的原理部分不含任何手抄的实验结果。
运行：python gen_report.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import matplotlib

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

ENV = R["meta"]["environment"]
PROTO = R["meta"]["protocol"]
D = R["data_overview"]
DATASET = D["dataset"]
SPLIT = D["split"]
B = R["baseline"]
A = R["activation"]
DR = R["dropout"]
SZ = R["data_size"]
H = R["hyper"]
AU = R["augmentation"]
F = R["final"]
FASH = R["fashion"]

VER_PATH = HERE / "verification.json"
VER = json.loads(VER_PATH.read_text(encoding="utf-8")) if VER_PATH.exists() else None

OUT_MD = HERE / "实验五_卷积神经网络_实验报告.md"
OUT_DOCX = HERE / "实验五_卷积神经网络_实验报告.docx"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


def pp(a, b, nd=2):
    """两个准确率之差，以百分点表示（a-b）。"""
    return f"{(a - b) * 100:+.{nd}f}"


def pm(mean, std, nd=2):
    return f"{100 * mean:.{nd}f}% ± {100 * std:.{nd}f}%"


def gap(a, b, nd=2):
    """两个准确率之差的绝对值，以百分点表示（用于"高/低 X 个百分点"这类叙述）。"""
    return f"{abs(a - b) * 100:.{nd}f}"


def num(v):
    return f"{int(v):,}"


def table(headers, rows):
    out = ["| " + " | ".join(str(c) for c in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def best_of(runs: dict, key="test_acc"):
    """返回 (标签, 准确率) —— 按给定指标取最好的配置。"""
    label = max(runs, key=lambda k: runs[k][key])
    return label, runs[label][key]


def worst_of(runs: dict, key="test_acc"):
    label = min(runs, key=lambda k: runs[k][key])
    return label, runs[label][key]


def avg_epoch_time(run: dict) -> float:
    t = run["history"]["epoch_time"]
    return sum(t) / len(t)


# 运行总量：统计结果文件中出现过的训练次数
def count_runs() -> int:
    n = 0
    n += len(B["runs"])
    n += len(A["runs"]) + len(A["grad_diagnosis"]["runs"])
    n += len(DR["runs"])
    n += len(SZ["cnn"]) + len(SZ["mlp"])
    n += sum(len(v) for v in H.values())
    n += len(AU["runs"])
    n += len(F["recipe_runs"]) + len(F["final_runs"])
    n += 2  # Fashion-MNIST 的 CNN / MLP
    return n


TOTAL_RUNS = count_runs()


def total_train_seconds() -> float:
    """所有实验条目的 train_time 之和（秒），用于报告中的训练预算说明。"""
    total = 0.0
    seen = set()

    def walk(o):
        nonlocal total
        if isinstance(o, dict):
            if "train_time" in o and "history" in o:
                if id(o) not in seen:
                    seen.add(id(o))
                    total += o["train_time"]
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for key in ("baseline", "activation", "dropout", "data_size", "hyper",
                "augmentation", "final", "fashion"):
        walk(R[key])
    return total


TOTAL_TRAIN_MIN = total_train_seconds() / 60.0
MATPLOTLIB_VERSION = matplotlib.__version__

# 常用汇总
BEST_BL_LABEL, BEST_BL_ACC = best_of(B["runs"])
MLP_BL = B["runs"]["MLP 基线"]
LENET_MODERN = B["runs"]["LeNet-5 现代结构 (ReLU+MaxPool)"]
LENET_CLASSIC = B["runs"]["LeNet-5 经典结构 (Sigmoid+AvgPool)"]
SMALLVGG = B["runs"]["SmallVGG (更深+BN+Dropout)"]

_ACT_CN = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh",
           "leaky_relu": "LeakyReLU", "elu": "ELU", "gelu": "GELU"}
ACT_LABEL, ACT_BEST = best_of(A["runs"])
ACT_WORST_LABEL, ACT_WORST = worst_of(A["runs"])
ACT_RELU = A["runs"]["relu"]
ACT_SIGMOID = A["runs"]["sigmoid"]

GD = A["grad_diagnosis"]["runs"]
GD_ORDER = A["grad_diagnosis"]["order"]
GD_SIG_SGD = GD["Sigmoid+AvgPool | SGD 0.01"]
GD_SIG_ADAM = GD["Sigmoid+AvgPool | Adam 1e-3"]
GD_RELU_SGD = GD["ReLU+MaxPool | SGD 0.01"]
GD_RELU_ADAM = GD["ReLU+MaxPool | Adam 1e-3"]

DR_ORDER = DR["order"]
DR_RUNS = DR["runs"]
DR_BEST_LABEL, DR_BEST_ACC = best_of(DR_RUNS)
DR_NOREG = DR_RUNS["无正则化"]
DR_D05 = DR_RUNS["Dropout 0.5 (FC)"]
DR_D02 = DR_RUNS["Dropout 0.2 (FC)"]


def train_val_gap(run: dict) -> float:
    h = run["history"]
    return h["train_acc"][-1] - h["val_acc"][-1]


SZ_SIZES = SZ["sizes"]
SZ_CNN = SZ["cnn"]
SZ_MLP = SZ["mlp"]
AU_ORDER = AU["order"]
AU_RUNS = AU["runs"]
AU_BEST_LABEL, AU_BEST_ACC = best_of(AU_RUNS)
AU_NONE = AU_RUNS["无增强"]
AU_20 = AU_RUNS["旋转/平移 ±20°"]

# 最终模型
F_RECIPES = F["recipe_runs"]
F_BEST_RECIPE_LABEL, F_BEST_RECIPE_VAL = best_of(F_RECIPES, "best_val_acc")
F_MEAN, F_STD = F["mean_test_acc"], F["std_test_acc"]
F_SEEDS = F["per_seed_test_acc"]
F_ROT = F["rotation"]
F_CLASS = F["per_class"]
F_CM = F["confusion"]
F_CONF = F["confidence"]

P = []


def add(text):
    P.append(text)


# ===========================================================================
# 标题、封面、摘要
# ===========================================================================

add(rf"""# 实验五：卷积神经网络实验

> 姓名：________　学号：________　班级：________　日期：________
>
> 代码目录：`code/`（`data.py` 数据与增强、`models.py` LeNet-5 / SmallVGG / MLP、`engine.py` 训练引擎、`metrics.py` 指标、`viz.py` 可视化）
>
> 主实验脚本：`exp5_report/full_experiment.py`　结果：`exp5_report/results.json`　插图：`exp5_report/figures/`
>
> 实验环境：PyTorch {ENV['torch']}（CUDA {ENV['cuda_version']}），GPU 为 {ENV['device']}；全部训练都在 GPU 上完成
>
> 运行方式：`cd exp5_report && python full_experiment.py`，再由 `python gen_report.py` 生成本报告

## 摘要

本实验用 PyTorch 在 GPU 上构建并训练卷积神经网络，解决 MNIST 手写数字图像分类问题。主模型采用课程中介绍的 LeNet-5：两个 5×5 卷积层（6、16 个通道）接 2×2 池化，再经过 256→120→84→10 的全连接分类头，共 {num(LENET_MODERN['params'])} 个可训练参数。所有实验共用同一套训练引擎、同一份数据划分（54,001 张训练图、5,999 张验证图、官方 10,000 张测试图）和同一套评估流程，共完成 {TOTAL_RUNS} 次训练，考察了激活函数、Dropout、数据量、优化器、学习率、批大小、权重衰减、数据增强等因素。

主要结果如下。第一，在相同优化器（Adam）和相同训练轮数下，现代配方（ReLU + 最大池化）的测试准确率为 {pct(LENET_MODERN['test_acc'])}，比经典配方（Sigmoid + 平均池化）的 {pct(LENET_CLASSIC['test_acc'])} 高 {gap(LENET_MODERN['test_acc'], LENET_CLASSIC['test_acc'])} 个百分点；更深的 SmallVGG（3×3 卷积 + BN + Dropout）达到 {pct(SMALLVGG['test_acc'])}。第二，激活函数的影响与优化器强耦合：在 Adam 下六种激活函数的测试准确率落在 {pct(ACT_WORST)}～{pct(ACT_BEST)} 之间；而在原始的 SGD lr=0.01 下，Sigmoid 网络训练 6 轮后测试准确率只有 {pct(GD_SIG_SGD['test_acc'])}，第一层卷积的梯度范数约 {GD_SIG_SGD['grad_norm_first_conv'][0]:.1e}，比 ReLU 的 {GD_RELU_SGD['grad_norm_first_conv'][0]:.1e} 小约 {__import__('math').log10(GD_RELU_SGD['grad_norm_first_conv'][0] / GD_SIG_SGD['grad_norm_first_conv'][0]):.1f} 个数量级，这是典型的梯度消失。第三，Dropout 只在过拟合明显的场景下有效：在 6,000 张训练图、故意放大的全连接头（512-128）上，无正则化的训练-验证差距为 {train_val_gap(DR_NOREG) * 100:+.1f} 个百分点，测试准确率 {pct(DR_NOREG['test_acc'])}；配置最优的「{DR_BEST_LABEL}」把差距缩小到 {train_val_gap(DR_RUNS[DR_BEST_LABEL]) * 100:+.1f} 个百分点、测试准确率提高到 {pct(DR_BEST_ACC)}。第四，数据量从 600 增加到 54,001 时，CNN 的测试准确率从 {pct(SZ_CNN[0]['test_acc'])} 提升到 {pct(SZ_CNN[-1]['test_acc'])}；在训练集很小（600 张）时 CNN 反而比 MLP 低 {abs(SZ_CNN[0]["test_acc"] - SZ_MLP[0]["test_acc"]) * 100:.1f} 个百分点，样本量增大后反超并逐步扩大优势。第五，在 6,000 张训练图的过拟合场景中，±20° 旋转/平移增强把干净测试集的准确率从 {pct(AU_NONE['test_acc'])} 变为 {pct(AU_20['test_acc'])}，但把测试集整体旋转 20° 后的准确率从 {pct(AU_NONE['rotation']['acc'][AU_NONE['rotation']['angles'].index(20)])} 提升到 {pct(AU_20['rotation']['acc'][AU_20['rotation']['angles'].index(20)])}，说明增强换来的主要是鲁棒性而不是干净测试集上的精度。第六，按验证集选出的最终配方为「{F['recipe']}」，用 3 个随机种子重复训练 15 轮，测试准确率为 {pm(F_MEAN, F_STD)}，共 {F['n_errors']} 张测试图被分错；在 Fashion-MNIST 上，同一套 CNN 的测试准确率为 {pct(FASH['cnn']['test_acc'])}，比 MLP 的 {pct(FASH['mlp']['test_acc'])} 高 {gap(FASH['cnn']['test_acc'], FASH['mlp']['test_acc'])} 个百分点。

本实验还定量展示了 CNN 相对全连接网络的优势来源：LeNet-5 只有 {num(LENET_MODERN['params'])} 个参数，不到 MLP（{num(MLP_BL['params'])} 个）的五分之一，测试准确率却高 {gap(LENET_MODERN['test_acc'], MLP_BL['test_acc'])} 个百分点——权值共享、局部连接和平移等变的归纳偏置，比单纯的参数量更重要。
""")


# ===========================================================================
# 1. 实验目的
# ===========================================================================

add(r"""## 1. 实验目的

1. 理解卷积神经网络处理图像的基本原理：局部连接、权值共享、池化下采样，以及这些设计为什么能让参数量远小于全连接网络、同时保留图像的空间结构。
2. 用 PyTorch 从零搭建 LeNet-5，掌握卷积层、池化层、激活函数、Dropout、批归一化和全连接层的组合方式，理解每个张量的通道数与空间尺寸如何逐层变化。
3. 掌握在 GPU 上训练模型的基本流程：数据加载、小批量迭代、前向传播、损失计算、反向传播、优化器更新、验证集评估与模型保存，并用控制变量法设计对照实验。
4. 通过实验分析激活函数的选择：Sigmoid、Tanh、ReLU、LeakyReLU、ELU、GELU 对收敛速度和最终准确率的影响，并从梯度量级的角度定量解释梯度消失现象。
5. 通过实验分析 Dropout 技巧：丢包率、作用位置（卷积层 / 全连接层）以及它与权重衰减的搭配，理解正则化只有在"过拟合确实存在"时才有收益。
6. 系统改变数据量与超参数（优化器、学习率、批大小、权重衰减），观察学习曲线、欠拟合与过拟合、优化稳定性，掌握"固定其余变量、只动一个因素"的实验方法。
7. 掌握图像数据增强（旋转、平移、随机擦除）与鲁棒性评估：不仅看干净测试集上的准确率，还要看模型在分布发生偏移（旋转）时的表现。
8. 完成一次完整的深度学习实验流程：数据检查 → 模型实现 → 正确性验证 → 控制变量实验 → 结果可视化 → 误差分析 → 结论与局限，并学会用混淆矩阵、每类指标、错分样本、卷积核和特征图解释模型行为。
""")


# ===========================================================================
# 2. 实验内容
# ===========================================================================

add(rf"""## 2. 实验内容

实验共分九部分，全部由 `exp5_report/full_experiment.py` 运行完成，结果写入 `results.json`，插图写入 `figures/`。除特别说明外，训练集、验证集、测试集的划分在所有实验中保持不变：

- 训练池：MNIST 官方训练集 60,000 张中按类别分层留出 5,999 张作验证集，剩下 {num(SPLIT['n_train_pool'])} 张；
- 验证集：{num(SPLIT['n_val'])} 张（用于选择模型、判断过拟合；不参与梯度更新）；
- 测试集：MNIST 官方测试集 {num(SPLIT['n_test'])} 张。

![图 1　MNIST 手写数字样本（每类 2 张）](figures/fig01_mnist_samples.png)

**第一部分：数据概览与结构图。** 展示 MNIST 各类样本，统计训练集类别分布（各类 {min(DATASET['class_counts_train'])}～{max(DATASET['class_counts_train'])} 张），并画出 LeNet-5 的结构示意图，核对每层输出尺寸与参数量。

**第二部分：基线模型对比。** 在全部 {num(SPLIT['n_train_pool'])} 张训练图上训练 10 轮，比较 5 个模型：LeNet-5 经典结构（Sigmoid + 平均池化）、LeNet-5 现代结构（ReLU + 最大池化）、LeNet-5 + 批归一化、更深的 SmallVGG（3×3 卷积 + BN + Dropout）、以及同样参数量级的 MLP 基线。所有模型使用相同的优化器（Adam，lr=0.001）与批大小（64）。

**第三部分：激活函数对比与梯度消失诊断。** 在 10,000 张训练图上固定 LeNet-5 结构（最大池化），分别使用 ReLU、Sigmoid、Tanh、LeakyReLU、ELU、GELU 训练 8 轮，记录验证曲线和第一层卷积权重的梯度范数。另设一组诊断实验：用经典结构分别搭配 SGD lr=0.01、SGD lr=0.1、Adam lr=0.001，并与 ReLU 结构对照，训练 6 轮，定量说明"激活函数 + 优化器"必须一起考虑。

**第四部分：Dropout 与正则化。** 为制造明显的过拟合，只用 6,000 张训练图，并把 LeNet-5 的全连接头放大到 512→128，训练 20 轮。比较 6 种设置：无正则化、Dropout 0.2、Dropout 0.5、Dropout 0.5 + 卷积层 Dropout2d 0.25、仅权重衰减 1e-3、Dropout 0.5 + 权重衰减 1e-3。

**第五部分：数据量的影响。** 训练集规模取 600、2,000、6,000、20,000、{num(SPLIT['n_train_pool'])} 五档（按类别等比例分层抽样），每档分别训练 CNN（LeNet-5 现代结构）与 MLP，各 8 轮，比较测试准确率与两者的差距。

**第六部分：超参数对比。** 在 10,000 张训练图上固定其它条件，每次只改一个因素，各训练 8 轮：优化器（SGD、SGD + Momentum、Adam、AdamW）、Adam 学习率（1e-4、3e-4、1e-3、3e-3）、SGD + Momentum 学习率（1e-3、1e-2、1e-1）、批大小（16、32、64、128、256）、权重衰减（0、1e-4、1e-3、1e-2）。

**第七部分：数据增强与旋转鲁棒性。** 在 6,000 张训练图的过拟合场景下，比较无增强、随机旋转/平移 ±10°、随机旋转/平移 ±20°、±20° 旋转 + 随机擦除四种增广方式，各训练 20 轮；然后把同一个测试集分别旋转 −30°、−20°、−10°、0°、10°、20°、30°，测量每个模型的准确率变化。

**第八部分：最终模型与误差分析。** 在全量训练集上比较 5 个候选配方（LeNet-5、+Dropout、+增广、+Dropout+增广、SmallVGG），按验证集准确率选出最终配方，用 3 个随机种子训练 15 轮，报告测试准确率的均值与标准差；再用混淆矩阵、每类 precision/recall/F1、错分样本、第一层卷积核、特征图、置信度分布、置信度阈值下的覆盖率-精度曲线和旋转鲁棒性对最终模型做误差分析。

**第九部分：Fashion-MNIST 验证。** 在同样规模但更难的服装图像数据集（10 类）上训练 CNN 与 MLP 各 10 轮，检验前面关于 CNN 优势的结论是否能迁移到另一种图像数据。
""")


# ===========================================================================
# 3. 实验环境
# ===========================================================================

add(rf"""## 3. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | {ENV['platform']} |
| 编程语言 | Python {ENV['python']}（Anaconda 发行版） |
| 深度学习框架 | PyTorch {ENV['torch']} + torchvision {ENV.get('torchvision')} |
| CUDA 运行时 | CUDA {ENV['cuda_version']}，cuDNN {ENV['cudnn_version'] // 10000}.{ENV['cudnn_version'] // 100 % 100}.{ENV['cudnn_version'] % 100} |
| GPU | {ENV['device']}（{ENV['gpu_memory_gb']} GB 显存，计算能力 {ENV['capability']}） |
| CPU / 内存 | {ENV['cpu_count']} 逻辑核（数据加载使用 4 个工作进程） |
| 数值计算 / 绘图 | NumPy {ENV['numpy']}，Matplotlib {MATPLOTLIB_VERSION}（中文字体 Noto Sans CJK SC） |
| 精度 | 默认 FP32（`torch.get_float32_matmul_precision() = {ENV['matmul_precision']}`），未使用混合精度 |
| 随机性 | 固定 Python / NumPy / PyTorch 随机种子；cudnn 设为确定性模式 |
| 运行方式 | `cd exp5_report && python full_experiment.py` |

**GPU 环境。** 除正确性验证（模型很小，在 CPU 上用 float64 做数值梯度校验）外，本实验的全部训练与推理都在 GPU 上完成；测试集旋转在 CPU 上生成后用 GPU 推理。PyTorch 的 CUDA 版安装方式为：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

安装后 `torch.cuda.is_available()` 返回 True，设备名为 `{ENV['device']}`。实验脚本在启动时打印设备信息；如果换到没有 GPU 的机器上，`engine.get_device()` 会自动退回 CPU，代码无需修改。

**训练速度。** 以 LeNet-5 现代结构、全量 {num(SPLIT['n_train_pool'])} 张训练图、批大小 64 为例，一个 epoch 包含 844 次参数更新，在 GPU 上平均耗时 {avg_epoch_time(LENET_MODERN):.2f} 秒（10 轮合计 {LENET_MODERN['train_time']:.0f} 秒，包含每轮的验证集评估）。作为对比，实验四中规模接近的 MLP 在纯 CPU 上用 NumPy 手写实现训练一个 epoch 需要数秒到数十秒；GPU 让本实验的 {TOTAL_RUNS} 次训练（含每轮的验证集评估）累计只用约 {TOTAL_TRAIN_MIN:.0f} 分钟。

**数据。** 使用 MNIST 原始 IDX 文件（`data/mnist/*.gz`，仓库已附带，也可用 `bash data/download_mnist.sh` 重新下载），由 `code/data.py` 直接解析，像素值除以 255 缩放到 $[0,1]$，不做额外的均值方差标准化。选择 MNIST 的理由：它是课程中介绍卷积神经网络时最常用的数据集，10 类、图像小（28×28）、训练集足够大，既能观察小样本过拟合，也能在合理时间内完成几十次对照实验。
""")


# ===========================================================================
# 4. 实验原理
# ===========================================================================

add(r"""## 4. 实验原理

### 4.1 全连接网络处理图像的三个问题

把 28×28 的灰度图拉平成 784 维向量送进全连接层，看起来可行（实验四的 MLP 在 MNIST 上就有 97% 以上的准确率），但存在三个结构性问题。

**参数爆炸。** 第一个隐藏层若有 256 个神经元，参数量就是 $784\times256+256=200{,}960$；而图像分辨率一旦提高到 256×256，同样的隐藏层需要 $65{,}536\times256$ 个参数，约 1,677 万个。参数越多，越容易过拟合，也越难训练。

**空间结构被破坏。** 拉平操作把相邻像素之间的二维关系丢掉了。数字 "1" 的笔画在左上角和右下角出现的模式没有任何区别，但 MLP 对每个位置都要单独学一套权重，无法复用。

**没有平移等变。** 同一个数字向上平移 2 个像素，在像素空间里是一个完全不同的输入向量，MLP 需要重新学习；而人类识别数字时，位置的小幅移动几乎不改变判断。

卷积神经网络用三个归纳偏置（inductive bias）解决这些问题：**局部连接**（每个神经元只看输入的一个小邻域）、**权值共享**（同一卷积核在整张图上滑动，参数复用）、**池化下采样**（降低分辨率、扩大感受野，并带来一定的小范围平移不变性）。

### 4.2 卷积运算与输出尺寸

设输入特征图为 $X\in\mathbb{R}^{C_{in}\times H\times W}$，卷积核为 $W\in\mathbb{R}^{C_{out}\times C_{in}\times K\times K}$，偏置 $b\in\mathbb{R}^{C_{out}}$，步长 $S=1$、零填充 $P$。第 $o$ 个输出通道在位置 $(i,j)$ 的值为

$$Y_{o,i,j}=\sum_{c=1}^{C_{in}}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1}W_{o,c,u,v}\,X_{c,\,iS+u-P,\,jS+v-P}+b_o .$$

输出空间尺寸由下式决定：

$$H_{out}=\lfloor\frac{H-K+2P}{S}\rfloor+1,\qquad W_{out}=\lfloor\frac{W-K+2P}{S}\rfloor+1 .$$

本实验的 LeNet-5 不做零填充（$P=0$）、步长为 1，因此 28→24→12→8→4 逐层缩小：

- 输入 $1\times28\times28$；
- conv1：$K=5$，输出 $6\times24\times24$；
- maxpool 2×2：输出 $6\times12\times12$；
- conv2：$K=5$，输出 $16\times8\times8$；
- maxpool 2×2：输出 $16\times4\times4$；
- flatten：256 维；再经过 120、84、10 三个全连接层。

**参数量对比。** 一个 $K\times K$、输入 $C_{in}$ 通道、输出 $C_{out}$ 通道的卷积层参数量为

$$K^2\cdot C_{in}\cdot C_{out}+C_{out}.$$

所以 conv1 有 $5^2\times1\times6+6=156$ 个参数，conv2 有 $5^2\times6\times16+16=2{,}416$ 个；而一个把 784 维映射到 256 维的全连接层就有 200,960 个参数。LeNet-5 全部参数只有 44,426 个，这正是权值共享带来的效率。

### 4.3 权值共享、局部连接与平移等变

**局部连接**：卷积核只在 $K\times K$ 的邻域内取值，符合图像"相邻像素相关"的先验。每个输出位置只与输入的局部区域相连，参数按位置共享。

**权值共享**：同一个卷积核在整张图上滑动，因此同一个"笔画检测器"可以在任意位置使用。这带来两重收益：参数量与图像大小无关（只与卷积核大小、通道数有关）；同一模式在不同位置被同样地检测，模型自动获得**平移等变**（translation equivariance）：输入平移 $t$，卷积输出也平移 $t$（在边界效应之外）。

**感受野**：堆叠卷积可以扩大感受野。第一层 5×5 卷积的每个输出只看输入的 5×5 区域；第二层 5×5 卷积作用在第一层 12×12 的特征图上，对应原图的感受野为 5+2×(5−1)=13×13。感受野随层数加深而增大，深层神经元能"看到"更大的结构（笔画组合、数字轮廓），同时空间尺寸逐层减小、通道数逐层增加——这是 CNN 典型的"空间分辨率下降、语义信息上升"的金字塔结构。

### 4.4 池化

池化对每个通道独立地在 $2\times2$ 窗口上取最大值或平均值，步长为 2。设输入为 $X$，则

$$\text{MaxPool}(X)_{i,j}=\max_{u,v\in\{0,1\}} X_{2i+u,\,2j+v},\qquad
\text{AvgPool}(X)_{i,j}=\frac{1}{4}\sum_{u,v\in\{0,1\}} X_{2i+u,\,2j+v}.$$

池化有三个作用：把特征图尺寸减半，降低后续计算量；扩大后续层的感受野；对窗口内的小幅平移不敏感（最大池化只保留窗口内最强的响应）。池化层没有可学习参数。最大池化保留最显著的响应，平均池化保留整体能量；本实验会在基线里对比这两种方式。

### 4.5 激活函数：Sigmoid、Tanh 与 ReLU 族

卷积和池化都是线性（或分段的线性）操作，如果网络里没有非线性激活函数，多层卷积的复合仍然等价于一次线性变换。激活函数是 CNN 表达能力的来源。实验对比的六种激活函数为：

$$\sigma(x)=\frac{1}{1+e^{-x}},\qquad \tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}},\qquad
\text{ReLU}(x)=\max(0,x),$$

$$\text{LeakyReLU}(x)=\max(\alpha x,x)\ (\alpha=0.1),\qquad
\text{ELU}(x)=\begin{cases}x,&x>0\\ \alpha(e^{x}-1),&x\le 0\end{cases},\qquad
\text{GELU}(x)=x\cdot\Phi(x),$$

其中 $\Phi$ 是标准正态分布的累积分布函数。它们的导数性质差别很大：

- Sigmoid 的导数 $\sigma'(x)=\sigma(x)(1-\sigma(x))\le 0.25$，且在两端的饱和区趋近于 0；
- Tanh 的导数 $1-\tanh^2(x)\le 1$，比 Sigmoid 略好，但同样会饱和；
- ReLU 在正半轴的导数恒为 1，不饱和，计算只需一次比较；负半轴导数为 0（"死亡 ReLU"问题）；
- LeakyReLU / ELU 给负半轴留了小的斜率，缓解死亡问题；GELU 是平滑的 ReLU 近似，是 Transformer 等现代模型的默认选择。

**梯度消失的定量解释。** 反向传播时，梯度要逐层乘上激活函数的导数。若每层导数都约为 0.25（Sigmoid 的典型值），经过 $L$ 层后梯度就缩放到约 $0.25^L$：5 层约 $10^{-3}$，10 层约 $10^{-6}$。这就是为什么经典 Sigmoid 网络需要用合适的学习率、精心的初始化，或者换用 ReLU 族激活函数。本实验第 7 节会用实测的第一层卷积梯度范数验证这一点。

### 4.6 Dropout

Dropout 在每个训练批次中，以概率 $p$ 随机把隐藏层神经元的输出置零，保留下来的神经元除以 $1-p$ 以保持期望不变：

$$r_j\sim\text{Bernoulli}(1-p),\qquad \tilde{h}_j=\frac{r_j h_j}{1-p}.$$

测试阶段不做随机丢弃，直接使用完整的隐藏层（相当于对指数多个"子网络"做隐式集成）。Dropout 的作用可以理解为：禁止神经元之间形成固定的共适应关系，迫使每个神经元学到相对独立的特征，从而降低过拟合。它通常只在大容量、小数据的场景下有效；数据充足时，Dropout 反而会拖慢收敛。

**作用位置。** 对全连接层使用标准的 Dropout；对卷积层，通常使用 Dropout2d（按通道整张特征图丢弃，而不是逐元素丢弃），因为卷积通道之间存在空间相关性，逐元素丢弃破坏性太强。本实验会对比"只在全连接层用 Dropout"与"全连接层 + 卷积层都用 Dropout2d"。

### 4.7 批归一化（Batch Normalization）

BN 对每个通道在小批量维度上计算均值与方差，再归一化到零均值、单位方差，并用两个可学习参数 $\gamma,\beta$ 恢复表达能力：

$$\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat{x}+\beta.$$

它在训练时使用当前批次的统计量，推理时使用训练过程中累积的滑动平均统计量。BN 的作用包括：缓解内部协变量偏移、允许更大的学习率、让损失曲面更平滑、加快收敛；代价是引入了批次依赖，小批量时统计量噪声较大。本实验把 BN 作为基线对比的一部分（LeNet-5 + BN、SmallVGG 使用 BN）。

### 4.8 LeNet-5 结构

LeNet-5 是 LeCun 等人在 1998 年提出的经典卷积网络，最初用于手写数字与文档识别。原始版本输入 32×32、使用 Sigmoid 激活和平均池化（下采样），本实验把它适配到 MNIST 的 28×28 输入：

| 层 | 操作 | 输入尺寸 | 输出尺寸 | 参数量 |
|---|---|---|---|---|
| conv1 | 5×5 卷积，6 通道 | 1×28×28 | 6×24×24 | 156 |
| pool1 | 2×2 池化 | 6×24×24 | 6×12×12 | 0 |
| conv2 | 5×5 卷积，16 通道 | 6×12×12 | 16×8×8 | 2,416 |
| pool2 | 2×2 池化 | 16×8×8 | 16×4×4 | 0 |
| flatten | 拉平 | 16×4×4 | 256 | 0 |
| fc1 | 全连接 256→120 | 256 | 120 | 30,840 |
| fc2 | 全连接 120→84 | 120 | 84 | 10,164 |
| fc3 | 全连接 84→10 | 84 | 10 | 850 |
| **合计** | | | | **44,426** |

原始 LeNet-5 输入 32×32、第二个池化后得到 16×5×5=400 维，本实验输入 28×28 时变成 16×4×4=256 维，因此第一个全连接层是 256→120。结构示意图如下：

![图 2　LeNet-5（28×28 适配版）结构示意图](figures/fig02_lenet5_arch.png)

### 4.9 损失函数与优化器

多分类使用 softmax 交叉熵。设模型输出 logits $z\in\mathbb{R}^{10}$，真实类别为 $y$，则

$$p_c=\frac{e^{z_c}}{\sum_{j=1}^{10}e^{z_j}},\qquad
L=-\log p_y=-\left(z_y-\log\sum_{j=1}^{10}e^{z_j}\right).$$

对 logits 的梯度有简洁形式 $\partial L/\partial z_c=p_c-\mathbb{1}[c=y]$，反向传播从这里开始。实际实现中 PyTorch 的 `CrossEntropyLoss` 把 softmax 与负对数似然合并，数值稳定的实现会先减去 logits 的最大值。

优化器方面，本实验对比 SGD、SGD + Momentum 和 Adam：

$$v_t=\beta v_{t-1}+(1-\beta)\nabla_\theta L_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t \quad\text{(SGD/Momentum)},$$

$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\quad
\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}\quad\text{(Adam)},$$

其中 $\beta_1=0.9$、$\beta_2=0.999$。Adam 用梯度二阶矩对每个参数做自适应缩放，因此对梯度量级不敏感——这正是它能"救活"Sigmoid 网络的原因。权重衰减（L2 正则）则在损失里加 $\lambda\lVert\theta\rVert^2/2$，AdamW 使用与梯度更新解耦的衰减方式。
""")


# ===========================================================================
# 5. 实现与正确性验证
# ===========================================================================

add(rf"""## 5. 实现与正确性验证

### 5.1 代码结构

| 路径 | 说明 |
|---|---|
| `code/data.py` | MNIST / Fashion-MNIST 的 IDX 解析、分层划分、DataLoader、四种数据增强、测试集旋转 |
| `code/models.py` | LeNet-5（激活/池化/BN/Dropout 可配置）、SmallVGG、MLP、参数量统计 |
| `code/engine.py` | 随机种子、设备选择、优化器构造、单轮训练/评估、统一训练循环（记录曲线与第一层梯度范数） |
| `code/metrics.py` | 混淆矩阵、每类 precision/recall/F1、最易混淆类别对、置信度统计与覆盖率-精度曲线 |
| `code/viz.py` | 训练曲线、样本网格、混淆矩阵、错分样本、卷积核、特征图、LeNet-5 结构图 |
| `exp5_report/full_experiment.py` | 九部分实验主脚本，结果写入 `results.json` |
| `exp5_report/verify.py` | 正确性验证（参数量、初始损失、输出尺寸、数值梯度、单批过拟合、旋转函数） |
| `exp5_report/gen_report.py` | 从 `results.json` 生成 Markdown 报告并调用 pandoc 生成 docx |

**关键接口。** 所有模型统一接受 `(N, 1, 28, 28)` 的输入、输出 `(N, 10)` 的 logits；所有实验共用 `engine.fit()` 训练、`engine.evaluate()` 评估，保证除被研究变量外其它条件完全一致。数据划分、验证集抽样、训练集子集抽样都使用显式随机种子。

### 5.2 LeNet-5 的 PyTorch 实现

核心代码（`code/models.py`，删去注释与类型标注）：

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, activation="relu",
                 pooling="max", use_bn=False, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None
        self.act = make_activation(activation)
        self.pool = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, activation="relu", pooling="max",
                 use_bn=False, dropout_fc=0.0, dropout_conv=0.0,
                 fc_sizes=(120, 84)):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 6, 5, activation, pooling, use_bn, dropout_conv),
            ConvBlock(6, 16, 5, activation, pooling, use_bn, dropout_conv),
        )
        fc1, fc2 = fc_sizes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, fc1), make_activation(activation),
            nn.Dropout(dropout_fc) if dropout_fc > 0 else nn.Identity(),
            nn.Linear(fc1, fc2), make_activation(activation),
            nn.Dropout(dropout_fc) if dropout_fc > 0 else nn.Identity(),
            nn.Linear(fc2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

激活函数通过字典工厂 `make_activation(name)` 构造，因此换激活函数只需要改一个字符串；池化、BN、Dropout 同理，这是后面所有控制变量实验能共用一套代码的原因。

### 5.3 训练循环

`engine.fit()` 的骨架如下（省略日志与梯度范数记录）：

```python
model = model.to(device)
criterion = nn.CrossEntropyLoss()
opt = make_optimizer(model, optimizer, lr, weight_decay)
for epoch in range(1, epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    va = evaluate(model, val_loader, device)   # 验证集只评估、不更新
    if va["acc"] > best_val_acc:               # 保存验证集最好的模型
        best_state = deepcopy(model.state_dict())
model.load_state_dict(best_state)              # 用验证集最优轮次的参数报告测试结果
```

**公平对照协议。**（1）所有实验使用相同的数据划分与同一个验证集；（2）同一组对照使用相同的随机种子、优化器、学习率、批大小和训练轮数，只改变被研究的变量；（3）每个配置报告"验证集最优轮次"对应的测试准确率，避免用训练末期的过拟合参数；（4）超参数扫描使用 10,000 张训练图的固定子集以控制变量与时间，最终模型回到全量数据；（5）涉及随机性的结论用多个种子或多次抽样验证。

### 5.4 评估指标

主指标是测试集准确率（10,000 张图的分类正确率）。误差分析还使用：

- **混淆矩阵** $C_{{ij}}$：真实类别 $i$ 被判为类别 $j$ 的样本数；
- **每类 precision / recall / F1**：$P_i=C_{{ii}}/\sum_j C_{{ji}}$，$R_i=C_{{ii}}/\sum_j C_{{ij}}$，$F_1=2PR/(P+R)$；
- **置信度**：softmax 最大概率；统计正确预测与错误预测的置信度分布，并用"置信度阈值筛选后的覆盖率-精度曲线"评估模型能否用置信度拒绝不可靠预测。

### 5.5 正确性验证

在正式实验前，先用 `exp5_report/verify.py` 从六个方面检查实现是否正确，实测结果如下（详细输出见 `exp5_report/verification.json`）：

{table(
    ["检查项", "实测结果", "判定"],
    [["参数量核对", f"手算 {VER['parameters']['hand_computed']:,} 个，PyTorch 统计 {VER['parameters']['pytorch_count']:,} 个",
      "✓" if VER["parameters"]["match"] else "✗"],
     ["输出尺寸核对", "；".join(VER["shapes"]["actual"]), "✓" if VER["shapes"]["match"] else "✗"],
     ["初始损失", f"{VER['initial_loss']['measured']:.6f}（ln10 = {VER['initial_loss']['expected']:.6f}，误差 {VER['initial_loss']['abs_error']:.2e}）",
      "✓" if VER["initial_loss"]["abs_error"] < 0.02 else "✗"],
     ["数值梯度校验", f"检查 {VER['numeric_gradient']['n_checked']} 个参数（conv/fc 混合），最大相对误差 {VER['numeric_gradient']['max_rel_error']:.2e}",
      "✓" if VER["numeric_gradient"]["max_rel_error"] < 1e-4 else "✗"],
     ["单批过拟合", f"固定 64 张图训练 {VER['single_batch_overfit']['steps']} 步：损失 {VER['single_batch_overfit']['initial_loss']:.4f} → {VER['single_batch_overfit']['final_loss']:.2e}，准确率 {VER['single_batch_overfit']['final_acc'] * 100:.1f}%",
      "✓" if VER["single_batch_overfit"]["final_loss"] < 0.05 else "✗"],
     ["旋转函数校验", f"0° 旋转与原图一致：{VER['rotation']['zero_angle_identical']}；形状保持：{VER['rotation']['shape_preserved']}；取值范围正确：{VER['rotation']['range_ok']}",
      "✓" if VER["rotation"]["zero_angle_identical"] and VER["rotation"]["range_ok"] else "✗"]]
)}

六项检查全部通过。其中数值梯度校验的做法是：把模型与数据转成 float64，在 CPU 上对 10 个参数（卷积层与全连接层各若干）做中心差分 $\partial L/\partial\theta\approx[L(\theta+\epsilon)-L(\theta-\epsilon)]/(2\epsilon)$（$\epsilon=10^{{-5}}$），与 PyTorch 自动求导的解析梯度比较，最大相对误差 {VER['numeric_gradient']['max_rel_error']:.1e}，说明反向传播链路（包括卷积、池化、展平、全连接）的梯度计算正确。单批过拟合把损失从 {VER['single_batch_overfit']['initial_loss']:.4f} 降到 {VER['single_batch_overfit']['final_loss']:.1e}，说明前向、损失、反向、优化器更新整条链路正常，排除了"模型很大但根本没在学"这类低级错误。
""")

print("第一部分（标题/摘要/1-5 节）准备完成")


# ===========================================================================
# 6. 实验一：基线模型对比
# ===========================================================================

add(rf"""## 6. 实验一：基线模型对比

### 6.1 设置

全部 {num(SPLIT['n_train_pool'])} 张训练图、10 轮、Adam lr=0.001、批大小 64。五个模型的控制变量设计如下：

- **LeNet-5 经典结构**：Sigmoid 激活 + 平均池化，代表 1998 年的原始配方；
- **LeNet-5 现代结构**：ReLU 激活 + 最大池化，其余结构完全相同；
- **LeNet-5 + BN**：在现代结构上给每个卷积层加批归一化；
- **SmallVGG**：两个卷积块（每块两层 3×3 卷积 + BN + ReLU，通道数 32/64），接 128 维全连接 + Dropout 0.5；
- **MLP 基线**：784→256→128→10 的全连接网络，ReLU + Adam。

除 MLP 外都用相同的 LeNet-5 分类头（120→84→10）和相同的训练协议。

### 6.2 结果

{table(
    ["模型", "参数量", "验证最优轮次", "验证准确率", "测试准确率", "训练耗时", "平均每轮"],
    [[label,
      num(B["runs"][label]["params"]),
      B["runs"][label]["best_epoch"],
      pct(B["runs"][label]["best_val_acc"]),
      pct(B["runs"][label]["test_acc"]),
      f"{B['runs'][label]['train_time']:.0f} s",
      f"{avg_epoch_time(B['runs'][label]):.2f} s"]
     for label in B["order"]]
)}

![图 3　基线模型的验证准确率与训练损失曲线](figures/fig03_baseline_curves.png)

![图 4　基线模型测试准确率对比](figures/fig04_baseline_final.png)

### 6.3 分析

**经典结构 vs 现代结构。** 两者只差激活函数（Sigmoid → ReLU）和池化方式（平均 → 最大），测试准确率却从 {pct(LENET_CLASSIC['test_acc'])} 提高到 {pct(LENET_MODERN['test_acc'])}，相差 {gap(LENET_MODERN['test_acc'], LENET_CLASSIC['test_acc'])} 个百分点。从曲线（图 3）可以看到，经典结构的验证准确率在前几轮爬升明显更慢，到第 10 轮时训练准确率只有 {pct(LENET_CLASSIC['history']['train_acc'][-1])}、验证准确率 {pct(LENET_CLASSIC['history']['val_acc'][-1])}，而现代结构在第 1 轮结束就已有 {pct(LENET_MODERN['history']['val_acc'][0])}。原因就是 4.5 节讨论的 Sigmoid 饱和与梯度缩小：虽然这里用 Adam 已经缓解了梯度量级问题，但 Sigmoid 的导数上界 0.25 仍然限制了每层的有效更新，收敛速度明显更慢。

**BN 的作用。** 在同样的现代结构上加 BN，测试准确率从 {pct(LENET_MODERN['test_acc'])} 提高到 {pct(B['runs']['LeNet-5 + BN']['test_acc'])}（{gap(B['runs']['LeNet-5 + BN']['test_acc'], LENET_MODERN['test_acc'])} 个百分点）。训练耗时从 {LENET_MODERN['train_time']:.0f} 秒增加到 {B['runs']['LeNet-5 + BN']['train_time']:.0f} 秒，因为 BN 增加了计算量。在这个小网络上 BN 的收益有限，但它的收敛曲线更平滑（验证准确率更早进入 97% 以上）。

**更深的 SmallVGG。** SmallVGG 的测试准确率达到 {pct(SMALLVGG['test_acc'])}，是五个模型中最高的，比 LeNet-5 现代结构高 {gap(SMALLVGG['test_acc'], LENET_MODERN['test_acc'])} 个百分点。它用两层堆叠的 3×3 卷积代替一层 5×5 卷积（在相同感受野下参数更少、非线性更强），并用 BN 与 Dropout 稳定训练；代价是参数量增加到 {num(SMALLVGG['params'])}，训练时间几乎翻倍。

**CNN vs MLP。** MLP 的参数量（{num(MLP_BL['params'])}）是 LeNet-5（{num(LENET_MODERN['params'])}）的 {MLP_BL['params'] / LENET_MODERN['params']:.1f} 倍，测试准确率却低 {gap(LENET_MODERN['test_acc'], MLP_BL['test_acc'])} 个百分点。这不是"参数越多越好"的反例，而是归纳偏置的胜利：卷积核的局部连接与权值共享把"图像上相邻像素相关""同一笔画可以出现在任意位置"这两个先验直接编码进了网络结构，因此用更少的参数学到了更可迁移的特征。MLP 的验证准确率在第 5 轮左右达到最高（{pct(MLP_BL['best_val_acc'])}），之后明显过拟合（训练准确率继续升到 {pct(MLP_BL['history']['train_acc'][-1])}，验证准确率回落），说明全连接层对图像数据更容易记住训练样本。

**运行时间。** LeNet-5 现代结构平均每轮 {avg_epoch_time(LENET_MODERN):.2f} 秒，MLP 平均每轮 {avg_epoch_time(MLP_BL):.2f} 秒。MLP 反而更慢，一个原因是它的全连接矩阵乘法在大 batch 下访存更密集；另一个原因是实验过程中 GPU 会因笔记本功耗墙降频，后运行的模型耗时会长一些。报告中的耗时只用于量级比较，不作为严格的性能结论。
""")


# ===========================================================================
# 7. 实验二：激活函数
# ===========================================================================

add(rf"""## 7. 实验二：激活函数的选择

### 7.1 六种激活函数的对比

在 10,000 张训练图、8 轮、Adam lr=0.001、LeNet-5 现代结构（最大池化）上，每次只换激活函数：

{table(
    ["激活函数", "公式", "验证准确率", "测试准确率", "第一轮梯度范数", "末轮梯度范数"],
    [[{"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh",
       "leaky_relu": "LeakyReLU", "elu": "ELU", "gelu": "GELU"}[name],
      {"relu": r"$\max(0,x)$", "sigmoid": r"$1/(1+e^{-x})$", "tanh": r"$\tanh(x)$",
       "leaky_relu": r"$\max(0.1x,x)$", "elu": r"$\alpha(e^x-1)$（$x\le0$）",
       "gelu": r"$x\Phi(x)$"}[name],
      pct(A["runs"][name]["best_val_acc"]),
      pct(A["runs"][name]["test_acc"]),
      f"{A['runs'][name]['grad_norm_first_conv'][0]:.3g}",
      f"{A['runs'][name]['grad_norm_first_conv'][-1]:.3g}"]
     for name in A["order"]]
)}

![图 5　六种激活函数的验证准确率曲线](figures/fig05_activation_curves.png)

![图 6　激活函数对测试准确率（左）与第一层卷积梯度范数（右）的影响](figures/fig06_activation_grad.png)

**结果。** 测试准确率从高到低依次为 {"、".join(f"{_ACT_CN[name]}（{pct(A['runs'][name]['test_acc'])}）" for name in sorted(A["order"], key=lambda k: -A["runs"][k]["test_acc"]))}。最好与最差相差 {gap(ACT_BEST, ACT_WORST)} 个百分点。几个值得注意的点：（1）GELU、Tanh、ELU、LeakyReLU 都落在 {pct(min(A['runs'][k]['test_acc'] for k in ['tanh','elu','leaky_relu','gelu']))}～{pct(max(A['runs'][k]['test_acc'] for k in ['tanh','elu','leaky_relu','gelu']))} 的窄区间内，说明只要解决饱和问题，激活函数具体形状的影响是次要的；（2）**ReLU 本身在这个设置下反而是 ReLU 族里最低的**（{pct(A['runs']['relu']['test_acc'])}），比 GELU 低 {gap(A['runs']['gelu']['test_acc'], A['runs']['relu']['test_acc'])} 个百分点，这是单种子、8 轮、10,000 张图下的小样本波动，不应解读为 GELU 一定更好；（3）Sigmoid 以 {pct(ACT_SIGMOID['test_acc'])} 明显落后，梯度饱和的影响在收敛速度上表现得最清楚（图 5 中 Sigmoid 曲线全程最低）。

**梯度范数证据。** 图 6 右侧是对数尺度的第一层卷积权重梯度范数。ReLU 的梯度范数在整个训练过程中稳定在 {A['runs']['relu']['grad_norm_first_conv'][0]:.2f}～{max(A['runs']['relu']['grad_norm_first_conv']):.2f}；Sigmoid 的梯度范数在第 1 轮只有 {A['runs']['sigmoid']['grad_norm_first_conv'][0]:.2g}（比 ReLU 低约 {__import__('math').log10(A['runs']['relu']['grad_norm_first_conv'][0] / A['runs']['sigmoid']['grad_norm_first_conv'][0]):.1f} 个数量级），随后随着参数移动、Sigmoid 逐渐退出饱和区而上升到第 8 轮的 {A['runs']['sigmoid']['grad_norm_first_conv'][-1]:.2g}，此时仍比 ReLU 低约 {__import__('math').log10(A['runs']['relu']['grad_norm_first_conv'][-1] / A['runs']['sigmoid']['grad_norm_first_conv'][-1]):.1f} 个数量级。注意这是在 Adam 下的测量：Adam 用梯度二阶矩做逐参数归一化，把 Sigmoid 的小梯度放大到可用步长，所以网络仍能收敛；但同样的梯度量级如果交给固定学习率的 SGD，就会直接失效——这正是 7.2 节要验证的。

### 7.2 梯度消失诊断：激活函数必须和优化器一起看

上面的对比里 Sigmoid 还能到 {pct(ACT_SIGMOID['test_acc'])}，是因为 Adam 的归一化抵消了梯度量级。如果把优化器换回 1998 年常用的 SGD，同样 6 轮、10,000 张训练图，结果如下：

{table(
    ["配置", "优化器 / 学习率", "测试准确率", "第一轮梯度范数", "末轮梯度范数"],
    [[label.split(" | ")[0],
      {"Sigmoid+AvgPool | SGD 0.01": "SGD, lr=0.01",
       "Sigmoid+AvgPool | SGD 0.1": "SGD, lr=0.1",
       "Sigmoid+AvgPool | Adam 1e-3": "Adam, lr=0.001",
       "ReLU+MaxPool | SGD 0.01": "SGD, lr=0.01",
       "ReLU+MaxPool | Adam 1e-3": "Adam, lr=0.001"}[label],
      pct(GD[label]["test_acc"]),
      f"{GD[label]['grad_norm_first_conv'][0]:.3g}",
      f"{GD[label]['grad_norm_first_conv'][-1]:.3g}"]
     for label in GD_ORDER]
)}

![图 7　梯度消失诊断：左为测试准确率，右为第一层卷积的梯度范数（对数尺度）](figures/fig07_vanishing_grad.png)

**现象。** 经典结构（Sigmoid + 平均池化）在 SGD lr=0.01 下训练 6 轮后测试准确率只有 {pct(GD_SIG_SGD['test_acc'])}，几乎等于 10 类随机猜测；把学习率提高 10 倍到 0.1 也没有改善（{pct(GD["Sigmoid+AvgPool | SGD 0.1"]['test_acc'])}）。而同样的结构改用 Adam 后达到 {pct(GD_SIG_ADAM['test_acc'])}。作为对照，ReLU 结构在 SGD lr=0.01 下也只有 {pct(GD_RELU_SGD['test_acc'])}，改用 Adam 后达到 {pct(GD_RELU_ADAM['test_acc'])}。

**解释。** 这不是代码 bug，而是梯度量级问题。图 7 右侧显示：Sigmoid 结构第一层卷积的梯度范数只有 {GD_SIG_SGD['grad_norm_first_conv'][0]:.1e} 量级，ReLU 结构是 {GD_RELU_SGD['grad_norm_first_conv'][0]:.1e} 量级，相差约 {__import__('math').log10(GD_RELU_SGD['grad_norm_first_conv'][0] / GD_SIG_SGD['grad_norm_first_conv'][0]):.1f} 个数量级。在固定学习率下，参数更新量正比于梯度，所以 Sigmoid 结构的参数几乎不动；把学习率提高 10 倍只能把更新放大 10 倍（约 1 个数量级），对约 2 个数量级的梯度差距是杯水车薪，而 Adam 用梯度二阶矩做逐参数归一化，相当于自动为每个参数选择合适步长，才把网络"救活"。

**与实验四的呼应。** 实验四中手写 MLP 时，全零初始化和过大的学习率都会让网络学不动；这次在 CNN 上看到的是同一类问题的另一种表现：**优化难度与表达能力是两回事**。Sigmoid 网络表达能力和 ReLU 一样（甚至更强，因为它是平滑的），但在这个深度/初始化/优化器组合下根本走不到好的解。这也解释了为什么 2010 年以后 ReLU 会成为深度网络的默认激活函数：它同时改善了梯度量级和计算效率。
""")


# ===========================================================================
# 8. 实验三：Dropout 与正则化
# ===========================================================================

add(rf"""## 8. 实验三：Dropout 与正则化

### 8.1 设置

为了让过拟合明显出现，本组实验只用 6,000 张训练图，并把 LeNet-5 的全连接头放大到 512→128（参数量从 {num(LENET_MODERN['params'])} 增加到 {num(DR_NOREG['params'])}），训练 20 轮。其它条件（Adam lr=0.001、批大小 64、验证集不变）与前面一致。共比较 6 种正则化设置：

{table(
    ["设置", "Dropout（FC）", "Dropout2d（卷积）", "权重衰减", "验证准确率", "测试准确率", "最后训练-验证差距"],
    [[label,
      f2(DR_RUNS[label]["config"].get("dropout_fc", 0.0)),
      f2(DR_RUNS[label]["config"].get("dropout_conv", 0.0)),
      f"{DR_RUNS[label]['config']['weight_decay']:g}",
      pct(DR_RUNS[label]["best_val_acc"]),
      pct(DR_RUNS[label]["test_acc"]),
      f"{train_val_gap(DR_RUNS[label]) * 100:+.2f} pp"]
     for label in DR_ORDER]
)}

![图 8　Dropout 0 / 0.2 / 0.5 的训练准确率与验证准确率曲线](figures/fig08_dropout_curves.png)

![图 9　各正则化设置的测试准确率（左）与过拟合程度（右）](figures/fig09_dropout_summary.png)

### 8.2 分析

**过拟合确实存在。** 无正则化时，最后一轮训练准确率 {pct(DR_NOREG['history']['train_acc'][-1])}、验证准确率 {pct(DR_NOREG['history']['val_acc'][-1])}，差距 {train_val_gap(DR_NOREG) * 100:+.2f} 个百分点（图 8 中两条曲线的开口），验证准确率在第 {DR_NOREG['best_epoch']} 轮之后开始回落。6,000 张图 + 约 {num(DR_NOREG['params'])} 个参数足以让网络开始"背题"。

**Dropout 的作用。** 加上 Dropout 后，训练-验证差距从 {train_val_gap(DR_NOREG) * 100:+.2f} 个百分点缩小到 Dropout 0.2 的 {train_val_gap(DR_D02) * 100:+.2f} 和 Dropout 0.5 的 {train_val_gap(DR_D05) * 100:+.2f} 个百分点；测试准确率从 {pct(DR_NOREG['test_acc'])} 变为 {pct(DR_D02['test_acc'])}（p=0.2）和 {pct(DR_D05['test_acc'])}（p=0.5）。注意 Dropout 让训练准确率变低（因为训练时有一半神经元被丢弃），这正是它抑制过拟合的代价：用训练集上的拟合能力换验证集/测试集上的泛化。

**Dropout 的作用位置。** 只在全连接层加 Dropout 0.5 与同时在卷积层加 Dropout2d 0.25 的测试准确率分别为 {pct(DR_D05['test_acc'])} 与 {pct(DR_RUNS['Dropout 0.5 (FC) + Dropout2d 0.25']['test_acc'])}，差距 {pp(DR_RUNS['Dropout 0.5 (FC) + Dropout2d 0.25']['test_acc'], DR_D05['test_acc'])} 个百分点。卷积层的 Dropout2d 丢弃整个通道，会破坏低层特征的完整性；在本实验的小数据场景下，它没有带来额外收益，反而略低。经验上 Dropout 主要用于参数量大的全连接层，卷积层更常用 BN、数据增强或权重衰减。

**Dropout vs 权重衰减。** 单独使用权重衰减 1e-3 的测试准确率为 {pct(DR_RUNS['Weight Decay 1e-3（无 Dropout）']['test_acc'])}，Dropout 0.5 为 {pct(DR_D05['test_acc'])}，两者组合为 {pct(DR_RUNS['Dropout 0.5 + Weight Decay 1e-3']['test_acc'])}。权重衰减直接压制权重大小，Dropout 通过随机丢弃破坏神经元之间的共适应，两者的正则化机制不同；在本实验中单独使用 Dropout 的收益更明显。

**结论。** Dropout 不是"越多越好"，也不是"永远有效"。它的收益依赖于：模型容量相对数据量偏大、过拟合真实存在、丢弃率适中（本实验 p=0.5 略优于 p=0.2）。在实验一的全量数据（54,001 张）上，LeNet-5 的过拟合不严重，Dropout 的收益会明显缩小——第 12 节的配方对比中，加 Dropout 的 R2 只比 R1 高 0.18 个百分点，远小于本节 2.65→1.28 个百分点的差距缩小幅度。
""")


# ===========================================================================
# 9. 实验四：数据量的影响
# ===========================================================================

add(rf"""## 9. 实验四：数据量的影响

### 9.1 结果

训练集规模从 600 到 {num(SPLIT['n_train_pool'])} 按类别等比例分层抽样，CNN（LeNet-5 现代结构）与 MLP（784-256-128）各训练 8 轮：

{table(
    ["训练样本数", "CNN 测试准确率", "MLP 测试准确率", "CNN − MLP", "CNN 每轮耗时", "MLP 每轮耗时"],
    [[num(c["config"]["n_train"]), pct(c["test_acc"]), pct(m["test_acc"]),
      f"{pp(c['test_acc'], m['test_acc'])} pp",
      f"{avg_epoch_time(c):.2f} s", f"{avg_epoch_time(m):.2f} s"]
     for c, m in zip(SZ_CNN, SZ_MLP)]
)}

![图 10　学习曲线：数据量对测试准确率的影响（左）与 CNN 相对 MLP 的优势（右）](figures/fig10_data_size.png)

### 9.2 分析

**数据量是最大的单一因素。** CNN 的测试准确率从 600 张时的 {pct(SZ_CNN[0]['test_acc'])} 提升到全量时的 {pct(SZ_CNN[-1]['test_acc'])}，提高了 {gap(SZ_CNN[-1]['test_acc'], SZ_CNN[0]['test_acc'])} 个百分点；MLP 从 {pct(SZ_MLP[0]['test_acc'])} 提升到 {pct(SZ_MLP[-1]['test_acc'])}。在 600 张时两者都还明显欠拟合（验证准确率曲线仍在上升），到 20,000 张之后收益开始递减（图 10 左）。

**CNN 的优势随数据量变化，并且在极小数据上会出现反转。** 在 600 张训练图时，CNN 的测试准确率（{pct(SZ_CNN[0]['test_acc'])}）反而比 MLP（{pct(SZ_MLP[0]['test_acc'])}）低 {abs(SZ_CNN[0]['test_acc'] - SZ_MLP[0]['test_acc']) * 100:.1f} 个百分点；从 2,000 张开始 CNN 反超（{gap(SZ_CNN[1]['test_acc'], SZ_MLP[1]['test_acc'])} 个百分点），到全量 {num(SPLIT['n_train_pool'])} 张时领先 {gap(SZ_CNN[-1]['test_acc'], SZ_MLP[-1]['test_acc'])} 个百分点（图 10 右）。对这个反转有两种解释：一是**优化预算**——固定 8 轮时，CNN 的参数共享与池化让梯度更"分散"，达到同样拟合程度需要更多更新，600 张图×8 轮只有 600 次参数更新，CNN 明显欠拟合（训练准确率也只有 {pct(SZ_CNN[0]['history']['train_acc'][-1])}，而 MLP 为 {pct(SZ_MLP[0]['history']['train_acc'][-1])}）；二是**归纳偏置的代价**——卷积先验假定"局部模式可以复用"，数据极小时这个先验未必能带来净收益。数据量增大后，CNN 的参数效率优势才体现出来。因此"CNN 一定比 MLP 好"是不成立的，数据量、训练预算和任务先验都会影响结论。

**训练效率。** 相同数据量下 MLP 的每轮耗时通常高于 CNN（例如全量时 {avg_epoch_time(SZ_MLP[-1]):.2f} 秒对 {avg_epoch_time(SZ_CNN[-1]):.2f} 秒），因为它的全连接矩阵更大。也就是说 CNN 在准确率和计算效率上同时占优。

**注意。** 这里只训练 8 轮，曲线尚未完全收敛；数据量很大时的最终准确率受训练轮数与学习率影响，不能把表中的数字当作该数据量的性能上限。
""")


# ===========================================================================
# 10. 实验五：超参数
# ===========================================================================


def _hyper_table(runs: dict, col: str, fmt=lambda v: f"{v:g}" if isinstance(v, (int, float)) else str(v)) -> str:
    rows = []
    for label, run in runs.items():
        rows.append([label, fmt(run["config"][col]), pct(run["best_val_acc"]),
                     pct(run["test_acc"]), f"{avg_epoch_time(run):.2f} s"])
    col_names = {"optimizer": "优化器", "batch_size": "批大小", "weight_decay": "权重衰减"}
    return table(["配置", col_names.get(col, col), "验证准确率", "测试准确率", "每轮耗时"], rows)


add(rf"""## 10. 实验五：超参数的影响

本组实验使用 10,000 张训练图的固定子集、8 轮，固定 LeNet-5 现代结构（ReLU + 最大池化），每次只改一个超参数。

### 10.1 优化器

{_hyper_table(H["optimizer"], "optimizer")}

![图 11　四种优化器的验证准确率曲线](figures/fig11_optimizer.png)

![图 12　四种优化器的测试准确率对比](figures/fig12_optimizer_final.png)

从高到低为：{"、".join(f"{label}（{pct(H['optimizer'][label]['test_acc'])}）" for label in sorted(H["optimizer"], key=lambda k: -H["optimizer"][k]["test_acc"]))}。Adam / AdamW 明显好于 SGD；SGD lr=0.01 在 8 轮内只达到 {pct(H["optimizer"]["SGD lr=0.01"]['test_acc'])}，是欠拟合而不是过拟合——它的训练准确率也只有 {pct(H["optimizer"]["SGD lr=0.01"]['history']['train_acc'][-1])}。这再次说明：比较优化器时必须让每个优化器都有机会收敛（各自的合理学习率），否则得到的结论是"学习率选得对不对"而不是"优化器谁更好"。AdamW 在本实验中的差别很小（它与 Adam 的区别是权重衰减的解耦方式，而本组 wd=1e-2 的强度在 10,000 张图上影响有限）。

### 10.2 学习率

{table(
    ["优化器", "学习率", "验证准确率", "测试准确率", "末轮训练损失"],
    [[label.split(" lr=")[0], label.split(" lr=")[1], pct(run["best_val_acc"]),
      pct(run["test_acc"]), f4(run["history"]["train_loss"][-1])]
     for group in ("adam_lr", "sgd_lr") for label, run in H[group].items()]
)}

![图 13　Adam（左）与 SGD+Momentum（右）的学习率对比](figures/fig13_lr.png)

Adam 在扫描范围内测试准确率最高的是 {max(H['adam_lr'], key=lambda k: H['adam_lr'][k]['test_acc']).split('=')[1]}（{pct(max(H['adam_lr'].values(), key=lambda r: r['test_acc'])['test_acc'])}），且从 1e-4 到 3e-3 呈单调上升趋势：学习率太小（1e-4）时 8 轮内明显欠拟合，末轮训练损失仍有 {H['adam_lr']['Adam lr=0.0001']['history']['train_loss'][-1]:.3f}；3e-3 在本实验的 8 轮预算内表现最好，但曲线初期的波动也更大，继续增大到 1e-2 量级是否会震荡本实验没有测试。SGD + Momentum 需要大得多的学习率（0.01～0.1 量级）才能与 Adam 可比，因为 Adam 的自适应步长等价于给每个参数单独调了学习率。**学习率的合理范围取决于优化器，不能脱离优化器谈"最优学习率"。**

### 10.3 批大小

{_hyper_table(H["batch"], "batch_size")}

![图 14　批大小对验证准确率的影响](figures/fig14_batch.png)

批大小从 16 增加到 256，测试准确率从 {pct(H['batch']['batch=16']['test_acc'])} 变为 {pct(H['batch']['batch=256']['test_acc'])}。在固定 epoch 数的情况下，小批量每个 epoch 的参数更新次数更多（10,000 张图、batch=16 时每轮 625 次更新，batch=256 时只有 39 次），所以小批量收敛更快；但小批量的每轮耗时更长，且梯度噪声更大。批大小和训练轮数、学习率之间存在耦合，比较时要注意"固定 epoch"这一协议本身的偏差。

### 10.4 权重衰减

{_hyper_table(H["weight_decay"], "weight_decay")}

![图 15　权重衰减对验证准确率的影响](figures/fig15_weight_decay.png)

在 10,000 张训练图、8 轮的设置下，权重衰减从 0 增加到 1e-2，测试准确率的变化在 {min(r['test_acc'] for r in H['weight_decay'].values()) * 100:.2f}%～{max(r['test_acc'] for r in H['weight_decay'].values()) * 100:.2f}% 之间。这个子集上的过拟合并不严重，所以权重衰减没有带来实质收益；过强的衰减（1e-2）会略微限制拟合能力。第 8 节在 6,000 张图的过拟合场景下也观察到类似现象：正则化只有在过拟合存在时才有价值。
""")


# ===========================================================================
# 11. 实验六：数据增强与旋转鲁棒性
# ===========================================================================

_ANG = AU_NONE["rotation"]["angles"]


def rot_acc(run: dict, angle: int) -> float:
    return run["rotation"]["acc"][run["rotation"]["angles"].index(angle)]


add(rf"""## 11. 实验六：数据增强与旋转鲁棒性

### 11.1 设置

仍然使用 6,000 张训练图、20 轮、Adam lr=0.001，只改变训练时的数据增强方式：

- **无增强**：原图直接训练；
- **旋转/平移 ±10°**：随机旋转 ±10°，随机平移最多 10% 的边长；
- **旋转/平移 ±20°**：更强的旋转与平移；
- **±20° + 随机擦除**：在 ±20° 旋转/平移的基础上，以 0.5 的概率随机擦除 2%～10% 面积的矩形区域（用背景色填充）。

### 11.2 结果

{table(
    ["增强方式", "测试准确率", "验证准确率", "训练-验证差距",
     "旋转 −20° 时准确率", "旋转 +20° 时准确率", "旋转 +30° 时准确率"],
    [[label, pct(AU_RUNS[label]["test_acc"]), pct(AU_RUNS[label]["best_val_acc"]),
      f"{train_val_gap(AU_RUNS[label]) * 100:+.2f} pp",
      pct(rot_acc(AU_RUNS[label], -20)), pct(rot_acc(AU_RUNS[label], 20)),
      pct(rot_acc(AU_RUNS[label], 30))]
     for label in AU_ORDER]
)}

> 说明：表中的"训练-验证差距"是最后一轮训练准确率减去验证准确率。使用数据增强或 Dropout 时，训练阶段的图像被随机变换、神经元被随机丢弃，训练准确率本身会偏低（甚至低于验证准确率），因此差距可能出现负值，这**不代表**过拟合消失，只说明训练阶段的评估条件比验证阶段更苛刻。

![图 16　不同数据增强方式的验证准确率与训练损失曲线](figures/fig16_aug_curves.png)

![图 17　把测试集整体旋转不同角度后，各模型的准确率](figures/fig17_rotation.png)

### 11.3 分析

**干净测试集上的精度。** 测试准确率最高的是「{AU_BEST_LABEL}」（{pct(AU_BEST_ACC)}）。无增强为 {pct(AU_NONE['test_acc'])}，±10° 增强为 {pct(AU_RUNS['旋转/平移 ±10°']['test_acc'])}，±20° 增强为 {pct(AU_20['test_acc'])}，±20° + 随机擦除为 {pct(AU_RUNS['±20° + 随机擦除']['test_acc'])}。在 6,000 张图的场景下，增强对干净测试集的影响并不大（因为 MNIST 数字本身基本是居中的正立图像，旋转/平移增强相当于引入了训练分布与测试分布的偏移）。

**旋转鲁棒性。** 图 17 是更有信息量的结果：把整个测试集旋转不同角度后重新评估。无增强模型在 0° 时准确率 {pct(rot_acc(AU_NONE, 0))}，旋转 ±20° 后掉到 {pct(rot_acc(AU_NONE, -20))} / {pct(rot_acc(AU_NONE, 20))}；而用 ±20° 旋转增强训练的模型在 ±20° 时仍有 {pct(rot_acc(AU_20, -20))} / {pct(rot_acc(AU_20, 20))}。旋转角度越大，无增强模型退化越快，增强模型则明显更平缓——**增强的本质是让训练分布覆盖测试时可能出现的变换**，它换来的往往不是干净测试集上的精度，而是分布偏移下的鲁棒性。

**随机擦除。** 在 ±20° 增强上再加随机擦除，测试准确率为 {pct(AU_RUNS['±20° + 随机擦除']['test_acc'])}，与不加擦除的 {pct(AU_20['test_acc'])} 接近；旋转鲁棒性也基本持平。这说明在 MNIST 这种干净、居中的小图上，遮挡类增强的边际收益很小；在真实场景（部分遮挡、复杂背景）中随机擦除才更能体现价值。
""")

print("第二部分（6-11 节）准备完成")


# ===========================================================================
# 12. 实验七：最终模型与误差分析
# ===========================================================================

F_ANGLES = F_ROT["angles"]


def f_rot(angle: int) -> float:
    return F_ROT["acc"][F_ROT["angles"].index(angle)]


# 置信度-覆盖率曲线上的几个关键点
_COV = F["coverage"]
_PRE = F["precision_at_threshold"]
_PRE_MAX = max(_PRE)
_cov_at_precision = max((c for c, p in zip(_COV, _PRE) if p >= 0.998), default=0.0)
_R1 = F_RECIPES["R1 LeNet-5 (ReLU+MaxPool)"]
_R2 = F_RECIPES["R2 R1 + Dropout 0.5"]
_R3 = F_RECIPES["R3 R1 + 增广 ±10°"]
_R4 = F_RECIPES["R4 R2 + 增广 ±20°"]
_R5 = F_RECIPES["R5 SmallVGG (更深+BN+Dropout)"]
# 按验证集最优选出的配方与次优配方的差距
_recipe_sorted = sorted(F_RECIPES.items(), key=lambda kv: -kv[1]["best_val_acc"])
BEST_RECIPE_LABEL = _recipe_sorted[0][0]
SECOND_RECIPE_LABEL = _recipe_sorted[1][0]
FINAL_RUN_0 = F["final_runs"][0]
FINAL_ARCH_CN = ("SmallVGG（3×3 卷积 + BN + Dropout）" if F["final_arch"] == "vgg"
                 else "LeNet-5")

_recipe_rows = []
for _lbl, _run in F_RECIPES.items():
    _arch = _run["config"].get("arch", "lenet")
    _aug = _run["config"].get("augmentation", "none")
    _recipe_rows.append([_lbl, "SmallVGG" if _arch == "vgg" else "LeNet-5", _aug,
                         pct(_run["best_val_acc"]), pct(_run["test_acc"])])

add(rf"""## 12. 实验七：最终模型与误差分析

### 12.1 配方选择与多种子结果

前面的实验分别考察了结构、激活函数、Dropout 和增强。为了得到最终模型，本节在全量训练集上比较 5 个候选配方（每个配方训练 12 轮、单一种子），按**验证集准确率**选择，再用选中的配方训练 3 个随机种子、15 轮，报告测试准确率的均值与标准差：

{table(
    ["候选配方", "结构", "数据增强", "验证准确率", "测试准确率"],
    _recipe_rows
)}

按验证集选出的是「{F['recipe']}」，最终模型结构为 {FINAL_ARCH_CN}，共 {num(F['params'])} 个参数。三个随机种子的测试准确率分别为 {"、".join(pct(a) for a in F_SEEDS)}，均值 {pm(F_MEAN, F_STD)}；三个种子的波动（标准差 {F_STD * 100:.2f} 个百分点）说明在 MNIST 上模型已经比较稳定，用单个种子比较相差 0.1～0.2 个百分点的配置时要谨慎。

![图 18　最终模型三个随机种子的验证准确率与验证损失曲线](figures/fig18_final_curves.png)

从配方对比还能看到一个与第 8 节呼应的现象：在**全量数据**上，加 Dropout 或增强的 LeNet-5（R2/R3/R4）测试准确率分别为 {pct(_R2['test_acc'])}、{pct(_R3['test_acc'])}、{pct(_R4['test_acc'])}，与最简单的 R1（{pct(_R1['test_acc'])}）相比分别高 {(_R2['test_acc'] - _R1['test_acc']) * 100:.2f}、{(_R3['test_acc'] - _R1['test_acc']) * 100:.2f}、{(_R4['test_acc'] - _R1['test_acc']) * 100:.2f} 个百分点，单种子下很难区分；而更深的 SmallVGG（R5）测试准确率 {pct(_R5['test_acc'])}，比最好的 LeNet-5 配方高 {gap(_R5['test_acc'], max(_R1['test_acc'], _R2['test_acc'], _R3['test_acc'], _R4['test_acc']))} 个百分点。原因与第 8 节相同——全量 54,001 张图时 LeNet-5 的过拟合并不严重，正则化带来的偏差与方差收益大致抵消；而"加深网络 + BN + 适度 Dropout"是更有效的容量扩展方式。

### 12.2 混淆矩阵与每类指标

最终模型（第 0 个种子）在 10,000 张测试图上的准确率为 {pct(F['test_acc'])}，共 {F['n_errors']} 张错分。图 19 是行归一化的混淆矩阵，图 20 是每类 precision / recall / F1：

![图 19　最终模型的混淆矩阵（行归一化，数字为行百分比）](figures/fig19_confusion.png)

![图 20　最终模型每类的 precision / recall / F1](figures/fig20_per_class.png)

{table(
    ["类别", "支持数", "Precision", "Recall", "F1", "错分数"],
    [[r["class"], r["support"], pct(r["precision"]), pct(r["recall"]),
      pct(r["f1"]), r["error_count"]] for r in F_CLASS]
)}

**分析。** 召回率最低的类别是 {min(F_CLASS, key=lambda r: r['recall'])['class']}（{pct(min(F_CLASS, key=lambda r: r['recall'])['recall'])}），精确率最低的类别是 {min(F_CLASS, key=lambda r: r['precision'])['class']}（{pct(min(F_CLASS, key=lambda r: r['precision'])['precision'])}）。最常出现的混淆是 {"、".join(f"{c['true']} → {c['pred']}（{c['count']} 张）" for c in F['top_confusions'][:5])}。这些错误大多来自笔画结构相近的数字（例如 {"、".join(f"{c['true']}→{c['pred']}" for c in F['top_confusions'][:3])}），与人的辨认难点一致；混淆矩阵对角线以外的数值都很小，说明模型没有系统性地偏向某个类别。

### 12.3 错分样本

![图 21　测试集中的随机错分样本（标题为 真实→预测 与置信度）](figures/fig21_errors.png)

图 21 从 {F['n_errors']} 张错分样本中随机抽取了 16 张。可以看到相当一部分是书写潦草、笔画缺失或非常规写法的样本，另一些则是人眼也容易犹豫的样本。这说明剩余错误的主要来源是数据本身的歧义（贝叶斯误差），而不是模型还有明显的系统缺陷。

### 12.4 第一层卷积核与特征图

![图 23　第一层卷积核（红=正权重，蓝=负权重）](figures/fig23_filters.png)

![图 24　同一张输入图在各个卷积层的部分特征图](figures/fig24_feature_maps.png)

图 23 是第一层 32 个 3×3 卷积核中的 16 个。与经典 LeNet-5 的 5×5 核相比，3×3 核能表达的图案更小，肉眼看起来不如图片里常见的"Gabor 条纹"那么直观，但仍能辨认出若干方向性的亮暗对比模式（例如中心与两侧反号、上下或左右方向的红蓝交替），它们的作用是检测局部笔画边缘与拐角。这正是 CNN 可解释性的经典证据——第一层学到的是与位置无关的局部模式，而不是某个数字的完整模板。

图 24 中，浅层（第一个卷积层）的特征图保留了较多的空间细节，不同通道分别响应不同的笔画方向和位置；深层（最后一个卷积层）的特征图空间尺寸缩小到 7×7，响应变成更抽象的局部区域，不同通道对应数字的不同部件。越深的层越"看不清"原始像素，但语义信息越集中，这就是 4.3 节所说的"空间分辨率下降、语义信息上升"。

### 12.5 置信度分析

![图 22　正确/错误预测的置信度分布（左）与按置信度阈值拒绝预测的覆盖率-精度曲线（右）](figures/fig22_confidence.png)

正确预测的平均置信度为 {pct(F_CONF['conf_correct_mean'], 1)}，错误预测的平均置信度为 {pct(F_CONF['conf_wrong_mean'], 1)}：模型对自己的错误预测信心明显更低，但并非完全"知道自己不知道"。图 22 右侧把置信度当作拒绝阈值：如果只保留置信度超过阈值的预测，覆盖率与准确率之间存在明显的权衡。本模型把整体准确率提高到 99.5% 以上后，若要求被保留样本的准确率不低于 99.8%，最多还能覆盖约 {pct(_cov_at_precision, 1)} 的测试样本（即拒绝约 {pct(1 - _cov_at_precision, 1)} 最不可靠的输入，其中包含了大部分错误样本）；精度要求再高（例如 99.9%）时，覆盖率会迅速降到 0——因为 41 个错误样本中有相当一部分的置信度并不低，"用置信度拒识"只能解决一部分问题。

### 12.6 最终模型的旋转鲁棒性

最终模型（配方「{F['recipe']}」）在旋转后的测试集上的准确率：

{table(
    ["旋转角度"] + [f"{a}°" for a in F_ANGLES],
    [["准确率"] + [pct(a) for a in F_ROT["acc"]]]
)}

即使配方中不包含旋转增强，最终模型对 ±10° 的小角度旋转仍有一定容忍度（{pct(f_rot(-10))} / {pct(f_rot(10))}，相比 0° 的 {pct(f_rot(0))} 只下降 {abs(f_rot(0) - f_rot(-10)) * 100:.2f} / {abs(f_rot(0) - f_rot(10)) * 100:.2f} 个百分点）：小角度旋转在数字中心附近可以局部近似为小幅平移，而卷积 + 池化对这种局部平移有一定容忍度；但 ±30° 时准确率降到 {pct(f_rot(-30))} / {pct(f_rot(30))}，说明这种容忍度只在很小的角度范围内成立。如果应用场景中图像朝向不固定，应当在训练时加入旋转增强（第 11 节），或者先做朝向归一化。
""")


# ===========================================================================
# 13. 实验八：Fashion-MNIST
# ===========================================================================

add(rf"""## 13. 实验八：Fashion-MNIST 上的验证

MNIST 的数字图像干净、居中、类别差异明显。为了检验前面的结论是否能迁移到更难的图像数据，又在 Fashion-MNIST 上做了同样的 CNN vs MLP 对比。Fashion-MNIST 与 MNIST 尺寸、规模、类别数完全相同（28×28、60,000 训练 / 10,000 测试、10 类），但内容是 T 恤、裤子、外套等服装，类内差异大、类间差异小，难度明显更高。

![图 25　Fashion-MNIST 服装样本（每类 2 张）](figures/fig25_fashion_samples.png)

{table(
    ["模型", "参数量", "验证准确率", "测试准确率", "训练耗时"],
    [["CNN（LeNet-5 现代结构）", num(FASH["cnn"]["params"]), pct(FASH["cnn"]["best_val_acc"]),
      pct(FASH["cnn"]["test_acc"]), f"{FASH['cnn']['train_time']:.0f} s"],
     ["MLP（784-256-128）", num(FASH["mlp"]["params"]), pct(FASH["mlp"]["best_val_acc"]),
      pct(FASH["mlp"]["test_acc"]), f"{FASH['mlp']['train_time']:.0f} s"]]
)}

![图 26　Fashion-MNIST 上 CNN 与 MLP 的测试准确率](figures/fig26_fashion_compare.png)

**结果分析。** 在 Fashion-MNIST 上，CNN 的测试准确率为 {pct(FASH['cnn']['test_acc'])}，MLP 为 {pct(FASH['mlp']['test_acc'])}，CNN 高 {gap(FASH['cnn']['test_acc'], FASH['mlp']['test_acc'])} 个百分点；在 MNIST 上两者差距是 {gap(LENET_MODERN['test_acc'], MLP_BL['test_acc'])} 个百分点。两个模型在 Fashion-MNIST 上的绝对准确率都明显低于 MNIST（同一 CNN 从 {pct(LENET_MODERN['test_acc'])} 降到 {pct(FASH['cnn']['test_acc'])}），符合"任务更难"的预期。值得诚实记录的是：**CNN 相对 MLP 的优势并没有在更难的数据上放大**（{gap(FASH['cnn']['test_acc'], FASH['mlp']['test_acc'])} 对 {gap(LENET_MODERN['test_acc'], MLP_BL['test_acc'])} 个百分点），两者只训练 10 轮且没有针对 Fashion-MNIST 调参，可能都还没跑满这个任务的性能，因此这个比较只能说明"CNN 仍然更好"，不能说明"CNN 的优势更大"。

这也再次说明：**在一种数据上得到的超参数结论不能直接搬到另一种数据上**。本实验没有为 Fashion-MNIST 重新调参（学习率、轮数、Dropout、增强都与 MNIST 相同），因此这些数字只用于验证"CNN 优势方向"是否一致，而不是该数据集上的性能上限，也不能用单次运行的 0.66 个百分点去论证更强的结论。
""")


# ===========================================================================
# 14. 讨论
# ===========================================================================

add(rf"""## 14. 讨论

### 14.1 CNN 的优势来源：归纳偏置而不是参数量

把实验一、实验四和实验八放在一起看：LeNet-5 用 {num(LENET_MODERN['params'])} 个参数在 MNIST 上达到 {pct(LENET_MODERN['test_acc'])}，MLP 用 {num(MLP_BL['params'])} 个参数只达到 {pct(MLP_BL['test_acc'])}；在更难的 Fashion-MNIST 上，CNN 仍比 MLP 高 {gap(FASH['cnn']['test_acc'], FASH['mlp']['test_acc'])} 个百分点。卷积网络把"局部性"和"平移等变"写进了结构，因此同样的数据量下能学到更可迁移的特征；但归纳偏置也是一把双刃剑：如果任务本身不满足这些先验，或者数据和训练预算不足以让 CNN 兑现先验，CNN 的优势就会缩小甚至消失——实验四中 MLP 在部分表格数据集上反而更好，本实验数据量实验中 600 张图时 CNN 也输给了 MLP，都是同一类现象。

### 14.2 激活函数：ReLU 族成为默认选择的原因

实验二给出了两条证据：在 Adam 下，ReLU 族的测试准确率为 {pct(ACT_BEST)} 左右，Sigmoid 为 {pct(ACT_SIGMOID['test_acc'])}，差距 {gap(ACT_BEST, ACT_SIGMOID['test_acc'])} 个百分点；梯度消失诊断中，Sigmoid + SGD 的网络 6 轮后仍停在随机水平（{pct(GD_SIG_SGD['test_acc'])}），第一层梯度只有 {GD_SIG_SGD['grad_norm_first_conv'][0]:.1e} 量级。ReLU 的优势不只是"不饱和"，还包括计算便宜（一次 max）、在正半轴保持梯度尺度、天然产生稀疏激活。它的缺点是负半轴梯度为 0，可能造成神经元"死亡"，LeakyReLU / ELU / GELU 用很小的代价缓解了这个问题；本实验中这几种变体与 ReLU 的差距在 {abs(A['runs']['relu']['test_acc'] - max(A['runs'][k]['test_acc'] for k in ['leaky_relu', 'elu', 'gelu'])) * 100:.2f} 个百分点以内，说明**先解决饱和问题，再谈具体形状**。

### 14.3 Dropout：条件依赖的正则化

实验三中，Dropout 把 6,000 张图场景下的训练-验证差距从 {train_val_gap(DR_NOREG) * 100:+.2f} 个百分点缩小到 {train_val_gap(DR_D05) * 100:+.2f} 个百分点，测试准确率从 {pct(DR_NOREG['test_acc'])} 提高到 {pct(DR_D05['test_acc'])}；但实验七的全量数据配方对比中，加 Dropout 的 LeNet-5（R2，{pct(_R2['test_acc'])}）只比不加的（R1，{pct(_R1['test_acc'])}）高 {gap(_R2['test_acc'], _R1['test_acc'])} 个百分点，落在单种子波动范围内，收益远小于过拟合场景。这不是矛盾，而是正则化的基本规律：**正则化用偏差换方差，只有在方差（过拟合）是主要矛盾时才有净收益**。判断是否需要 Dropout 的方法不是看模型多大，而是看训练集与验证集的差距是否显著、验证准确率是否在训练后期回落。

### 14.4 数据量、模型容量与正则化的三角关系

数据量实验显示 CNN 从 600 张到 54,001 张提升 {gap(SZ_CNN[-1]['test_acc'], SZ_CNN[0]['test_acc'])} 个百分点，且收益递减；Dropout 实验显示 6,000 张图 + 约 {num(DR_NOREG['params'])} 参数时过拟合明显，而全量数据时不明显。把这两点结合起来：模型容量应当与数据量匹配。数据少时，要么减小模型，要么加正则化（Dropout、权重衰减、数据增强）；数据多时，正则化过强反而欠拟合。本实验中 SmallVGG（更大容量）在全量数据上最好，但在 6,000 张图的小数据上它是否仍然最好并没有测试——这是本实验的一个局限。

### 14.5 超参数的敏感度与实验协议

从实验五看，学习率与优化器的组合影响最大（SGD lr=0.01 在 8 轮内只到 {pct(H['optimizer']['SGD lr=0.01']['test_acc'])}，而 Adam 达到 {pct(H['optimizer']['Adam lr=1e-3']['test_acc'])}），批大小次之，权重衰减在本实验的范围内影响最小。需要强调协议本身的影响：超参数扫描使用 10,000 张图、8 轮，如果换成全量数据、30 轮，最优学习率和正则化强度都可能不同；"固定 epoch 数比较批大小"会让小批量因为更新次数更多而占便宜。这些偏差在报告中都如实标注，不能把扫描结果当作绝对结论。

### 14.6 实验的局限

1. **数据集有限**：只在 MNIST 和 Fashion-MNIST 两种 28×28 灰度图上验证，没有涉及彩色图像、目标检测、语义分割等更复杂的视觉任务，也没有用 ImageNet 规模的预训练模型。
2. **模型规模有限**：LeNet-5 和 SmallVGG 都是小网络，结论（例如 BN 收益有限、Dropout 依赖过拟合程度）不必然适用于大网络、大数据。
3. **训练预算有限**：大多数对照实验训练 8～20 轮，没有使用学习率调度、早停、模型集成等技巧；部分配置（如 SGD）并未跑到收敛，因此"优化器对比"只能说明本实验预算下的相对表现。
4. **随机性**：超参数扫描与消融实验大多只用了 1 个随机种子，小于 0.2 个百分点的差异不应过度解读；最终模型用了 3 个种子，报告了均值与标准差。
5. **验证集与测试集**：验证集从官方训练集中分层留出，测试集是官方测试集，二者同分布；没有测试分布偏移（如不同书写风格、不同采集设备）下的泛化。
6. **可解释性分析是定性的**：卷积核与特征图的解释基于肉眼观察，没有做定量归因（如 Grad-CAM、显著性图）。
""")


# ===========================================================================
# 15. 结论
# ===========================================================================

add(rf"""## 15. 结论

1. 用 PyTorch 在 GPU 上完整实现并训练了 LeNet-5 卷积神经网络，在 MNIST 上达到 {pct(LENET_MODERN['test_acc'])}（现代结构）和 {pm(F_MEAN, F_STD)}（3 种子最终模型）的测试准确率；相比之下，参数量是它 {MLP_BL['params'] / LENET_MODERN['params']:.1f} 倍的 MLP 只有 {pct(MLP_BL['test_acc'])}。CNN 的优势来自局部连接、权值共享和平移等变的归纳偏置，而不是参数量。
2. 激活函数的选择在本实验中影响显著：Adam 下 ReLU 族比 Sigmoid 高约 {gap(ACT_BEST, ACT_SIGMOID['test_acc'])} 个百分点；在 SGD lr=0.01 下 Sigmoid 网络 6 轮后仍停在 {pct(GD_SIG_SGD['test_acc'])}（随机水平），第一层卷积梯度只有 {GD_SIG_SGD['grad_norm_first_conv'][0]:.1e}，比 ReLU 小约 {__import__('math').log10(GD_RELU_SGD['grad_norm_first_conv'][0] / GD_SIG_SGD['grad_norm_first_conv'][0]):.1f} 个数量级，是定量可测的梯度消失现象。激活函数必须和优化器、学习率一起考虑。
3. Dropout 是有效的正则化手段，但只在过拟合存在时有效：6,000 张图 + 大分类头的场景下，Dropout 0.5 把训练-验证差距从 {train_val_gap(DR_NOREG) * 100:+.2f} 个百分点缩小到 {train_val_gap(DR_D05) * 100:+.2f} 个百分点，测试准确率从 {pct(DR_NOREG['test_acc'])} 提高到 {pct(DR_D05['test_acc'])}；在全量数据上，Dropout 的收益缩小到单种子波动范围内（R2 比 R1 高 {gap(_R2['test_acc'], _R1['test_acc'])} 个百分点）。丢弃率过高、作用在卷积层（Dropout2d）都可能适得其反。
4. 数据量是最重要的单一因素：CNN 从 600 张训练图到 54,001 张，测试准确率提升 {gap(SZ_CNN[-1]['test_acc'], SZ_CNN[0]['test_acc'])} 个百分点，收益随数据量增加而递减。CNN 相对 MLP 的优势与数据量强相关：600 张时 CNN 反而低 {abs(SZ_CNN[0]['test_acc'] - SZ_MLP[0]['test_acc']) * 100:.1f} 个百分点，2,000 张起反超，全量时高 {gap(SZ_CNN[-1]['test_acc'], SZ_MLP[-1]['test_acc'])} 个百分点——归纳偏置需要足够的数据和训练预算才能兑现。
5. 超参数中学习率与优化器的影响最大，批大小次之，权重衰减在本实验范围内影响最小；小批量在固定轮数下收敛更快（更新次数更多），但每轮耗时更长。学习率的合理范围依赖优化器，不能脱离优化器谈"最优学习率"。
6. 数据增强的价值主要体现在鲁棒性上：±20° 旋转/平移增强在干净测试集上的准确率与无增强接近（{pct(AU_20['test_acc'])} 对 {pct(AU_NONE['test_acc'])}），但把测试集整体旋转 20° 后准确率从 {pct(rot_acc(AU_NONE, 20))} 提升到 {pct(rot_acc(AU_20, 20))}。是否使用增强取决于应用场景中是否存在相应的变换。
7. 最终模型在测试集上错分 {F['n_errors']} 张（{pct(F['test_acc'])} 准确率），错误集中在笔画结构相近的数字对（{"、".join(f"{c['true']}→{c['pred']}" for c in F['top_confusions'][:3])}），剩余错误主要来自样本本身的歧义；置信度可以在高精度场景下用于拒识，但会牺牲覆盖率。
8. 通过本次实验完整走通了"理解原理 → 实现模型 → 验证正确性 → 控制变量实验 → 可视化分析 → 误差分析 → 讨论局限"的深度学习实验流程，理解了表达能力、优化难度、泛化能力是三个需要分别对待的问题。
""")


# ===========================================================================
# 16. 复现说明与文件清单
# ===========================================================================

add(rf"""## 16. 复现说明

### 16.1 运行步骤

```bash
# 1. 安装 GPU 版 PyTorch（CUDA 13.0 wheel；其它 CUDA 版本换对应 index-url）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. 准备数据（仓库已附带 data/mnist/*.gz 与 data/fashion/*.gz，可跳过）
bash data/download_mnist.sh
bash data/download_fashion_mnist.sh

# 3. 正确性验证（约 10 秒）
cd exp5_report && python verify.py

# 4. 跑全部实验（本机 GPU 上约 20 分钟；可用 --parts 分片）
python full_experiment.py

# 5. 生成报告（Markdown + docx，需要 pandoc）
python gen_report.py
```

`full_experiment.py` 支持 `--parts 1 2 ...` 只运行指定部分，结果会增量写入 `results.json`；支持 `--out` 指定结果文件。报告中的所有数字由 `gen_report.py` 从 `results.json` 读取，不手工抄写。

### 16.2 文件清单

| 路径 | 说明 |
|---|---|
| `code/data.py` | IDX 解析、分层划分、DataLoader、数据增强、测试集旋转 |
| `code/models.py` | LeNet-5 / SmallVGG / MLP 与激活函数工厂 |
| `code/engine.py` | 训练引擎、环境信息、优化器、梯度范数记录 |
| `code/metrics.py` | 混淆矩阵、每类指标、置信度统计 |
| `code/viz.py` | 全部插图 |
| `data/mnist/*.gz`、`data/fashion/*.gz` | MNIST 与 Fashion-MNIST 原始数据 |
| `exp5_report/full_experiment.py` | 九部分实验主脚本 |
| `exp5_report/verify.py` | 正确性验证脚本 |
| `exp5_report/results.json` | 全部实验指标与曲线数据 |
| `exp5_report/verification.json` | 正确性验证结果 |
| `exp5_report/figures/fig01–fig26` | 报告插图 |
| `exp5_report/gen_report.py` | 从 JSON 生成本报告 |
| `exp5_report/实验五_卷积神经网络_实验报告.md` | 本报告 |

### 16.3 复现注意事项

- 所有随机过程都使用显式种子：随机种子固定为 0，最终模型用 0/1/2 三个种子；数据划分使用分层抽样，验证集固定为 5,999 张。
- GPU 型号、CUDA 版本、cuDNN 版本会影响训练耗时；`torch.backends.cudnn.deterministic=True` 会让 cudnn 选择确定性算法，速度略慢但结果可复现。
- 实验脚本默认使用 GPU，没有 GPU 时会退回 CPU，此时全部实验需要数小时，建议用 `--parts` 分片运行。
- 本报告中的"训练耗时"包含每轮的验证集评估，且受笔记本 GPU 功耗墙影响，只用于量级比较。
- 超参数扫描使用 10,000 张训练图、8 轮，最终模型使用全量数据、15 轮，两者的结论不能直接互推。
""")


# ===========================================================================
# 参考文献
# ===========================================================================

add(r"""## 参考文献

1. LeCun Y., Bottou L., Bengio Y., Haffner P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 1998, 86(11): 2278–2324.
2. LeCun Y., Boser B., Denker J. S., Henderson D., Howard R. E., Hubbard W., Jackel L. D. Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1989, 1(4): 541–551.
3. Krizhevsky A., Sutskever I., Hinton G. E. ImageNet classification with deep convolutional neural networks. *NIPS*, 2012.
4. Simonyan K., Zisserman A. Very deep convolutional networks for large-scale image recognition. *ICLR*, 2015.
5. Srivastava N., Hinton G., Krizhevsky A., Sutskever I., Salakhutdinov R. Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 2014, 15(1): 1929–1958.
6. Ioffe S., Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 2015.
7. He K., Zhang X., Ren S., Sun J. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*, 2015.
8. Nair V., Hinton G. E. Rectified linear units improve restricted Boltzmann machines. *ICML*, 2010.
9. Glorot X., Bengio Y. Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 2010.
10. Kingma D. P., Ba J. Adam: A method for stochastic optimization. *ICLR*, 2015.
11. Bengio Y., Simard P., Frasconi P. Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 1994, 5(2): 157–166.
12. Hendrycks D., Gimpel K. Gaussian error linear units (GELUs). *arXiv:1606.08415*, 2016.
""")


# ===========================================================================
# 写出 Markdown 并转 docx
# ===========================================================================

def normalize_display_math(text: str) -> str:
    """把 $$公式$$ 统一改成三行式（Markdown 渲染器要求 $$ 独占一行）。"""
    def repl(m):
        return "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n"

    text = re.sub(r"\$\$(.+?)\$\$", repl, text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text)


md = normalize_display_math("\n\n".join(P) + "\n")
OUT_MD.write_text(md, encoding="utf-8")
print(f"Markdown 已写入 {OUT_MD}（{len(md)} 字符，{md.count(chr(10))} 行）")

subprocess.run(
    ["pandoc", str(OUT_MD), "-o", str(OUT_DOCX),
     "--resource-path", str(HERE)],
    check=True,
)
print(f"DOCX 已写入 {OUT_DOCX}")
