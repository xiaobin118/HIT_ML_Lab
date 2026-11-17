# HIT_ML_Lab

机器学习课程实验合集：六个实验全部从零手写核心算法，覆盖线性模型、聚类与 EM、多层感知机、卷积网络和生成模型。除实验五、六使用 PyTorch 做 GPU 训练外，实验一至四的实现不依赖任何自动微分框架，k-means、GMM/EM 与 MLP 的反向传播都是自己写的。

每个实验目录包含：可运行代码、完整实验报告（Markdown）、结果数据（`results.json`）和报告插图；数据集与模型权重因体积原因没有提交，按各目录 README 的说明下载即可复现。

## 实验一览

| # | 实验 | 实现方式 | 关键结果 | 代码 | 报告 |
|---|---|---|---|---|---|
| 1 | 多项式拟合正弦函数 | 纯 NumPy，手写梯度下降 + L2 | 大样本 $M=20$ 测试 MSE ≈ 0.0059；$N=15,M=15$ 严重过拟合 | [experiment1](experiment1/) | [报告](experiment1/实验一_多项式拟合正弦函数_实验报告.md) |
| 2 | 逻辑回归 | 纯 NumPy，梯度下降 + 牛顿法（IRLS） | 相关高斯数据 96.00% vs 高斯朴素贝叶斯 87.67%；Bank 90.56% vs 基线 89.16% | [experiment2](experiment2/) | [报告](experiment2/exp2_report/实验二_逻辑回归_实验报告.md) |
| 3 | k-means 与混合高斯 / EM | 纯 NumPy，禁用 scipy / sklearn | Iris 上 ARI 0.7302 → 0.9039；pendigits 上 0.5463 → 0.5650 | [experiment3](experiment3/) | [报告](experiment3/exp3_report/实验三_kmeans与GMM_实验报告.md) |
| 4 | 多层感知机 | 纯 NumPy，手写反向传播 + SGD/Momentum/Adam | MNIST 全量 98.16%，线性分类器 92.66% | [experiment4](experiment4/) | [报告](experiment4/exp4_report/实验四_多层感知机_实验报告.md) |
| 5 | 卷积神经网络 | PyTorch + CUDA，LeNet-5 / SmallVGG | SmallVGG 99.52% ± 0.07%；Fashion-MNIST 88.71% | [experiment5](experiment5/) | [报告](experiment5/exp5_report/实验五_卷积神经网络_实验报告.md) |
| 6 | 生成模型（DCGAN / VAE / DDPM） | PyTorch + CUDA | 扩散模型 FID 20.0 < DCGAN 92.7 < VAE 203.1 | [experiment6](experiment6/) | [报告](experiment6/exp6_report/实验六_基于生成模型的图像生成_实验报告.md) |

## 目录结构

```text
HIT_ML_Lab/
├── experiment1/                   # 实验一：多项式拟合正弦函数
│   ├── poly_sine_experiment.py    # 主实验
│   ├── step01…step12              # 分步骤练习脚本
│   ├── cpp_poly_sine/             # C++17 / CMake 重写（进行中）
│   └── README.md
├── experiment2/                   # 实验二：逻辑回归
│   ├── code/                      # LogisticRegression + GaussianNB
│   └── exp2_report/               # 复现脚本、结果、报告
├── experiment3/                   # 实验三：k-means 与 GMM/EM
│   ├── code/                      # 算法与数据加载
│   ├── docs/  learn/              # 原理推导讲义与练习
│   └── exp3_report/               # 报告、结果、9 张插图
├── experiment4/                   # 实验四：多层感知机
│   ├── code/                      # MLP、线性分类器、优化器、数据与可视化
│   ├── learn/                     # 配套讲义
│   ├── lab2_mlp/                  # 课程 Lab2：PyTorch 版 MLP（Ionosphere）
│   └── exp4_report/               # 报告、结果、24 张插图
├── experiment5/                   # 实验五：卷积神经网络
│   ├── code/                      # 数据、模型、训练引擎、指标与可视化
│   ├── learn/                     # 配套讲义
│   └── exp5_report/               # 报告、结果、26 张插图
├── experiment6/                   # 实验六：生成模型
│   ├── code/                      # DCGAN / VAE / 扩散模型
│   └── exp6_report/               # 报告、结果、39 张插图
└── README.md
```

## 复现

通用环境：Python 3.13、NumPy、Matplotlib。实验五、六额外需要 GPU 版 PyTorch：

```bash
pip install numpy matplotlib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**数据集未提交**（MNIST、Fashion-MNIST、Bank Marketing、UCI 数据集、Anime Face 合计数 GB）。各实验目录的 README 写明了下载地址或 `download_*.sh` 脚本；实验四的 `lab2_mlp` 自带 75 KB 的 Ionosphere 数据，可直接运行。

**Office 文档未提交**。所有报告以 Markdown 提供，正文数字由 `results.json` 生成，避免手工抄写造成的不一致；需要 Word 版时可用 pandoc 转换。

## 说明

- 实验三的 `code/gmm.py` 仍是带 `NotImplementedError` 的练习骨架，`run_experiment.py` 还留有 TODO；报告中的 GMM 结果来自 `exp3_report/full_experiment.py` 内的参考实现，报告第 6.2 节已如实标注。
- 实验一的 `cpp_poly_sine/` 是进行中的 C++ 重写，`linear_algebra.cpp` 尚未实现，暂不能完整运行。
- 报告开头的姓名 / 学号 / 班级 / 日期保留为占位符，仓库不包含个人信息。
