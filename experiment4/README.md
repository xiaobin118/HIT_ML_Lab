# 实验四：多层感知机（从零实现）

用纯 NumPy 实现 MLP 与线性分类器，不依赖 PyTorch / TensorFlow / 自动微分 / scikit-learn。
实验内容：二维四类数据的决策边界对比、数据量学习曲线、超参数扫描、MNIST 图像分类与可视化。

## 目录结构

```
experiment4/
├── data/                         # MNIST 原始 IDX 文件（未随仓库提交，用 download_mnist.sh 下载）
├── code/
│   ├── mlp.py                    # MLP：前向、反向传播、初始化、数值梯度校验
│   ├── linear.py                 # softmax 线性分类器（独立实现）
│   ├── train.py                  # SGD / Momentum / Adam + 小批量训练循环
│   ├── datasets.py               # 二维合成数据、MNIST 加载、分层抽样
│   ├── metrics.py                # 混淆矩阵、每类 P/R/F1、易混淆类别对
│   └── viz.py                    # 决策边界、曲线、混淆矩阵等绘图
├── learn/                        # 配套讲义与练习
├── lab2_mlp/                     # 课程 Lab2：PyTorch 版 MLP（Ionosphere 二分类）
└── exp4_report/
    ├── full_experiment.py        # 四组实验主脚本 → results.json + figures/
    ├── results.json              # 全部指标与曲线
    ├── gen_report.py             # 从 results.json 生成 Markdown 报告
    ├── figures/                  # 24 张图（含代码与运行截图）
    └── 实验四_多层感知机_实验报告.md
```

## 运行

```bash
bash data/download_mnist.sh        # 先下载 MNIST（约 11 MB）
cd exp4_report
python full_experiment.py          # 约 15 分钟（CPU）
python gen_report.py               # 从 results.json 重新生成报告
```

只跑某一部分：`python full_experiment.py --parts 1 2`，分片结果写到 `--out` 指定的文件。

## 实现要点

- **反向传播手写**：每层保存 `(A_prev, Z, A)`，从 `dZ = (P - Y)/N` 出发逐层回传；L2 只作用于权重。
- **数值校验**：`numeric_gradient_check` 用中心差分校验全部参数，误差在 `1e-10` 量级。
- **公平对照**：线性分类器与 MLP 实现相同的 `get_params/set_params/loss_and_grads/predict` 接口，共用同一个训练循环和优化器。
- **数值稳定**：softmax 减去行最大值；sigmoid 按正负分段计算。

## 主要结果

| 数据 | 线性分类器 | MLP |
|---|---|---|
| 四团高斯（线性可分） | 0.981 | 0.975 |
| 四同心圆环 | 0.247 | 0.991 |
| 四臂螺旋 | 0.267 | 0.997 |
| MNIST（全量 60k） | 92.66% | 98.16% |

详细分析见 [实验报告](exp4_report/实验四_多层感知机_实验报告.md)。
报告 4.7 节还在 Fashion-MNIST、Pendigits、Letter 等 6 个额外数据集上做了对照，复现脚本为 `learn/practice/extra_datasets.py`。
