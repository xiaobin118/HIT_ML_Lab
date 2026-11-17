# 实验五：卷积神经网络（PyTorch + GPU）

用 PyTorch 在 GPU 上实现并训练卷积神经网络（主模型 LeNet-5，另有更深的 SmallVGG 对照），解决 MNIST 手写数字图像分类问题；系统分析激活函数的选择、Dropout、数据量、优化器/学习率/批大小/权重衰减、数据增强等技巧，并做混淆矩阵、错分样本、卷积核与特征图、置信度等误差分析。

## 目录结构

```
experiment5/
├── code/
│   ├── data.py          # MNIST / Fashion-MNIST 的 IDX 解析、分层划分、DataLoader、数据增强、旋转测试
│   ├── models.py        # LeNet-5（激活/池化/BN/Dropout 可配置）、SmallVGG、MLP
│   ├── engine.py        # 随机种子、设备选择、优化器、训练循环、梯度范数记录、环境信息
│   ├── metrics.py       # 混淆矩阵、每类 P/R/F1、最易混淆类别对、置信度统计
│   └── viz.py           # 全部插图（曲线、样本、混淆矩阵、卷积核、特征图、结构图）
├── data/
│   ├── download_mnist.sh           # 下载 MNIST 原始 IDX 文件
│   ├── download_fashion_mnist.sh   # 下载 Fashion-MNIST 原始 IDX 文件
│   ├── mnist/                      # 自行下载（未随仓库提交）
│   └── fashion/                    # 自行下载（未随仓库提交）
└── exp5_report/
    ├── full_experiment.py   # 九部分实验主脚本 → results.json + figures/
    ├── verify.py            # 正确性验证（参数量/尺寸/初始损失/数值梯度/单批过拟合/旋转）
    ├── gen_report.py        # 从 results.json 生成 Markdown 报告
    ├── replot.py            # 只重画个别版式需要调整的插图（不重训）
    ├── results.json         # 全部实验指标与曲线
    ├── verification.json    # 正确性验证结果
    ├── figures/             # 26 张插图
    └── 实验五_卷积神经网络_实验报告.md
```

## 环境

- Python 3.13（Anaconda）
- **GPU 版 PyTorch**：torch 2.14.1+cu130、torchvision 0.29.1+cu130（CUDA 13.0）
- GPU：NVIDIA GeForce RTX 4050 Laptop GPU（6 GB 显存，计算能力 8.9）
- numpy、matplotlib；生成 docx 需要系统安装 pandoc

安装 GPU 版 PyTorch（与驱动匹配的 CUDA 版本可换 index-url）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

没有 GPU 时 `engine.get_device()` 会自动退回 CPU，代码无需修改，但训练会慢很多。

## 运行

```bash
# 1. 数据（未随仓库提交，先下载）
bash data/download_mnist.sh
bash data/download_fashion_mnist.sh

# 2. 正确性验证（约 10 秒）
cd exp5_report && python verify.py

# 3. 全部实验（GPU 上约 20 分钟；可用 --parts 只跑某几部分）
python full_experiment.py

# 4. 生成报告（Markdown；如需 docx 再装 pandoc）
python gen_report.py
```

`full_experiment.py --parts 1 2 3 ...` 会增量写入 `results.json`，不会覆盖其它部分的结果；`--out` 可指定结果文件路径。

## 实验内容

1. **数据概览与结构图**：MNIST 样本、类别分布、LeNet-5 结构图（每层尺寸与参数量）。
2. **基线对比**（全量 54,001 张训练图，10 轮）：LeNet-5 经典结构（Sigmoid+AvgPool）、现代结构（ReLU+MaxPool）、+BN、SmallVGG、MLP。
3. **激活函数**：ReLU / Sigmoid / Tanh / LeakyReLU / ELU / GELU，并测量第一层卷积的梯度范数；另做梯度消失诊断（Sigmoid/ReLU × SGD/Adam）。
4. **Dropout 与正则化**：6,000 张训练图 + 放大的全连接头制造过拟合，扫描 Dropout 丢弃率、作用位置（FC / 卷积 Dropout2d）以及权重衰减。
5. **数据量**：600 / 2,000 / 6,000 / 20,000 / 54,001 张，CNN vs MLP 学习曲线。
6. **超参数**：优化器（SGD、SGD+Momentum、Adam、AdamW）、学习率、批大小、权重衰减。
7. **数据增强**：旋转/平移 ±10°/±20°、随机擦除；把测试集整体旋转 −30°～+30° 评估鲁棒性。
8. **最终模型**：5 个候选配方按验证集选型，3 个随机种子重复训练 15 轮；混淆矩阵、每类指标、错分样本、卷积核、特征图、置信度、旋转鲁棒性。
9. **Fashion-MNIST**：更难的同尺寸数据集上的 CNN vs MLP 验证。

## 主要结果

| 实验 | 结论 |
|---|---|
| 基线（MNIST 测试集） | LeNet-5 经典结构 97.87%，现代结构 98.80%，+BN 98.96%，SmallVGG **99.47%**，MLP 97.69%（参数 23.5 万却不如 4.4 万参数的 LeNet-5） |
| 激活函数（10k 训练图，8 轮） | GELU 98.06%、Tanh 98.00%、ELU 97.77%、LeakyReLU 97.67%、ReLU 97.46%、Sigmoid 93.61%；第一层梯度 Sigmoid 起点比 ReLU 低约 3 个数量级 |
| 梯度消失诊断 | SGD lr=0.01 下 Sigmoid 网络 6 轮后仅 11.35%（随机水平），第一层卷积梯度 ~9e-5，比 ReLU 低约 2 个数量级；Adam 可救活（89.82%） |
| Dropout（6k 训练图，过拟合场景） | 无正则化 98.07% / 训练-验证差距 2.65pp；Dropout 0.5 → 98.19% / 差距 1.28pp；全量数据上 Dropout 收益缩到单种子波动内（R2 比 R1 高 0.18pp） |
| 数据量 | CNN 从 600 张的 78.14% 到全量的 98.82%；**600 张时 CNN 反而低于 MLP 9.7pp**，2,000 张起反超，全量时领先 1.25pp |
| 数据增强 | ±20° 旋转/平移：干净测试集 97.63%→97.91%，整体旋转 ±30° 时准确率从 69.5%/79.3% 提升到 86.4%/89.3% |
| 最终模型 | SmallVGG（3×3 卷积 + BN + Dropout），验证集选型，15 轮 × 3 种子，测试准确率 **99.52% ± 0.07%**（41 张错分） |
| Fashion-MNIST | CNN 88.71% vs MLP 88.05%（高 0.66pp）；更难的数据上 CNN 仍更好，但优势没有放大，两者都受 10 轮训练预算限制 |

全部数字、曲线与插图见 [实验报告](exp5_report/实验五_卷积神经网络_实验报告.md)。
