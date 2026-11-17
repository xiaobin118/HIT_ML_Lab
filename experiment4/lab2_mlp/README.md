# Lab2：用 PyTorch 构建多层感知机（Ionosphere 二分类）

## 1. 项目概况

用 PyTorch 实现一个含两个隐藏层的 MLP，在 UCI Ionosphere 数据集上完成二分类实验：

- 数据：351 个样本、34 个连续特征，标签 g（good）/ b（bad），g 225 个、b 126 个；
- 模型：34 → 隐藏层（ReLU）→ 1 个 logit，`BCEWithLogitsLoss`；
- 评估：分层五折交叉验证；封闭测试 = 训练集准确率，开放测试 = 验证折准确率；
- 调参：隐层层数、隐层维度、激活、Dropout、学习率、优化器、Batch、Epoch；
- 分析：训练曲线、正则化对比、混淆矩阵、预测概率与错分样本。

最终选出的配置：**2×32 + Dropout 0.2 + Adam（lr=1e-3, batch=32, 100 epoch）**，参数量 2,209；
五折开放准确率 **94.02% ± 3.30%**，封闭准确率 **99.72%**，OOF F1 **0.955**。

## 2. 环境

| 组件 | 版本 |
|---|---|
| 系统 | Ubuntu 24.04（WSL2，Linux 6.6.87.2） |
| Python | 3.13.12 |
| PyTorch | 2.14.1+cu130 |
| NumPy / Matplotlib | 2.5.2 / 3.11.0 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU（本实验在 CPU 上训练） |

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 数据

`data/ionosphere.data` 来自 UCI Machine Learning Repository：

<https://archive.ics.uci.edu/dataset/52/ionosphere>

每行 35 列：前 34 列为特征，最后一列为标签 `g` / `b`。脚本中把 `g` 映射为 1、`b` 映射为 0。

## 4. 目录结构

```text
lab2_mlp/
├── data/ionosphere.data          # 原始数据（UCI，75 KB）
├── src/
│   ├── data.py                   # 读取、分层五折、按折标准化、Dataset
│   ├── model.py                  # 可配置 MLP
│   ├── train.py                  # 单折训练、早停、指标、五折交叉验证
│   ├── run_experiments.py        # 表 1、表 2、扫描、正则化、误差分析
│   ├── make_figures.py           # 报告插图
│   ├── make_screenshots.py       # 终端截图
│   └── predict.py                # 加载模型做单样本预测
├── results/
│   ├── results.json              # 全部指标与曲线
│   └── model_best.pt             # 最优配置在全量数据上训练的权重
├── figures/                      # 报告用图
└── README.md
```

## 5. 运行

```bash
cd lab2_mlp

# 1) 跑全部实验（约 6 分钟，CPU），生成 results.json 与 model_best.pt
python src/run_experiments.py

# 2) 生成插图与终端截图
python src/make_figures.py
python src/make_screenshots.py

# 3) 单样本预测
python src/predict.py
```

`predict.py` 会加载 `results/model_best.pt`，对第一个样本输出预测类别和 P(g)。

## 6. 关键设置

- 随机种子：划分、初始化、DataLoader 均为 42；
- 五折：分层抽样，Fold 1 为 280 训练 / 71 验证，其余为 281 / 70；
- 标准化：每折只用训练集的均值和标准差；
- SGD：momentum=0.9；Adam 使用 PyTorch 默认的 β1/β2；
- 损失：`BCEWithLogitsLoss`（输出 logit，推理时 Sigmoid）；
- 指标：准确率、precision、recall、F1，正类为 g。

## 7. 主要结果

### 表 1（模板要求的 5 组配置，五折均值）

| # | 结构 | 激活 | Dropout | 优化器 | 参数量 | 封闭 | 开放 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1×10 | ReLU | 0 | SGD | 361 | 99.64% | 92.60% | 8.5s |
| 2 | 1×20 | ReLU | 0 | SGD | 721 | 99.72% | 92.60% | 7.0s |
| 3 | 1×30 | ReLU | 0 | SGD | 1081 | 99.72% | 93.17% | 7.0s |
| 4 | 2×32 | ReLU | 0.2 | Adam | 2209 | 99.72% | **94.02%** | 6.9s |
| 5 | 3×64 | Tanh | 0.3 | Adam | 10625 | 98.86% | 90.31% | 11.0s |

### 表 2（三组配置的五折开放精度）

| 配置 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | 均值 | 标准差 |
|---|---|---|---|---|---|---|---|
| 1 隐藏层, 16 | 90.14% | 94.29% | 88.57% | 97.14% | 88.57% | 91.74% | 3.41% |
| 2 隐藏层, 64 | 91.55% | 95.71% | 92.86% | 95.71% | 91.43% | 93.45% | 1.91% |
| 2 隐藏层, 64 + Dropout 0.3 | 90.14% | 97.14% | 92.86% | 95.71% | 91.43% | 93.46% | 2.61% |

最优配置的 OOF 混淆矩阵：TN=108、FP=18、FN=3、TP=222；
accuracy 94.02%、precision 92.50%、recall 98.67%、F1 0.955。

## 8. 说明

- 报告中的表格、图和数字由 `results/results.json` 自动生成，未手工抄录；
- 训练曲线、超参数扫描、正则化对比和误差分析的完整数据都在 `results/results.json`；
- 本实验只使用一个数据集、一组固定划分，94.02% 是在同一组五折上选超参数得到的偏乐观估计；
  更严谨的泛化估计需要嵌套交叉验证或额外留出测试集。
