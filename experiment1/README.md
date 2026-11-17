# 实验一：多项式拟合正弦函数（纯 NumPy）

从零实现多项式回归，拟合带噪声的正弦函数：

$$
y=\sin(2\pi x)+\varepsilon,\qquad x\sim U[-1,1],\quad \varepsilon\sim\mathcal N(0,\ 0.2^2)
$$

不使用 PyTorch / TensorFlow 等自动微分工具，MSE、半均方损失、解析梯度、全批量梯度下降与 L2 正则化全部手写。

## 实验内容

- 用 Vandermonde 矩阵构造特征 $\Phi=[1,x,x^2,\dots,x^M]$，对非截距列做 z-score 标准化（测试集复用训练集统计量）。
- 扫描阶数 $M$、样本量 $N$、L2 系数 $\lambda$ 与学习率 $\eta$，观察欠拟合、过拟合、偏差—方差权衡。
- 用训练 MSE、无噪声测试 MSE、权重范数 $\lVert w\rVert$ 与曲线综合诊断模型状态。

## 主要结果

| 观察 | 结果 |
|---|---|
| 大样本 + 高阶 | $N=10000,\ M=20$ 的测试 MSE 约 **0.0059**，接近真实正弦函数 |
| 小样本 + 高阶 | $N=15,\ M=15$ 严重过拟合，训练误差很低而测试误差高两个数量级 |
| L2 正则化 | $\lambda\approx0.01$ 时测试 MSE 最小；$\lambda$ 过大转为欠拟合 |
| 学习率 | $\eta=0.05$ 收敛最好；$\eta=0.2,0.5$ 直接发散 |
| 标准化 | 高阶多项式不做特征标准化时梯度下降难以收敛 |

## 目录

```text
experiment1/
├── poly_sine_experiment.py     # 主实验：三组对比 + 四合一分析图
├── plot_best_model.py          # 大样本最优模型的可视化
├── step01_numpy.py … step12_final_plot.py
│                               # 按步骤拆分的练习脚本（数据生成、特征、MSE、梯度下降、L2、绘图）
├── *.png                       # 各步骤与最终结果图
├── cpp_poly_sine/              # C++17 / CMake 重写版本（进行中，见下）
└── 实验一_多项式拟合正弦函数_实验报告.md
```

### C++ 重写（进行中）

`cpp_poly_sine/` 是用现代 C++17 + CMake 重写本实验的尝试：`Matrix`、数据生成、多项式特征已经实现，`src/linear_algebra.cpp` 的 `mat_vec_mul` / `mat_t_vec_mul` 仍是待实现的 TODO，因此目前还不能完整跑通。保留它是为了记录多文件编译、CMake、clangd 配置的练习过程。

## 运行

```bash
python poly_sine_experiment.py   # 主实验，生成 poly_sine_results.png
python plot_best_model.py        # 最优模型图
```

依赖：Python 3.13、NumPy、Matplotlib。数据由程序随机生成，无需外部数据集。
