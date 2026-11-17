# -*- coding: utf-8 -*-
"""
第 10 课：Matplotlib 可视化（骨架）

运行：
    python step10_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from step09_polynomial_l2 import train_polynomial_l2
from step04_standardize import transform_features
from step03_polynomial_features import polynomial_features
from step02_generate_data import generate_data, make_test_data
import matplotlib
matplotlib.use("Agg")  # 无显示器环境下也能保存图片


def basic_plot():
    """
    任务 1：Matplotlib 基础。

    要求：
    1. 生成 x = np.linspace(-1, 1, 300)；
    2. 计算 y = np.sin(2 * np.pi * x)；
    3. 创建 fig, ax = plt.subplots()；
    4. 用 ax.plot(x, y, label="true sine") 画线；
    5. 设置 xlabel、ylabel、title；
    6. 打开 ax.grid(alpha=0.3)；
    7. 显示图例 ax.legend()；
    8. 保存为 step10_basic.png。
    """
    # TODO
    x = np.linspace(-1, 1, 300)
    y = np.sin(2 * np.pi * x)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="true sine")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Basic Plot")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("step10_basic.png", dpi=150)


def plot_overfit_vs_l2():
    """
    任务 2：画训练点、真实曲线、未正则化高阶曲线、L2 曲线。

    要求：
    1. 生成 N=15 的训练数据；
    2. 生成 600 个无噪声测试点；
    3. 分别训练：
           degree=15, l2=0.0
           degree=15, l2=0.005
    4. 对测试点构造特征并标准化：
           Phi_plot = polynomial_features(x_plot, degree)
           Phi_plot_std = transform_features(Phi_plot, result["mean"], result["std"])
           y_plot = Phi_plot_std @ result["w"]
       注意：必须使用训练集得到的 mean/std。
    5. 画：
           - 训练点：ax.scatter(x_train, y_train, ...)
           - 真实曲线：ax.plot(x_plot, y_true, ...)
           - l2=0 曲线：ax.plot(x_plot, y_pred_0, ...)
           - l2=0.01 曲线：ax.plot(x_plot, y_pred_reg, ...)
    6. 设置标题、坐标轴、图例、grid；
    7. 保存为 step10_overfit.png。
    """
    # TODO
    N = 15
    x_train, y_train = generate_data(N, noise_std=0.2, seed=42)
    x_plot, y_true = make_test_data(num=600)

    degree = 15
    result_no_reg = train_polynomial_l2(
        x_train, y_train, x_plot, y_true, degree, l2=0.0
    )
    result_reg = train_polynomial_l2(
        x_train, y_train, x_plot, y_true, degree, l2=0.01
    )

    Phi_plot = polynomial_features(x_plot, degree)

    Phi_no_reg = transform_features(
        Phi_plot, result_no_reg["mean"], result_no_reg["std"]
    )
    Phi_reg = transform_features(
        Phi_plot, result_reg["mean"], result_reg["std"]
    )

    y_pred_no_reg = Phi_no_reg @ result_no_reg["w"]
    y_pred_reg = Phi_reg @ result_reg["w"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(x_train, y_train, color="black", label="Training data")
    ax.plot(x_plot, y_true, color="green", linestyle="--", label="True sine")
    ax.plot(x_plot, y_pred_no_reg, color="red", label="M=15, lambda=0")
    ax.plot(x_plot, y_pred_reg, color="orange", label="M=15, lambda=0.01")

    ax.set_title("Overfitting vs L2 Regularization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2.5, 2.5)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig("step10_overfit.png", dpi=150)


def main():
    basic_plot()
    plot_overfit_vs_l2()
    print("saved: step10_basic.png, step10_overfit.png")


if __name__ == "__main__":
    main()
