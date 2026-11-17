# -*- coding: utf-8 -*-
"""
绘制最优模型：
    N = 100
    degree = 15
    lambda = 0.0
"""

from step09_polynomial_l2 import train_polynomial_l2
from step04_standardize import transform_features
from step03_polynomial_features import polynomial_features
from step02_generate_data import generate_data, make_test_data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")


def main():
    N = 100
    degree = 15
    l2 = 0.0
    seed = 42

    # 训练数据：带噪声
    x_train, y_train = generate_data(N, noise_std=0.2, seed=seed)

    # 测试/绘图网格：无噪声
    x_plot, y_true = make_test_data(600)

    # 训练最优模型
    result = train_polynomial_l2(
        x_train, y_train, x_plot, y_true,
        degree=degree, l2=l2, lr=0.1, epochs=60000
    )

    # 对绘图网格做与训练一致的标准化
    Phi_plot = polynomial_features(x_plot, degree)
    Phi_plot_std = transform_features(
        Phi_plot, result["mean"], result["std"]
    )
    y_pred = Phi_plot_std @ result["w"]

    # 画图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        x_train, y_train,
        s=28, color="black", alpha=0.55,
        label=f"Training data (N={N})"
    )
    ax.plot(
        x_plot, y_true,
        color="green", linestyle="--", linewidth=2.0,
        label="True function: sin(2*pi*x)"
    )
    ax.plot(
        x_plot, y_pred,
        color="red", linewidth=2.0,
        label=f"Best model: degree={degree}, lambda={l2:g}"
    )

    ax.set_title(
        f"N={N}, degree={degree}, lambda={l2:g}  |  "
        f"train MSE={result['train_mse']:.5f}, "
        f"test MSE={result['test_mse']:.5f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-1.6, 1.6)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig("best_model.png", dpi=150)
    print("saved: best_model.png")
    print(f"train_mse = {result['train_mse']:.6f}")
    print(f"test_mse  = {result['test_mse']:.6f}")
    print(f"||w||     = {np.linalg.norm(result['w']):.4f}")


if __name__ == "__main__":
    main()
