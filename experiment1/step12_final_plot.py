# -*- coding: utf-8 -*-
"""
第 12 课：最终四合一实验图（骨架）

运行：
    python step12_final_plot.py
"""

from step09_polynomial_l2 import train_polynomial_l2
from step04_standardize import transform_features
from step03_polynomial_features import polynomial_features
from step02_generate_data import generate_data, make_test_data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")


def plot_curve(x_plot, result, degree, ax, label):
    """
    在 ax 上画一个训练好的多项式模型的预测曲线。

    步骤：
        Phi_plot = polynomial_features(x_plot, degree)
        Phi_plot_std = transform_features(Phi_plot, result["mean"], result["std"])
        y_plot = Phi_plot_std @ result["w"]
        ax.plot(x_plot, y_plot, label=label)

    注意：
        result["mean"] 和 result["std"] 必须来自训练集。
    """
    # TODO
    Phi_plot = polynomial_features(x_plot, degree)
    Phi_plot_std = transform_features(Phi_plot, result["mean"], result["std"])
    y_plot = Phi_plot_std @ result["w"]
    ax.plot(x_plot, y_plot, label=label)


def make_figure():
    """
    生成 2x2 四合一图：
        (a) 过拟合曲线 vs L2 曲线
        (b) test_mse vs degree，分别画 N=15, 30, 100
        (c) train_mse / test_mse vs l2，固定 N=15, degree=15
        (d) train_mse / test_mse vs N，固定 degree=15, l2=0

    要求：
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.tight_layout()
        fig.savefig("step12_final.png", dpi=150)
    """
    # TODO
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 所有子图共用的无噪声测试网格
    x_plot, y_true = make_test_data(600)

    # =========================================================
    # (a) 左上：过拟合 vs L2
    # =========================================================
    x_train_15, y_train_15 = generate_data(
        15, noise_std=0.2, seed=42
    )
    degree = 15

    res_no_reg = train_polynomial_l2(
        x_train_15, y_train_15, x_plot, y_true,
        degree=degree, l2=0.0
    )
    res_reg = train_polynomial_l2(
        x_train_15, y_train_15, x_plot, y_true,
        degree=degree, l2=0.01
    )

    ax = axes[0, 0]
    ax.scatter(
        x_train_15, y_train_15,
        color="black", s=30, label="Training data"
    )
    ax.plot(
        x_plot, y_true,
        color="green", linestyle="--", label="True sine"
    )
    plot_curve(x_plot, res_no_reg, degree, ax, "M=15, lambda=0")
    plot_curve(x_plot, res_reg, degree, ax, "M=15, lambda=0.01")

    ax.set_title("(a) Overfitting vs L2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2.5, 2.5)
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================
    # (b) 右上：test_mse vs degree
    # =========================================================
    ax = axes[0, 1]
    degrees = [1, 3, 5, 9, 15]

    for N in [15, 30, 100]:
        x_train, y_train = generate_data(
            N, noise_std=0.2, seed=42
        )

        test_mses = []
        for degree in degrees:
            result = train_polynomial_l2(
                x_train, y_train, x_plot, y_true,
                degree=degree, l2=0.0
            )
            test_mses.append(result["test_mse"])

        ax.semilogy(degrees, test_mses, "o-", label=f"N={N}")

    ax.set_title("(b) Test MSE vs degree")
    ax.set_xlabel("degree")
    ax.set_ylabel("Test MSE (log scale)")
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================
    # (c) 左下：train/test MSE vs lambda
    # =========================================================
    ax = axes[1, 0]
    l2_values = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

    train_mses = []
    test_mses = []

    for l2 in l2_values:
        result = train_polynomial_l2(
            x_train_15, y_train_15, x_plot, y_true,
            degree=15, l2=l2
        )
        train_mses.append(result["train_mse"])
        test_mses.append(result["test_mse"])

    xpos = np.arange(len(l2_values))

    ax.semilogy(xpos, train_mses, "o-", label="Train MSE")
    ax.semilogy(xpos, test_mses, "s-", label="Test MSE")

    ax.set_xticks(xpos)
    ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "0.1", "1"])
    ax.set_title("(c) Effect of L2")
    ax.set_xlabel("lambda")
    ax.set_ylabel("MSE (log scale)")
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================
    # (d) 右下：train/test MSE vs N
    # =========================================================
    ax = axes[1, 1]
    N_values = [15, 30, 100]

    train_mses = []
    test_mses = []

    for N in N_values:
        x_train, y_train = generate_data(
            N, noise_std=0.2, seed=42
        )

        result = train_polynomial_l2(
            x_train, y_train, x_plot, y_true,
            degree=15, l2=0.0
        )

        train_mses.append(result["train_mse"])
        test_mses.append(result["test_mse"])

    ax.semilogy(N_values, train_mses, "o-", label="Train MSE")
    ax.semilogy(N_values, test_mses, "s-", label="Test MSE")

    ax.set_title("(d) Effect of training size")
    ax.set_xlabel("N")
    ax.set_ylabel("MSE (log scale)")
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================
    # 统一收尾
    # =========================================================
    fig.tight_layout()
    fig.savefig("step12_final.png", dpi=150)


def main():
    make_figure()
    print("saved: a.png")


if __name__ == "__main__":
    main()
