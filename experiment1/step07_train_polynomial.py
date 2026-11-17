# -*- coding: utf-8 -*-
"""
第 7 课：把前面的模块接起来，训练多项式模型（骨架）

运行：
    python step07_train_polynomial.py
"""

import numpy as np

from step02_generate_data import generate_data, make_test_data
from step03_polynomial_features import polynomial_features
from step04_standardize import standardize_train_test
from step05_mse import mse
from step06_gradient_descent import train_gd, half_mse_loss


def train_polynomial(x_train, y_train, x_test, y_test, degree,
                     lr=0.1, epochs=60000, noise_std=0.2):
    """
    完整训练流程。

    任务：
    1. 构造训练/测试特征矩阵：
           Phi_train = polynomial_features(x_train, degree)
           Phi_test  = polynomial_features(x_test, degree)
    2. 标准化：
           Phi_train_std, Phi_test_std, mean, std = standardize_train_test(...)
    3. 训练：
           w, losses = train_gd(Phi_train_std, y_train, lr=lr, epochs=epochs)
    4. 预测：
           y_train_pred = Phi_train_std @ w
           y_test_pred  = Phi_test_std @ w
    5. 计算：
           train_mse = mse(y_train, y_train_pred)
           test_mse  = mse(y_test, y_test_pred)
    6. 返回一个 dict：
           {
               "w": w,
               "train_mse": train_mse,
               "test_mse": test_mse,
               "losses": losses,
               "mean": mean,
               "std": std,
           }
    """
    Phi_train = polynomial_features(x_train, degree)
    Phi_test = polynomial_features(x_test, degree)
    Phi_train_std, Phi_test_std, mean, std = standardize_train_test(Phi_train, Phi_test)
    w, losses = train_gd(Phi_train_std, y_train, lr=lr, epochs=epochs)
    y_train_pred = Phi_train_std @ w
    y_test_pred = Phi_test_std @ w
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)
    return {
        "w": w,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "losses": losses,
        "mean": mean,
        "std": std,
    }


def main():
    # 先固定小样本，专门观察高阶多项式过拟合
    n_train = 15
    x_train, y_train = generate_data(n_train, noise_std=0.2, seed=42)
    x_test, y_test = make_test_data(num=600)

    print(f"{'degree':>8s} {'train_mse':>14s} {'test_mse':>14s} {'||w||':>10s}")

    for degree in [1, 3, 5, 9, 15]:
        result = train_polynomial(
            x_train, y_train, x_test, y_test,
            degree=degree,
            lr=0.1,
            epochs=60000,
        )
        print(f"{degree:8d} {result['train_mse']:14.6f} "
              f"{result['test_mse']:14.6f} "
              f"{np.linalg.norm(result['w']):10.4f}")

    # 完成后检查：
    # 1. 高次多项式 train_mse 是否很低？
    # 2. 高次多项式 test_mse 是否反而很大？
    # 3. 观察哪个 degree 的 test_mse 最小。
    # 4. 检查 losses[-1] 是否约等于 0.5 * train_mse。


if __name__ == "__main__":
    main()
