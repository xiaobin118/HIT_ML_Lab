# -*- coding: utf-8 -*-
"""
第 9 课：把 L2 接回多项式训练流程（骨架）

运行：
    python step09_polynomial_l2.py
"""

import numpy as np

from step02_generate_data import generate_data, make_test_data
from step03_polynomial_features import polynomial_features
from step04_standardize import standardize_train_test
from step05_mse import mse
from step08_l2_regularization import train_gd_l2, l2_objective


def train_polynomial_l2(x_train, y_train, x_test, y_test, degree,
                        lr=0.1, epochs=60000, l2=0.0):
    """
    完整训练流程 + L2 正则化。

    任务：
    1. 构造 Phi_train、Phi_test；
    2. 标准化，得到 Phi_train_std、Phi_test_std；
    3. 调用 train_gd_l2(Phi_train_std, y_train,
                       lr=lr, epochs=epochs, l2=l2)；
    4. 计算 y_train_pred、y_test_pred；
    5. 计算 train_mse、test_mse；
    6. 返回：
         {
            "w": w,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "losses": losses,
            "mean": mean,
            "std": std,
         }
    """
    # TODO
    Phi_train = polynomial_features(x_train, degree)
    Phi_test = polynomial_features(x_test, degree)
    Phi_train_std, Phi_test_std, mean, std = standardize_train_test(Phi_train, Phi_test)
    w, losses = train_gd_l2(Phi_train_std, y_train, lr=lr, epochs=epochs, l2=l2)
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
    # 固定同一批数据，保证不同 lambda 公平比较
    n_train = 15
    degree = 15
    x_train, y_train = generate_data(n_train, noise_std=0.2, seed=42)
    x_test, y_test = make_test_data(num=600)

    print(f"{'lambda':>10s} {'train_mse':>14s} {'test_mse':>14s} {'||w||':>10s}")

    l2_values = [1e-2, 1e-1, 1.0, 2.0, 3.0, 4.0, 5.0]

    for l2 in l2_values:
        result = train_polynomial_l2(
            x_train, y_train, x_test, y_test,
            degree=degree,
            lr=0.1,
            epochs=60000,
            l2=l2,
        )
        print(f"{l2:10.4g} {result['train_mse']:14.6f} "
              f"{result['test_mse']:14.6f} "
              f"{np.linalg.norm(result['w']):10.4f}")

    # 完成后观察：
    # 1. lambda=0 时是否 train_mse 小、test_mse 大；
    # 2. 增大 lambda 后 test_mse 是否先下降后上升；
    # 3. ||w|| 是否随着 lambda 增大而减小；
    # 4. 哪个 lambda 的 test_mse 最小。


if __name__ == "__main__":
    main()
