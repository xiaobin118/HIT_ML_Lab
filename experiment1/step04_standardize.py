# -*- coding: utf-8 -*-
"""
第 4 课：特征标准化（骨架）

运行：
    python step04_standardize.py
"""

import numpy as np


def fit_standardizer(Phi_train):
    """
    在训练特征矩阵上计算每一列的均值和标准差。

    输入：
        Phi_train: shape (N, D)，D = degree + 1
    输出：
        mean: shape (D - 1,)
        std:  shape (D - 1,)

    任务：
    1. 只对第 1 列到最后一列计算均值和标准差，第 0 列是截距 1，不参与；
    2. 如果某一列 std 非常接近 0，把它设为 1.0，避免后面除以 0；
    3. 返回 mean, std。

    提示：
    - Phi_train[:, 1:]
    - .mean(axis=0)
    - .std(axis=0)
    - np.where(condition, value_if_true, value_if_false)
    """
    # TODO
    mean: np.ndarray = Phi_train[:, 1:].mean(axis=0)
    std: np.ndarray = Phi_train[:, 1:].std(axis=0)
    std: np.ndarray = np.where(std < 1e-8, 1.0, std)
    return mean, std


def transform_features(Phi, mean, std):
    """
    用训练集得到的 mean/std 对特征矩阵做标准化。

    输入：
        Phi: shape (N, D)
        mean: shape (D - 1,)
        std: shape (D - 1,)
    输出：
        Phi_std: shape (N, D)，第 0 列保持为 1

    任务：
    1. 不要直接修改传入的 Phi，先复制一份；
    2. 第 0 列保持不变；
    3. 第 1 列到最后一列执行 (Phi[:, 1:] - mean) / std；
    4. 返回新矩阵。

    注意：
    - NumPy 广播规则：mean/std 会按列广播到每一行。
    """
    # TODO
    Phi_std: np.ndarray = Phi.copy()
    Phi_std[:, 1:] = (Phi_std[:, 1:] - mean) / std
    return Phi_std


def standardize_train_test(Phi_train, Phi_test):
    """
    训练集拟合标准化参数，训练集和测试集都使用同一组参数变换。

    返回：
        Phi_train_std, Phi_test_std, mean, std
    """
    # TODO
    mean, std = fit_standardizer(Phi_train)
    Phi_train_std = transform_features(Phi_train, mean, std)
    Phi_test_std = transform_features(Phi_test, mean, std)
    return Phi_train_std, Phi_test_std, mean, std


def main():
    # 构造一个小例子，方便观察
    x_train = np.linspace(-1, 1, 15)
    x_test = np.linspace(-1, 1, 5)

    # 先用第 3 课的 polynomial_features 构造特征。
    # 这里先简单复制一份实现，避免 import 出现路径问题。
    def make_phi(x, degree):
        return np.vander(np.asarray(x, dtype=float), N=degree + 1, increasing=True)

    degree = 5
    Phi_train = make_phi(x_train, degree)
    Phi_test = make_phi(x_test, degree)

    # 写完后，调用 standardize_train_test，并检查：
    # 1. 训练集标准化后，Phi_train_std[:, 1:] 每列均值约等于 0；
    # 2. 训练集标准化后，Phi_train_std[:, 1:] 每列标准差约等于 1；
    # 3. Phi_train_std[:, 0] 仍然全为 1；
    # 4. Phi_test_std 的均值/标准差不必等于 0/1；
    #    因为它用的是训练集统计量，不能在测试集上重新拟合。
    Phi_train_std, Phi_test_std, mean, std = standardize_train_test(Phi_train, Phi_test)
    print("Phi_train_std[:, 1:] mean:", Phi_train_std[:, 1:].mean(axis=0))
    print("Phi_train_std[:, 1:] std:", Phi_train_std[:, 1:].
          std(axis=0))
    print("Phi_train_std[:, 0]:", Phi_train_std[:, 0])
    print("Phi_test_std[:, 1:] mean:", Phi_test_std[:, 1:].mean(axis=0))
    print("Phi_test_std[:, 1:] std:", Phi_test_std[:, 1:].std(axis=0))
    print("Phi_test_std[:, 0]:", Phi_test_std[:, 0])


if __name__ == "__main__":
    main()
