# -*- coding: utf-8 -*-
"""
第 3 课：多项式特征矩阵（骨架）

运行：
    python step03_polynomial_features.py
"""

import numpy as np


def polynomial_features(x, degree):
    """
    把一维输入 x 变成多项式特征矩阵 Phi。

    输入：
        x: shape (N,)，一维数组
        degree: 多项式最高次数 M

    输出：
        Phi: shape (N, M+1)

    其中第 i 行是：
        [1, x_i, x_i^2, ..., x_i^M]

    任务：
    1. 使用 np.vander 构造 Vandermonde 矩阵；
    2. 注意参数 increasing=True，让列的顺序从低次到高次；
    3. 返回 Phi。

    提示：
    - increasing=False 时，列顺序是 x^M, ..., x, 1；
    - 我们要的是 1, x, x^2, ..., x^M。
    """
    # TODO
    x: np.ndarray = np.asarray(x)
    Phi = np.vander(x, N=degree + 1, increasing=True)
    return Phi


def main():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    # 写完后，循环 M = 1, 3, 5，打印：
    # 1. Phi.shape
    # 2. Phi
    #
    # 然后增加检查：
    # np.allclose(Phi[:, 0], 1)
    # np.allclose(Phi[:, 1], x)
    # np.allclose(Phi[:, 2], x ** 2)
    #
    # 如果是 True，说明特征矩阵构造正确。
    for M in [1, 3, 5]:
        Phi = polynomial_features(x, M)
        print(f"Degree: {M}, Phi.shape: {Phi.shape}")
        print(Phi)
        print("Check:")
        print("Phi[:, 0] == 1:", np.allclose(Phi[:, 0], 1))
        print("Phi[:, 1] == x:", np.allclose(Phi[:, 1], x))
        if M >= 2:
            print("Phi[:, 2] == x^2:", np.allclose(Phi[:, 2], x ** 2))
        print()


if __name__ == "__main__":
    main()
