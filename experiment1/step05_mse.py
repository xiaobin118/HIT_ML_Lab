# -*- coding: utf-8 -*-
"""
第 5 课：预测函数与 MSE 损失（骨架）

运行：
    python step05_mse.py
"""

import numpy as np


def predict(Phi, w):
    """
    计算所有样本的预测值。

    输入：
        Phi: shape (N, D)
        w:   shape (D,)
    输出：
        y_pred: shape (N,)

    任务：
    1. 使用矩阵乘法 @ 计算 Phi @ w；
    2. 返回结果。

    思考：
    - (N, D) @ (D,) 的结果 shape 是什么？
    - 为什么不能用 *？
    """
    # TODO
    y_pred: np.ndarray = Phi @ w
    print(f"y_pred's shape: {y_pred.shape}")
    return y_pred


def mse(y_true, y_pred):
    """
    计算均方误差 MSE。

    公式：
        MSE = (1/N) * sum_i (y_true_i - y_pred_i)^2

    输入：
        y_true: shape (N,)
        y_pred: shape (N,)
    输出：
        scalar float

    任务：
    1. 先求误差 err = y_pred - y_true；
    2. 对 err 逐元素平方；
    3. 用 np.mean 求平均值；
    4. 返回 Python float。

    要求：
    - 不允许写 for 循环。
    """
    # TODO
    err: np.ndarray = y_pred - y_true
    mse_value: float = np.mean(err ** 2)
    return mse_value


def main():
    # 一个可以手算的小例子
    Phi = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])
    w = np.array([1.0, 2.0])
    y_true = np.array([1.0, 3.0, 6.0])

    # 写完后：
    # 1. 调用 predict，打印 y_pred 和 shape；
    # 2. 检查 y_pred 是否等于 [1, 3, 5]；
    # 3. 调用 mse，打印结果；
    # 4. 手算检查 MSE 是否等于 1/3；
    # 5. 思考：如果 y_pred 和 y_true 的 shape 不同，会发生什么？
    y_pred = predict(Phi, w)
    print("y_pred:", y_pred)
    print("Check y_pred == [1, 3, 5]:", np.allclose(y_pred, np.array([1, 3, 5])))
    mse_value = mse(y_true, y_pred)
    print("MSE:", mse_value)


if __name__ == "__main__":
    main()
