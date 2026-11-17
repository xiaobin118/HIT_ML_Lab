# -*- coding: utf-8 -*-
"""
第 6 课：手写梯度下降（骨架）

运行：
    python step06_gradient_descent.py
"""

import numpy as np


def half_mse_loss(Phi, y, w):
    """
    使用 1/(2N) 形式的目标函数：
        J(w) = 1/(2N) * sum_i (Phi_i @ w - y_i)^2

    输入：
        Phi: shape (N, D)
        y:   shape (N,)
        w:   shape (D,)
    输出：
        scalar float

    任务：
    1. 先算 err = Phi @ w - y；
    2. 返回 0.5 * np.mean(err ** 2)。

    说明：
    - 除以 2 是为了求导时和平方的 2 抵消，后面梯度更简洁。
    """
    err = Phi @ w - y
    return 0.5 * np.mean(err ** 2)


def gradient(Phi, y, w):
    """
    计算 J 对 w 的梯度。

    数学结果：
        grad = (1/N) * Phi.T @ (Phi @ w - y)

    输入：
        Phi: shape (N, D)
        y:   shape (N,)
        w:   shape (D,)
    输出：
        grad: shape (D,)

    任务：
    1. 算 err = Phi @ w - y；
    2. 算 Phi.T @ err；
    3. 除以样本数 N；
    4. 返回 grad。

    形状检查：
        Phi.T: (D, N)
        err:   (N,)
        结果:  (D,)
    """
    # TODO
    err = Phi @ w - y
    N = Phi.shape[0]
    grad = (1 / N) * (Phi.T @ err)
    return grad


def train_gd(Phi, y, lr=0.1, epochs=1000):
    """
    全批量梯度下降。

    输入：
        Phi: shape (N, D)
        y:   shape (N,)
        lr: 学习率
        epochs: 迭代次数
    输出：
        w: shape (D,)
        losses: shape (epochs,)，每次迭代后的损失

    任务：
    1. 初始化 w 为全零向量，shape (D,)；
    2. 循环 epochs 次：
        a. 计算梯度；
        b. 更新 w = w - lr * grad；
        c. 记录当前损失 half_mse_loss(Phi, y, w)；
    3. 返回 w 和 losses。

    注意：
    - 这里的梯度是解析梯度，不是数值近似；
    - 不要用 PyTorch/TensorFlow 自动微分。
    """
    # TODO
    N, D = Phi.shape
    w = np.zeros(D)
    losses = np.zeros(epochs)
    for epoch in range(epochs):
        grad = gradient(Phi, y, w)
        w -= lr * grad
        losses[epoch] = half_mse_loss(Phi, y, w)
    return w, losses


def main():
    # 一个可以精确求解的线性例子
    Phi = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])
    y = np.array([1.0, 3.0, 5.0])

    # 真实关系：y = 1 + 2x
    # 理论最优 w 应该是 [1, 2]

    # 写完后：
    # 1. 调用 train_gd(Phi, y, lr=0.1, epochs=1000)
    # 2. 打印最终 w
    # 3. 打印最终损失，应该接近 0
    # 4. 打印 losses[:10] 和 losses[-10:]，观察损失是否下降
    # 5. 额外实验：把 lr 改成 0.01、0.5、1.0，观察结果
    w, losses = train_gd(Phi, y, lr=0.1, epochs=1000)
    print("Final w:", w)
    print("Final loss:", losses[-1])
    print("Losses first 10:", losses[:10])
    print("Losses last 10:", losses[-10:])


if __name__ == "__main__":
    main()
