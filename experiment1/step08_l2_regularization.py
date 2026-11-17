# -*- coding: utf-8 -*-
"""
第 8 课：L2 正则化（骨架）

运行：
    python step08_l2_regularization.py
"""

import numpy as np


def l2_penalty(w, l2, reg_bias=False):
    """
    计算 L2 惩罚项：
        (l2 / 2) * sum(w_j^2)

    任务：
    1. 如果 reg_bias=False，不惩罚 w[0]（截距）；
    2. 如果 reg_bias=True，所有 w 都参与惩罚；
    3. 返回标量 float。

    提示：
    - 截距之外的权重：w[1:]
    - 平方和：np.sum(...)
    """
    # TODO
    if reg_bias:
        return (l2 / 2) * np.sum(w ** 2)
    else:
        return (l2 / 2) * np.sum(w[1:] ** 2)


def l2_objective(Phi, y, w, l2, reg_bias=False):
    """
    计算带 L2 的目标函数：
        J = 1/(2N) * ||Phi w - y||^2 + l2_penalty(w, l2)

    任务：
    1. 先算数据项 0.5 * np.mean((Phi @ w - y) ** 2)；
    2. 再加上 l2_penalty(...)；
    3. 返回标量 float。
    """
    # TODO
    err = Phi @ w - y
    mse_loss = 0.5 * np.mean(err ** 2)
    data_loss = mse_loss + l2_penalty(w, l2, reg_bias)
    return data_loss


def l2_gradient(Phi, y, w, l2, reg_bias=False):
    """
    计算带 L2 的梯度：
        grad = (1/N) * Phi.T @ (Phi @ w - y) + l2 * w_pen

    其中 w_pen 是惩罚部分：
        reg_bias=False 时，w_pen[0] = 0，其余等于 w；
        reg_bias=True 时，w_pen = w。

    任务：
    1. 先算原始梯度 (1/N) * Phi.T @ (Phi @ w - y)；
    2. 构造 reg = l2 * w；
    3. 若 reg_bias=False，令 reg[0] = 0；
    4. grad = 原始梯度 + reg；
    5. 返回 grad，shape 为 (D,)。
    """
    # TODO
    N = Phi.shape[0]
    err = Phi @ w - y
    original_grad = (1 / N) * (Phi.T @ err)
    reg = l2 * w
    if not reg_bias:
        reg[0] = 0
    grad = original_grad + reg
    return grad



def train_gd_l2(Phi, y, lr=0.1, epochs=1000, l2=0.0, reg_bias=False):
    """
    带 L2 正则化的全批量梯度下降。

    任务：
    1. w 初始化为全零，shape (D,)；
    2. 循环 epochs 次：
        a. grad = l2_gradient(Phi, y, w, l2, reg_bias)
        b. w = w - lr * grad
        c. 记录 l2_objective(Phi, y, w, l2, reg_bias)
    3. 返回 w, losses。
    """
    # TODO
    D = Phi.shape[1]
    w = np.zeros(D)
    losses = []
    for _ in range(epochs):
        grad = l2_gradient(Phi, y, w, l2, reg_bias)
        w = w - lr * grad
        loss = l2_objective(Phi, y, w, l2, reg_bias)
        losses.append(loss)
    return w, np.array(losses)


def main():
    # 简单线性例子：y = 1 + 2x
    Phi = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])
    y = np.array([1.0, 3.0, 5.0])

    # 写完后：
    # 1. 对 l2 = 0, 0.01, 0.1, 1, 10 分别训练；
    # 2. 打印最终的 w；
    # 3. 打印最终的训练 MSE（用 step05 的 mse 或自己算）；
    # 4. 观察 l2 增大时，w[1] 是否变小；
    # 5. 思考：为什么 l2=0 时损失可以到 0，而 l2>0 时通常不行？
    l2_values = [0, 0.01, 0.1, 1, 10]
    for l2 in l2_values:
        w, losses = train_gd_l2(Phi, y, lr=0.1, epochs=1000, l2=l2, reg_bias=False)
        train_mse = np.mean((Phi @ w - y) ** 2)
        print(f"l2={l2}: w={w}, train_mse={train_mse}")


if __name__ == "__main__":
    main()
