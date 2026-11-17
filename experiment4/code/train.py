"""优化器与训练循环（不依赖 torch.optim，全部手写）。

支持的优化器：
- SGD：p <- p - lr * g
- Momentum：v <- mu*v - lr*g ; p <- p + v
- Adam：一阶/二阶矩估计 + 偏差修正
"""

from __future__ import annotations

import time

import numpy as np


class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g

    def state_dict(self):
        return {}


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.momentum
            v -= self.lr * g
            p += v

    def state_dict(self):
        return {"momentum": self.momentum}


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            m_hat = m / (1.0 - b1 ** self.t)
            v_hat = v / (1.0 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {"beta1": self.beta1, "beta2": self.beta2}


def make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":
        return SGD(lr)
    if name == "momentum":
        return Momentum(lr, momentum=0.9)
    if name == "adam":
        return Adam(lr)
    raise ValueError(f"未知优化器: {name}")


def accuracy(model, X, y):
    if len(y) == 0:
        return float("nan")
    return float(np.mean(model.predict(X) == y))


def evaluate(model, X, y):
    loss, _ = model.loss_and_grads(X, y, l2=0.0)
    return {"loss": loss, "acc": accuracy(model, X, y)}


def fit(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    l2=0.0,
    optimizer="adam",
    seed=0,
    shuffle=True,
    eval_each_epoch=True,
):
    """小批量训练。

    batch_size >= 训练集大小 时退化为全批量梯度下降。
    返回 history: 每个 epoch 一行，记录损失、准确率与耗时。
    """
    rng = np.random.default_rng(seed)
    opt = make_optimizer(optimizer, lr)
    n = X_train.shape[0]
    bs = min(int(batch_size), n)
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n) if shuffle else np.arange(n)
        epoch_loss, correct = 0.0, 0
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            xb, yb = X_train[idx], y_train[idx]
            loss, grads = model.loss_and_grads(xb, yb, l2=l2)
            # 用更新前的参数统计训练准确率，避免高估
            correct += int(np.sum(model.predict(xb) == yb))
            opt.step(model.get_params(), grads)
            epoch_loss += loss * len(idx)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / n,
            "train_acc": correct / n,
            "time": time.perf_counter() - t0,
        }
        if eval_each_epoch and X_test is not None:
            ev = evaluate(model, X_test, y_test)
            row["test_loss"] = ev["loss"]
            row["test_acc"] = ev["acc"]
        history.append(row)

    return history
