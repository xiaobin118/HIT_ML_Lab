"""线性分类器：多项逻辑回归（softmax 回归），独立手写。

这是本实验的对照模型。它只有一个线性打分函数

    logits = X W + b ,      P(y=c|x) = softmax(logits)_c

决策边界是超平面；把 2 维输入映射到平面上，每一类的判定区域是若干半平面
的交，因此必然是**凸的、由直线围成**的区域。MLP 通过隐藏层把输入先映射到
一个新空间，相当于先学出一组非线性特征，再做线性划分，这是它能处理
环形、螺旋形等线性不可分数据的原因。
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """多项逻辑回归，用 (小批量) 梯度下降训练，支持 L2 正则。

    与 MLP 使用相同的优化器和训练循环，保证两者唯一的差别是模型本身，
    而不是训练技巧。
    """

    def __init__(self, n_features, n_classes, seed=0, init="zeros"):
        self.n_features = n_features
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        if init == "zeros":
            self.W = np.zeros((n_features, n_classes))
        elif init == "small":
            self.W = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        else:
            raise ValueError(init)
        self.b = np.zeros(n_classes)

    # 与 MLP 保持同样的参数接口，方便复用同一套优化器与训练循环
    @property
    def param_names(self):
        return ["W1", "b1"]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        self.W, self.b = params[0], params[1]

    def forward(self, X, cache=False):
        logits = X @ self.W + self.b
        if cache:
            return logits, X
        return logits

    def loss_and_grads(self, X, y, l2=0.0):
        logits = X @ self.W + self.b
        n = X.shape[0]
        m = logits.max(axis=1, keepdims=True)
        shifted = logits - m
        logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True)) + m
        loss = float(np.mean(logsumexp - logits[np.arange(n), y]))
        if l2:
            loss += 0.5 * l2 * float(np.sum(self.W * self.W))

        probs = softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        dW = X.T @ dlogits
        if l2:
            dW = dW + l2 * self.W
        db = dlogits.sum(axis=0)
        return loss, [dW, db]

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)

    def trainable_parameters(self):
        return self.W.size + self.b.size

    def count_units(self):
        return 0

    def decision_scores(self, X):
        """每个类别的线性打分，用于画平行等值线。"""
        return X @ self.W + self.b
