"""多层感知机（MLP）的 NumPy 实现。

本文件手写前向传播与反向传播，不使用 PyTorch / TensorFlow / JAX 等任何
自动微分框架，也不依赖 scikit-learn 的模型实现。所有梯度均由链式法则
逐层显式推出并写成矩阵乘法（见报告第 4 节）。

记号约定
--------
- 第 l 层（l = 1..L-1，0 号为输入层）：
      Z_l = A_{l-1} W_l + b_l ,    A_l = f(Z_l)
  其中 W_l 形状 (n_{l-1}, n_l)，b_l 形状 (n_l,)。
- 输出层不加激活函数，直接输出 logits，配合 softmax + 交叉熵。
- 损失（含 L2 正则，偏置不参与正则）：
      L = -1/N * sum_n log softmax(z_n)[y_n] + lambda/2 * sum_l ||W_l||_F^2
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 激活函数：每个都返回 (f, f')，且 f' 只依赖 z
# ---------------------------------------------------------------------------


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定写法：正负分两段，避免 exp 溢出。"""
    out = np.empty_like(z, dtype=z.dtype)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t


def identity(z: np.ndarray) -> np.ndarray:
    return z


def identity_grad(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "linear": (identity, identity_grad),
}


# ---------------------------------------------------------------------------
# softmax 与交叉熵
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """按行做 softmax，减去行最大值保证数值稳定（不改变结果）。"""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """返回 (平均交叉熵, 对 logits 的梯度 (N, C))。

    用 logsumexp 计算损失，避免 log(0)；对 logits 的梯度就是
    (softmax(z) - onehot(y)) / N，这一步是整条反向链的起点。
    """
    n = logits.shape[0]
    m = logits.max(axis=1, keepdims=True)
    shifted = logits - m
    logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True)) + m
    loss = np.mean(logsumexp - logits[np.arange(n), y])

    probs = softmax(logits)
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    return float(loss), dlogits


# ---------------------------------------------------------------------------
# 参数初始化
# ---------------------------------------------------------------------------


class MLP:
    """全连接多层感知机（多分类）。

    Parameters
    ----------
    layer_sizes : list[int]
        例如 [784, 256, 128, 10]：输入 784 维，两个隐藏层，10 类输出。
    activation : str
        隐藏层激活函数，取 "relu" / "tanh" / "sigmoid" / "linear"。
    init : str
        权重初始化方式：
        - "he"     : N(0, 2/fan_in)，适合 ReLU；
        - "xavier" : N(0, 2/(fan_in+fan_out))，适合 tanh/sigmoid；
        - "small"  : N(0, 0.01^2)；
        - "zeros"  : 全零（用来演示对称性无法打破）；
        - "auto"   : ReLU 用 he，其余用 xavier。
    seed : int
        初始化随机种子。
    """

    def __init__(self, layer_sizes, activation="relu", init="auto", seed=0):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes 至少要有输入维和输出维")
        if activation not in ACTIVATIONS:
            raise ValueError(f"未知激活函数: {activation}")
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.init = init
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.Ws, self.bs = [], []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.Ws.append(self._init_weight(fan_in, fan_out))
            self.bs.append(np.zeros(fan_out, dtype=np.float64))
        # 供可视化使用的训练步数计数
        self.step_count = 0

    def _init_weight(self, fan_in, fan_out):
        mode = self.init
        if mode == "auto":
            mode = "he" if self.activation == "relu" else "xavier"
        if mode == "he":
            std = np.sqrt(2.0 / fan_in)
        elif mode == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
        elif mode == "small":
            std = 0.01
        elif mode == "zeros":
            return np.zeros((fan_in, fan_out), dtype=np.float64)
        else:
            raise ValueError(f"未知初始化方式: {mode}")
        return self.rng.normal(0.0, std, size=(fan_in, fan_out))

    # -- 参数扁平化，便于优化器统一处理 -----------------------------------
    @property
    def param_names(self):
        # 顺序必须与 get_params() 一致：先所有 W，再所有 b
        names = [f"W{i + 1}" for i in range(len(self.Ws))]
        names += [f"b{i + 1}" for i in range(len(self.bs))]
        return names

    def get_params(self):
        return self.Ws + self.bs

    def set_params(self, params):
        n = len(self.Ws)
        self.Ws = list(params[:n])
        self.bs = list(params[n:])

    # -- 前向传播 ---------------------------------------------------------
    def forward(self, X, cache=False):
        """返回 logits；cache=True 时同时返回反向传播所需的中间量。"""
        a = X
        layers = []
        n_layers = len(self.Ws)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            if i < n_layers - 1:
                a_next = self.act(z)
            else:
                a_next = z  # 输出层：线性 logits
            if cache:
                layers.append((a, z, a_next))
            a = a_next
        if cache:
            return a, layers
        return a

    def loss_and_grads(self, X, y, l2=0.0):
        """返回 (损失, 梯度列表)，梯度顺序与 get_params() 一致。"""
        logits, layers = self.forward(X, cache=True)
        loss, dlogits = softmax_cross_entropy(logits, y)

        # L2 正则（只惩罚权重矩阵，不惩罚偏置）
        if l2:
            loss += 0.5 * l2 * sum(float(np.sum(W * W)) for W in self.Ws)

        n_layers = len(self.Ws)
        dWs = [None] * n_layers
        dbs = [None] * n_layers
        dZ = dlogits
        for i in range(n_layers - 1, -1, -1):
            a_prev, z, _ = layers[i]
            W = self.Ws[i]
            dWs[i] = a_prev.T @ dZ
            if l2:
                dWs[i] = dWs[i] + l2 * W
            dbs[i] = dZ.sum(axis=0)
            if i > 0:
                dA_prev = dZ @ W.T
                z_prev = layers[i - 1][1]
                dZ = dA_prev * self.act_grad(z_prev)
        return loss, dWs + dbs

    # -- 推理 -------------------------------------------------------------
    def predict_proba(self, X):
        return softmax(self.forward(X))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def trainable_parameters(self):
        return sum(p.size for p in self.get_params())

    def count_units(self):
        """隐藏层神经元数量之和（用于画「容量—精度」曲线）。"""
        return sum(self.layer_sizes[1:-1])


# ---------------------------------------------------------------------------
# 数值梯度校验
# ---------------------------------------------------------------------------


def numeric_gradient_check(model, X, y, l2=0.0, eps=1e-6):
    """中心差分逐元素检验解析梯度。

    Returns
    -------
    rows : list[dict]
        每个参数张量一行：名称、最大绝对误差、相对误差、是否通过。
    """
    _, grads = model.loss_and_grads(X, y, l2)
    params = model.get_params()
    rows = []
    for idx, (name, P) in enumerate(zip(model.param_names, params)):
        G = grads[idx]
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            k = it.multi_index
            old = P[k]
            P[k] = old + eps
            lp, _ = model.loss_and_grads(X, y, l2)
            P[k] = old - eps
            lm, _ = model.loss_and_grads(X, y, l2)
            P[k] = old
            num[k] = (lp - lm) / (2.0 * eps)
        abs_err = float(np.max(np.abs(num - G)))
        denom = max(float(np.max(np.abs(num) + np.abs(G))), 1e-12)
        rel_err = abs_err / denom
        rows.append(
            {
                "param": name,
                "shape": list(P.shape),
                "max_abs_err": abs_err,
                "rel_err": rel_err,
                "passed": abs_err < 1e-6,
            }
        )
    return rows
