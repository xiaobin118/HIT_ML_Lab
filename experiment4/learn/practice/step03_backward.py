"""第 3 步：反向传播。

目标：从 dZ = (P - Y) / N 出发，逐层回传，算出所有 dW、db，
并用数值梯度验证。

本文件继承第 2 步的 MLP（前向、初始化、predict 都已经有了），
只需要补激活函数导数和 loss_and_grads。

规则
----
1. 只改标了 TODO 的地方。
2. 运行 `python step03_backward.py`，七项自检必须 PASS。
3. 不要打开 `../code/mlp.py` 的 loss_and_grads 部分。
"""

from __future__ import annotations

import numpy as np

from common import finite_difference_error, make_blobs
from step01_linear import cross_entropy
from step02_mlp_forward import ACTIVATIONS, Array, MLP as ForwardMLP, sigmoid


# ---------------------------------------------------------------------------
# TODO 1：激活函数的导数
# ---------------------------------------------------------------------------
def relu_grad(z: Array) -> Array:
    """ReLU 的导数：z > 0 处为 1，否则为 0。

    Parameters
    ----------
    z : Array
        激活前的值，形状任意（通常是 (N, n_out)）。

    Returns
    -------
    Array
        与 z 形状相同的浮点数组，元素为 0.0 或 1.0。

    提示：`(z > 0)` 得到布尔数组，再 `.astype(float)` 转成 0/1。
    """
    # TODO 1a：一行。
    return (z > 0).astype(float)


def sigmoid_grad(z: Array) -> Array:
    """sigmoid 的导数：s * (1 - s)，其中 s = sigmoid(z)。

    Parameters
    ----------
    z : Array
        激活前的值。
    """
    # TODO 1b：可以先 s = sigmoid(z)，再返回 s * (1 - s)。
    s = sigmoid(z)
    return s * (1 - s)


def tanh_grad(z: Array) -> Array:
    """tanh 的导数：1 - tanh(z)^2。"""
    # TODO 1c：一行。
    return 1 - np.tanh(z) ** 2


def linear_grad(z: Array) -> Array:
    """线性激活的导数：处处为 1。已提供。"""
    return np.ones_like(z, dtype=float)


# 名字 -> 导数函数。forward 用 ACTIVATIONS，反向用这个。
ACTIVATION_GRADS = {
    "relu": relu_grad,
    "sigmoid": sigmoid_grad,
    "tanh": tanh_grad,
    "linear": linear_grad,
}


# ---------------------------------------------------------------------------
# TODO 2：反向传播
# ---------------------------------------------------------------------------
class MLP(ForwardMLP):
    """在第 2 步的前向网络上加一个 loss_and_grads 方法。"""

    def loss_and_grads(
        self, X: Array, y: Array, l2: float = 0.0
    ) -> tuple[float, list[Array]]:
        """返回 (loss, grads)。

        Parameters
        ----------
        X : Array
            输入，形状 (N, D)。
        y : Array
            真实类别，形状 (N,)，整数 0..C-1。
        l2 : float
            L2 正则系数 λ；只惩罚 W，不惩罚 b。

        Returns
        -------
        loss : float
            平均交叉熵 + 0.5 * l2 * sum(||W||^2)。
        grads : list[Array]
            顺序与 self.Ws + self.bs 一致：先所有 dW，再所有 db。
            对两层网络是 [dW1, dW2, db1, db2]。

        步骤：
        1. logits, layers = self.forward(X, cache=True)
        2. loss, dZ = cross_entropy(logits, y)
        3. 从最后一层往前循环：
               dW = a_prev.T @ dZ
               db = dZ.sum(axis=0)
               如果 l > 0：
                   dA_prev = dZ @ W.T
                   dZ = dA_prev * ACTIVATION_GRADS[self.activation](layers[l-1][1])
        4. 若 l2 > 0：loss += 0.5*l2*sum(W^2)，每个 dW += l2*W
        5. return loss, dWs + dbs
        """
        # TODO 2：按上面的步骤实现。
        logits, layers = self.forward(X, cache=True)
        loss, dZ = cross_entropy(logits, y)

        n_layers = len(self.Ws)
        dWs: List[Array] = [None] * n_layers
        dbs: List[Array] = [None] * n_layers
        grad_fn = ACTIVATION_GRADS[self.activation]

        for l in range(n_layers - 1, -1, -1):
            a_prev, z, a = layers[l]
            W = self.Ws[l]
            dWs[l] = a_prev.T @ dZ
            dbs[l] = dZ.sum(axis=0)
            if l > 0:
                dA_prev = dZ @ W.T
                dZ = dA_prev * grad_fn(layers[l - 1][1])

        if l2 > 0:
            loss += 0.5 * l2 * sum(float(np.sum(W * W)) for W in self.Ws)
            for l in range(n_layers):
                dWs[l] += l2 * self.Ws[l]

        return loss, dWs + dbs


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check(name: str, passed: bool, detail: str = "") -> bool:
    """打印 PASS/FAIL 并返回结果。"""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(passed)


def grad_error(model: MLP, X: Array, y: Array, l2: float) -> float:
    """用中心差分检查 model 的解析梯度，返回最大绝对误差。"""
    loss, grads = model.loss_and_grads(X, y, l2=l2)

    def loss_only() -> float:
        logits = model.forward(X)          # cache=False，只返回 logits
        value, _ = cross_entropy(logits, y)
        if l2:
            value += 0.5 * l2 * sum(float(np.sum(W * W)) for W in model.Ws)
        return value

    return finite_difference_error(loss_only, model.Ws + model.bs, grads)


def self_check() -> bool:
    print("=" * 70)
    print("第 3 步自检")
    print("=" * 70)
    rng = np.random.default_rng(0)
    ok = True

    # 1. 激活导数
    rg = relu_grad(np.array([-1.0, 0.0, 2.0]))
    sg = sigmoid_grad(np.array([0.0]))
    tg = tanh_grad(np.array([0.0]))
    ok1 = (np.allclose(rg, [0.0, 0.0, 1.0])
           and abs(sg[0] - 0.25) < 1e-12
           and abs(tg[0] - 1.0) < 1e-12)
    ok &= check("1. 激活导数", ok1, f"relu'={rg}, sigmoid'(0)={sg[0]:.4f}, tanh'(0)={tg[0]:.4f}")

    # 2~4. 三种激活的梯度检查
    X = rng.normal(size=(12, 3))
    y = rng.integers(0, 4, size=12)
    for idx, activation in enumerate(["relu", "tanh", "sigmoid"], start=2):
        model = MLP([3, 5, 4], activation=activation, seed=0)
        err = grad_error(model, X, y, l2=0.0)
        ok &= check(f"{idx}. {activation} 梯度", err < 1e-6, f"最大误差 = {err:.2e}")

    # 5. L2 梯度
    model = MLP([3, 5, 4], activation="tanh", seed=0)
    err_l2 = grad_error(model, X, y, l2=0.01)
    ok &= check("5. L2 梯度", err_l2 < 1e-6, f"最大误差 = {err_l2:.2e}")

    # 6. 梯度形状与顺序
    model = MLP([3, 5, 4], activation="relu", seed=0)
    _, grads = model.loss_and_grads(X, y)
    params = model.Ws + model.bs
    shapes_ok = (len(grads) == len(params)
                 and all(g.shape == p.shape for g, p in zip(grads, params)))
    ok &= check("6. 梯度形状", shapes_ok,
                f"{[tuple(g.shape) for g in grads]}")

    # 7. 用自己算的梯度做 30 步全批量更新，损失应该下降
    Xb, yb = make_blobs(n_per_class=80, seed=1)
    model = MLP([2, 16, 4], activation="relu", seed=0)
    loss0, _ = model.loss_and_grads(Xb, yb)
    for _ in range(30):
        loss, grads = model.loss_and_grads(Xb, yb)
        for p, g in zip(model.Ws + model.bs, grads):
            p -= 0.2 * g
    loss1, _ = model.loss_and_grads(Xb, yb)
    ok &= check("7. 损失下降", loss1 < loss0 * 0.7,
                f"{loss0:.3f} -> {loss1:.3f}")

    print("-" * 70)
    print("全部通过。" if ok else "还有检查未通过。")
    return ok


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
def demo() -> None:
    print("\n梯度检查误差的量级：")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = rng.integers(0, 4, size=12)
    for activation in ["relu", "tanh", "sigmoid"]:
        model = MLP([3, 5, 4], activation=activation, seed=0)
        err = grad_error(model, X, y, l2=0.01)
        print(f"  {activation:8s} 最大误差 = {err:.2e}")
    print("  < 1e-6 说明解析梯度与数值梯度一致，反向传播写对了。")


if __name__ == "__main__":
    try:
        all_ok = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if all_ok:
        demo()
