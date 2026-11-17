"""第 4 步：优化器与小批量训练。

目标：
    1. 实现 SGD / Momentum / Adam 三个优化器；
    2. 实现 fit：每个 epoch 打乱数据、切小批量、更新参数、记录曲线；
    3. 在 MNIST 子集上训练到 95% 以上。

本文件复用第 3 步的 MLP（已经有前向、反向、损失），
以及主项目的 MNIST 加载器（只加载数据，不涉及模型实现）。

运行：python step04_train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from common import make_blobs, train_test_split
from step03_backward import MLP

# 复用主项目的数据加载器：把 experiment4/code 加入 import 路径。
# 这里只用它的 load_mnist / stratified_subsample，模型部分仍然是你在 step03 写的。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from datasets import load_mnist, stratified_subsample  # noqa: E402

Array = np.ndarray


# ---------------------------------------------------------------------------
# TODO 1：三个优化器
# ---------------------------------------------------------------------------
class SGD:
    """最朴素的梯度下降：p <- p - lr * g。"""

    def __init__(self, lr: float) -> None:
        """lr : float，学习率。"""
        self.lr = lr

    def step(self, params: list[Array], grads: list[Array]) -> None:
        """原地更新参数。

        Parameters
        ----------
        params : list[Array]
            参数列表，顺序是 model.Ws + model.bs。
        grads : list[Array]
            与 params 一一对应的梯度。
        """
        # TODO 1a：一行循环，p -= self.lr * g。
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    """带动量的 SGD：v <- mu*v - lr*g，p <- p + v。"""

    def __init__(self, lr: float, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        # 第一次 step 时再按参数形状初始化，见 TODO。
        self.velocity: list[Array] | None = None

    def step(self, params: list[Array], grads: list[Array]) -> None:
        """原地更新参数。

        第一次调用时初始化速度：
            self.velocity = [np.zeros_like(p) for p in params]
        之后每一步：
            v = self.momentum * v - self.lr * g
            p += v
        """
        # TODO 1b：实现速度的初始化与更新。
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * g
            params[i] += self.velocity[i]


class Adam:
    """Adam：一阶矩 + 二阶矩 + 偏差修正。"""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: list[Array] | None = None
        self.v: list[Array] | None = None
        self.t = 0

    def step(self, params: list[Array], grads: list[Array]) -> None:
        """原地更新参数。

        第一次调用时初始化：
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        每次调用 self.t += 1，然后对每个参数：
            m = beta1*m + (1-beta1)*g
            v = beta2*v + (1-beta2)*g*g
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            p -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        # TODO 1c：实现 Adam 的状态初始化、时间步与更新。
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, lr: float) -> SGD | Momentum | Adam:
    """按名字创建优化器。已提供。"""
    if name == "sgd":
        return SGD(lr)
    if name == "momentum":
        return Momentum(lr)
    if name == "adam":
        return Adam(lr)
    raise ValueError(f"未知优化器: {name}")


# ---------------------------------------------------------------------------
# TODO 2：训练循环
# ---------------------------------------------------------------------------
def accuracy(model: MLP, X: Array, y: Array) -> float:
    """返回预测准确率。已提供。"""
    return float(np.mean(model.predict(X) == y))


def fit(
    model: MLP,
    X: Array,
    y: Array,
    Xte: Array,
    yte: Array,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    optimizer: str = "adam",
    l2: float = 0.0,
    record_test_loss: bool = False,
    seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """小批量训练。

    Parameters
    ----------
    model : MLP
        第 3 步的模型，会原地更新。
    X, y : Array
        训练数据，X 形状 (N, D)，y 形状 (N,)。
    Xte, yte : Array
        测试数据，每个 epoch 结束后算一次准确率。
    epochs : int
        完整遍历训练集多少遍。
    batch_size : int
        每个 batch 的样本数；最后一个 batch 可能不足。
    lr : float
        学习率，传给优化器。
    optimizer : str
        "sgd" / "momentum" / "adam"。
    l2 : float
        L2 正则系数 λ；0 表示不正则（第 6 步会用到）。
    record_test_loss : bool
        为 True 时每个 epoch 额外计算一次测试损失并记入 history["test_loss"]。
        默认 False，避免为调用方增加不必要的测试集前向/反向。
    seed : int
        每个 epoch 打乱数据用的随机种子。
    verbose : bool
        为 True 时每轮打印一行。

    Returns
    -------
    history : list[dict]
        每个 epoch 一个字典，包含 epoch、train_loss、train_acc、test_acc。
    """
    # TODO 2：按下面的步骤实现：
    #   rng = np.random.default_rng(seed)
    #   opt = make_optimizer(optimizer, lr)
    #   N = len(X)
    #   for epoch in range(1, epochs + 1):
    #       order = rng.permutation(N)
    #       epoch_loss = 0.0
    #       for start in range(0, N, batch_size):
    #           idx = order[start:start + batch_size]
    #           loss, grads = model.loss_and_grads(X[idx], y[idx])
    #           opt.step(model.Ws + model.bs, grads)
    #           epoch_loss += loss * len(idx)
    #       history.append({...})
    #   return history
    rng = np.random.default_rng(seed)
    opt = make_optimizer(optimizer, lr)
    N = len(X)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(N)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            idx = order[start:start + batch_size]
            loss, grads = model.loss_and_grads(X[idx], y[idx], l2=l2)
            opt.step(model.Ws + model.bs, grads)
            epoch_loss += loss * len(idx)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / N,
            "train_acc": accuracy(model, X, y),
            "test_acc": accuracy(model, Xte, yte),
        }
        if record_test_loss:
            test_loss, _ = model.loss_and_grads(Xte, yte)
            row["test_loss"] = float(test_loss)
        history.append(row)
        if verbose:
            print(f"Epoch {epoch:2d}: "
                  f"train_loss={epoch_loss / N:.4f}, "
                  f"train_acc={train_acc:.4f}, "
                  f"test_acc={test_acc:.4f}")

    return history


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(passed)


def self_check() -> bool:
    print("=" * 70)
    print("第 4 步自检")
    print("=" * 70)
    ok = True

    # 1. SGD
    p = np.array([1.0])
    SGD(0.1).step([p], [np.array([2.0])])
    ok &= check("1. SGD", abs(p[0] - 0.8) < 1e-12, f"p = {p[0]:.4f}（应为 0.8）")

    # 2. Momentum：g=1，lr=0.1，mu=0.9
    p = np.array([0.0])
    mom = Momentum(0.1, momentum=0.9)
    mom.step([p], [np.array([1.0])])
    mom.step([p], [np.array([1.0])])
    # 第一步 v=-0.1, p=-0.1；第二步 v=-0.09-0.1=-0.19, p=-0.29
    ok &= check("2. Momentum", abs(p[0] + 0.29) < 1e-12, f"p = {p[0]:.4f}（应为 -0.29）")

    # 3. Adam：g=1，第一步的更新量约为 -lr
    p = np.array([0.0])
    Adam(0.1).step([p], [np.array([1.0])])
    ok &= check("3. Adam", abs(p[0] + 0.1) < 1e-6, f"p = {p[0]:.6f}（应约为 -0.1）")

    # 4. 每个 epoch 的 batch 数：10 个样本，batch=4 -> 3 次更新
    class CountingMLP(MLP):
        def __init__(self) -> None:
            super().__init__([2, 8, 2], seed=0)
            self.calls = 0

        def loss_and_grads(
            self, X: Array, y: Array, l2: float = 0.0
        ) -> tuple[float, list[Array]]:
            self.calls += 1
            return super().loss_and_grads(X, y, l2)

    counter = CountingMLP()
    Xs = np.random.default_rng(0).normal(size=(10, 2))
    ys = np.zeros(10, dtype=int)
    fit(counter, Xs, ys, Xs, ys, epochs=1, batch_size=4, lr=0.01)
    ok &= check("4. 更新次数", counter.calls == 3, f"调用 {counter.calls} 次（应为 3）")

    # 5. 二维小数据训练
    X, y = make_blobs(n_per_class=80, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=1)
    model = MLP([2, 16, 4], seed=0)
    hist = fit(model, Xtr, ytr, Xte, yte, epochs=30, batch_size=32,
               lr=1e-2, optimizer="adam")
    acc = accuracy(model, Xte, yte)
    ok &= check("5. 小数据训练", acc > 0.95, f"测试准确率 = {acc:.3f}")

    # 6. history 结构
    keys_ok = all({"epoch", "train_loss", "train_acc", "test_acc"} <= set(h) for h in hist)
    loss_down = hist[-1]["train_loss"] < hist[0]["train_loss"]
    ok &= check("6. history", len(hist) == 30 and keys_ok and loss_down,
                f"{len(hist)} 轮，损失 {hist[0]['train_loss']:.3f} -> {hist[-1]['train_loss']:.3f}")

    # 7. MNIST 子集
    Xtr_m, ytr_m, Xte_m, yte_m = load_mnist(ROOT / "data")
    Xs, ys = stratified_subsample(Xtr_m, ytr_m, 3000, seed=0)
    Xv, yv = stratified_subsample(Xte_m, yte_m, 1000, seed=1)
    model = MLP([784, 128, 10], seed=0)
    fit(model, Xs, ys, Xv, yv, epochs=10, batch_size=128, lr=1e-3, optimizer="adam")
    acc_m = accuracy(model, Xv, yv)
    ok &= check("7. MNIST 子集", acc_m > 0.88, f"测试准确率 = {acc_m:.4f}")

    print("-" * 70)
    print("全部通过。" if ok else "还有检查未通过。")
    return ok


# ---------------------------------------------------------------------------
# 演示：正式一点的 MNIST 训练
# ---------------------------------------------------------------------------
def demo() -> None:
    Xtr, ytr, Xte, yte = load_mnist(ROOT / "data")
    Xs, ys = stratified_subsample(Xtr, ytr, 10000, seed=0)
    model = MLP([784, 128, 10], seed=0)
    t0 = time.perf_counter()
    hist = fit(model, Xs, ys, Xte, yte, epochs=20, batch_size=128,
               lr=1e-3, optimizer="adam")
    acc = accuracy(model, Xte, yte)
    print(f"\nMNIST 10000 张，20 轮：测试准确率 = {acc:.4f}，"
          f"耗时 {time.perf_counter() - t0:.1f}s")
    print(f"训练损失：{hist[0]['train_loss']:.3f} -> {hist[-1]['train_loss']:.3f}")


if __name__ == "__main__":
    try:
        all_ok = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if all_ok:
        demo()
