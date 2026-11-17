"""第 1 步：线性分类器（softmax 回归）。

目标：不借助任何机器学习库，实现
    打分 → softmax → 交叉熵 → 对 W,b 的梯度 → 预测
并用数值梯度检查验证自己的梯度。

规则
----
1. 只修改标了 `TODO` 的地方，其他代码不要动。
2. 写完后运行 `python step01_linear.py`，所有自检必须 PASS。
3. 不要打开 `../code/linear.py`。先自己写，卡住 20 分钟以上再来问。

如果对 axis / argmax / arange / 花式索引不熟：
    先读 `learn/docs/00_NumPy函数速查.md`，再跑 `python numpy_warmup.py`。
每个参数的含义和类型见 `learn/docs/00b_名词与类型速查.md`。
"""

from __future__ import annotations

import numpy as np

from common import (
    accuracy,
    finite_difference_error,
    make_blobs,
    make_spiral,
    plot_decision_boundary,
    train_test_split,
)

# 类型别名：Array 就是 np.ndarray，任何形状的 NumPy 数组。
Array = np.ndarray


# ---------------------------------------------------------------------------
# TODO 1：数值稳定的 softmax
# ---------------------------------------------------------------------------
def softmax(logits: Array) -> Array:
    """把 (N, C) 的打分变成每行和为 1 的概率。

    Parameters
    ----------
    logits : Array
        形状 (N, C)。N 是样本数，C 是类别数；每个元素是"未归一化的打分"。

    Returns
    -------
    Array
        形状 (N, C)，每一行加起来等于 1，每个元素在 (0, 1) 内。

    公式：p[n, c] = exp(z[n, c]) / sum_j exp(z[n, j])

    要求：
    - 输入 (N, C)，输出 (N, C)，dtype 保持 float64；
    - 每一行加起来等于 1（误差 < 1e-12）；
    - logits 整体加 1000 不溢出、结果不变（先减去每行最大值）。
    """
    # TODO 1：写 3~5 行。提示：z = logits - logits.max(axis=1, keepdims=True)
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    softmax_probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    return softmax_probs


# ---------------------------------------------------------------------------
# TODO 2：交叉熵损失 + 对 logits 的梯度
# ---------------------------------------------------------------------------
def cross_entropy(logits: Array, y: Array) -> tuple[float, Array]:
    """返回 (loss, dlogits)。

    Parameters
    ----------
    logits : Array
        形状 (N, C) 的打分。
    y : Array
        形状 (N,) 的整数数组，每个元素是 0..C-1 的真实类别下标。

    Returns
    -------
    loss : float
        平均交叉熵，一个 Python 标量（不是数组）。
    dlogits : Array
        形状 (N, C)，损失对 logits 的梯度 (P - onehot(y)) / N。

    loss = -1/N * sum_n log softmax(logits)[n, y[n]]
    dlogits = (softmax(logits) - onehot(y)) / N      # 形状 (N, C)

    要求：
    - loss 是一个 Python float；
    - dlogits 形状与 logits 相同；
    - 当 logits 全为 0、类别数 C=4 时，loss 应为 ln 4（均匀分布）。
    """
    # TODO 2：可以先调用上面的 softmax，也可以自己用 logsumexp 写。
    #        注意：不要直接 log(0)。先减最大值再算 log-sum-exp。
    # N: 样本数；C: 类别数（C 在后面算 dlogits 时用形状 (N,C) 体现）
    N, C = logits.shape
    m = logits.max(axis=1, keepdims=True)   # (N,1) 每行最大值
    e = np.exp(logits - m)                  # (N,C) 每行最大值变成 1
    s = e.sum(axis=1, keepdims=True)        # (N,1) 每行的分母
    P = e / s                               # (N,C) softmax 概率

    logsumexp = np.log(s[:, 0]) + m[:, 0]   # (N,)，注意 [:, 0]
    loss = float(np.mean(logsumexp - logits[np.arange(N), y]))  # 没有负号

    dlogits = P.copy()
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    return loss, dlogits


# ---------------------------------------------------------------------------
# TODO 3 / TODO 4：线性分类器
# ---------------------------------------------------------------------------
class LinearClassifier:
    """logits = X @ W + b，Y 用 one-hot 或类别下标都可以（本文件用下标）。"""

    def __init__(self, n_features: int, n_classes: int, seed: int = 0) -> None:
        """构造线性分类器。

        Parameters
        ----------
        n_features : int
            输入特征维数 D。二维数据 D=2，MNIST 拉平后 D=784。
        n_classes : int
            类别数 C。二维四类 C=4，MNIST C=10。
        seed : int
            随机种子。线性模型用全零初始化，本步暂时用不到；
            保留它是为了和后面 MLP 的接口一致。

        Attributes
        ----------
        self.W : Array
            权重矩阵，形状 (n_features, n_classes) = (D, C)。
        self.b : Array
            偏置向量，形状 (n_classes,) = (C,)。
        """
        # 线性模型用 0 初始化即可（没有隐藏层，不存在对称性问题）。
        self.W: Array = np.zeros((n_features, n_classes))
        self.b: Array = np.zeros(n_classes)

    def scores(self, X: Array) -> Array:
        """返回 (N, C) 的 logits。这一行已经写好，不用改。

        Parameters
        ----------
        X : Array
            输入，形状 (N, D)。

        Returns
        -------
        Array
            形状 (N, C) 的打分矩阵 X @ W + b。
        """
        return X @ self.W + self.b

    def loss_and_grads(self, X: Array, y: Array, l2: float = 0.0) -> tuple[float, Array, Array]:
        """返回 (loss, dW, db)。

        Parameters
        ----------
        X : Array
            输入，形状 (N, D)。
        y : Array
            真实类别，形状 (N,)，整数 0..C-1。
        l2 : float
            L2 正则系数 λ。0 表示不正则；越大权重被压得越小。

        Returns
        -------
        loss : float
            平均交叉熵 + 0.5 * l2 * ||W||_F^2。
        dW : Array
            形状 (D, C)，对 W 的梯度。
        db : Array
            形状 (C,)，对 b 的梯度。

        L = 平均交叉熵 + 0.5 * l2 * ||W||_F^2
        dW = X^T @ dlogits + l2 * W        # 注意 dlogits 已经除过 N
        db = sum_n dlogits[n]              # 对第 0 维求和，形状 (C,)

        要求：
        - dW 形状 (D, C)，db 形状 (C,)；
        - L2 只惩罚 W，不惩罚 b；
        - l2=0 时也要能正常工作。
        """
        # TODO 3：调用 cross_entropy，再算 dW、db。
        #         形状提示：X (N,D), W (D,C), dlogits (N,C) -> X.T @ dlogits = (D,C)
        loss, dlogits = cross_entropy(self.scores(X), y)   # dlogits: (N,C)
        loss += 0.5 * l2 * np.sum(self.W ** 2)            # L2 只加在 loss 上
        dW = X.T @ dlogits + l2 * self.W                  # (D,N)@(N,C) -> (D,C)
        db = np.sum(dlogits, axis=0)                      # 对 N 求和 -> (C,)
        return loss, dW, db

    def predict(self, X: Array) -> Array:
        """返回 (N,) 的类别下标：每个样本取 logits 最大的那一类。

        Parameters
        ----------
        X : Array
            输入，形状 (N, D)。

        Returns
        -------
        Array
            形状 (N,) 的整数数组，元素在 0..C-1。
        """
        # argmax: Returns the indices of the maximum values along an axis.
        return np.argmax(self.scores(X), axis=1)


# ---------------------------------------------------------------------------
# 以下代码是提供的训练循环，不需要你写
# ---------------------------------------------------------------------------
def train_full_batch(
    model: "LinearClassifier",
    X: Array,
    y: Array,
    steps: int = 800,
    lr: float = 0.5,
    l2: float = 0.0,
    verbose_every: int = 0,
) -> float | None:
    """最简单的全批量梯度下降：每次用全部训练样本算一次梯度。

    Parameters
    ----------
    model : LinearClassifier
        要训练的模型。会被原地修改（W、b 变小）。
    X, y : Array
        训练数据，X 形状 (N, D)，y 形状 (N,)。
    steps : int
        参数更新次数。
    lr : float
        学习率，每一步走多远。
    l2 : float
        L2 正则系数，透传给 loss_and_grads。
    verbose_every : int
        大于 0 时，每这么多步打印一次损失；0 表示不打印。

    Returns
    -------
    float | None
        最后一次更新前的损失；steps=0 时返回 None。
    """
    loss: float | None = None
    for step in range(1, steps + 1):
        loss, dW, db = model.loss_and_grads(X, y, l2=l2)
        model.W -= lr * dW
        model.b -= lr * db
        if verbose_every and step % verbose_every == 0:
            print(f"  step {step:4d}  loss = {loss:.4f}")
    return loss


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def self_check() -> bool:
    print("=" * 62)
    print("第 1 步自检")
    print("=" * 62)
    rng = np.random.default_rng(0)
    ok = True

    # 检查 1：softmax 行和为 1
    z = rng.normal(size=(5, 4))
    p = softmax(z)
    # np.abs 逐元素取绝对值；sum(axis=1) 每行求和；max() 取整体最大误差
    row_err = float(np.abs(p.sum(axis=1) - 1).max())
    passed = p.shape == z.shape and row_err < 1e-12
    ok &= passed
    print(f"[1/5] softmax 行和误差 = {row_err:.2e}   {'PASS' if passed else 'FAIL'}")

    # 检查 2：整体平移不变、不溢出
    p_shift = softmax(z + 1000.0)
    # np.isfinite(...).all()：检查是否全是有限值（没有 inf/nan）
    shift_err = float(np.abs(p_shift - p).max())
    passed = np.isfinite(p_shift).all() and shift_err < 1e-12
    ok &= passed
    print(f"[2/5] softmax(z+1000) 误差 = {shift_err:.2e}   {'PASS' if passed else 'FAIL'}")

    # 检查 3：均匀 logits 的交叉熵 = ln C
    loss_uniform, _ = cross_entropy(np.zeros((3, 4)), np.array([0, 1, 2]))
    passed = abs(loss_uniform - np.log(4)) < 1e-12
    ok &= passed
    print(f"[3/5] 均匀分布交叉熵 = {loss_uniform:.6f}（应为 {np.log(4):.6f}）"
          f"   {'PASS' if passed else 'FAIL'}")

    # 检查 4：数值梯度
    model = LinearClassifier(2, 4, seed=1)
    model.W = rng.normal(size=(2, 4))
    model.b = rng.normal(size=4)
    X = rng.normal(size=(16, 2))
    y = rng.integers(0, 4, size=16)
    loss, dW, db = model.loss_and_grads(X, y, l2=0.01)
    err = finite_difference_error(
        lambda: model.loss_and_grads(X, y, l2=0.01)[0],
        [model.W, model.b],
        [dW, db],
    )
    passed = err < 1e-6
    ok &= passed
    print(f"[4/5] 梯度检查最大误差 = {err:.2e}（应 < 1e-6）   {'PASS' if passed else 'FAIL'}")

    # 检查 5：极端 logits 下损失必须是有限值（不能 log(0)）
    # 真实类别的概率会因为 exp(-1000) 下溢成 0，直接 log(0) 会得到 -inf。
    big_logits = np.array([[0.0, 1000.0, 0.0, 0.0]])
    big_y = np.array([0])
    loss_big, _ = cross_entropy(big_logits, big_y)
    passed = np.isfinite(loss_big) and loss_big > 100
    ok &= passed
    print(f"[5/5] 极端 logits 的损失 = {loss_big}（应约为 1000，不能是 inf）"
          f"   {'PASS' if passed else 'FAIL'}")

    print("-" * 62)
    if ok:
        print("全部通过。继续看下面的训练演示。")
    else:
        print("还有检查未通过。先修到全部 PASS，再看训练结果。")
    return ok


# ---------------------------------------------------------------------------
# 训练演示：同一套代码，在两种数据上训练
# ---------------------------------------------------------------------------
def demo() -> None:
    for name, maker, lr, steps in [
        ("四团高斯（线性可分）", make_blobs, 0.5, 800),
        ("四臂螺旋（线性不可分）", make_spiral, 0.5, 800),
    ]:
        X, y = maker(seed=0)
        Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=0)
        model = LinearClassifier(2, 4, seed=0)
        first_loss, _ = cross_entropy(model.scores(Xtr), ytr)
        last_loss = train_full_batch(model, Xtr, ytr, steps=steps, lr=lr)
        acc_train = accuracy(model.predict, Xtr, ytr)
        acc_test = accuracy(model.predict, Xte, yte)

        print(f"\n=== {name} ===")
        print(f"  训练损失：{first_loss:.4f} -> {last_loss:.4f}")
        print(f"  训练准确率：{acc_train:.4f}")
        print(f"  测试准确率：{acc_test:.4f}")
        if "螺旋" in name:
            plot_decision_boundary(model.predict, Xtr, ytr, "step01_spiral.png",
                                   title=f"线性分类器在螺旋上：{acc_test:.3f}")
        else:
            plot_decision_boundary(model.predict, Xtr, ytr, "step01_blobs.png",
                                   title=f"线性分类器在四团高斯上：{acc_test:.3f}")


if __name__ == "__main__":
    try:
        ok = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if ok:
        demo()
