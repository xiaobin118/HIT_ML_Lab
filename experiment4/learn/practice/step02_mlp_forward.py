"""第 2 步：MLP 的前向传播（不写反向传播）。

目标：实现
    H = relu(X W1 + b1)
    Z = H W2 + b2
并把每层的 (输入, 仿射输出, 激活输出) 缓存下来，供第 3 步反向传播使用。

规则
----
1. 只改标了 TODO 的地方。
2. 运行 `python step02_mlp_forward.py`，六项自检必须 PASS。
3. 不要打开 `../code/mlp.py` 的 forward 部分。

读代码时如果不知道某个参数/变量的含义，先看：
    learn/docs/00b_名词与类型速查.md
本文件里每个函数都写了类型标注和参数说明。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from common import accuracy  # noqa: F401  （本步暂时用不到，留给下一步）

# 类型别名：让下面的标注短一点。
#   Array       = np.ndarray，任何形状的 NumPy 数组
#   LayerCache  = 一层的缓存 (a_prev, z, a)
#   ForwardCache= 所有层的缓存，每层一个 LayerCache
Array = np.ndarray
LayerCache = tuple[Array, Array, Array]
ForwardCache = list[LayerCache]


# ---------------------------------------------------------------------------
# TODO 1：激活函数
# ---------------------------------------------------------------------------
def relu(z: Array) -> Array:
    """逐元素 max(0, z)。输入什么形状，输出什么形状。

    Parameters
    ----------
    z : Array
        任意形状的浮点数组，形状 (...,)，通常是某一层的仿射输出。

    Returns
    -------
    Array
        与 z 形状相同的数组，负数位置变成 0。
    """
    # TODO 1a：一行。
    return np.maximum(0, z)


def sigmoid(z: Array) -> Array:
    """逐元素 1 / (1 + exp(-z))，要求不溢出。

    Parameters
    ----------
    z : Array
        任意形状的浮点数组。注意：np.empty_like(z) 会沿用 z 的 dtype，
        如果传整数数组，结果会被截断成整数，所以调用时请传浮点数组。

    Returns
    -------
    Array
        与 z 形状相同，每个元素在 (0, 1) 内。

    提示：按 z >= 0 和 z < 0 分两段计算：
        z >= 0：1 / (1 + exp(-z))
        z < 0 ：exp(z) / (1 + exp(z))
    """
    # TODO 1b：写 4~6 行。
    out = np.empty_like(z)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    out[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
    return out


def identity(z: Array) -> Array:
    """线性激活 f(z)=z，用于验证"去掉非线性就退化成线性模型"。已提供。

    Parameters
    ----------
    z : Array
        任意形状的数组。

    Returns
    -------
    Array
        原样返回 z。
    """
    return z


ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": np.tanh,
    "linear": identity,
}


# ---------------------------------------------------------------------------
# TODO 2 / 3 / 4：MLP
# ---------------------------------------------------------------------------
class MLP:
    """全连接网络，输出层不加激活函数（直接给 logits）。"""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "relu",
        init: str = "he",
        seed: int = 0,
    ) -> None:
        """构造网络并初始化所有层的 W、b。

        Parameters
        ----------
        layer_sizes : Sequence[int]
            每层的神经元个数，按"输入 -> 隐藏 -> ... -> 输出"排列。
            例如 [784, 128, 10] 表示：
                输入层 784 维（一张 28x28 图拉平），
                隐藏层 128 个神经元，
                输出层 10 维（10 个数字类别）。
            长度至少为 2；相邻两个数决定一层的 W 形状。

        activation : str
            隐藏层激活函数的名字，必须是 ACTIVATIONS 的键：
                "relu" / "sigmoid" / "tanh" / "linear"。
            输出层永远不加激活，所以这个参数只影响隐藏层。

        init : str
            权重初始化方式：
                "he"     : W ~ N(0, 2/fan_in)，适合 ReLU；
                "xavier" : W ~ N(0, 2/(fan_in+fan_out))，适合 tanh/sigmoid；
                "zeros"  : W 全零，只用来演示"对称性无法打破"。
            偏置 b 永远初始化为 0。

        seed : int
            随机种子。同一个 seed 会得到同一组初始权重，结果可复现。

        Attributes
        ----------
        self.layer_sizes : list[int]
            把传入的 layer_sizes 转成 list 保存。
        self.activation : str
            保存激活函数名，forward 里用它查 ACTIVATIONS。
        self.init : str
            保存初始化方式（本方法用完就不再需要，但方便调试）。
        self.rng : np.random.Generator
            NumPy 随机数生成器；self.rng.normal(...) 抽正态随机数。
        self.Ws : list[Array]
            权重矩阵列表。self.Ws[0] 是第 1 层的 W，形状 (fan_in, fan_out)；
            对 [784, 128, 10] 来说：
                self.Ws[0].shape == (784, 128)
                self.Ws[1].shape == (128, 10)
        self.bs : list[Array]
            偏置向量列表。self.bs[0] 形状 (128,)，self.bs[1] 形状 (10,)。
        """
        self.layer_sizes: list[int] = list(layer_sizes)
        self.activation: str = activation
        self.init: str = init
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.Ws: list[Array] = []
        self.bs: list[Array] = []

        # TODO 2：按层初始化参数。
        #
        # 这一段在做什么：
        #   layer_sizes[:-1] = [784, 128]   每一层的输入维 fan_in
        #   layer_sizes[1:]  = [128, 10]    每一层的输出维 fan_out
        #   zip(...) 把它们按位置配对：
        #       第 1 次 fan_in=784, fan_out=128
        #       第 2 次 fan_in=128, fan_out=10
        #
        # 每一层生成：
        #   W : 形状 (fan_in, fan_out)，元素从 N(0, std^2) 抽；
        #   b : 形状 (fan_out,)，全零。
        #
        # std 的计算：
        #   He      : std = sqrt(2 / fan_in)
        #   Xavier  : std = sqrt(2 / (fan_in + fan_out))
        #   zeros   : std = 0（W 全零）
        #
        # self.rng.normal(loc, scale, size) 的参数：
        #   loc   = 0.0      正态分布的均值
        #   scale = std      正态分布的标准差
        #   size  = (fan_in, fan_out)  输出数组的形状
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # fan_in  = 这一层的输入个数（上一层的神经元数）
            # fan_out = 这一层的输出个数（这一层的神经元数）
            if self.init == "he":
                # He：std = sqrt(2/fan_in)，适合 ReLU
                std = np.sqrt(2.0 / fan_in)
            elif self.init == "xavier":
                # Xavier：std = sqrt(2/(fan_in+fan_out))，适合 tanh/sigmoid
                std = np.sqrt(2.0 / (fan_in + fan_out))
            elif self.init == "zeros":
                # 全零：只用于演示"对称性无法打破"，正常训练不要用
                std = 0.0
            else:
                raise ValueError(f"Unknown init method: {self.init}")
            # 权重形状 (fan_in, fan_out)：行数是输入维，列数是输出维
            W = self.rng.normal(0.0, std, size=(fan_in, fan_out))
            # 偏置形状 (fan_out,)：每个输出神经元一个偏置
            b = np.zeros(fan_out)
            self.Ws.append(W)   # 追加到列表末尾，Ws[i] 就是第 i 层
            self.bs.append(b)

    def forward(
        self, X: Array, cache: bool = False
    ) -> Array | tuple[Array, ForwardCache]:
        """前向传播：把输入 X 一路算到 logits。

        Parameters
        ----------
        X : Array
            输入数据，形状 (N, D)。N 是样本数，D 是特征数。
            对 MNIST，D = 784；对二维数据，D = 2。
        cache : bool
            是否保存反向传播需要的中间量。
            False：只返回 logits，日常预测用这个。
            True ：返回 (logits, layers)，第 3 步算梯度时用。

        Returns
        -------
        Array | tuple[Array, ForwardCache]
            cache=False 时返回 logits，形状 (N, C)，C 是类别数。
            cache=True 时返回 (logits, layers)，layers 是一个列表，
            layers[i] = (a_prev, z, a)：
                a_prev : 第 i 层的输入，形状 (N, fan_in)
                z      : 第 i 层的仿射输出 a_prev @ W + b，形状 (N, fan_out)
                a      : 第 i 层的输出；隐藏层是激活后的值，
                         最后一层就是 logits 本身。

        注意：输出层不做激活，所以 logits 可以是负数，直接交给 softmax。
        """
        # TODO 3：循环每一层。
        #
        # 变量的含义：
        #   a      : 当前层的输入，形状 (N, fan_in)。循环开始时就是 X。
        #   z      : 当前层的仿射输出 a @ W + b，形状 (N, fan_out)。
        #   a_next : 当前层的输出。隐藏层 = 激活(z)；输出层 = z（不激活）。
        #   layers : 缓存列表，每项 (a_prev, z, a)，第 3 步反向传播要用。
        #   n_layers = len(self.Ws)，即层数（不含输入层）。
        #   enumerate(...) 同时给出下标 i 和 (W, b) 对；
        #   i < n_layers - 1 表示"不是最后一层"，只有这些层才做激活。
        a: Array = X
        layers: ForwardCache = []
        n_layers: int = len(self.Ws)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            # z: (N, fan_in) @ (fan_in, fan_out) + (fan_out,) -> (N, fan_out)
            z = a @ W + b
            # 隐藏层做激活；最后一层（i == n_layers-1）保持线性，直接给 logits
            a_next = ACTIVATIONS[self.activation](z) if i < n_layers - 1 else z
            if cache:
                # 存 (这一层的输入, 激活前, 激活后)，第 3 步反向传播要用
                layers.append((a, z, a_next))
            a = a_next
        return (a, layers) if cache else a

    def predict(self, X: Array) -> Array:
        """返回 (N,) 的预测类别。

        Parameters
        ----------
        X : Array
            输入，形状 (N, D)。

        Returns
        -------
        Array
            形状 (N,) 的整数数组，每个元素是 0..C-1 的类别下标。
        """
        # TODO 4a：一行。
        return np.argmax(self.forward(X), axis=1)

    def count_parameters(self) -> int:
        """返回所有 W、b 的元素个数之和。

        Returns
        -------
        int
            例如 [784, 128, 10] 返回 101770：
                784*128 + 128 = 100480
                128*10  + 10  = 1290
        """
        # TODO 4b：提示 sum(p.size for p in ...)。
        return sum(p.size for p in self.Ws + self.bs)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check(name: str, passed: bool, detail: str = "") -> bool:
    """打印一行 PASS/FAIL，并把布尔结果返回给调用者。

    Parameters
    ----------
    name : str
        检查项的名字，例如 "1. 激活函数"。
    passed : bool
        是否通过。
    detail : str
        额外信息，打印在名字后面，通常是实际形状/数值。
    """
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(passed)


def self_check() -> bool:
    print("=" * 66)
    print("第 2 步自检")
    print("=" * 66)
    rng = np.random.default_rng(0)
    ok = True

    # 1. 激活函数（最先检查，对应 TODO 1）
    r = relu(np.array([-1.0, 0.0, 2.0]))
    s = sigmoid(np.array([0.0, 1000.0, -1000.0]))
    ok1 = (np.allclose(r, [0.0, 0.0, 2.0])
           and abs(s[0] - 0.5) < 1e-12
           and s[1] == 1.0 and s[2] == 0.0
           and np.isfinite(s).all())
    ok &= check("1. 激活函数", ok1, f"relu={r}, sigmoid={s}")

    # 2. 参数形状 + forward 形状
    m = MLP([3, 5, 4], seed=0)
    shapes = [(tuple(W.shape), tuple(b.shape)) for W, b in zip(m.Ws, m.bs)]
    expected = [((3, 5), (5,)), ((5, 4), (4,))]
    X = rng.normal(size=(7, 3))
    out = m.forward(X)
    ok &= check("2. 形状", shapes == expected and out.shape == (7, 4),
                f"W/b = {shapes}, logits = {out.shape}")

    # 3. 去掉激活后退化为线性层
    X2 = rng.normal(size=(7, 2))
    lin = MLP([2, 5, 4], activation="linear", seed=0)
    out2 = lin.forward(X2)
    W_eff = lin.Ws[0] @ lin.Ws[1]
    b_eff = lin.bs[0] @ lin.Ws[1] + lin.bs[1]
    expected2 = X2 @ W_eff + b_eff
    err2 = float(np.abs(out2 - expected2).max())
    ok &= check("3. 线性塌缩", err2 < 1e-10,
                f"与 X(W1W2)+(b1W2+b2) 的最大差 = {err2:.2e}")

    # 4. 参数量
    m = MLP([784, 128, 10], seed=0)
    n = m.count_parameters()
    ok &= check("4. 参数量", n == 101770, f"784-128-10 = {n:,}（应为 101,770）")

    # 5. 初始化
    m = MLP([784, 128, 10], init="he", seed=0)
    std = float(m.Ws[0].std())
    want = np.sqrt(2.0 / 784)
    he_ok = abs(std - want) < 0.05 * want
    z = MLP([3, 5, 4], init="zeros", seed=0)
    out_z = z.forward(rng.normal(size=(7, 3)))
    zeros_ok = np.allclose(out_z, 0.0)
    ok &= check("5. 初始化", he_ok and zeros_ok,
                f"He std={std:.4f}（约 {want:.4f}），zeros 输出全零={zeros_ok}")

    # 6. cache
    m = MLP([3, 5, 4], seed=0)
    X = rng.normal(size=(7, 3))
    logits, layers = m.forward(X, cache=True)
    cache_ok = len(layers) == 2
    if cache_ok:
        a0, z0, a1 = layers[0]
        a1b, z1, a2 = layers[1]
        cache_ok = (np.allclose(a0, X)
                    and np.allclose(a1b, a1)
                    and np.allclose(a2, logits)
                    and z0.shape == (7, 5) and a1.shape == (7, 5)
                    and z1.shape == (7, 4) and a2.shape == (7, 4))
    ok &= check("6. cache 结构", cache_ok,
                "layers[i] = (a_prev, z, a)，且相邻层首尾相接")

    print("-" * 66)
    print("全部通过。" if ok else "还有检查未通过。")
    return ok


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
def demo() -> None:
    print("\n本实验会用到的两个结构：")
    for sizes in ([2, 32, 32, 4], [784, 256, 128, 10]):
        m = MLP(sizes, seed=0)
        print(f"  {sizes} -> {m.count_parameters():,} 个参数")

    print("\n随机初始化下，ReLU 隐藏层的非零比例（输入服从标准正态）：")
    m = MLP([784, 256, 128, 10], seed=0)
    X = np.random.default_rng(1).normal(size=(2000, 784))
    _, layers = m.forward(X, cache=True)
    for i, (_, _, a) in enumerate(layers[:-1], start=1):
        print(f"  第 {i} 个隐藏层：非零 {(a > 0).mean():.2%}")
    print("  约 50% 的单元被激活，是 ReLU + He 初始化的典型现象。")


if __name__ == "__main__":
    try:
        all_ok = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if all_ok:
        demo()
