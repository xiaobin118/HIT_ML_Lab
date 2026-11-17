"""第 3 步：把 MNIST 装进 DataLoader。

目标：从原始的 .gz 字节文件出发，自己解析 IDX 格式 → 得到 Tensor → 分层划分训练/验证集
→ 包装成 Dataset/DataLoader → 按 batch 取数据。这是训练循环之前的"数据管道"。

规则
----
1. 只修改标了 `TODO` 的地方，其它代码不要动。
2. 写完后运行 `python step03_data.py`，全部 7 项自检必须 PASS。
3. 不要打开 `../code/data.py`。先自己写，卡住 20 分钟以上再来问。
4. 讲义在 `../docs/03_MNIST数据管道.md`。

数据已经放在 `experiment5/data/mnist/` 下（4 个 .gz 文件）。
"""

from __future__ import annotations

import gzip
import struct
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 本文件在 experiment5/learn/practice/ 下，数据在 experiment5/data/mnist/
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mnist"

CAPTION = "MNIST 数据管道"


# ===========================================================================
# TODO 1：解析 IDX 文件
# ===========================================================================
def read_idx_images(path: str | Path) -> np.ndarray:
    """读取 IDX3 图像文件，返回 uint8 数组，形状 (N, H, W)。

    文件格式（大端序）：
        16 字节头：magic(4) + n(4) + rows(4) + cols(4)
        然后是 n*rows*cols 个 uint8 像素
    magic 必须等于 0x00000803，否则抛 ValueError。

    提示：gzip.open(path, "rb")；struct.unpack(">IIII", f.read(16))；
          np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)。
    """
    # TODO 1：写 8~10 行
    image = gzip.open(path, "rb")
    magic, n, rows, cols = struct.unpack(">IIII", image.read(16))
    if magic != 0x00000803:
        raise ValueError(f"magic 不对，应为 0x00000803，实际 {magic:#010x}")
    buf = image.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path: str | Path) -> np.ndarray:
    """读取 IDX1 标签文件，返回 int64 数组，形状 (N,)。

    文件格式（大端序）：
        8 字节头：magic(4) + n(4)
        然后是 n 个 uint8 标签（0..9）
    magic 必须等于 0x00000801，否则抛 ValueError。
    """
    # TODO 1：写 6~8 行
    label = gzip.open(path, "rb")
    magic, n = struct.unpack(">II", label.read(8))
    if magic != 0x00000801:
        raise ValueError(f"magic 不对，应为 0x00000801，实际 {magic:#010x}")
    buf = label.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


# ===========================================================================
# TODO 2：加载 MNIST
# ===========================================================================
def load_mnist(root: str | Path = DATA_DIR) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载 MNIST，返回 (X_train, y_train, X_test, y_test)。

    要求
    ----
    - X：形状 (N, 1, 28, 28)、dtype float32、数值在 [0,1]（除以 255，并补通道维）；
    - y：形状 (N,)、dtype int64；
    - 四个文件名：train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、
      t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz。

    提示：read_idx_images / read_idx_labels + torch.from_numpy + .float() / 255.0 + .unsqueeze(1)。
    """
    # TODO 2：写 6~10 行
    X_train = read_idx_images(root / "train-images-idx3-ubyte.gz")
    y_train = read_idx_labels(root / "train-labels-idx1-ubyte.gz")
    X_test = read_idx_images(root / "t10k-images-idx3-ubyte.gz")
    y_test = read_idx_labels(root / "t10k-labels-idx1-ubyte.gz")
    X_train = torch.from_numpy(X_train).float() / 255.0
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float() / 255.0
    y_test = torch.from_numpy(y_test).long()
    return X_train.unsqueeze(1), y_train, X_test.unsqueeze(1), y_test



# ===========================================================================
# TODO 3：统计每类样本数
# ===========================================================================
def class_counts(y: torch.Tensor, n_classes: int = 10) -> list[int]:
    """返回长度为 n_classes 的列表，第 c 个元素是标签 c 的样本数。

    提示：torch.bincount(y, minlength=n_classes)；记得转成 Python int 列表。
    """
    # TODO 3：写 1~2 行
    return torch.bincount(y, minlength=n_classes).tolist()


# ===========================================================================
# TODO 4：分层划分训练/验证集
# ===========================================================================
def stratified_split(
    y: torch.Tensor, n_val: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """按类别分层地把数据集切成 (训练索引, 验证索引)。

    要求
    ----
    1. 每个类别分别打乱（用显式 seed 的 torch.Generator），
       取 round(该类样本数 * n_val / 总样本数) 个作为验证集，其余为训练集；
    2. 返回两个 int64 索引张量，二者不相交，合起来覆盖所有样本；
    3. 同一个 seed 两次调用结果完全一样。

    提示：torch.unique(y) 遍历类别；torch.nonzero(y == c).flatten() 取该类索引；
          torch.randperm(len(idx), generator=g) 打乱；最后 torch.cat 合并。
    """
    # TODO 4：写 12~18 行
    g = torch.Generator().manual_seed(seed)
    train_idx_list = []
    val_idx_list = []
    for c in torch.unique(y):
        idx = torch.nonzero(y == c).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_val_c = round(len(idx) * n_val / len(y))
        val_idx_list.append(idx[:n_val_c])
        train_idx_list.append(idx[n_val_c:])
    train_idx = torch.cat(train_idx_list)
    val_idx = torch.cat(val_idx_list)
    return train_idx, val_idx


# ===========================================================================
# TODO 5：构造 DataLoader
# ===========================================================================
def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """把 X、y 包装成 DataLoader。

    要求
    ----
    1. 用 TensorDataset(X, y) 包装；
    2. batch_size 默认 64；
    3. shuffle=True 时用显式 seed 的 Generator 保证可复现（同一个 seed 两次得到的顺序相同）；
    4. shuffle=False 时保持原顺序。

    提示：g = torch.Generator().manual_seed(seed)；
          DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)。
    """
    # TODO 5：写 4~6 行
    ds = TensorDataset(X, y)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


# ===========================================================================
# TODO 6：计算归一化统计量
# ===========================================================================
def normalization_stats(X: torch.Tensor) -> tuple[float, float]:
    """返回 X 的 (均值, 标准差)，都是 Python float。

    注意：PyTorch 的 `X.std()` 默认是**样本标准差**（除以 n-1），而 NumPy 的
    `np.std` 默认是**总体标准差**（除以 n）。这里要求总体标准差，保持和 NumPy 一致：
    用 `X.std(unbiased=False)`。在 600 万像素的数据集上两者差不到 1e-7，
    但在 [0,1] 这种小例子上差很多，测试会抓到。
    """
    # TODO 6：写 1~2 行
    return float(X.mean()), float(X.std(unbiased=False))


# ===========================================================================
# TODO 7：归一化
# ===========================================================================
def normalize(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """返回 (X - mean) / std，形状不变、dtype 保持 float32。

    归一化后整个数据集的均值应约等于 0、标准差约等于 1（用 normalization_stats 验证）。
    """
    # TODO 7：写 1 行
    return (X - mean) / std


# ===========================================================================
# 自检代码：不要修改下面的内容
# ===========================================================================


CHECKS: list[tuple[str, callable]] = []
_CACHE: dict = {}


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


def _data():
    """只调用一次 load_mnist，给多个测试复用。"""
    if "data" not in _CACHE:
        _CACHE["data"] = load_mnist()
    return _CACHE["data"]


@check("TODO 1  read_idx_images / read_idx_labels：形状、dtype、magic 校验")
def _test_read_idx() -> None:
    X = read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    assert isinstance(X, np.ndarray), f"应返回 np.ndarray，实际 {type(X)}"
    assert X.shape == (60000, 28, 28), f"训练图像形状应为 (60000,28,28)，实际 {X.shape}"
    assert X.dtype == np.uint8, f"dtype 应为 uint8，实际 {X.dtype}"
    assert int(X.min()) == 0 and int(X.max()) == 255, "像素值应在 0~255"

    y = read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    assert y.shape == (60000,), f"标签形状应为 (60000,)，实际 {y.shape}"
    assert y.dtype == np.int64, f"标签 dtype 应为 int64，实际 {y.dtype}"
    assert set(np.unique(y).tolist()) == set(range(10)), "标签应包含 0~9 共 10 类"

    # 用测试集文件验证 n 的解析也正确
    Xte = read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    assert Xte.shape == (10000, 28, 28), f"测试图像形状应为 (10000,28,28)，实际 {Xte.shape}"


@check("TODO 2  load_mnist：形状 (N,1,28,28) float32、标签 int64、值域 [0,1]")
def _test_load_mnist() -> None:
    Xtr, ytr, Xte, yte = _data()
    for name, t, shape in [("Xtr", Xtr, (60000, 1, 28, 28)), ("ytr", ytr, (60000,)),
                           ("Xte", Xte, (10000, 1, 28, 28)), ("yte", yte, (10000,))]:
        assert tuple(t.shape) == shape, f"{name} 形状应为 {shape}，实际 {tuple(t.shape)}"
    assert Xtr.dtype == torch.float32 and Xte.dtype == torch.float32, "X 应为 float32"
    assert ytr.dtype == torch.int64 and yte.dtype == torch.int64, "y 应为 int64"
    assert float(Xtr.min()) >= 0.0 and float(Xtr.max()) <= 1.0, "像素应缩放到 [0,1]"
    # 与原始文件对照：第 0 张图除以 255 后应一致
    raw0 = read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz")[0].astype(np.float32) / 255.0
    assert torch.allclose(Xtr[0, 0], torch.from_numpy(raw0), atol=1e-6), "第 0 张图与原始像素不一致"


@check("TODO 3  class_counts：每类计数正确")
def _test_class_counts() -> None:
    y = torch.tensor([0, 1, 1, 2, 2, 2, 9])
    got = class_counts(y)
    assert isinstance(got, list) and len(got) == 10, f"应返回长度 10 的 list，实际 {got!r}"
    assert got == [1, 2, 3, 0, 0, 0, 0, 0, 0, 1], f"计数不对：{got}"


@check("TODO 4  stratified_split：不相交、覆盖全部、每类比例正确、可复现")
def _test_stratified() -> None:
    _, ytr, _, _ = _data()
    train_idx, val_idx = stratified_split(ytr, 6000, seed=0)
    assert train_idx.dtype == torch.int64 and val_idx.dtype == torch.int64, "索引应为 int64"
    all_idx = torch.cat([train_idx, val_idx])
    assert len(all_idx) == 60000, f"训练+验证应覆盖 60000 个样本，实际 {len(all_idx)}"
    assert len(torch.unique(all_idx)) == 60000, "训练/验证索引有重叠"
    n_val = len(val_idx)
    assert 5900 <= n_val <= 6100, f"验证集大小应接近 6000（四舍五入后可能是 5999），实际 {n_val}"

    counts = torch.bincount(ytr, minlength=10).float()
    for c in range(10):
        c_val = int((ytr[val_idx] == c).sum())
        expected = float(counts[c] * n_val / 60000)
        assert abs(c_val - expected) <= 1.5, f"类别 {c} 的验证样本应为约 {expected:.1f}，实际 {c_val}"

    train2, val2 = stratified_split(ytr, 6000, seed=0)
    assert torch.equal(train_idx, train2) and torch.equal(val_idx, val2), "同一个 seed 两次划分应完全一致"
    _, val3 = stratified_split(ytr, 6000, seed=1)
    assert not torch.equal(val_idx, val3), "不同 seed 的划分应该不同"


@check("TODO 5  make_loader：batch 形状、顺序、可复现的 shuffle")
def _test_make_loader() -> None:
    X = torch.zeros(100, 1, 28, 28)
    y = torch.arange(100)
    loader = make_loader(X, y, batch_size=16, shuffle=False)
    batches = [batch for batch in loader]
    assert len(batches) == 7, f"100 个样本、batch=16 应有 7 个 batch，实际 {len(batches)}"
    xb, yb = batches[0]
    assert tuple(xb.shape) == (16, 1, 28, 28), f"第一个 batch 形状应为 (16,1,28,28)，实际 {tuple(xb.shape)}"
    assert tuple(batches[-1][1].shape) == (4,), f"最后一个 batch 应有 4 个样本，实际 {tuple(batches[-1][1].shape)}"
    assert torch.equal(torch.cat([b[1] for b in batches]), y), "shuffle=False 时应保持原顺序"

    order1 = torch.cat([b[1] for b in make_loader(X, y, batch_size=16, shuffle=True, seed=0)])
    order2 = torch.cat([b[1] for b in make_loader(X, y, batch_size=16, shuffle=True, seed=0)])
    order3 = torch.cat([b[1] for b in make_loader(X, y, batch_size=16, shuffle=True, seed=1)])
    assert torch.equal(order1, order2), "同一个 seed 的 shuffle 顺序应一致"
    assert not torch.equal(order1, order3), "不同 seed 的 shuffle 顺序应该不同"


@check("TODO 6  normalization_stats：真实 MNIST 均值≈0.1307、标准差≈0.3081")
def _test_norm_stats() -> None:
    Xtr, _, _, _ = _data()
    mean, std = normalization_stats(Xtr)
    assert isinstance(mean, float) and isinstance(std, float), "应返回两个 Python float"
    assert abs(mean - 0.1307) < 0.01, f"MNIST 均值应约 0.1307，实际 {mean:.4f}"
    assert abs(std - 0.3081) < 0.01, f"MNIST 标准差应约 0.3081，实际 {std:.4f}"
    m2, s2 = normalization_stats(torch.tensor([0.0, 1.0]))
    assert abs(m2 - 0.5) < 1e-6 and abs(s2 - 0.5) < 1e-6, f"[0,1] 的均值/标准差应为 0.5/0.5，实际 {m2}/{s2}"


@check("TODO 7  normalize：归一化后均值≈0、标准差≈1")
def _test_normalize() -> None:
    Xtr, _, _, _ = _data()
    mean, std = normalization_stats(Xtr)
    z = normalize(Xtr, mean, std)
    assert tuple(z.shape) == tuple(Xtr.shape), f"形状应保持 {tuple(Xtr.shape)}，实际 {tuple(z.shape)}"
    assert z.dtype == torch.float32, f"dtype 应保持 float32，实际 {z.dtype}"
    assert abs(float(z.mean())) < 1e-4, f"归一化后均值应约 0，实际 {float(z.mean()):.2e}"
    assert abs(float(z.std()) - 1.0) < 1e-4, f"归一化后标准差应约 1，实际 {float(z.std()):.4f}"

    toy = normalize(torch.tensor([0.0, 1.0]), 0.5, 0.5)
    assert torch.allclose(toy, torch.tensor([-1.0, 1.0])), f"[0,1] 用均值 0.5/std 0.5 归一化应为 [-1,1]，实际 {toy}"


def main() -> None:
    print("=" * 74)
    print("第 3 步自检：MNIST 数据管道")
    print("=" * 74)
    n_pass = 0
    failures = []
    for name, fn in CHECKS:
        try:
            fn()
        except NotImplementedError as e:
            print(f"❌ {name}\n     {e}")
            failures.append(name)
        except AssertionError as e:
            print(f"❌ {name}\n     {e}")
            failures.append(name)
        except Exception:
            print(f"❌ {name}\n     {traceback.format_exc(limit=2).strip()}")
            failures.append(name)
        else:
            print(f"✅ {name}")
            n_pass += 1
    print("-" * 74)
    print(f"结果：{n_pass}/{len(CHECKS)} PASS")
    if failures:
        print("未通过：")
        for name in failures:
            print("  -", name)
        print("\n按上面的报错逐个修改 TODO。改完重新运行本脚本。")
        raise SystemExit(1)
    print("\n全部通过！把输出、load_mnist 的代码、思考题答案发给老师，")
    print("就可以进入第 4 步：正式的训练循环。")


if __name__ == "__main__":
    main()
