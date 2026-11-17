"""数据模块：MNIST / Fashion-MNIST 的 IDX 解析、训练/验证/测试划分、数据增强。

本项目把原始 IDX 文件放在 data/mnist 与 data/fashion 下（可离线复现），
不依赖 torchvision.datasets 的联网下载；数据增强使用 torchvision.transforms.v2，
它可以直接作用在 torch.Tensor 上，也就能在 GPU 上对整个 batch 做增广。

数据集约定
----------
- MNIST：60,000 张训练图 + 10,000 张测试图，28×28 灰度，10 类；
  实验时从训练集里分层留出 6,000 张做验证集，最终测试用官方 10,000 张测试图。
- Fashion-MNIST：同样的尺寸与规模，内容换成 10 类服装，用来验证结论不只在
  "干净的手写数字"上成立。
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 数据集描述：每类的中文名，用于画图和报告
MNIST_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
FASHION_CLASSES = [
    "T恤", "裤子", "套头衫", "连衣裙", "外套",
    "凉鞋", "衬衫", "运动鞋", "包", "短靴",
]


# ---------------------------------------------------------------------------
# IDX 文件解析（与实验四相同的数据格式）
# ---------------------------------------------------------------------------


def _read_idx_images(path: Path) -> np.ndarray:
    """读取 IDX3 图像文件，返回 uint8 数组 (N, H, W)。"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"{path} 不是 IDX3 图像文件（magic={magic:#x}）")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """读取 IDX1 标签文件，返回 int64 数组 (N,)。"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"{path} 不是 IDX1 标签文件（magic={magic:#x}）")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_raw_dataset(name: str = "mnist"):
    """加载原始数据集，返回 (X_train, y_train, X_test, y_test)。

    X 为 float32 张量，形状 (N, 1, 28, 28)，像素值缩放到 [0, 1]；
    y 为 int64 张量。整个过程只做 /255 归一化，不额外做均值方差标准化，
    这样输入保持"图像"的稀疏结构，卷积核的可视化也更直观。
    """
    folder = DATA_ROOT / name
    files = {
        "Xtr": folder / "train-images-idx3-ubyte.gz",
        "ytr": folder / "train-labels-idx1-ubyte.gz",
        "Xte": folder / "t10k-images-idx3-ubyte.gz",
        "yte": folder / "t10k-labels-idx1-ubyte.gz",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少数据文件：" + ", ".join(missing) + "\n"
            "请运行 bash data/download_mnist.sh 或 bash data/download_fashion_mnist.sh"
        )

    Xtr = torch.from_numpy(_read_idx_images(files["Xtr"]).astype(np.float32) / 255.0).unsqueeze(1)
    ytr = torch.from_numpy(_read_idx_labels(files["ytr"]))
    Xte = torch.from_numpy(_read_idx_images(files["Xte"]).astype(np.float32) / 255.0).unsqueeze(1)
    yte = torch.from_numpy(_read_idx_labels(files["yte"]))
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# 训练 / 验证 / 测试划分
# ---------------------------------------------------------------------------


def stratified_split_indices(
    y: torch.Tensor, n_val: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """按类别分层地把训练集切成 (训练, 验证) 两部分。

    每一类都按相同比例抽取验证样本，保证验证集的类别分布与训练集一致。
    返回两个索引张量。
    """
    if n_val >= len(y):
        raise ValueError(f"验证集大小 {n_val} 不能超过训练集 {len(y)}")
    g = torch.Generator().manual_seed(seed)
    val_idx, train_idx = [], []
    for c in torch.unique(y).tolist():
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        k = int(round(len(idx) * n_val / len(y)))
        val_idx.append(idx[:k])
        train_idx.append(idx[k:])
    return torch.cat(train_idx), torch.cat(val_idx)


def stratified_subset_indices(
    y: torch.Tensor, n: int, seed: int = 0
) -> torch.Tensor:
    """从训练集中按类别等比例抽取 n 个样本的索引（用于数据量实验）。"""
    if n >= len(y):
        return torch.arange(len(y))
    g = torch.Generator().manual_seed(seed)
    picked = []
    classes = torch.unique(y).tolist()
    per_class = n / len(classes)
    for c in classes:
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        picked.append(idx[: int(round(per_class))])
    out = torch.cat(picked)
    # 补足或截断到恰好 n 个
    n_extra = n - len(out)
    if n_extra > 0:
        all_idx = torch.randperm(len(y), generator=g)
        used = set(out.tolist())
        extra = torch.tensor([i for i in all_idx.tolist() if i not in used][:n_extra])
        out = torch.cat([out, extra])
    elif n_extra < 0:
        out = out[torch.randperm(len(out), generator=g)][:n]
    return out


class TensorDataset(Dataset):
    """把 (X, y) 张量包装成 Dataset；可选地在 __getitem__ 上施加 v2 增强。"""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None):
        assert len(X) == len(y)
        self.X, self.y = X, y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    transform=None,
    num_workers: int = 4,
    seed: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """构造 DataLoader；训练集打乱顺序，验证/测试集保持固定顺序。"""
    ds = TensorDataset(X, y, transform)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        generator=gen if shuffle else None,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------

# 所有增广都在 [0,1] 的 float 张量上做，v2 的旋转/平移默认用 0 填充，
# 对 MNIST 的黑底白字来说正好等价于"背景不引入假笔画"。
_AUG_SPECS = {
    "none": [],
    "rotate10": [v2.RandomAffine(degrees=10, translate=(0.1, 0.1))],
    "rotate20": [v2.RandomAffine(degrees=20, translate=(0.1, 0.1))],
    "rotate20_erase": [
        v2.RandomAffine(degrees=20, translate=(0.1, 0.1)),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0.0),
    ],
}


def make_train_transform(spec: str):
    """返回训练阶段的增广变换；spec 为 _AUG_SPECS 的键。"""
    if spec not in _AUG_SPECS:
        raise ValueError(f"未知增广配置 {spec!r}，可选：{list(_AUG_SPECS)}")
    ops = _AUG_SPECS[spec]
    if not ops:
        return None
    return v2.Compose(ops)


def rotate_batch(X: torch.Tensor, angle: float) -> torch.Tensor:
    """把一批图像绕中心旋转 angle 度（用于测试模型对旋转的鲁棒性）。

    v2.functional.rotate 支持 batch 维，因此可以一次处理整个测试集。
    """
    return v2.functional.rotate(X, angle, interpolation=v2.InterpolationMode.BILINEAR)


def describe(make: str = "mnist") -> dict:
    """返回数据集的元信息，写进 results.json 供报告使用。"""
    Xtr, ytr, Xte, yte = load_raw_dataset(make)
    return {
        "name": "MNIST" if make == "mnist" else "Fashion-MNIST",
        "n_train_raw": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "shape": list(Xtr.shape[1:]),
        "n_classes": int(len(torch.unique(ytr))),
        "class_counts_train": torch.bincount(ytr, minlength=10).tolist(),
    }


if __name__ == "__main__":
    # 直接运行本文件时做一次自检：形状、范围、类别分布
    for name in ("mnist", "fashion"):
        Xtr, ytr, Xte, yte = load_raw_dataset(name)
        print(f"{name}: train {tuple(Xtr.shape)} {Xtr.dtype}, test {tuple(Xte.shape)}")
        print(f"  像素范围 [{Xtr.min().item():.3f}, {Xtr.max().item():.3f}]"
              f"  类别计数 {torch.bincount(ytr, minlength=10).tolist()}")
    Xtr, ytr, _, _ = load_raw_dataset("mnist")
    tr, va = stratified_split_indices(ytr, 6000, seed=0)
    print(f"分层划分: train {len(tr)}, val {len(va)}, "
          f"val 类别计数 {torch.bincount(ytr[va], minlength=10).tolist()}")
    for n in (600, 2000, 6000, 20000):
        idx = stratified_subset_indices(ytr, n, seed=0)
        print(f"子集 n={n}: 实际 {len(idx)}, 类别计数 {torch.bincount(ytr[idx], minlength=10).tolist()}")
