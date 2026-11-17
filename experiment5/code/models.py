"""模型模块：LeNet-5（可换激活函数 / 池化 / BN / Dropout）、小型 VGG 与 MLP 对照。

所有模型都接受 (N, 1, 28, 28) 的输入、输出 (N, 10) 的 logits，
接口统一，方便用同一套训练引擎做控制变量实验。
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 激活函数表
# ---------------------------------------------------------------------------

# key -> (构造函数, 报告里用的公式/说明)
ACTIVATIONS = {
    "relu": (lambda: nn.ReLU(inplace=False), "max(0, x)"),
    "sigmoid": (lambda: nn.Sigmoid(), "1 / (1 + e^{-x})"),
    "tanh": (lambda: nn.Tanh(), "tanh(x)"),
    "leaky_relu": (lambda: nn.LeakyReLU(negative_slope=0.1, inplace=False), "max(0.1x, x)"),
    "elu": (lambda: nn.ELU(alpha=1.0, inplace=False), "x (x>0) 或 α(e^x - 1) (x≤0)"),
    "gelu": (lambda: nn.GELU(), "x·Φ(x)（高斯误差线性单元）"),
}


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"未知激活函数 {name!r}，可选：{list(ACTIVATIONS)}")
    return ACTIVATIONS[name][0]()


# ---------------------------------------------------------------------------
# 基础模块：卷积块 / 分类头
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """conv -> [BN] -> activation -> pooling。

    默认不做 padding，遵循 LeNet-5 的原始设计，让特征图逐层缩小（28→24→12→8→4）。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        activation: str = "relu",
        pooling: str = "max",
        use_bn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None
        self.act = make_activation(activation)
        if pooling == "max":
            self.pool = nn.MaxPool2d(2)
        elif pooling == "avg":
            self.pool = nn.AvgPool2d(2)
        elif pooling == "none":
            self.pool = nn.Identity()
        else:
            raise ValueError(f"未知池化方式 {pooling!r}")
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class LeNet5(nn.Module):
    """LeNet-5 的 28×28 适配版。

    原始 LeNet-5（LeCun et al., 1998）输入 32×32、用平均池化和 Sigmoid，
    本实验把它适配到 MNIST 的 28×28（去掉输入 padding），因此第二层池化后
    的特征图是 16×4×4 而不是 16×5×5，第一层全连接是 256→120。
    结构参数全部可配置，用于做激活函数 / 池化 / BN / Dropout 的对照实验：

        conv5x5(1→6)  -> pool -> conv5x5(6→16) -> pool -> flatten(256)
        -> fc(256→120) -> fc(120→84) -> fc(84→10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        activation: str = "relu",
        pooling: str = "max",
        use_bn: bool = False,
        dropout_fc: float = 0.0,
        dropout_conv: float = 0.0,
        fc_sizes: Iterable[int] = (120, 84),
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 6, 5, activation, pooling, use_bn, dropout_conv),
            ConvBlock(6, 16, 5, activation, pooling, use_bn, dropout_conv),
        )
        fc1, fc2 = tuple(fc_sizes)
        self.flatten_dim = 16 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, fc1),
            make_activation(activation),
            nn.Dropout(dropout_fc) if dropout_fc > 0 else nn.Identity(),
            nn.Linear(fc1, fc2),
            make_activation(activation),
            nn.Dropout(dropout_fc) if dropout_fc > 0 else nn.Identity(),
            nn.Linear(fc2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回各层特征图，供可视化用（conv1 输出、pool1 输出、conv2 输出...）。"""
        outs = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, ConvBlock):
                outs.append(x)
        return outs


class SmallVGG(nn.Module):
    """更深的 VGG 风格小网络：两个卷积块（每块两层 3×3 卷积 + BN + ReLU + MaxPool）。

    结构与 LeNet-5 不同：只使用 3×3 卷积、通道数更多（32→64）、每个卷积块
    内连续两层卷积，最后接一个 128 维全连接 + Dropout。用来观察"加深网络 +
    BN + Dropout"相对 LeNet-5 能带来多少提升。
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """全连接基线：784 -> 256 -> 128 -> 10，用于和 CNN 对比。"""

    def __init__(
        self,
        num_classes: int = 10,
        in_dim: int = 28 * 28,
        hidden: Iterable[int] = (256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), make_activation(activation)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown(model: LeNet5) -> list[tuple[str, int]]:
    """把 LeNet-5 每个参数张量的形状与数量列出来，用于报告中的结构表。"""
    rows = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            rows.append((name, tuple(p.shape), p.numel()))
    return rows


def build_model(name: str, **kwargs) -> nn.Module:
    """按名字构造模型，便于实验脚本统一调用。"""
    name = name.lower()
    if name in ("lenet", "lenet5", "lenet-5"):
        return LeNet5(**kwargs)
    if name in ("vgg", "smallvgg", "small_vgg"):
        return SmallVGG(**kwargs)
    if name == "mlp":
        return MLP(**kwargs)
    raise ValueError(f"未知模型 {name!r}")


if __name__ == "__main__":
    # 自检：打印结构、参数量、输出尺寸；确认初始 logits 接近均匀
    for name, kw in [
        ("lenet_classic", dict(activation="sigmoid", pooling="avg")),
        ("lenet_relu", dict(activation="relu", pooling="max")),
        ("lenet_bn", dict(activation="relu", pooling="max", use_bn=True)),
        ("smallvgg", dict()),
        ("mlp", dict()),
    ]:
        model_name = "mlp" if name == "mlp" else ("vgg" if name == "smallvgg" else "lenet")
        m = build_model(model_name, **kw)
        n = count_parameters(m)
        x = torch.randn(2, 1, 28, 28)
        y = m(x)
        print(f"{name:14s} 参数量 {n:>8,d}  输出 {tuple(y.shape)}  logits 均值 {y.mean().item():+.4f}")
    print("\nLeNet-5 参数明细：")
    m = LeNet5()
    total = 0
    for pname, shape, num in param_breakdown(m):
        total += num
        print(f"  {pname:24s} {str(shape):20s} {num:>8,d}")
    print(f"  {'合计':24s} {'':20s} {total:>8,d}")
