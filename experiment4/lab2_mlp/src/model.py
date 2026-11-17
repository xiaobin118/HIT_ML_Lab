"""多层感知机模型定义（PyTorch）。"""

from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"未知激活函数: {name}")


class MLP(nn.Module):
    """可配置深度/宽度/激活/Dropout/BatchNorm 的二分类 MLP。

    输入 34 维，输出 1 个 logit（配合 BCEWithLogitsLoss 使用）。
    """

    def __init__(
        self,
        in_dim: int = 34,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状 (batch,) 的 logit。"""
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """可训练参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
