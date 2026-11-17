"""模型模块：DCGAN 的生成器/判别器（激活函数、Dropout 可配置）与卷积 VAE。

全部模型从零实现，只依赖 PyTorch 的基础层：

    Generator     z(100) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64（tanh，输出 [-1,1]）
    Discriminator 64×64 → 32 → 16 → 8 → 4 → 1（输出 logit，配合 BCEWithLogitsLoss）
    VAE           conv 编码器 → (μ, logσ²) → 重参数化采样 → conv 解码器（sigmoid 输出 [0,1]）

生成器/判别器的激活函数、判别器 Dropout 都做成参数，供控制变量实验使用。
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 激活函数表
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "relu": (lambda: nn.ReLU(inplace=False), "ReLU: max(0, x)"),
    "leaky_relu": (lambda: nn.LeakyReLU(negative_slope=0.2, inplace=False), "LeakyReLU: max(0.2x, x)"),
    "elu": (lambda: nn.ELU(alpha=1.0, inplace=False), "ELU: x (x>0) 或 α(e^x−1) (x≤0)"),
    "gelu": (lambda: nn.GELU(), "GELU: x·Φ(x)"),
    "sigmoid": (lambda: nn.Sigmoid(), "Sigmoid: 1/(1+e^-x)"),
}


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"未知激活函数 {name!r}，可选：{list(ACTIVATIONS)}")
    return ACTIVATIONS[name][0]()


# ---------------------------------------------------------------------------
# DCGAN 权值初始化（Radford et al., 2015）：Conv 用 N(0, 0.02)，BN 权重 N(1, 0.02)
# ---------------------------------------------------------------------------


def dcgan_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# DCGAN 生成器
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """DCGAN 生成器：5 层转置卷积，把 100 维噪声逐步放大到 64×64 彩色图。

    4×4 → 8×8 → 16×16 → 32×32 → 64×64，通道数 512 → 256 → 128 → 64 → 3。
    除最后一层用 tanh 外，每层都是 [转置卷积 → 批归一化 → 激活]。
    """

    def __init__(self, z_dim: int = 100, base: int = 64, activation: str = "relu", use_bn: bool = True):
        super().__init__()
        self.z_dim = z_dim
        c = [base * 8, base * 4, base * 2, base]
        layers: list[nn.Module] = []

        def block(in_ch: int, out_ch: int, act: bool = True) -> list[nn.Module]:
            mods: list[nn.Module] = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
            ]
            if use_bn:
                mods.append(nn.BatchNorm2d(out_ch))
            if act:
                mods.append(make_activation(activation))
            return mods

        # 第 1 层：z(1,1) → 4×4，stride 1、kernel 4 的特殊上采样
        layers.append(nn.ConvTranspose2d(z_dim, c[0], kernel_size=4, stride=1, padding=0, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(c[0]))
        layers.append(make_activation(activation))
        # 第 2~4 层：每层空间尺寸 ×2
        for in_ch, out_ch in zip(c[:-1], c[1:]):
            layers.extend(block(in_ch, out_ch))
        # 输出层：64×64×3，tanh 限制到 [-1,1]
        layers.append(nn.ConvTranspose2d(c[-1], 3, kernel_size=4, stride=2, padding=1, bias=True))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(z)


# ---------------------------------------------------------------------------
# DCGAN 判别器
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """DCGAN 判别器：5 层卷积把 64×64 图片压成 1 个 logit。

    输出不接 sigmoid，训练时直接用 BCEWithLogitsLoss（数值更稳定）。
    第一层不接 BN（DCGAN 的原始设计），之后每层 [卷积 → BN → LeakyReLU]，可插入 Dropout。
    """

    def __init__(
        self,
        base: int = 64,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        use_bn: bool = True,
    ):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8]
        layers: list[nn.Module] = []
        # 第一层：3×64×64 → base×32×32（无 BN）
        layers += [
            nn.Conv2d(3, channels[0], kernel_size=4, stride=2, padding=1, bias=True),
            make_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        # 第 2~4 层：空间尺寸 32→16→8→4
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        # 输出层：4×4 → 1×1 的 logit
        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0, bias=True))
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0))


# ---------------------------------------------------------------------------
# 卷积 VAE
# ---------------------------------------------------------------------------


class VAE(nn.Module):
    """卷积变分自编码器：编码器输出 (μ, logσ²)，重参数化采样后由解码器重建。

    编码器/解码器结构互为镜像（64→32→16→8→4 与 4→8→16→32→64），
    隐藏层用 BatchNorm，解码器最后一层输出 logits，配合
    binary_cross_entropy_with_logits 计算重建项（等价于对 [0,1] 像素做伯努利建模）。
    """

    def __init__(self, z_dim: int = 128, base: int = 64, beta: float = 1.0):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        ch = [base, base * 2, base * 4, base * 8]        # 64,128,256,512

        # --- 编码器 ---
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch[0], 4, 2, 1), nn.LeakyReLU(0.2),                    # 64→32
            nn.Conv2d(ch[0], ch[1], 4, 2, 1), nn.BatchNorm2d(ch[1]), nn.LeakyReLU(0.2),  # 32→16
            nn.Conv2d(ch[1], ch[2], 4, 2, 1), nn.BatchNorm2d(ch[2]), nn.LeakyReLU(0.2),  # 16→8
            nn.Conv2d(ch[2], ch[3], 4, 2, 1), nn.BatchNorm2d(ch[3]), nn.LeakyReLU(0.2),  # 8→4
        )
        self.enc_flat = ch[3] * 4 * 4
        self.fc_mu = nn.Linear(self.enc_flat, z_dim)
        self.fc_logvar = nn.Linear(self.enc_flat, z_dim)

        # --- 解码器 ---
        self.fc_dec = nn.Linear(z_dim, self.enc_flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch[3], ch[2], 4, 2, 1), nn.BatchNorm2d(ch[2]), nn.ReLU(),   # 4→8
            nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1), nn.BatchNorm2d(ch[1]), nn.ReLU(),   # 8→16
            nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1), nn.BatchNorm2d(ch[0]), nn.ReLU(),   # 16→32
            nn.ConvTranspose2d(ch[0], 3, 4, 2, 1),                                         # 32→64（logits）
        )

    # -- 基本部件 ---------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 512, 4, 4)
        return self.dec(h)     # logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n: int, device: torch.device, seed: int | None = None,
               batch: int = 256) -> torch.Tensor:
        """从先验 N(0, I) 采样生成图片，返回 [0,1] 的 CPU 张量。"""
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        outs = []
        with torch.no_grad():
            for start in range(0, n, batch):
                b = min(batch, n - start)
                z = torch.randn(b, self.z_dim, device=device,
                                generator=gen if gen is not None else None)
                outs.append(torch.sigmoid(self.decode(z)).cpu())
        return torch.cat(outs, dim=0)

    @staticmethod
    def elbo_loss(
        recon_logits: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (总损失, 重建损失/图, KL 损失/图)，损失按 batch 平均。

        解码器输出 logits，经 sigmoid 得到像素均值；重建项用平方误差
        （等价于固定方差的高斯似然），比逐像素 BCE 在本数据上更稳定：
        BCE 的梯度尺度太大，会让编码器停在"高噪声 + 输出平均脸"的局部解，
        而平方误差能稳定收敛到清晰重建（见正确性验证）。
        KL = -0.5 * Σ(1 + logσ² − μ² − σ²) / batch。
        """
        recon = F.mse_loss(torch.sigmoid(recon_logits), x, reduction="none")
        recon_loss = recon.flatten(1).sum(1).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return recon_loss + beta * kl, recon_loss, kl


# ---------------------------------------------------------------------------
# 参数量与结构摘要
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: tuple[int, ...], device: str = "cpu") -> list[dict]:
    """用前向 hook 记录每个叶子层（卷积/转置卷积/线性/BN/池化/激活）的输出尺寸。"""
    records: list[dict] = []
    hooks = []

    def make_hook(name: str, module: nn.Module):
        def hook(_m, _inp, out):
            if torch.is_tensor(out):
                shape = list(out.shape[1:])
            else:
                shape = None
            records.append({
                "name": name,
                "type": type(module).__name__,
                "out_shape": shape,
                "params": sum(p.numel() for p in module.parameters(recurse=False)),
            })
        return hook

    leaves = [(n, m) for n, m in model.named_modules() if len(list(m.children())) == 0]
    for name, module in leaves:
        hooks.append(module.register_forward_hook(make_hook(name, module)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape, device=device)
        model(dummy)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return records
