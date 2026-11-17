"""扩散模型模块：DDPM 的 U-Net、余弦噪声调度、EMA、DDPM/DDIM 采样。

实现的是 Ho et al. (2020) DDPM 的标准结构 + Nichol & Dhariwal (2021) 的余弦噪声调度：

    U-Net：64×64×3 输入，4 个分辨率级别（64/32/16/8），每级 2 个带时间步条件的
           ResBlock；16×16 与 8×8 处插入空间自注意力；最后零初始化卷积输出 ε 预测。
    EMA：维护参数的指数滑动平均副本，采样时使用 EMA 权重（DDPM 的标配）。
    采样：DDPM（1000 步，随机）与 DDIM（20/50/100 步，确定性，η=0）。

训练目标是 ε-预测的均方误差：
    L = E_{x0, ε, t} || ε − ε_θ( √ᾱ_t · x0 + √(1−ᾱ_t) · ε, t ) ||²
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 正弦时间步嵌入
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = time.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ---------------------------------------------------------------------------
# 基础块
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv → 加时间条件 → GN → SiLU → Dropout → Conv → 残差。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 dropout: float = 0.1, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """空间多头自注意力（用 scaled_dot_product_attention 加速）。"""

    def __init__(self, channels: int, heads: int = 4, groups: int = 32):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (b, heads, hw, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """DDPM 标准 U-Net（64×64 输入；空间尺寸 64→32→16→8，通道 base×[1,2,2,4]）。"""

    def __init__(self, in_ch: int = 3, base: int = 96,
                 channel_mult: tuple[int, ...] = (1, 2, 2, 4),
                 num_res_blocks: int = 2, attn_resolutions: tuple[int, ...] = (16, 8),
                 time_dim: int = 128, dropout: float = 0.1, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        channels = [base * m for m in channel_mult]
        self.pos_emb = SinusoidalPositionEmbeddings(time_dim)
        self.time_embed = TimeEmbedding(time_dim, base * 4)
        self.init_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # ---- 下采样 ----
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch, res = base, image_size
        skip_channels: list[int] = [base]  # init_conv 的输出也要参与跳跃连接
        for level, c in enumerate(channels):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, c, base * 4, dropout))
                ch = c
                attns.append(AttentionBlock(c) if res in attn_resolutions else nn.Identity())
                skip_channels.append(c)
            self.down_blocks.append(blocks)
            self.down_attn.append(attns)
            if level != len(channels) - 1:
                self.downsamples.append(Downsample(c))
                skip_channels.append(c)
                res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # ---- 中间层 ----
        self.mid1 = ResBlock(ch, ch, base * 4, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, base * 4, dropout)

        # ---- 上采样 ----
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            c = channels[level]
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_c = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_c, c, base * 4, dropout))
                ch = c
                attns.append(AttentionBlock(c) if res in attn_resolutions else nn.Identity())
            self.up_blocks.append(blocks)
            self.up_attn.append(attns)
            if level != 0:
                self.upsamples.append(Upsample(c))
                res *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.final = nn.Sequential(nn.GroupNorm(32, base), nn.SiLU(),
                                   nn.Conv2d(base, in_ch, 3, padding=1))
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(self.pos_emb(t))
        h = self.init_conv(x)
        skips = [h]
        for blocks, attns, down in zip(self.down_blocks, self.down_attn, self.downsamples):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, t_emb))
                skips.append(h)
            h = down(h)
            if not isinstance(down, nn.Identity):
                skips.append(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        for blocks, attns, up in zip(self.up_blocks, self.up_attn, self.upsamples):
            for block, attn in zip(blocks, attns):
                h = attn(block(torch.cat([h, skips.pop()], dim=1), t_emb))
            h = up(h)
        return self.final(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 噪声调度与扩散过程
# ---------------------------------------------------------------------------


def make_beta_schedule(T: int, schedule: str = "cosine", s: float = 0.008) -> torch.Tensor:
    """余弦调度（Nichol & Dhariwal, 2021）或线性调度（Ho et al., 2020）。"""
    if schedule == "cosine":
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 1e-5, 0.9999).float()
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, T).float()
    raise ValueError(f"未知调度 {schedule!r}")


class GaussianDiffusion:
    """前向加噪 q(x_t|x_0) 与反向采样所需的所有系数。"""

    def __init__(self, T: int = 1000, schedule: str = "cosine", device: torch.device | None = None):
        self.T = T
        self.schedule = schedule
        self.device = device or torch.device("cpu")
        betas = make_beta_schedule(T, schedule).to(self.device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (betas * (1.0 - torch.cat([
            torch.ones(1, device=self.device), self.alphas_cumprod[:-1]]))
            / (1.0 - self.alphas_cumprod))

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.device = device
        for name in ("betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "posterior_variance"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0
                + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise)

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t.float())
        return F.mse_loss(pred, noise)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class EMA:
    """参数的指数滑动平均：采样时用平均权重，能明显提升 DDPM 的生成质量。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for e, p in zip(self.ema.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for e, p in zip(self.ema.buffers(), model.buffers()):
            e.copy_(p)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ddpm_step(model, diffusion, x, i):
    n = x.shape[0]
    t = torch.full((n,), i, device=x.device, dtype=torch.float)
    eps = model(x, t)
    ab = diffusion.alphas_cumprod[i]
    ab_prev = diffusion.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=x.device)
    alpha = diffusion.alphas[i]
    beta = diffusion.betas[i]
    x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
    mean = (torch.sqrt(ab_prev) * beta / (1 - ab)) * x0 + \
           (torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab)) * x
    if i > 0:
        var = diffusion.posterior_variance[i]
        return mean + torch.sqrt(var) * torch.randn_like(x)
    return mean


@torch.no_grad()
def sample_ddpm(model: nn.Module, diffusion: GaussianDiffusion, n: int,
                device: torch.device, seed: int | None = None, batch: int = 128,
                progress: bool = False) -> torch.Tensor:
    """完整 1000 步 DDPM 采样，返回 [0,1] 的 CPU 张量 (n,3,64,64)。"""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    outs = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        x = torch.randn(b, 3, 64, 64, device=device)
        for i in reversed(range(diffusion.T)):
            x = _ddpm_step(model, diffusion, x, i)
        outs.append(((x + 1) / 2).clamp(0, 1).cpu())
        if progress:
            print(f"  DDPM 采样 {start + b}/{n}", flush=True)
    return torch.cat(outs)


@torch.no_grad()
def sample_ddim(model: nn.Module, diffusion: GaussianDiffusion, n: int,
                device: torch.device, steps: int = 50, eta: float = 0.0,
                seed: int | None = None, batch: int = 256) -> torch.Tensor:
    """DDIM 确定性采样（η=0），可用 20~100 步逼近 1000 步 DDPM 的质量。"""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    ts = torch.linspace(diffusion.T - 1, 0, steps + 1).long()
    outs = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        x = torch.randn(b, 3, 64, 64, device=device)
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            eps = model(x, torch.full((b,), float(t), device=device))
            ab_t = diffusion.alphas_cumprod[t]
            ab_next = diffusion.alphas_cumprod[t_next]
            x0 = ((x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)).clamp(-1, 1)
            if t_next > 0:
                sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t)) * \
                        torch.sqrt(1 - ab_t / ab_next)
                x = (torch.sqrt(ab_next) * x0
                     + torch.sqrt(torch.clamp(1 - ab_next - sigma ** 2, min=0)) * eps
                     + sigma * torch.randn_like(x))
            else:
                x = x0
        outs.append(((x + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(outs)


@torch.no_grad()
def sample_ddim_trajectory(model: nn.Module, diffusion: GaussianDiffusion,
                           n: int, device: torch.device, steps: int = 50,
                           seed: int = 0) -> torch.Tensor:
    """DDIM 采样过程中的若干帧，用于展示"从纯噪声逐步生成"的过程。"""
    model.eval()
    torch.manual_seed(seed)
    ts = torch.linspace(diffusion.T - 1, 0, steps + 1).long()
    x = torch.randn(n, 3, 64, 64, device=device)
    frames = []
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        eps = model(x, torch.full((n,), float(t), device=device))
        ab_t = diffusion.alphas_cumprod[t]
        ab_next = diffusion.alphas_cumprod[t_next]
        x0 = ((x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)).clamp(-1, 1)
        if t_next > 0:
            x = torch.sqrt(ab_next) * x0 + torch.sqrt(torch.clamp(1 - ab_next, min=0)) * eps
        else:
            x = x0
        frames.append(((x + 1) / 2).clamp(0, 1).cpu())
    return torch.stack(frames)
