"""训练引擎：随机种子、设备、DCGAN 训练循环、VAE 训练循环与采样工具。

所有对照实验都复用这里的同一个训练函数，保证除被研究的变量（激活函数、
Dropout、数据量、超参数）之外，其它条件完全一致。训练循环按"优化步数"
（而不是按 epoch）控制长度，这样不同数据量的实验可以比较相同的梯度更新次数。
"""

from __future__ import annotations

import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from data import make_loader, to_gan_range


# ---------------------------------------------------------------------------
# 随机性与设备
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True 让 cuDNN 自动挑最快的卷积算法（本实验在笔记本 GPU 上把
    # 单步耗时从 134 ms 降到 88 ms）；牺牲逐位可复现，但固定随机种子后结论稳定
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_report() -> dict:
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:                                    # pragma: no cover
        tv = None
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": tv,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device_count": torch.cuda.device_count(),
        "capability": (
            ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None
        ),
        "gpu_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
            if torch.cuda.is_available() else 0.0
        ),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except Exception:                                    # pragma: no cover
        info["matplotlib"] = None
    return info


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class GANConfig:
    """一次 DCGAN 训练的全部配置，写进 results.json 便于复现。"""

    max_steps: int = 6000
    batch_size: int = 128
    z_dim: int = 100
    g_base: int = 64
    d_base: int = 64
    g_act: str = "relu"
    d_act: str = "leaky_relu"
    d_dropout: float = 0.0
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    seed: int = 0
    real_smooth: float = 0.0      # 单边标签平滑：真实样本标签取 1-smooth
    log_every: int = 25

    def to_json(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


@dataclass
class VAEConfig:
    max_steps: int = 4000
    batch_size: int = 128
    z_dim: int = 128
    base: int = 64
    beta: float = 1.0
    lr: float = 1e-3
    seed: int = 0
    log_every: int = 25

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _grad_norm(parameters):
    """参数梯度总范数（返回 GPU 上的 0 维张量，避免每步都做 .item() 同步）。"""
    grads = [p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.sqrt(torch.stack(grads).sum())


def _run_callback(cb, step: int, model: nn.Module) -> None:
    """在 eval + no_grad 状态下执行回调（采样/FID），结束后恢复训练模式。"""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            cb(step, model)
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# DCGAN 训练循环
# ---------------------------------------------------------------------------


def train_dcgan(
    G: nn.Module,
    D: nn.Module,
    images: torch.Tensor,
    train_idx: np.ndarray,
    cfg: GANConfig,
    device: torch.device,
    snapshot_steps: tuple[int, ...] = (),
    snapshot_cb=None,
    score_cb=None,
    progress: bool = True,
) -> dict:
    """训练一对生成器/判别器，返回逐步记录的训练历史。

    每个优化步先更新 D（真实图标签→1，生成图标签→0），再更新 G（希望 D 把
    生成图判成 1）。损失用 BCEWithLogitsLoss，输出层不接 sigmoid。
    snapshot_cb(step, G)：在指定的 step 上执行（保存样本网格 / 计算 FID 等）；
    score_cb(step, G)：可选的额外评估钩子。
    """
    set_seed(cfg.seed)
    G.to(device)
    D.to(device)
    G.train()
    D.train()

    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    loader = make_loader(images, train_idx, batch_size=cfg.batch_size,
                         shuffle=True, seed=cfg.seed)
    data_iter = _cycle(loader)

    fixed_z = torch.randn(64, cfg.z_dim, device=device,
                          generator=torch.Generator(device=device).manual_seed(cfg.seed + 1))

    history = {
        "step": [], "loss_d": [], "loss_g": [],
        "d_real": [], "d_fake": [],
        "d_acc_real": [], "d_acc_fake": [],
        "grad_norm_g": [], "grad_norm_d": [],
        "time_per_step_ms": [],
    }
    keys = ("loss_d", "loss_g", "d_real", "d_fake", "d_acc_real", "d_acc_fake",
            "grad_norm_g", "grad_norm_d")
    acc = {k: torch.zeros((), device=device) for k in keys}
    n_logged = 0

    snapshot_set = set(int(s) for s in snapshot_steps)
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        # ---------------- 判别器 ----------------
        real = next(data_iter).to(device, non_blocking=True)
        real = to_gan_range(real)
        b = real.size(0)
        ones = torch.ones(b, device=device) * (1.0 - cfg.real_smooth)
        zeros = torch.zeros(b, device=device)

        z = torch.randn(b, cfg.z_dim, device=device)
        fake = G(z).detach()

        opt_d.zero_grad(set_to_none=True)
        d_real = D(real)
        d_fake = D(fake)
        loss_d = bce(d_real, ones) + bce(d_fake, zeros)
        loss_d.backward()
        grad_d = _grad_norm(D.parameters())
        opt_d.step()

        # ---------------- 生成器 ----------------
        z = torch.randn(b, cfg.z_dim, device=device)
        fake = G(z)
        opt_g.zero_grad(set_to_none=True)
        d_fake_g = D(fake)
        loss_g = bce(d_fake_g, torch.ones(b, device=device))
        loss_g.backward()
        grad_g = _grad_norm(G.parameters())
        opt_g.step()

        # ---------------- 记录（保持在 GPU 上累加，只在打日志时同步一次） ----------
        with torch.no_grad():
            acc["loss_d"] += loss_d.detach()
            acc["loss_g"] += loss_g.detach()
            acc["d_real"] += torch.sigmoid(d_real).mean()
            acc["d_fake"] += torch.sigmoid(d_fake).mean()
            acc["d_acc_real"] += (d_real > 0).float().mean()
            acc["d_acc_fake"] += (d_fake < 0).float().mean()
            acc["grad_norm_g"] += grad_g
            acc["grad_norm_d"] += grad_d
        n_logged += 1

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            history["step"].append(step)
            for k in keys:
                history[k].append(round(float((acc[k] / n_logged).item()), 5))
            history["time_per_step_ms"].append(round((time.time() - t0) / step * 1000, 3))
            acc = {k: torch.zeros((), device=device) for k in keys}
            n_logged = 0
            if progress and (step % (cfg.log_every * 20) == 0 or step == cfg.max_steps):
                print(f"    step {step:5d}/{cfg.max_steps}  "
                      f"lossD {history['loss_d'][-1]:.3f}  lossG {history['loss_g'][-1]:.3f}  "
                      f"D(real) {history['d_real'][-1]:.3f}  D(fake) {history['d_fake'][-1]:.3f}")

        if step in snapshot_set and snapshot_cb is not None:
            _run_callback(snapshot_cb, step, G)
        if step in snapshot_set and score_cb is not None:
            _run_callback(score_cb, step, G)

    history["train_time_s"] = round(time.time() - t0, 2)
    history["steps_per_s"] = round(cfg.max_steps / max(history["train_time_s"], 1e-9), 2)
    history["n_train_images"] = int(len(train_idx))
    history["epochs_equivalent"] = round(cfg.max_steps * cfg.batch_size / len(train_idx), 2)
    return history


# ---------------------------------------------------------------------------
# VAE 训练循环
# ---------------------------------------------------------------------------


def train_vae(
    model: nn.Module,
    images: torch.Tensor,
    train_idx: np.ndarray,
    cfg: VAEConfig,
    device: torch.device,
    snapshot_steps: tuple[int, ...] = (),
    snapshot_cb=None,
    progress: bool = True,
) -> dict:
    """训练 VAE，记录总损失、重建损失、KL 损失随优化步的变化。"""
    set_seed(cfg.seed)
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = make_loader(images, train_idx, batch_size=cfg.batch_size,
                         shuffle=True, seed=cfg.seed)
    data_iter = _cycle(loader)

    history = {"step": [], "loss": [], "recon": [], "kl": [],
               "recon_per_pixel": [], "kl_per_dim": [], "time_per_step_ms": []}
    keys = ("loss", "recon", "kl")
    acc = {k: torch.zeros((), device=device) for k in keys}
    n_logged = 0
    t0 = time.time()
    snapshot_set = set(int(s) for s in snapshot_steps)

    for step in range(1, cfg.max_steps + 1):
        x = next(data_iter).to(device, non_blocking=True)
        recon_logits, mu, logvar = model(x)
        loss, recon, kl = model.elbo_loss(recon_logits, x, mu, logvar, beta=cfg.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc["loss"] += loss.detach()
            acc["recon"] += recon.detach()
            acc["kl"] += kl.detach()
        n_logged += 1

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            means = {k: float((acc[k] / n_logged).item()) for k in keys}
            history["step"].append(step)
            history["loss"].append(round(means["loss"], 5))
            history["recon"].append(round(means["recon"], 5))
            history["kl"].append(round(means["kl"], 5))
            history["recon_per_pixel"].append(round(means["recon"] / (3 * 64 * 64), 6))
            history["kl_per_dim"].append(round(means["kl"] / cfg.z_dim, 5))
            history["time_per_step_ms"].append(round((time.time() - t0) / step * 1000, 3))
            acc = {k: torch.zeros((), device=device) for k in keys}
            n_logged = 0
            if progress and (step % (cfg.log_every * 20) == 0 or step == cfg.max_steps):
                print(f"    step {step:5d}/{cfg.max_steps}  loss {history['loss'][-1]:.2f}  "
                      f"recon {history['recon'][-1]:.2f}  KL {history['kl'][-1]:.2f}")

        if step in snapshot_set and snapshot_cb is not None:
            _run_callback(snapshot_cb, step, model)

    history["train_time_s"] = round(time.time() - t0, 2)
    history["steps_per_s"] = round(cfg.max_steps / max(history["train_time_s"], 1e-9), 2)
    history["n_train_images"] = int(len(train_idx))
    history["epochs_equivalent"] = round(cfg.max_steps * cfg.batch_size / len(train_idx), 2)
    return history


# ---------------------------------------------------------------------------
# 采样工具
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_gan(G: nn.Module, n: int, device: torch.device, seed: int = 0,
               batch: int = 256) -> torch.Tensor:
    """从生成器采样 n 张图，返回 [0,1] 的 CPU 张量。"""
    G.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    outs = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        z = torch.randn(b, G.z_dim, device=device, generator=gen)
        outs.append((torch.tanh(G(z)) + 1.0) / 2.0)
    return torch.cat(outs).cpu()


@torch.no_grad()
def interpolate_gan(G: nn.Module, device: torch.device, n_steps: int = 10,
                    seed: int = 0) -> torch.Tensor:
    """在潜空间里对两个随机噪声做线性插值，返回 (n_steps, 3, 64, 64) 的 [0,1] 图片。"""
    G.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    z1 = torch.randn(1, G.z_dim, device=device, generator=gen)
    z2 = torch.randn(1, G.z_dim, device=device, generator=gen)
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
    z = z1 * (1 - alphas) + z2 * alphas
    return ((torch.tanh(G(z)) + 1.0) / 2.0).cpu()


@torch.no_grad()
def reconstruct_vae(model: nn.Module, images: torch.Tensor, indices: np.ndarray,
                    device: torch.device, n: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """用 VAE 重建 n 张真实图片，返回 (原图, 重建图)，均为 [0,1] 的 CPU 张量。"""
    rng = np.random.RandomState(seed)
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    x = images[torch.as_tensor(np.asarray(indices))].float().div_(255.0).to(device)
    model.eval()
    recon_logits, _, _ = model(x)
    return x.cpu(), torch.sigmoid(recon_logits).cpu()


@torch.no_grad()
def interpolate_vae(model: nn.Module, device: torch.device, n_steps: int = 10,
                    seed: int = 0) -> torch.Tensor:
    """VAE 潜空间插值：从先验采两个点，线性插值后解码。"""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    z1 = torch.randn(1, model.z_dim, device=device, generator=gen)
    z2 = torch.randn(1, model.z_dim, device=device, generator=gen)
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
    z = z1 * (1 - alphas) + z2 * alphas
    return torch.sigmoid(model.decode(z)).cpu()
