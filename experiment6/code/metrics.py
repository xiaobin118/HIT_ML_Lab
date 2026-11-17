"""评估模块：FID（Inception-v3 特征距离）、最近邻记忆化分析、多样性与 C2ST 判别测试。

生成模型没有"准确率"这种天然指标，本实验用三类互补的定量指标：

1. FID（Fréchet Inception Distance）：生成样本与真实样本在 Inception-v3 的
   2048 维池化特征空间中的均值/协方差距离，越低越好；它是本实验最主要的指标。
2. 最近邻距离：每张生成图到训练集的最近像素 L2 距离，与"真实留出图到训练集"
   的同一距离做对比，用于判断小数据量下模型是否只是在背训练图（记忆化）。
3. C2ST（分类器双样本检验）：训练一个二分类小 CNN 区分真实图与生成图，
   测试准确率越接近 50% 说明生成样本越难与真实样本区分。

另外还提供特征空间多样性（检测模式崩溃）、颜色多样性、VAE 重建的 MSE/PSNR。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


def _sqrtm_trace_product(sigma1: np.ndarray, sigma2: np.ndarray) -> float:
    """计算 trace(sqrt(sigma1 @ sigma2))，用于 FID 公式。

    sigma1、sigma2 都是半正定协方差矩阵。先做 SVD：
        sigma1 @ sigma2 的特征值（可能非对称矩阵，但特征值为非负实数）
        等于其奇异值，因此 trace(sqrt(A)) = Σ sqrt(奇异值)。
    这样避免引入 scipy，也不需要处理复数。
    """
    sv = np.linalg.svd(sigma1 @ sigma2, compute_uv=False)
    return float(np.sqrt(np.clip(sv, 0, None)).sum())


def fid_from_stats(real_stats: dict, fake_stats: dict) -> float:
    """Fréchet Inception Distance：||μr−μg||² + Tr(Σr+Σg−2(ΣrΣg)^{1/2})。"""
    mu_r, mu_g = real_stats["mu"], fake_stats["mu"]
    s_r, s_g = real_stats["sigma"], fake_stats["sigma"]
    diff = float(np.sum((mu_r - mu_g) ** 2))
    tr = np.trace(s_r) + np.trace(s_g) - 2.0 * _sqrtm_trace_product(s_r, s_g)
    return round(float(diff + tr), 4)


class FIDEvaluator:
    """用 torchvision 预训练的 Inception-v3 计算 FID。

    取 Inception-v3 全局平均池化后的 2048 维特征（去掉 fc 层），输入按
    pytorch-fid 的惯例缩放到 299×299、归一化到 [-1,1]。
    """

    def __init__(self, device: torch.device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device
        self.model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=False, aux_logits=True
        )
        self.model.fc = nn.Identity()
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def features(self, images: torch.Tensor, batch: int = 128) -> np.ndarray:
        """images: [0,1] 的 (N,3,64,64) 张量，返回 (N,2048) 的 float64 特征。"""
        feats = []
        for start in range(0, len(images), batch):
            x = images[start:start + batch].to(self.device, non_blocking=True)
            x = F.interpolate(x, size=(299, 299), mode="bilinear",
                              align_corners=False, antialias=True)
            x = (x - 0.5) / 0.5
            feats.append(self.model(x).float().cpu())
        return torch.cat(feats).numpy().astype(np.float64)

    def stats(self, images: torch.Tensor) -> dict:
        f = self.features(images)
        mu = f.mean(axis=0)
        sigma = np.cov(f, rowvar=False)
        return {"mu": mu, "sigma": sigma}

    def score(self, real_stats: dict, fake_images: torch.Tensor) -> float:
        return fid_from_stats(real_stats, self.stats(fake_images))


def reference_stats(
    evaluator: FIDEvaluator,
    images_uint8: torch.Tensor,
    ref_idx: np.ndarray,
    cache_path: Path,
    force: bool = False,
) -> dict:
    """计算（并缓存）真实参考集的 FID 统计量，避免每次实验重复算。"""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        blob = np.load(cache_path)
        return {
            "mu": blob["mu"], "sigma": blob["sigma"],
            "n": int(blob["n"]), "features": None,
        }
    x = images_uint8[torch.as_tensor(np.asarray(ref_idx))].float().div_(255.0)
    st = evaluator.stats(x)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mu=st["mu"], sigma=st["sigma"], n=len(ref_idx))
    return {"mu": st["mu"], "sigma": st["sigma"], "n": int(len(ref_idx)), "features": None}


# ---------------------------------------------------------------------------
# 最近邻（记忆化）分析
# ---------------------------------------------------------------------------


@torch.no_grad()
def min_distances(source: torch.Tensor, pool: torch.Tensor,
                  device: torch.device, batch: int = 64) -> np.ndarray:
    """source 中每张图到 pool 的最近像素 L2 距离（输入均为 [0,1] 的 (N,3,64,64)）。

    pool 一次性放到 GPU（约 0.7 GB），source 分块计算，避免显存峰值过高。
    """
    pool = pool.to(device).float().flatten(1)            # (P, 12288)
    out = []
    for start in range(0, len(source), batch):
        chunk = source[start:start + batch].to(device).float().flatten(1)
        dist = torch.cdist(chunk, pool, p=2)
        out.append(dist.min(dim=1).values.cpu().numpy())
    return np.concatenate(out)


def nn_summary(dist: np.ndarray) -> dict:
    return {
        "mean": round(float(dist.mean()), 4),
        "std": round(float(dist.std()), 4),
        "median": round(float(np.median(dist)), 4),
        "min": round(float(dist.min()), 4),
        "p10": round(float(np.percentile(dist, 10)), 4),
        "p90": round(float(np.percentile(dist, 90)), 4),
    }


# ---------------------------------------------------------------------------
# 多样性与 C2ST
# ---------------------------------------------------------------------------


def feature_diversity(features: np.ndarray, n: int = 500, seed: int = 0) -> float:
    """在 Inception 特征空间里随机抽 n 个样本，计算两两 L2 距离的均值。

    值越小说明样本越"挤在一起"，常用于辅助判断模式崩溃。
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(features), size=min(n, len(features)), replace=False)
    f = features[idx]
    dist = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)
    triu = np.triu_indices(len(f), k=1)
    return round(float(dist[triu].mean()), 4)


def color_diversity(images: torch.Tensor, n: int = 500, seed: int = 0) -> dict:
    """统计生成图的颜色丰富度：每张图的主色调（色相）直方图的熵，以及亮度的标准差。"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(images), size=min(n, len(images)), replace=False)
    x = images[idx].float()                                   # (n,3,64,64) [0,1]
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx = x.max(dim=1).values
    mn = x.min(dim=1).values
    delta = (mx - mn).clamp_min(1e-6)
    hue = torch.zeros_like(mx)
    mask_r = (mx == r)
    mask_g = (mx == g) & ~mask_r
    mask_b = (mx == b) & ~mask_r & ~mask_g
    hue[mask_r] = ((g - b)[mask_r] / delta[mask_r]) % 6
    hue[mask_g] = ((b - r)[mask_g] / delta[mask_g]) + 2
    hue[mask_b] = ((r - g)[mask_b] / delta[mask_b]) + 4
    hue = hue / 6.0
    sat = delta / mx.clamp_min(1e-6)
    # 只统计饱和度足够高的像素的色相，避免灰白背景主导
    sel = sat > 0.15
    hist = []
    for i in range(len(x)):
        h = hue[i][sel[i]]
        hist.append(torch.histc(h, bins=24, min=0, max=1))
    hist = torch.stack(hist).mean(dim=0)
    p = hist / hist.sum().clamp_min(1e-9)
    entropy = float(-(p * (p + 1e-12).log()).sum())
    return {
        "hue_entropy": round(entropy, 4),
        "hue_entropy_norm": round(entropy / np.log(24), 4),
        "brightness_std": round(float(x.mean(dim=1).std()), 4),
    }


class _C2STNet(nn.Module):
    """C2ST 用的小型卷积二分类器（32×32 输入）。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                               # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                               # 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.fc = nn.Linear(128 * 2 * 2, 1)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1)).view(-1)


def c2st_accuracy(
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    seed: int = 0,
    n_train: int = 2000,
    n_test: int = 800,
    epochs: int = 4,
) -> dict:
    """训练小 CNN 区分真实/生成图，返回测试准确率（0.5 = 完全分不出）。

    这是对生成样本质量的"整体难度"检验：C2ST 准确率越低，说明生成图在
    低层纹理/颜色上越接近真实图（但注意它和 FID 关注点不同）。
    """
    import random

    g = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)

    def take(x, n):
        idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
        return x[idx].float()

    real_tr, real_te = take(real, n_train), take(real, n_test)
    fake_tr, fake_te = take(fake, n_train), take(fake, n_test)
    x_tr = torch.cat([real_tr, fake_tr])
    y_tr = torch.cat([torch.zeros(len(real_tr)), torch.ones(len(fake_tr))])
    x_te = torch.cat([real_te, fake_te])
    y_te = torch.cat([torch.zeros(len(real_te)), torch.ones(len(fake_te))])

    # 32×32 下采样，降低计算量
    def prep(x):
        return F.interpolate(x, size=(32, 32), mode="bilinear",
                             align_corners=False, antialias=True)

    x_tr, x_te = prep(x_tr), prep(x_te)
    perm = torch.randperm(len(x_tr), generator=g)
    x_tr, y_tr = x_tr[perm], y_tr[perm]

    model = _C2STNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    bs = 128
    for _ in range(epochs):
        model.train()
        for start in range(0, len(x_tr), bs):
            xb = x_tr[start:start + bs].to(device, non_blocking=True)
            yb = y_tr[start:start + bs].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x_te), 256):
            xb = x_te[start:start + 256].to(device)
            yb = y_te[start:start + 256].to(device)
            pred = (model(xb) > 0).float()
            correct += int((pred == yb).sum().item())
    return {
        "accuracy": round(correct / len(x_te), 4),
        "n_train": int(len(x_tr)),
        "n_test": int(len(x_te)),
    }


# ---------------------------------------------------------------------------
# VAE 重建指标
# ---------------------------------------------------------------------------


def reconstruction_metrics(real: torch.Tensor, recon: torch.Tensor) -> dict:
    """VAE 重建质量：像素 MSE 与 PSNR（图片范围 [0,1]，PSNR = 10log10(1/MSE)）。"""
    mse = float(F.mse_loss(recon, real).item())
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    return {"mse": round(mse, 6), "psnr": round(psnr, 3)}
