"""数据模块：Anime Face 64×64 数据集的去重、内存缓存、划分与 DataLoader 构造。

原始数据来自 Kaggle「Anime Face Dataset」（21,551 张 64×64 RGB 动漫头像，
经 HuggingFace 镜像下载，CC0 许可）。原始文件里有大量字节级重复图片，
本模块按文件 md5 去重，并把去重后的图片一次性解码成 uint8 张量缓存到磁盘，
后续所有实验都从内存缓存读取，避免每个 epoch 反复解码 PNG。

划分方式：固定随机种子下把去重后的图片分成
    - 训练池（train）：用于所有生成模型的训练；
    - 参考集（ref，3,000 张）：不参与训练，只用于计算 FID 的实图统计量、
      以及"生成图到真实训练集最近邻距离"的实图基线。
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "anime-faces"
IMAGE_DIR = DATA_DIR / "images"
CACHE_PATH = DATA_DIR / "images_uint8.pt"
DEDUP_PATH = DATA_DIR / "unique_files.json"

IMAGE_SIZE = 64
REF_SIZE = 3000          # 留出的参考集大小（不参与训练）
SPLIT_SEED = 0


# ---------------------------------------------------------------------------
# 文件枚举与去重
# ---------------------------------------------------------------------------


def list_files() -> list[Path]:
    """按数字编号排序列出原始 PNG 文件（1.png, 2.png, ...）。"""
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"找不到图片目录 {IMAGE_DIR}；请先运行 data/download_anime_faces.sh 下载并解压数据集"
        )
    files = sorted(IMAGE_DIR.glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"{IMAGE_DIR} 中没有 PNG 图片")
    return files


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def unique_files(force: bool = False) -> tuple[list[Path], dict]:
    """按 md5 去掉字节级重复的图片，返回 (保留文件列表, 去重统计)。

    去重结果缓存到 unique_files.json，避免每次实验都重新算 md5。
    """
    if DEDUP_PATH.exists() and not force:
        info = json.loads(DEDUP_PATH.read_text(encoding="utf-8"))
        files = [IMAGE_DIR / name for name in info["kept"]]
        return files, info["stats"]

    files = list_files()
    print(f"[data] 计算 {len(files)} 个文件的 md5 ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        hashes = list(ex.map(_file_md5, files))

    seen: dict[str, Path] = {}
    kept: list[Path] = []
    for path, h in zip(files, hashes):
        if h in seen:
            continue
        seen[h] = path
        kept.append(path)

    stats = {
        "n_raw": len(files),
        "n_unique": len(kept),
        "n_duplicate": len(files) - len(kept),
        "duplicate_ratio": round((len(files) - len(kept)) / len(files), 4),
    }
    DEDUP_PATH.write_text(
        json.dumps({"kept": [p.name for p in kept], "stats": stats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[data] 去重：{stats['n_raw']} → {stats['n_unique']} "
          f"（重复 {stats['n_duplicate']} 张，占 {stats['duplicate_ratio']:.1%}）")
    return kept, stats


# ---------------------------------------------------------------------------
# 解码并缓存为 uint8 张量（N, 3, 64, 64）
# ---------------------------------------------------------------------------


def _decode(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (IMAGE_SIZE, IMAGE_SIZE):
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def load_clean_images(force: bool = False) -> tuple[torch.Tensor, list[str], dict]:
    """加载去重后的全部图片，返回 (uint8 NCHW 张量, 文件名列表, 去重统计)。

    第一次调用会并行解码 17k 张 PNG 并缓存为 .pt（约 210 MB），之后直接读缓存。
    """
    if CACHE_PATH.exists() and not force:
        blob = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        return blob["images"], blob["names"], blob["dedup_stats"]

    files, dedup_stats = unique_files(force=force)
    print(f"[data] 解码 {len(files)} 张图片 ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        arrays = list(ex.map(_decode, files))
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W) uint8
    names = [p.name for p in files]
    torch.save({"images": images, "names": names, "dedup_stats": dedup_stats}, CACHE_PATH)
    print(f"[data] 已缓存 {tuple(images.shape)} → {CACHE_PATH.name}")
    return images, names, dedup_stats


# ---------------------------------------------------------------------------
# 训练/参考集划分、子集抽样
# ---------------------------------------------------------------------------


def split_indices(n: int, n_ref: int = REF_SIZE, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """随机划分训练池与参考集，返回 (train_idx, ref_idx)，均为升序索引。"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    ref = np.sort(perm[:n_ref])
    train = np.sort(perm[n_ref:])
    return train, ref


def subset_indices(indices: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """从给定索引集合中随机抽 n 个（用于"不同数据量"实验）。"""
    if n >= len(indices):
        return np.asarray(indices)
    rng = np.random.RandomState(seed)
    picked = rng.choice(len(indices), size=n, replace=False)
    return np.sort(np.asarray(indices)[picked])


# ---------------------------------------------------------------------------
# Dataset 与 DataLoader
# ---------------------------------------------------------------------------


class ImageTensorDataset(Dataset):
    """把内存中的 uint8 NCHW 图片按索引转成 [0,1] 的 float CHW 张量。"""

    def __init__(self, images: torch.Tensor, indices: np.ndarray | None = None):
        self.images = images
        self.indices = np.arange(len(images)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.images[int(self.indices[i])].float().div_(255.0)


def make_loader(
    images: torch.Tensor,
    indices: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """构造 DataLoader；图片已在内存里，num_workers=0 反而最快。"""
    ds = ImageTensorDataset(images, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# 归一化约定与统计
# ---------------------------------------------------------------------------


def to_gan_range(x: torch.Tensor) -> torch.Tensor:
    """[0,1] → [-1,1]，GAN 生成器输出用 tanh，因此真实图也要缩放到同一区间。"""
    return x * 2.0 - 1.0


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1]，用于把 GAN 生成结果保存成图片。"""
    return (x + 1.0) / 2.0


def dataset_stats(images_uint8: torch.Tensor, indices: np.ndarray) -> dict:
    """统计选中图片的像素均值/标准差、亮度、饱和度，用于报告的数据概览。"""
    x = images_uint8[indices].float().div_(255.0)          # (N,3,H,W)
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    brightness = x.mean(dim=1)                              # (N,H,W)
    mx = x.max(dim=1).values
    mn = x.min(dim=1).values
    saturation = ((mx - mn) / mx.clamp_min(1e-6)).mean()
    return {
        "n": int(len(indices)),
        "mean_rgb": [round(float(v), 4) for v in mean],
        "std_rgb": [round(float(v), 4) for v in std],
        "mean_brightness": round(float(brightness.mean()), 4),
        "mean_saturation": round(float(saturation), 4),
    }


def save_uint8_grid(path: Path, images_uint8: torch.Tensor, nrow: int = 8, scale: int = 2) -> None:
    """把 uint8 NCHW 图片拼成网格 PNG（不在报告里用，留给快速目检）。"""
    from torchvision.utils import make_grid

    grid = make_grid(images_uint8.float().div_(255.0), nrow=nrow, padding=2)
    ndarr = (grid.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(ndarr).resize(
        (ndarr.shape[1] * scale, ndarr.shape[0] * scale), Image.NEAREST
    ).save(path)
