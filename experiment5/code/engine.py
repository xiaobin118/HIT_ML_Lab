"""训练引擎：随机种子、设备选择、优化器构造、单轮训练/评估与统一训练循环。

训练循环只做一件事：给定模型、数据和超参数，跑满 epochs 轮，记录每轮的
训练/验证损失与准确率、学习率、耗时，以及（可选的）第一层卷积权重的梯度范数。
所有对照实验都复用这一个函数，保证除被研究的变量外其它条件完全一致。
"""

from __future__ import annotations

import copy
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from data import make_loader


# ---------------------------------------------------------------------------
# 随机性与设备
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """固定所有随机源，保证同一配置重复运行得到相同结果。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn 的 benchmark 会在不同输入尺寸下选不同算法，牺牲一点速度换可复现性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """优先用 GPU；没有 CUDA 时退回 CPU，便于在别的机器上复现。"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_report() -> dict:
    """收集实验环境信息，写入 results.json 并展示在报告第 3 节。"""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
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
        "cpu_count": __import__("os").cpu_count(),
        "numpy": np.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover - torchvision 一定存在，这里只是兜底
        info["torchvision"] = None
    return info


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------

OPTIMIZERS = ("sgd", "momentum", "adam", "adamw")


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float = 0.0):
    """构造优化器。momentum 表示 SGD + 0.9 动量。"""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知优化器 {name!r}，可选：{OPTIMIZERS}")


# ---------------------------------------------------------------------------
# 单轮训练与评估
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    """一轮小批量训练，返回 (平均损失, 准确率, 第一步的梯度范数均值[可选])。"""
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.numel()
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device, return_details: bool = False):
    """在验证/测试集上评估，返回损失、准确率；需要时额外返回预测与概率。"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_pred, all_label, all_prob = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_n += y.numel()
        if return_details:
            all_pred.append(pred.cpu())
            all_label.append(y.cpu())
            all_prob.append(prob.cpu())
    out = {"loss": total_loss / total_n, "acc": total_correct / total_n, "n": total_n}
    if return_details:
        out["pred"] = torch.cat(all_pred).numpy()
        out["label"] = torch.cat(all_label).numpy()
        out["prob"] = torch.cat(all_prob).numpy()
    return out


# ---------------------------------------------------------------------------
# 统一训练循环
# ---------------------------------------------------------------------------


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int = 10,
    optimizer: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    seed: int = 0,
    grad_clip: float | None = None,
    track_grad_norm: bool = False,
    scheduler: str | None = None,
    log_prefix: str = "",
) -> dict:
    """训练模型并返回历史记录。

    返回的 dict 包含：
        history      每轮的 train_loss/train_acc/val_loss/val_acc/epoch_time/lr
        best_val_acc 验证集最好准确率（按轮取最大，模型已被恢复到该轮）
        best_epoch   取得最好验证准确率的轮次（从 1 开始）
        total_time   训练总耗时（秒）
        params       可训练参数数量
        grad_norm_first_conv  每轮第一层卷积权重梯度范数的均值（可选）
    """
    device = device or get_device()
    set_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    sched = None
    if scheduler == "cosine":
        # 余弦退火：学习率从 lr 平滑降到 0，后期步长变小，有助于收敛到更平的极小值
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler is not None:
        raise ValueError(f"未知学习率调度器 {scheduler!r}，可选：cosine / None")

    history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                               "lr", "epoch_time")}
    grad_norms: list[float] = []       # 当前轮内每个 step 的梯度范数
    grad_epoch_means: list[float] = []  # 每轮的平均值，写进结果

    # 想观察"第一层卷积的梯度"是否消失：在它的权重上注册反向 hook
    first_conv = None
    if track_grad_norm:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break
        if first_conv is not None:
            def _hook(grad):
                grad_norms.append(float(grad.detach().norm(2).item()))
            first_conv.weight.register_hook(_hook)

    best_val_acc, best_state, best_epoch = -1.0, None, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        grad_norms.clear()
        lr_used = opt.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device, grad_clip)
        va = evaluate(model, val_loader, device)
        if sched is not None:
            sched.step()
        dt = time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["lr"].append(lr_used)
        history["epoch_time"].append(dt)
        if track_grad_norm and grad_norms:
            grad_epoch_means.append(sum(grad_norms) / len(grad_norms))

        if va["acc"] > best_val_acc:
            best_val_acc, best_epoch = va["acc"], epoch
            best_state = copy.deepcopy(model.state_dict())

        prefix = f"{log_prefix} " if log_prefix else ""
        print(f"{prefix}epoch {epoch:2d}/{epochs}  "
              f"train_loss {tr_loss:.4f}  train_acc {tr_acc * 100:5.2f}%  "
              f"val_loss {va['loss']:.4f}  val_acc {va['acc'] * 100:5.2f}%  "
              f"{dt:.1f}s", flush=True)

    total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time": total_time,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if grad_epoch_means:
        result["grad_norm_first_conv"] = grad_epoch_means
    return result
