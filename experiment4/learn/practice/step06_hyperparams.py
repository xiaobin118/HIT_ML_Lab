"""第 6 步：超参数扫描。

目标：固定其他条件，每次只改一个因素，观察它对 MNIST 测试准确率的影响：
    宽度、深度、学习率、初始化、批大小、L2。

本文件复用第 4 步的 MLP 和 fit，只写两个函数：
    run_config：跑一组配置，返回准确率与参数量
    sweep     ：对一组取值循环调用 run_config

运行：python step06_hyperparams.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step04_train import MLP, fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from datasets import load_mnist, stratified_subsample  # noqa: E402

Array = np.ndarray

# MNIST 只加载一次；不同训练规模按 n_train 缓存子集。
_XTR, _YTR, _XTE, _YTE = load_mnist(ROOT / "data")
_CACHE: dict[tuple[int, int], tuple[Array, Array, Array, Array]] = {}


def get_data(n_train: int, n_test: int) -> tuple[Array, Array, Array, Array]:
    """按类别分层抽取训练/测试子集，并缓存。已提供。"""
    key = (n_train, n_test)
    if key not in _CACHE:
        Xs, ys = stratified_subsample(_XTR, _YTR, n_train, seed=0)
        Xv, yv = stratified_subsample(_XTE, _YTE, n_test, seed=1)
        _CACHE[key] = (Xs, ys, Xv, yv)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# TODO 1：跑一组配置
# ---------------------------------------------------------------------------
def run_config(
    sizes: list[int],
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    init: str = "he",
    l2: float = 0.0,
    n_train: int = 6000,
    n_test: int = 3000,
    seed: int = 0,
) -> dict:
    """训练一个 MLP，返回最终训练/测试准确率。

    Parameters
    ----------
    sizes : list[int]
        网络结构，例如 [784, 128, 10]。
    epochs : int
        训练轮数。
    batch_size : int
        批大小。
    lr : float
        学习率。
    init : str
        "he" / "xavier" / "small" / "zeros"。
    l2 : float
        L2 正则系数。
    n_train, n_test : int
        训练/测试子集大小。
    seed : int
        初始化与打乱数据的种子。

    Returns
    -------
    dict
        至少包含：
            "test_acc"  : float
            "train_acc" : float
            "history"   : fit 返回的每个 epoch 记录
            "params"    : int，模型参数量

    步骤：
        1. Xtr, ytr, Xte, yte = get_data(n_train, n_test)
        2. model = MLP(sizes, activation="relu", init=init, seed=seed)
        3. hist = fit(model, Xtr, ytr, Xte, yte,
                      epochs=epochs, batch_size=batch_size,
                      lr=lr, optimizer="adam", l2=l2, seed=seed)
        4. 返回 hist[-1] 里的 train_acc/test_acc，加上 history 和参数数量
    """
    # TODO 1：按上面的步骤实现。
    Xtr, ytr, Xte, yte = get_data(n_train, n_test)
    model = MLP(sizes, activation="relu", init=init, seed=seed)
    hist = fit(model , Xtr, ytr, Xte, yte,
               epochs=epochs, batch_size=batch_size,
               lr=lr, optimizer="adam", l2=l2, seed=seed)
    return {
        "test_acc": hist[-1]["test_acc"],
        "train_acc": hist[-1]["train_acc"],
        "history": hist,
        "params": model.count_params(),
    }


# ---------------------------------------------------------------------------
# TODO 2：单因素扫描
# ---------------------------------------------------------------------------
def sweep(param: str, values: list, **base) -> list[dict]:
    """对某个参数的一组取值分别训练，其余参数保持不变。

    Parameters
    ----------
    param : str
        要扫描的参数名，必须是 run_config 的某个参数名，
        例如 "sizes"、"lr"、"init"、"batch_size"、"l2"。
    values : list
        该参数的候选值。
    **base
        其余固定参数，原样传给 run_config。

    Returns
    -------
    list[dict]
        每个元素是 {"参数名": 取值, "test_acc": ..., "train_acc": ..., ...}。
    """
    # TODO 2：循环 values，用 {**base, param: v} 调用 run_config。
    results = []
    for v in values:
        config = {**base, param: v}
        result = run_config(**config)
        result[param] = v
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 画图（已提供）
# ---------------------------------------------------------------------------
def plot_sweep(
    xs: list[int | float | str],
    test_accs: list[float],
    train_accs: list[float] | None,
    xlabel: str,
    title: str,
    path: str,
    logx: bool = False,
) -> None:
    """画一条测试准确率曲线，可选第二条训练准确率。"""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    idx = np.arange(len(xs))
    ax.plot(idx, test_accs, "o-", color="#d62728", label="测试准确率")
    if train_accs is not None:
        ax.plot(idx, train_accs, "s--", color="#1f77b4", alpha=0.65,
                label="训练准确率", markersize=4)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(x) for x in xs], fontsize=8)
    if logx:
        ax.set_xlabel(xlabel + "（对数刻度，点等距）")
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("准确率")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    for i, v in enumerate(test_accs):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存到 {path}")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(passed)


def self_check() -> bool:
    print("=" * 72)
    print("第 6 步自检（缩小规模：3000 张训练、8 轮，只为验证趋势）")
    print("=" * 72)
    ok = True
    t0 = time.perf_counter()

    # 先跑一次，确保 run_config（TODO 1）已经实现，
    # 否则 sweep 会先报 TODO 2，掩盖第一个 TODO。
    _ = run_config([784, 8, 10], epochs=2, n_train=1000, n_test=500)

    base = {"epochs": 8, "n_train": 3000, "n_test": 1000}

    # 1. 宽度
    rows = sweep("sizes", [[784, 4, 10], [784, 32, 10], [784, 128, 10]], **base)
    accs = [r["test_acc"] for r in rows]
    ok &= check("1. 宽度：单元越多越好", accs[-1] - accs[0] > 0.05,
                f"4 -> {accs[0]:.3f}, 128 -> {accs[-1]:.3f}")

    # 2. 学习率
    rows = sweep("lr", [1e-4, 1e-2], sizes=[784, 128, 10], **base)
    accs = [r["test_acc"] for r in rows]
    ok &= check("2. 学习率：太小会欠拟合", accs[1] - accs[0] > 0.05,
                f"1e-4 -> {accs[0]:.3f}, 1e-2 -> {accs[1]:.3f}")

    # 3. 初始化
    rows = sweep("init", ["zeros", "he"], sizes=[784, 128, 10], **base)
    accs = {r["init"]: r["test_acc"] for r in rows}
    ok &= check("3. 全零初始化学不动",
                accs["zeros"] < 0.20 and accs["he"] - accs["zeros"] > 0.50,
                f"zeros={accs['zeros']:.3f}, he={accs['he']:.3f}")

    # 4. 批大小
    rows = sweep("batch_size", [32, 3000], sizes=[784, 128, 10], **base)
    accs = [r["test_acc"] for r in rows]
    ok &= check("4. 固定轮数下小批量更新更多", accs[0] - accs[1] > 0.05,
                f"batch=32 -> {accs[0]:.3f}, full -> {accs[1]:.3f}")

    print("-" * 72)
    print(f"全部通过（{time.perf_counter() - t0:.1f}s）。" if ok else "还有检查未通过。")
    return ok


# ---------------------------------------------------------------------------
# 演示：六组扫描
# ---------------------------------------------------------------------------
def demo() -> None:
    print("\n正式扫描：6000 张训练、3000 张测试、12 轮")
    t0 = time.perf_counter()
    base = {"epochs": 12, "n_train": 6000, "n_test": 3000}

    # 宽度
    widths = [4, 8, 16, 32, 64, 128, 256]
    rows = sweep("sizes", [[784, w, 10] for w in widths], **base)
    plot_sweep(widths, [r["test_acc"] for r in rows],
               [r["train_acc"] for r in rows], "隐藏单元数",
               "宽度的影响", "step06_width.png", logx=True)

    # 深度
    depths = [[784] + [64] * d + [10] for d in [1, 2, 3]]
    rows = sweep("sizes", depths, **base)
    plot_sweep([f"{len(s)-2} 层" for s in depths],
               [r["test_acc"] for r in rows], None, "隐藏层数",
               "深度的影响（每层 64）", "step06_depth.png")

    # 学习率
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    rows = sweep("lr", lrs, sizes=[784, 128, 10], **base)
    plot_sweep([f"{v:g}" for v in lrs], [r["test_acc"] for r in rows], None,
               "学习率", "学习率的影响", "step06_lr.png", logx=True)

    # 初始化
    inits = ["zeros", "xavier", "he"]
    rows = sweep("init", inits, sizes=[784, 128, 10], **base)
    plot_sweep(inits, [r["test_acc"] for r in rows], None,
               "初始化", "初始化的影响", "step06_init.png")

    # 批大小
    batches = [16, 64, 128, 6000]
    rows = sweep("batch_size", batches, sizes=[784, 128, 10], **base)
    plot_sweep([str(b) for b in batches], [r["test_acc"] for r in rows], None,
               "批大小", "批大小的影响（12 轮）", "step06_batch.png")

    # L2（小数据，容易过拟合）
    l2s = [0.0, 1e-3, 1e-2, 1e-1]
    rows = sweep("l2", l2s, sizes=[784, 128, 10],
                 epochs=20, n_train=1000, n_test=3000)
    plot_sweep([f"{v:g}" for v in l2s],
               [r["test_acc"] for r in rows],
               [r["train_acc"] for r in rows],
               "L2 系数 λ", "L2 正则（1000 张训练，20 轮）", "step06_l2.png")

    print(f"\n六组扫描完成，总耗时 {time.perf_counter() - t0:.1f}s。")
    print("读图时先看趋势，再看具体数字；不同机器/种子的绝对值会略有差异。")


if __name__ == "__main__":
    try:
        ok = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if ok:
        demo()
