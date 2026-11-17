#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验四主实验脚本：多层感知机 vs 线性分类器。

运行方式：
    cd exp4_report && python full_experiment.py            # 全部实验
    cd exp4_report && python full_experiment.py --parts 1  # 只跑第 1 部分

输出：
    exp4_report/results.json   全部指标
    exp4_report/figures/*.png  全部插图
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import (
    DATASETS_2D,
    DATASET_TITLES,
    load_mnist,
    stratified_subsample,
    train_test_split,
)
from linear import SoftmaxClassifier
from metrics import accuracy, confusion_matrix, per_class_report, top_confusions
from mlp import MLP, numeric_gradient_check
from train import fit
from viz import (
    CLASS_COLORS,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_history,
    plot_mnist_grid,
    savefig,
)

FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
SEED = 0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def to_py(obj):
    """把 numpy 标量 / 数组递归转换成可 JSON 序列化的 Python 对象。"""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run(model, Xtr, ytr, Xte, yte, epochs, batch_size, lr, l2=0.0, opt="adam", seed=SEED):
    t0 = time.perf_counter()
    hist = fit(
        model,
        Xtr,
        ytr,
        Xte,
        yte,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        l2=l2,
        optimizer=opt,
        seed=seed,
    )
    return {
        "history": hist,
        "train_acc": hist[-1]["train_acc"],
        "test_acc": hist[-1]["test_acc"],
        "train_loss": hist[-1]["train_loss"],
        "test_loss": hist[-1]["test_loss"],
        "time": time.perf_counter() - t0,
        "params": int(model.trainable_parameters()),
    }


def timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    print(f"[done] {label}  ({time.perf_counter() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# 第 0 部分：梯度校验
# ---------------------------------------------------------------------------


def part0_gradcheck():
    print("\n=== 第 0 部分：反向传播的数值梯度校验 ===")
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(6, 3))
    y = rng.integers(0, 4, size=6)
    out = {"model": "[3, 5, 4]", "n_samples": 6, "eps": 1e-6, "l2": 0.01, "runs": []}
    for act in ["relu", "tanh", "sigmoid"]:
        model = MLP([3, 5, 4], activation=act, init="xavier", seed=1)
        rows = numeric_gradient_check(model, X, y, l2=0.01, eps=1e-6)
        out["runs"].append({"activation": act, "rows": rows,
                            "max_abs_err": max(r["max_abs_err"] for r in rows)})
        print(f"  {act:8s} max|numeric-analytic| = {out['runs'][-1]['max_abs_err']:.2e}")
    return out


# ---------------------------------------------------------------------------
# 第 1 部分：二维四类数据，线性 vs MLP
# ---------------------------------------------------------------------------

SYNTH_EPOCHS = 600
SYNTH_LINEAR_EPOCHS = 1500
SYNTH_BATCH = 64
SYNTH_SEEDS = [0, 1, 2]
LINEAR_LR_2D = 1e-1
MLP_LR_2D = 1e-2
MLP_2D_SIZES = [2, 32, 32, 4]


def part1_2d():
    print("\n=== 第 1 部分：二维四类数据上的线性分类器与 MLP ===")
    results = {"config": {"mlp_epochs": SYNTH_EPOCHS, "linear_epochs": SYNTH_LINEAR_EPOCHS,
                          "batch_size": SYNTH_BATCH, "linear_lr": LINEAR_LR_2D,
                          "mlp_lr": MLP_LR_2D, "mlp_sizes": MLP_2D_SIZES,
                          "optimizer": "adam", "seeds": SYNTH_SEEDS},
               "datasets": {}}
    models = {}

    for name in ["blobs", "rings", "spiral"]:
        test_acc = {"linear": [], "mlp": []}
        train_acc = {"linear": [], "mlp": []}
        test_loss = {"linear": [], "mlp": []}
        times = {"linear": [], "mlp": []}
        seed0 = {}
        for s in SYNTH_SEEDS:
            X, y = DATASETS_2D[name](seed=s)
            Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=s)
            lin = SoftmaxClassifier(2, 4, seed=s)
            mlp = MLP(MLP_2D_SIZES, activation="relu", init="he", seed=s)
            r_lin = run(lin, Xtr, ytr, Xte, yte, SYNTH_LINEAR_EPOCHS, SYNTH_BATCH, LINEAR_LR_2D)
            r_mlp = run(mlp, Xtr, ytr, Xte, yte, SYNTH_EPOCHS, SYNTH_BATCH, MLP_LR_2D)
            for tag, r in [("linear", r_lin), ("mlp", r_mlp)]:
                test_acc[tag].append(r["test_acc"])
                train_acc[tag].append(r["train_acc"])
                test_loss[tag].append(r["test_loss"])
                times[tag].append(r["time"])
            if s == SEED:
                models[name] = (lin, mlp, (Xtr, ytr, Xte, yte))
                seed0 = {"linear_test": r_lin["test_acc"], "mlp_test": r_mlp["test_acc"],
                         "n_train": int(len(ytr)), "n_test": int(len(yte)),
                         "linear_history": r_lin["history"], "mlp_history": r_mlp["history"]}

        d = {"title": DATASET_TITLES[name], "seed0": seed0}
        for tag in ["linear", "mlp"]:
            d[tag] = {
                "test_acc_mean": float(np.mean(test_acc[tag])),
                "test_acc_std": float(np.std(test_acc[tag])),
                "train_acc_mean": float(np.mean(train_acc[tag])),
                "test_accs": test_acc[tag],
                "test_loss_mean": float(np.mean(test_loss[tag])),
                "time_mean": float(np.mean(times[tag])),
                "params": int(models[name][0 if tag == "linear" else 1].trainable_parameters()),
            }
        d["gap_mean"] = d["mlp"]["test_acc_mean"] - d["linear"]["test_acc_mean"]
        results["datasets"][name] = d
        print(f"  {name:7s} linear={d['linear']['test_acc_mean']:.4f}±{d['linear']['test_acc_std']:.4f} "
              f"mlp={d['mlp']['test_acc_mean']:.4f}±{d['mlp']['test_acc_std']:.4f} "
              f"gap={d['gap_mean']:+.4f}")

    # 图 1：三个数据集的散点
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    for ax, name in zip(axes, ["blobs", "rings", "spiral"]):
        Xtr, ytr, _, _ = models[name][2]
        for c in range(4):
            m = ytr == c
            ax.scatter(Xtr[m, 0], Xtr[m, 1], s=9, color=CLASS_COLORS[c], label=f"类别 {c}")
        ax.set_title(f"{DATASET_TITLES[name]}\n训练样本 {len(ytr)} 个", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="lower right", markerscale=1.4, framealpha=0.85)
    fig.suptitle("二维四类合成数据（每类 200~250 个样本，含噪声）", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig01_datasets.png")

    # 图 2：决策边界对比
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.0))
    for j, name in enumerate(["blobs", "rings", "spiral"]):
        lin, mlp, (Xtr, ytr, Xte, yte) = models[name]
        d = results["datasets"][name]
        plot_decision_boundary(axes[0, j], lin, Xtr, ytr,
                               title=f"线性分类器：测试准确率 {d['seed0']['linear_test']:.3f}")
        plot_decision_boundary(axes[1, j], mlp, Xtr, ytr,
                               title=f"MLP(32-32)：测试准确率 {d['seed0']['mlp_test']:.3f}")
        axes[1, j].set_xlabel(DATASET_TITLES[name], fontsize=9)
    fig.suptitle("线性分类器（上）与 MLP（下）的决策区域：线性区域只能由直线围成，"
                 "MLP 的边界是分段折线", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig02_boundaries.png")

    # 图 3：螺旋数据上的训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    lin_h = results["datasets"]["spiral"]["seed0"]["linear_history"]
    mlp_h = results["datasets"]["spiral"]["seed0"]["mlp_history"]
    for h, tag, col in [(lin_h, "线性分类器", "#1f77b4"), (mlp_h, "MLP(32-32)", "#d62728")]:
        axes[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], color=col,
                     label=f"{tag} 训练损失", linewidth=1.6)
        axes[0].plot([r["epoch"] for r in h], [r["test_loss"] for r in h], color=col,
                     linestyle="--", alpha=0.7, label=f"{tag} 测试损失", linewidth=1.3)
        axes[1].plot([r["epoch"] for r in h], [r["train_acc"] for r in h], color=col,
                     label=f"{tag} 训练", linewidth=1.6)
        axes[1].plot([r["epoch"] for r in h], [r["test_acc"] for r in h], color=col,
                     linestyle="--", alpha=0.7, label=f"{tag} 测试", linewidth=1.3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_yscale("log")
    axes[0].set_title("损失曲线（实线训练 / 虚线测试）")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, ncol=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].set_ylim(0.1, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7.5, ncol=2)
    fig.suptitle(f"四臂螺旋数据（seed 0）：线性分类器最终测试准确率 "
                 f"{results['datasets']['spiral']['seed0']['linear_test']:.3f}，"
                 f"MLP 为 {results['datasets']['spiral']['seed0']['mlp_test']:.3f}", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig03_spiral_curves.png")

    return results


# ---------------------------------------------------------------------------
# 第 2 部分：二维数据上的「训练集大小」实验
# ---------------------------------------------------------------------------


def part2_data_size_2d():
    print("\n=== 第 2 部分：训练集大小的影响（二维螺旋） ===")
    X, y = DATASETS_2D["spiral"](n_per_class=750, noise=0.05, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=1)
    sizes = [30, 60, 120, 250, 500, 900, 1500, 2100]
    sub_seeds = [0, 1, 2]
    out = {"sizes": sizes, "n_test": int(len(yte)), "sub_seeds": sub_seeds,
           "linear": [], "mlp": [], "epochs": {"linear": SYNTH_LINEAR_EPOCHS, "mlp": 400}}

    for n in sizes:
        accs = {"linear": [], "mlp": []}
        trains = {"linear": [], "mlp": []}
        times = {"linear": [], "mlp": []}
        for s in sub_seeds:
            Xs, ys = stratified_subsample(Xtr, ytr, n, seed=s)
            lin = SoftmaxClassifier(2, 4, seed=s)
            mlp = MLP(MLP_2D_SIZES, activation="relu", init="he", seed=s)
            rl = run(lin, Xs, ys, Xte, yte, SYNTH_LINEAR_EPOCHS, min(64, n), LINEAR_LR_2D)
            rm = run(mlp, Xs, ys, Xte, yte, 400, min(64, n), MLP_LR_2D)
            for tag, r in [("linear", rl), ("mlp", rm)]:
                accs[tag].append(r["test_acc"])
                trains[tag].append(r["train_acc"])
                times[tag].append(r["time"])
        row = {"n": int(n)}
        for tag in ["linear", "mlp"]:
            row[tag] = {
                "test_acc_mean": float(np.mean(accs[tag])),
                "test_acc_std": float(np.std(accs[tag])),
                "train_acc_mean": float(np.mean(trains[tag])),
                "test_accs": accs[tag],
                "time_mean": float(np.mean(times[tag])),
            }
        out["linear"].append({"n": int(n), **row["linear"]})
        out["mlp"].append({"n": int(n), **row["mlp"]})
        print(f"  n={n:5d}  linear={row['linear']['test_acc_mean']:.3f}±"
              f"{row['linear']['test_acc_std']:.3f}  "
              f"mlp={row['mlp']['test_acc_mean']:.3f}±{row['mlp']['test_acc_std']:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, tag, col, marker in [("linear", "线性分类器", "#1f77b4", "o"),
                                  ("mlp", "MLP(32-32)", "#d62728", "s")]:
        ns = [d["n"] for d in out[key]]
        te = [d["test_acc_mean"] for d in out[key]]
        te_std = [d["test_acc_std"] for d in out[key]]
        tr = [d["train_acc_mean"] for d in out[key]]
        ax.errorbar(ns, te, yerr=te_std, marker=marker, color=col, capsize=3,
                    label=f"{tag} 测试准确率（3 次重复）", linewidth=1.7)
        ax.plot(ns, tr, marker=marker, color=col, linestyle="--", alpha=0.55,
                label=f"{tag} 训练准确率", linewidth=1.2, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("训练样本数（对数坐标）")
    ax.set_ylabel("准确率")
    ax.set_ylim(0.15, 1.03)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("四臂螺旋：准确率随训练样本数的变化", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig04_data_size_2d.png")
    return out


# ---------------------------------------------------------------------------
# 第 3 部分：MNIST 上的「训练集大小」实验
# ---------------------------------------------------------------------------

MNIST_SMALL_MLP = [784, 128, 10]


def part3_data_size_mnist(Xtr, ytr, Xte, yte):
    print("\n=== 第 3 部分：训练集大小的影响（MNIST） ===")
    sizes = [100, 300, 1000, 3000, 10000, 30000, 60000]
    out = {"sizes": sizes, "n_test": int(len(yte)), "linear": [], "mlp": [],
           "epochs": 20, "mlp_sizes": MNIST_SMALL_MLP}

    for n in sizes:
        Xs, ys = stratified_subsample(Xtr, ytr, n, seed=SEED)
        lin = SoftmaxClassifier(784, 10, seed=SEED)
        mlp = MLP(MNIST_SMALL_MLP, activation="relu", init="he", seed=SEED)
        rl = run(lin, Xs, ys, Xte, yte, 20, min(128, max(16, n // 4)), 1e-3)
        rm = run(mlp, Xs, ys, Xte, yte, 20, min(128, max(16, n // 4)), 1e-3)
        out["linear"].append({"n": int(n), "test_acc": rl["test_acc"],
                              "train_acc": rl["train_acc"], "time": rl["time"]})
        out["mlp"].append({"n": int(n), "test_acc": rm["test_acc"],
                           "train_acc": rm["train_acc"], "time": rm["time"]})
        print(f"  n={n:6d}  linear={rl['test_acc']:.4f}  mlp={rm['test_acc']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for key, tag, col, marker in [("linear", "线性分类器", "#1f77b4", "o"),
                                  ("mlp", "MLP(128)", "#d62728", "s")]:
        ns = [d["n"] for d in out[key]]
        axes[0].plot(ns, [d["test_acc"] for d in out[key]], marker=marker, color=col,
                     label=tag, linewidth=1.7)
        axes[0].plot(ns, [d["train_acc"] for d in out[key]], marker=marker, color=col,
                     linestyle="--", alpha=0.55, linewidth=1.2, label=f"{tag} 训练")
        axes[1].plot(ns, [d["time"] for d in out[key]], marker=marker, color=col,
                     label=tag, linewidth=1.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("训练样本数（对数坐标）")
    axes[0].set_ylabel("准确率")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[0].set_title("MNIST 测试准确率 vs 训练样本数")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练样本数（对数坐标）")
    axes[1].set_ylabel("训练总耗时 / s")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    axes[1].set_title("训练耗时（20 轮，CPU）")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig05_data_size_mnist.png")
    return out


# ---------------------------------------------------------------------------
# 第 4 部分：MNIST 子集上的超参数实验
# ---------------------------------------------------------------------------

HP_EPOCHS = 20


def hp_run(Xtr, ytr, Xte, yte, sizes, **kwargs):
    mlp = MLP(sizes, activation=kwargs.pop("activation", "relu"),
              init=kwargs.pop("init", "he"), seed=kwargs.pop("seed", SEED))
    res = run(mlp, Xtr, ytr, Xte, yte, kwargs.pop("epochs", HP_EPOCHS),
              kwargs.pop("batch_size", 128), kwargs.pop("lr", 1e-3),
              kwargs.pop("l2", 0.0), kwargs.pop("opt", "adam"))
    return res


def part4_hyperparams(Xtr, ytr, Xte, yte):
    print("\n=== 第 4 部分：超参数对比（MNIST 子集） ===")
    Xs, ys = stratified_subsample(Xtr, ytr, 10000, seed=SEED)
    Xv, yv = stratified_subsample(Xte, yte, 5000, seed=1)
    out = {"n_train": int(len(ys)), "n_test": int(len(yv)), "epochs": HP_EPOCHS,
           "base": {"sizes": [784, 128, 10], "batch_size": 128, "lr": 1e-3,
                    "optimizer": "adam", "l2": 0.0}}

    # -- 宽度 --------------------------------------------------------------
    print("  [宽度]")
    widths = [4, 8, 16, 32, 64, 128, 256]
    out["width"] = {"units": widths, "test_acc": [], "train_acc": [], "params": [],
                    "time": [], "history": []}
    for w in widths:
        r = hp_run(Xs, ys, Xv, yv, [784, w, 10])
        out["width"]["test_acc"].append(r["test_acc"])
        out["width"]["train_acc"].append(r["train_acc"])
        out["width"]["params"].append(r["params"])
        out["width"]["time"].append(r["time"])
        out["width"]["history"].append(r["history"])
        print(f"    width={w:3d} test={r['test_acc']:.4f} train={r['train_acc']:.4f}")

    # -- 深度 --------------------------------------------------------------
    print("  [深度]")
    depths = [1, 2, 3, 4]
    out["depth"] = {"depths": depths, "sizes": [], "test_acc": [], "train_acc": [],
                    "params": [], "time": [], "history": []}
    for d in depths:
        sizes = [784] + [64] * d + [10]
        r = hp_run(Xs, ys, Xv, yv, sizes, epochs=20)
        out["depth"]["sizes"].append(sizes)
        out["depth"]["test_acc"].append(r["test_acc"])
        out["depth"]["train_acc"].append(r["train_acc"])
        out["depth"]["params"].append(r["params"])
        out["depth"]["time"].append(r["time"])
        out["depth"]["history"].append(r["history"])
        print(f"    depth={d} test={r['test_acc']:.4f} train={r['train_acc']:.4f}")

    # -- 激活函数 ----------------------------------------------------------
    print("  [激活函数]")
    acts = ["relu", "tanh", "sigmoid"]
    out["activation"] = {"names": acts, "test_acc": [], "train_acc": [], "history": [],
                         "params": [], "time": []}
    for act in acts:
        r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], activation=act,
                   init="auto", epochs=20)
        out["activation"]["test_acc"].append(r["test_acc"])
        out["activation"]["train_acc"].append(r["train_acc"])
        out["activation"]["params"].append(r["params"])
        out["activation"]["time"].append(r["time"])
        out["activation"]["history"].append(r["history"])
        print(f"    {act:8s} test={r['test_acc']:.4f} train={r['train_acc']:.4f}")

    # -- 学习率 ------------------------------------------------------------
    print("  [学习率]")
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    out["lr"] = {"lrs": lrs, "test_acc": [], "train_acc": [], "history": [],
                 "test_loss": [], "time": []}
    for lr in lrs:
        r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], lr=lr, epochs=20)
        out["lr"]["test_acc"].append(r["test_acc"])
        out["lr"]["train_acc"].append(r["train_acc"])
        out["lr"]["test_loss"].append(r["test_loss"])
        out["lr"]["history"].append(r["history"])
        out["lr"]["time"].append(r["time"])
        print(f"    lr={lr:g} test={r['test_acc']:.4f} loss={r['test_loss']:.4f}")

    # -- 初始化 ------------------------------------------------------------
    print("  [初始化]")
    inits = ["zeros", "small", "xavier", "he"]
    out["init"] = {"names": inits, "test_acc": [], "train_acc": [], "history": [],
                   "time": []}
    for init in inits:
        r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], init=init, epochs=20)
        out["init"]["test_acc"].append(r["test_acc"])
        out["init"]["train_acc"].append(r["train_acc"])
        out["init"]["history"].append(r["history"])
        out["init"]["time"].append(r["time"])
        print(f"    {init:7s} test={r['test_acc']:.4f} train={r['train_acc']:.4f}")

    # -- 批大小 ------------------------------------------------------------
    print("  [批大小]")
    batches = [16, 64, 128, 512, 10000]
    out["batch"] = {"sizes": batches, "test_acc": [], "train_acc": [], "history": [],
                    "time": []}
    for bs in batches:
        r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], batch_size=bs, epochs=20)
        out["batch"]["test_acc"].append(r["test_acc"])
        out["batch"]["train_acc"].append(r["train_acc"])
        out["batch"]["history"].append(r["history"])
        out["batch"]["time"].append(r["time"])
        print(f"    batch={bs:6d} test={r['test_acc']:.4f}")

    # 全批量再用 300 轮，说明「更新次数」才是瓶颈
    r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], batch_size=10000, epochs=300)
    out["batch"]["full_300ep"] = {"test_acc": r["test_acc"], "train_acc": r["train_acc"],
                                  "history": r["history"], "time": r["time"]}
    print(f"    full-batch 300ep test={r['test_acc']:.4f}")

    # -- L2 正则（小数据、易过拟合） --------------------------------------
    print("  [L2 正则，小数据]")
    Xsm, ysm = stratified_subsample(Xs, ys, 1000, seed=2)
    l2s = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    out["l2"] = {"l2": l2s, "test_acc": [], "train_acc": [], "history": [],
                 "n_train": int(len(ysm)), "epochs": 30}
    for lam in l2s:
        r = hp_run(Xsm, ysm, Xv, yv, [784, 128, 10], l2=lam, epochs=30, batch_size=64)
        out["l2"]["test_acc"].append(r["test_acc"])
        out["l2"]["train_acc"].append(r["train_acc"])
        out["l2"]["history"].append(r["history"])
        print(f"    l2={lam:g} test={r['test_acc']:.4f} train={r['train_acc']:.4f}")

    # -- 优化器 ------------------------------------------------------------
    print("  [优化器]")
    opts = [("sgd", 1e-2), ("momentum", 1e-2), ("adam", 1e-3)]
    out["optimizer"] = {"names": [], "lrs": [], "test_acc": [], "train_acc": [],
                        "history": [], "time": []}
    for name, lr in opts:
        r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], opt=name, lr=lr, epochs=20)
        out["optimizer"]["names"].append(name)
        out["optimizer"]["lrs"].append(lr)
        out["optimizer"]["test_acc"].append(r["test_acc"])
        out["optimizer"]["train_acc"].append(r["train_acc"])
        out["optimizer"]["history"].append(r["history"])
        out["optimizer"]["time"].append(r["time"])
        print(f"    {name:8s} lr={lr:g} test={r['test_acc']:.4f}")

    # -- 随机种子稳定性 ----------------------------------------------------
    print("  [随机种子]")
    seeds = [0, 1, 2]
    accs = []
    for s in seeds:
        r = hp_run(Xs, ys, Xv, yv, [784, 128, 10], seed=s, epochs=20)
        accs.append(r["test_acc"])
    out["seed"] = {"seeds": seeds, "test_acc": accs,
                   "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    print(f"    seeds={seeds} mean={np.mean(accs):.4f} std={np.std(accs):.4f}")

    plot_hyperparam_figures(out)
    return out


def plot_hyperparam_figures(hp):
    # 图 6：宽度与深度
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    w = hp["width"]
    axes[0].plot(w["units"], w["test_acc"], "o-", color="#d62728", label="测试准确率")
    axes[0].plot(w["units"], w["train_acc"], "s--", color="#d62728", alpha=0.55,
                 label="训练准确率", markersize=4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("隐藏层神经元数")
    axes[0].set_ylabel("准确率")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("单隐藏层宽度的影响")
    for x, yv in zip(w["units"], w["test_acc"]):
        axes[0].annotate(f"{yv:.3f}", (x, yv), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7.5)

    d = hp["depth"]
    labels = [f"{n} 层\n" + "-".join(str(s) for s in sz[1:-1]) for n, sz in zip(d["depths"], d["sizes"])]
    axes[1].bar(range(len(labels)), d["test_acc"], color="#4c72b0", width=0.6)
    for i, v in enumerate(d["test_acc"]):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylim(0.90, 0.97)
    axes[1].text(0.02, 0.905, "纵轴起点 0.90", transform=axes[1].transAxes, fontsize=7,
                 color="#555555")
    axes[1].set_ylabel("测试准确率")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("深度的影响（每层 64 个单元）")

    all_params = list(hp["width"]["params"]) + list(hp["depth"]["params"])
    all_acc = list(hp["width"]["test_acc"]) + list(hp["depth"]["test_acc"])
    axes[2].scatter(hp["width"]["params"], hp["width"]["test_acc"], color="#d62728",
                    marker="o", label="改变宽度")
    axes[2].scatter(hp["depth"]["params"], hp["depth"]["test_acc"], color="#4c72b0",
                    marker="^", label="改变深度")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("可训练参数个数")
    axes[2].set_ylabel("测试准确率")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=8)
    axes[2].set_title("参数量与准确率")
    fig.suptitle("容量实验：MNIST 1 万张训练样本，ReLU + Adam，20 轮", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig06_capacity.png")

    # 图 7：激活函数
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {"relu": "#d62728", "tanh": "#1f77b4", "sigmoid": "#2ca02c"}
    for name, hist, acc in zip(hp["activation"]["names"], hp["activation"]["history"],
                               hp["activation"]["test_acc"]):
        ax.plot([r["epoch"] for r in hist], [r["test_acc"] for r in hist],
                color=colors[name], label=f"{name}（最终 {acc:.4f}）", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("测试准确率")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("激活函数对比：128 单元单隐藏层，20 轮", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig07_activation.png")

    # 图 8：学习率
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    lrs = hp["lr"]["lrs"]
    axes[0].plot(lrs, hp["lr"]["test_acc"], "o-", color="#d62728")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("学习率（对数坐标）")
    axes[0].set_ylabel("测试准确率")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title("学习率对最终准确率的影响")
    for x, y in zip(lrs, hp["lr"]["test_acc"]):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7.5)
    for lr, hist, col in zip(lrs, hp["lr"]["history"],
                             plt.cm.viridis(np.linspace(0, 0.9, len(lrs)))):
        axes[1].plot([r["epoch"] for r in hist], [r["test_loss"] for r in hist],
                     color=col, label=f"lr={lr:g}", linewidth=1.4)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("测试交叉熵")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7.5, ncol=2)
    axes[1].set_title("不同学习率的收敛过程")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig08_lr.png")

    # 图 9：初始化
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors2 = {"zeros": "#7f7f7f", "small": "#ff7f0e", "xavier": "#1f77b4", "he": "#2ca02c"}
    for name, hist, acc in zip(hp["init"]["names"], hp["init"]["history"],
                               hp["init"]["test_acc"]):
        axes[0].plot([r["epoch"] for r in hist], [r["test_acc"] for r in hist],
                     color=colors2[name], label=f"{name}（{acc:.4f}）", linewidth=1.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("测试准确率")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8.5)
    axes[0].set_title("初始化方式：测试准确率")
    axes[1].bar(range(len(hp["init"]["names"])), hp["init"]["test_acc"],
                color=[colors2[n] for n in hp["init"]["names"]], width=0.6)
    for i, v in enumerate(hp["init"]["test_acc"]):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    axes[1].set_xticks(range(len(hp["init"]["names"])))
    axes[1].set_xticklabels(hp["init"]["names"])
    axes[1].set_ylim(min(hp["init"]["test_acc"]) - 0.03, min(1.0, max(hp["init"]["test_acc"]) + 0.02))
    axes[1].set_ylabel("测试准确率")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("最终准确率")
    fig.suptitle("初始化方式对比：ReLU 网络，128 单元，20 轮", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig09_init.png")

    # 图 10：批大小与 L2
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    b = hp["batch"]
    xs = list(range(len(b["sizes"])))
    axes[0].plot(xs, b["test_acc"], "o-", color="#d62728")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([str(s) for s in b["sizes"]])
    axes[0].set_xlabel("批大小（最后一档为全批量）")
    axes[0].set_ylabel("测试准确率")
    axes[0].grid(alpha=0.3)
    for x, y in zip(xs, b["test_acc"]):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_title("批大小的影响（均为 20 轮）")

    l2 = hp["l2"]
    idx = np.arange(len(l2["l2"]))
    axes[1].plot(idx, l2["test_acc"], "o-", color="#d62728", label="测试准确率")
    axes[1].plot(idx, l2["train_acc"], "s--", color="#1f77b4", alpha=0.7,
                 label="训练准确率", markersize=4)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(["0"] + [f"{v:g}" for v in l2["l2"][1:]])
    axes[1].set_xlabel("L2 系数 λ")
    axes[1].set_ylabel("准确率")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_title(f"L2 正则（{l2['n_train']} 张训练图，30 轮）")
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig10_batch_l2.png")

    # 图 11：优化器
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors3 = {"sgd": "#ff7f0e", "momentum": "#1f77b4", "adam": "#2ca02c"}
    for name, lr, hist, acc in zip(hp["optimizer"]["names"], hp["optimizer"]["lrs"],
                                   hp["optimizer"]["history"], hp["optimizer"]["test_acc"]):
        ax.plot([r["epoch"] for r in hist], [r["train_loss"] for r in hist],
                color=colors3[name], label=f"{name} (lr={lr:g}, 测试 {acc:.4f})",
                linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("训练交叉熵")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    ax.set_title("优化器对比：SGD / Momentum / Adam（20 轮）", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig11_optimizer.png")


# ---------------------------------------------------------------------------
# 第 5 部分：MNIST 正式实验（全量训练集）
# ---------------------------------------------------------------------------

MNIST_BIG_MLP = [784, 256, 128, 10]


def part5_mnist(Xtr, ytr, Xte, yte):
    print("\n=== 第 5 部分：MNIST 全量训练集 ===")
    out = {"epochs": 20, "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "mlp_sizes": MNIST_BIG_MLP}

    lin = SoftmaxClassifier(784, 10, seed=SEED)
    r_lin = run(lin, Xtr, ytr, Xte, yte, 20, 128, 1e-3)
    print(f"  线性分类器 test={r_lin['test_acc']:.4f}")

    mlp = MLP(MNIST_BIG_MLP, activation="relu", init="he", seed=SEED)
    r_mlp = run(mlp, Xtr, ytr, Xte, yte, 20, 128, 1e-3)
    print(f"  MLP {MNIST_BIG_MLP} test={r_mlp['test_acc']:.4f}")

    for tag, r in [("linear", r_lin), ("mlp", r_mlp)]:
        out[tag] = {k: r[k] for k in ["train_acc", "test_acc", "train_loss",
                                      "test_loss", "time", "params"]}
        out[tag]["history"] = r["history"]

    # 混淆矩阵与每类指标
    pred_lin = lin.predict(Xte)
    pred_mlp = mlp.predict(Xte)
    out["linear"]["per_class"] = per_class_report(yte, pred_lin, 10)
    out["mlp"]["per_class"] = per_class_report(yte, pred_mlp, 10)
    out["linear"]["confusion"] = confusion_matrix(yte, pred_lin, 10).tolist()
    out["mlp"]["confusion"] = confusion_matrix(yte, pred_mlp, 10).tolist()
    out["mlp"]["top_confusions"] = top_confusions(yte, pred_mlp, 10, k=8)

    # 图 12：MNIST 概览
    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
    ax_samples = fig.add_subplot(gs[0, 0])
    ax_samples.axis("off")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xte), 16, replace=False)
    grid = np.zeros((4 * 28, 4 * 28))
    for k, i in enumerate(idx):
        r, c = divmod(k, 4)
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = Xte[i].reshape(28, 28)
    ax_samples.imshow(grid, cmap="gray_r")
    ax_samples.set_title("测试集样本（4×4）", fontsize=10)

    ax_loss = fig.add_subplot(gs[0, 1])
    for r, tag, col in [(r_lin, "线性分类器", "#1f77b4"), (r_mlp, "MLP(256-128)", "#d62728")]:
        ax_loss.plot([h["epoch"] for h in r["history"]], [h["train_loss"] for h in r["history"]],
                     color=col, label=f"{tag} 训练")
        ax_loss.plot([h["epoch"] for h in r["history"]], [h["test_loss"] for h in r["history"]],
                     color=col, linestyle="--", alpha=0.7, label=f"{tag} 测试")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("交叉熵")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=7)
    ax_loss.set_title("训练 / 测试损失", fontsize=10)

    ax_acc = fig.add_subplot(gs[0, 2])
    for r, tag, col in [(r_lin, "线性分类器", "#1f77b4"), (r_mlp, "MLP(256-128)", "#d62728")]:
        ax_acc.plot([h["epoch"] for h in r["history"]], [h["test_acc"] for h in r["history"]],
                    color=col, label=tag)
        ax_acc.plot([h["epoch"] for h in r["history"]], [h["train_acc"] for h in r["history"]],
                    color=col, linestyle="--", alpha=0.6)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("准确率")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8)
    ax_acc.set_title("训练（虚线）/ 测试（实线）准确率", fontsize=10)

    ax_cm1 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix(ax_cm1, np.array(out["linear"]["confusion"]), title="线性分类器混淆矩阵",
                          normalize=True)
    ax_cm2 = fig.add_subplot(gs[1, 1])
    plot_confusion_matrix(ax_cm2, np.array(out["mlp"]["confusion"]), title="MLP 混淆矩阵",
                          normalize=True)
    ax_bar = fig.add_subplot(gs[1, 2])
    per = out["mlp"]["per_class"]
    x = np.arange(10)
    ax_bar.bar(x - 0.2, [p["precision"] for p in per], width=0.4, label="precision", color="#4c72b0")
    ax_bar.bar(x + 0.2, [p["recall"] for p in per], width=0.4, label="recall", color="#dd8452")
    ax_bar.set_xticks(x)
    ax_bar.set_ylim(0.85, 1.005)
    ax_bar.set_xlabel("数字")
    ax_bar.grid(alpha=0.3, axis="y")
    ax_bar.legend(fontsize=8)
    ax_bar.set_title("MLP 每类 precision / recall", fontsize=10)
    fig.suptitle(f"MNIST 结果：线性分类器 {out['linear']['test_acc']:.3f} vs "
                 f"MLP {out['mlp']['test_acc']:.3f}", fontsize=13)
    savefig(fig, FIG_DIR / "fig12_mnist_overview.png")

    # 图 13：单独的大混淆矩阵（便于阅读）
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    plot_confusion_matrix(axes[0], np.array(out["linear"]["confusion"]),
                          title=f"线性分类器（准确率 {out['linear']['test_acc']:.4f}）", normalize=True)
    plot_confusion_matrix(axes[1], np.array(out["mlp"]["confusion"]),
                          title=f"MLP（准确率 {out['mlp']['test_acc']:.4f}）", normalize=True)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig13_mnist_confusion.png")

    # 图 14：MLP 错分样本
    wrong = np.flatnonzero(pred_mlp != yte)
    rng.shuffle(wrong)
    sel = wrong[:16]
    fig = plt.figure(figsize=(11.0, 3.2))
    for k, i in enumerate(sel):
        ax = fig.add_subplot(2, 8, k + 1)
        ax.imshow(Xte[i].reshape(28, 28), cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"{yte[i]}→{pred_mlp[i]}", fontsize=9, color="crimson")
    fig.suptitle("MLP 分错的测试样本（标题为 真实→预测）", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig14_mnist_errors.png")

    # 图 15：第一层权重（每个隐藏单元 28x28 的模板）
    W1 = mlp.Ws[0]
    norms = np.linalg.norm(W1, axis=0)
    top = np.argsort(norms)[::-1][:32]
    vmax = np.abs(W1[:, top]).max()
    fig, axes = plt.subplots(4, 8, figsize=(11.0, 6.0))
    for ax, j in zip(axes.ravel(), top):
        ax.imshow(W1[:, j].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"#{j}", fontsize=7)
    fig.suptitle("MLP 第一层权重（按范数取前 32 个单元；红正蓝负）", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig15_mnist_filters.png")

    # 图 16：原始像素 PCA vs 第一隐藏层表示 PCA
    Xs_, ys_ = stratified_subsample(Xte, yte, 3000, seed=3)
    # 重新取第一层激活
    Z1 = Xs_ @ mlp.Ws[0] + mlp.bs[0]
    H1 = np.maximum(Z1, 0.0)
    Z2 = H1 @ mlp.Ws[1] + mlp.bs[1]
    H2 = np.maximum(Z2, 0.0)

    def pca2(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    for ax, Xrep, title in [
        (axes[0], pca2(Xs_), "原始像素的 PCA 前两维"),
        (axes[1], pca2(H2), "MLP 第二隐藏层（128 维）的 PCA 前两维"),
    ]:
        for c in range(10):
            m = ys_ == c
            ax.scatter(Xrep[m, 0], Xrep[m, 1], s=4, color=CLASS_COLORS[c], alpha=0.7,
                       label=str(c))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        ax.set_aspect("equal")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   markerscale=2.0, title="数字", title_fontsize=7)
    fig.suptitle("隐藏层把原始像素映射到更易线性分开的表示", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fig16_hidden_pca.png")

    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--out", default="results.json",
                        help="结果文件名（默认 results.json，可指定分片文件）")
    args = parser.parse_args()
    parts = set(args.parts)

    t_start = time.perf_counter()
    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seed": SEED,
        }
    }

    if "0" in parts:
        results["gradcheck"] = part0_gradcheck()
    if "1" in parts:
        results["part1_2d"] = part1_2d()
    if "2" in parts:
        results["part2_size_2d"] = part2_data_size_2d()

    need_mnist = bool(parts & {"3", "4", "5"})
    if need_mnist:
        print("\n加载 MNIST ...")
        Xtr, ytr, Xte, yte = load_mnist(DATA_DIR)
        print(f"  train={Xtr.shape}, test={Xte.shape}")
    if "3" in parts:
        results["part3_size_mnist"] = part3_data_size_mnist(Xtr, ytr, Xte, yte)
    if "4" in parts:
        results["part4_hyper"] = part4_hyperparams(Xtr, ytr, Xte, yte)
    if "5" in parts:
        results["part5_mnist"] = part5_mnist(Xtr, ytr, Xte, yte)

    # 如果只跑了部分，合并已有的 results.json，避免覆盖
    out_path = HERE / args.out
    if parts != {"0", "1", "2", "3", "4", "5"} and out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        old.update(results)
        results = old

    out_path.write_text(json.dumps(to_py(results), ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n结果已写入 {out_path}，总耗时 {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
