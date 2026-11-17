#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验五主实验脚本：用 GPU 上的 PyTorch 训练卷积神经网络（LeNet-5 及其变体）。

实验分九部分，全部由本脚本一次运行完成，结果写入 results.json，插图写入 figures/：

    1. 数据概览与 LeNet-5 结构图
    2. 基线对比：LeNet-5 经典配方 / 现代配方 / +BN / SmallVGG / MLP
    3. 激活函数对比：ReLU、Sigmoid、Tanh、LeakyReLU、ELU、GELU（含第一层梯度范数）
    4. Dropout 与正则化：小数据过拟合场景下扫描丢弃率与作用位置，对比权重衰减
    5. 数据量：600 → 54,001 张训练图，CNN vs MLP 学习曲线
    6. 超参数：优化器、学习率、批大小、权重衰减
    7. 数据增强：旋转/平移/随机擦除，以及旋转鲁棒性测试
    8. 最终模型：候选配方对比 → 3 个随机种子训练 → 混淆矩阵/错分样本/卷积核/特征图/置信度
    9. Fashion-MNIST 上的验证与样本展示

运行方式：
    cd exp5_report && python full_experiment.py              # 全部实验
    cd exp5_report && python full_experiment.py --parts 1 2  # 只跑指定部分（增量写入 results.json）

GPU 要求：PyTorch + CUDA（本实验在 RTX 4050 Laptop 上运行）；没有 GPU 时自动退回 CPU。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import viz
from data import (
    FASHION_CLASSES,
    MNIST_CLASSES,
    describe,
    load_raw_dataset,
    make_loader,
    make_train_transform,
    rotate_batch,
    stratified_split_indices,
    stratified_subset_indices,
)
from engine import environment_report, evaluate, fit, get_device, set_seed
from metrics import confidence_stats, confusion_matrix, per_class_report, top_confusions
from models import LeNet5, MLP, SmallVGG, build_model, count_parameters

FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_PATH = HERE / "results.json"
SEED = 0
N_VAL = 6000
BATCH = 64
DEVICE = get_device()
viz.set_fig_dir(FIG_DIR)

# 结果里所有浮点数保留 6 位，避免 JSON 过大
def fnum(x):
    return round(float(x), 6)


def to_py(obj):
    """递归地把 numpy / torch 标量转成可 JSON 序列化的 Python 对象。"""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


# ---------------------------------------------------------------------------
# 数据与公共工具
# ---------------------------------------------------------------------------


class DataBundle:
    """把 MNIST 的原始张量与固定划分打包在一起，供各实验复用。"""

    def __init__(self, name: str = "mnist", n_val: int = N_VAL, seed: int = SEED):
        self.name = name
        self.Xtr, self.ytr, self.Xte, self.yte = load_raw_dataset(name)
        self.tr_idx, self.va_idx = stratified_split_indices(self.ytr, n_val, seed=seed)
        self.class_names = MNIST_CLASSES if name == "mnist" else FASHION_CLASSES
        self.n_classes = int(self.ytr.max().item()) + 1

    def loaders(self, n_train: int | None = None, batch: int = BATCH,
                aug: str = "none", num_workers: int = 4, seed: int = SEED):
        """构造 train / val / test 三个 DataLoader。

        n_train 为 None 时用全部 54,001 张训练图；否则从训练池里分层抽样。
        """
        tr_idx = self.tr_idx
        if n_train is not None and n_train < len(tr_idx):
            sub = stratified_subset_indices(self.ytr[tr_idx], n_train, seed=seed)
            tr_idx = tr_idx[sub]
        train = make_loader(
            self.Xtr[tr_idx], self.ytr[tr_idx], batch_size=batch, shuffle=True,
            transform=make_train_transform(aug), num_workers=num_workers, seed=seed,
        )
        val = make_loader(self.Xtr[self.va_idx], self.ytr[self.va_idx], batch_size=512,
                          shuffle=False, num_workers=2)
        test = make_loader(self.Xte, self.yte, batch_size=512, shuffle=False, num_workers=2)
        return {"train": train, "val": val, "test": test,
                "n_train": len(tr_idx), "n_val": len(self.va_idx)}

    def test_loader(self):
        return make_loader(self.Xte, self.yte, batch_size=512, shuffle=False, num_workers=2)


def run_experiment(label: str, model: torch.nn.Module, loaders: dict, *,
                   epochs: int, optimizer: str = "adam", lr: float = 1e-3,
                   weight_decay: float = 0.0, seed: int = SEED,
                   track_grad: bool = False, scheduler: str | None = None,
                   extra_config: dict | None = None) -> dict:
    """训练一个配置并返回完整结果（含测试集评估）。"""
    t0 = time.perf_counter()
    res = fit(model, loaders["train"], loaders["val"], epochs=epochs, optimizer=optimizer,
              lr=lr, weight_decay=weight_decay, device=DEVICE, seed=seed,
              track_grad_norm=track_grad, scheduler=scheduler, log_prefix=f"[{label}]")
    te = evaluate(model, loaders["test"], DEVICE, return_details=True)
    config = {
        "epochs": epochs, "optimizer": optimizer, "lr": lr,
        "weight_decay": weight_decay, "batch_size": loaders["train"].batch_size,
        "n_train": loaders["n_train"], "seed": seed,
        "params": res["params"], "scheduler": scheduler,
    }
    if extra_config:
        config.update(extra_config)
    out = {
        "label": label,
        "config": config,
        "history": res["history"],
        "best_val_acc": fnum(res["best_val_acc"]),
        "best_epoch": int(res["best_epoch"]),
        "test_acc": fnum(te["acc"]),
        "test_loss": fnum(te["loss"]),
        "train_time": fnum(time.perf_counter() - t0),
        "params": int(res["params"]),
    }
    if track_grad and "grad_norm_first_conv" in res:
        out["grad_norm_first_conv"] = [fnum(g) for g in res["grad_norm_first_conv"]]
    print(f"  -> {label}: val {res['best_val_acc'] * 100:.2f}%  "
          f"test {te['acc'] * 100:.2f}%  ({out['train_time']:.1f}s)\n", flush=True)
    return out


# ---------------------------------------------------------------------------
# 第 1 部分：数据概览与结构图
# ---------------------------------------------------------------------------


def part1_data_overview(lib: dict) -> dict:
    print("\n[part 1] 数据概览与结构图", flush=True)
    D: DataBundle = lib["mnist"]
    set_seed(SEED)

    # 每类两张样本，按 2 行 × 10 列排列
    fig, axes = plt.subplots(2, 10, figsize=(10.5, 2.8))
    g = torch.Generator().manual_seed(0)
    for c in range(10):
        idx = torch.nonzero(D.ytr == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)[:2]]
        for r in range(2):
            axes[r, c].imshow(D.Xtr[idx[r], 0], cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(D.class_names[c], fontsize=10)
    fig.suptitle("MNIST 手写数字样本（每类 2 张）", y=1.02)
    viz.savefig(fig, "fig01_mnist_samples.png")

    # LeNet-5 结构示意图
    viz.plot_architecture("fig02_lenet5_arch.png")

    return {
        "dataset": describe("mnist"),
        "split": {"n_train_pool": len(D.tr_idx), "n_val": len(D.va_idx),
                  "n_test": len(D.yte), "n_train_full": len(D.tr_idx)},
        "class_names": D.class_names,
    }


# ---------------------------------------------------------------------------
# 第 2 部分：基线模型对比
# ---------------------------------------------------------------------------


def part2_baseline(lib: dict) -> dict:
    print("\n[part 2] 基线对比（全量训练集，10 轮）", flush=True)
    D: DataBundle = lib["mnist"]
    loaders = D.loaders(n_train=None, aug="none")
    specs = [
        ("LeNet-5 经典结构 (Sigmoid+AvgPool)", "lenet",
         dict(activation="sigmoid", pooling="avg"), "adam", 1e-3),
        ("LeNet-5 现代结构 (ReLU+MaxPool)", "lenet",
         dict(activation="relu", pooling="max"), "adam", 1e-3),
        ("LeNet-5 + BN", "lenet", dict(activation="relu", pooling="max", use_bn=True),
         "adam", 1e-3),
        ("SmallVGG (更深+BN+Dropout)", "vgg", dict(dropout_fc=0.5), "adam", 1e-3),
        ("MLP 基线", "mlp", dict(hidden=(256, 128)), "adam", 1e-3),
    ]
    runs = {}
    for label, arch, kw, opt, lr in specs:
        set_seed(SEED)
        model = LeNet5(**kw) if arch == "lenet" else (SmallVGG(**kw) if arch == "vgg" else MLP(**kw))
        runs[label] = run_experiment(label, model, loaders, epochs=10, optimizer=opt,
                                     lr=lr, seed=SEED)

    # 图 3：验证准确率与训练损失曲线
    viz.plot_curves({k: v["history"] for k, v in runs.items()},
                    ("val_acc", "train_loss"), "fig03_baseline_curves.png")
    # 图 4：最终测试准确率
    viz.plot_grouped_final(
        [{"label": k, "value": v["test_acc"]} for k, v in runs.items()],
        "fig04_baseline_final.png", title="基线模型测试准确率（全量训练集，10 轮）",
        ylim=(0.92, 1.005), rotate=15,
    )
    return {"order": [s[0] for s in specs], "runs": runs}


# ---------------------------------------------------------------------------
# 第 3 部分：激活函数对比
# ---------------------------------------------------------------------------


def part3_activation(lib: dict) -> dict:
    print("\n[part 3] 激活函数对比（10,000 张训练图，8 轮）", flush=True)
    D: DataBundle = lib["mnist"]
    loaders = D.loaders(n_train=10000, aug="none")
    act_names = ["relu", "sigmoid", "tanh", "leaky_relu", "elu", "gelu"]
    runs = {}
    for act in act_names:
        set_seed(SEED)
        model = LeNet5(activation=act, pooling="max")
        label = act
        runs[label] = run_experiment(label, model, loaders, epochs=8, optimizer="adam",
                                     lr=1e-3, seed=SEED, track_grad=True,
                                     extra_config={"activation": act, "pooling": "max"})

    viz.plot_curves({k: v["history"] for k, v in runs.items()},
                    ("val_acc",), "fig05_activation_curves.png")

    # 图 6：最终测试准确率 + 第一层卷积权重的梯度范数
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    labels = list(runs)
    colors = [viz.PALETTE[i % len(viz.PALETTE)] for i in range(len(labels))]
    axes[0].bar(range(len(labels)), [runs[k]["test_acc"] for k in labels], color=colors)
    for i, k in enumerate(labels):
        axes[0].text(i, runs[k]["test_acc"] + 0.0008, f"{runs[k]['test_acc'] * 100:.2f}%",
                     ha="center", fontsize=8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylim(0.93, 1.0)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("测试准确率（10,000 张训练图，8 轮）")
    for i, k in enumerate(labels):
        g = runs[k].get("grad_norm_first_conv", [])
        if g:
            axes[1].plot(range(1, len(g) + 1), g, marker="o", markersize=3,
                         color=colors[i], label=k)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练轮次 (epoch)")
    axes[1].set_ylabel("第一层卷积权重梯度范数（对数尺度）")
    axes[1].set_title("第一层卷积的梯度大小（越小越接近梯度消失）")
    axes[1].legend(fontsize=8)
    viz.savefig(fig, "fig06_activation_grad.png")

    # 3.2 梯度消失诊断：固定同一个 LeNet-5 结构，只改「激活函数 + 优化器 + 学习率」
    #     经典配方（Sigmoid + 平均池化）在 SGD 下几乎学不动，这里把第一层卷积的
    #     梯度范数量出来，说明问题出在梯度量级而不是数据或代码。
    print("  [3.2] 梯度消失诊断（10,000 训练图，6 轮）", flush=True)
    diag_specs = [
        ("Sigmoid+AvgPool | SGD 0.01", dict(activation="sigmoid", pooling="avg"), "sgd", 0.01),
        ("Sigmoid+AvgPool | SGD 0.1", dict(activation="sigmoid", pooling="avg"), "sgd", 0.1),
        ("Sigmoid+AvgPool | Adam 1e-3", dict(activation="sigmoid", pooling="avg"), "adam", 1e-3),
        ("ReLU+MaxPool | SGD 0.01", dict(activation="relu", pooling="max"), "sgd", 0.01),
        ("ReLU+MaxPool | Adam 1e-3", dict(activation="relu", pooling="max"), "adam", 1e-3),
    ]
    diag_runs = {}
    for label, kw, opt, lr in diag_specs:
        set_seed(SEED)
        model = LeNet5(**kw)
        diag_runs[label] = run_experiment(label, model, loaders, epochs=6, optimizer=opt,
                                          lr=lr, seed=SEED, track_grad=True,
                                          extra_config=dict(kw))
    labels = list(diag_runs)
    colors = [viz.PALETTE[i % len(viz.PALETTE)] for i in range(len(labels))]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].barh(range(len(labels)), [diag_runs[k]["test_acc"] for k in labels], color=colors)
    for i, k in enumerate(labels):
        axes[0].text(diag_runs[k]["test_acc"] + 0.005, i,
                     f"{diag_runs[k]['test_acc'] * 100:.2f}%", va="center", fontsize=8)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlim(0, 1.0)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("测试准确率（6 轮）")
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, k in enumerate(labels):
        g = diag_runs[k].get("grad_norm_first_conv", [])
        if g:
            axes[1].plot(range(1, len(g) + 1), g, marker="o", markersize=3,
                         linestyle=styles[i % len(styles)], color=colors[i], label=k)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练轮次 (epoch)")
    axes[1].set_ylabel("第一层卷积权重梯度范数（对数尺度）")
    axes[1].set_title("第一层卷积权重的梯度量级（对数尺度）")
    axes[1].legend(fontsize=7)
    viz.savefig(fig, "fig07_vanishing_grad.png")
    return {"order": act_names, "runs": runs,
            "grad_diagnosis": {"order": labels, "runs": diag_runs}}


# ---------------------------------------------------------------------------
# 第 4 部分：Dropout 与正则化
# ---------------------------------------------------------------------------


def part4_dropout(lib: dict) -> dict:
    print("\n[part 4] Dropout / 正则化对比（6,000 张训练图，20 轮，故意过拟合）", flush=True)
    D: DataBundle = lib["mnist"]
    loaders = D.loaders(n_train=6000, aug="none")
    specs = [
        ("无正则化", dict(dropout_fc=0.0), 0.0),
        ("Dropout 0.2 (FC)", dict(dropout_fc=0.2), 0.0),
        ("Dropout 0.5 (FC)", dict(dropout_fc=0.5), 0.0),
        ("Dropout 0.5 (FC) + Dropout2d 0.25", dict(dropout_fc=0.5, dropout_conv=0.25), 0.0),
        ("Weight Decay 1e-3（无 Dropout）", dict(dropout_fc=0.0), 1e-3),
        ("Dropout 0.5 + Weight Decay 1e-3", dict(dropout_fc=0.5), 1e-3),
    ]
    runs = {}
    for label, kw, wd in specs:
        set_seed(SEED)
        model = LeNet5(activation="relu", pooling="max", fc_sizes=(512, 128), **kw)
        runs[label] = run_experiment(label, model, loaders, epochs=20, optimizer="adam",
                                     lr=1e-3, weight_decay=wd, seed=SEED,
                                     extra_config={"dropout_fc": kw["dropout_fc"],
                                                   "dropout_conv": kw.get("dropout_conv", 0.0)})

    # 图 7：训练/验证曲线（只看 Dropout 0 / 0.2 / 0.5 三条，图才不挤）
    show = ["无正则化", "Dropout 0.2 (FC)", "Dropout 0.5 (FC)"]
    viz.plot_curves({k: runs[k]["history"] for k in show},
                    ("train_acc", "val_acc"), "fig08_dropout_curves.png")

    # 图 8：测试准确率与"训练-验证差距"
    labels = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    colors = [viz.PALETTE[i % len(viz.PALETTE)] for i in range(len(labels))]
    axes[0].barh(range(len(labels)), [runs[k]["test_acc"] for k in labels], color=colors)
    for i, k in enumerate(labels):
        axes[0].text(runs[k]["test_acc"] - 0.003, i, f"{runs[k]['test_acc'] * 100:.2f}%",
                     va="center", ha="right", fontsize=8, color="white")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlim(0.96, 1.0)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("测试准确率（6,000 张训练图，20 轮）")
    gaps = []
    for k in labels:
        h = runs[k]["history"]
        # 最后一轮相对"验证最优轮"的泛化差距：用最后 epoch 训练准确率 - 验证准确率
        gaps.append(h["train_acc"][-1] - h["val_acc"][-1])
    axes[1].barh(range(len(labels)), gaps, color=colors)
    for i, g in enumerate(gaps):
        axes[1].text(g + 0.002, i, f"{g * 100:+.2f}%", va="center", fontsize=8)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("最后一轮训练准确率 − 验证准确率（越大越过拟合）")
    axes[1].set_title("过拟合程度")
    viz.savefig(fig, "fig09_dropout_summary.png")
    return {"order": labels, "runs": runs}


# ---------------------------------------------------------------------------
# 第 5 部分：数据量
# ---------------------------------------------------------------------------


def part5_data_size(lib: dict) -> dict:
    print("\n[part 5] 数据量实验（CNN vs MLP，8 轮）", flush=True)
    D: DataBundle = lib["mnist"]
    sizes = [600, 2000, 6000, 20000, len(D.tr_idx)]
    cnn_runs, mlp_runs = [], []
    for n in sizes:
        loaders = D.loaders(n_train=n, aug="none")
        set_seed(SEED)
        cnn = LeNet5(activation="relu", pooling="max")
        cnn_runs.append(run_experiment(f"CNN n={n}", cnn, loaders, epochs=8, seed=SEED,
                                       extra_config={"n_train": loaders["n_train"]}))
        set_seed(SEED)
        mlp = MLP(hidden=(256, 128))
        mlp_runs.append(run_experiment(f"MLP n={n}", mlp, loaders, epochs=8, seed=SEED,
                                       extra_config={"n_train": loaders["n_train"]}))

    xs = [r["config"]["n_train"] for r in cnn_runs]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].plot(xs, [r["test_acc"] for r in cnn_runs], marker="o", label="CNN (LeNet-5)",
                 color=viz.PALETTE[0])
    axes[0].plot(xs, [r["test_acc"] for r in mlp_runs], marker="s", label="MLP (784-256-128)",
                 color=viz.PALETTE[1])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("训练样本数（对数尺度）")
    axes[0].set_ylabel("测试准确率")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("学习曲线：数据量 vs 测试准确率")
    axes[0].legend()
    gaps = [c["test_acc"] - m["test_acc"] for c, m in zip(cnn_runs, mlp_runs)]
    axes[1].plot(xs, gaps, marker="D", color=viz.PALETTE[2])
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("训练样本数（对数尺度）")
    axes[1].set_ylabel("CNN 准确率 − MLP 准确率")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("CNN 相对 MLP 的优势")
    viz.savefig(fig, "fig10_data_size.png")
    return {"sizes": sizes, "cnn": cnn_runs, "mlp": mlp_runs}


# ---------------------------------------------------------------------------
# 第 6 部分：超参数
# ---------------------------------------------------------------------------


def part6_hyper(lib: dict) -> dict:
    print("\n[part 6] 超参数对比（10,000 张训练图，8 轮）", flush=True)
    D: DataBundle = lib["mnist"]
    loaders = D.loaders(n_train=10000, aug="none")
    out: dict = {"optimizer": {}, "adam_lr": {}, "sgd_lr": {}, "batch": {}, "weight_decay": {}}

    def fresh():
        return LeNet5(activation="relu", pooling="max")

    # 6.1 优化器
    for opt, lr, tag in [("sgd", 0.01, "SGD lr=0.01"),
                         ("momentum", 0.01, "SGD+Momentum lr=0.01"),
                         ("adam", 1e-3, "Adam lr=1e-3"),
                         ("adamw", 1e-3, "AdamW lr=1e-3 wd=1e-2")]:
        set_seed(SEED)
        wd = 1e-2 if opt == "adamw" else 0.0
        out["optimizer"][tag] = run_experiment(tag, fresh(), loaders, epochs=8, optimizer=opt,
                                               lr=lr, weight_decay=wd, seed=SEED,
                                               extra_config={"optimizer": opt})

    # 6.2 学习率
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        set_seed(SEED)
        out["adam_lr"][f"Adam lr={lr:g}"] = run_experiment(
            f"Adam lr={lr:g}", fresh(), loaders, epochs=8, optimizer="adam", lr=lr,
            seed=SEED, extra_config={"optimizer": "adam"})
    for lr in [1e-3, 1e-2, 1e-1]:
        set_seed(SEED)
        out["sgd_lr"][f"SGD+Momentum lr={lr:g}"] = run_experiment(
            f"SGD+Momentum lr={lr:g}", fresh(), loaders, epochs=8, optimizer="momentum",
            lr=lr, seed=SEED, extra_config={"optimizer": "momentum"})

    # 6.3 批大小
    for bs in [16, 32, 64, 128, 256]:
        set_seed(SEED)
        ld = D.loaders(n_train=10000, batch=bs, aug="none")
        out["batch"][f"batch={bs}"] = run_experiment(
            f"batch={bs}", fresh(), ld, epochs=8, optimizer="adam", lr=1e-3, seed=SEED,
            extra_config={"batch_size": bs})

    # 6.4 权重衰减
    for wd in [0.0, 1e-4, 1e-3, 1e-2]:
        set_seed(SEED)
        out["weight_decay"][f"weight decay={wd:g}"] = run_experiment(
            f"weight decay={wd:g}", fresh(), loaders, epochs=8, optimizer="adam",
            lr=1e-3, weight_decay=wd, seed=SEED, extra_config={"weight_decay": wd})

    viz.plot_curves({k: v["history"] for k, v in out["optimizer"].items()},
                    ("val_acc",), "fig11_optimizer.png")
    viz.plot_grouped_final([{"label": k, "value": v["test_acc"]}
                            for k, v in out["optimizer"].items()],
                           "fig12_optimizer_final.png", ylim=(0.9, 1.0), rotate=10)

    # 图 12：学习率（Adam / SGD 两个子图）
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for i, (k, v) in enumerate(out["adam_lr"].items()):
        axes[0].plot(v["history"]["epoch"], v["history"]["val_acc"], marker="o", markersize=3,
                     color=viz.PALETTE[i], label=k)
    axes[0].set_title("Adam 学习率对验证准确率的影响")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("验证准确率")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(fontsize=8)
    for i, (k, v) in enumerate(out["sgd_lr"].items()):
        axes[1].plot(v["history"]["epoch"], v["history"]["val_acc"], marker="o", markersize=3,
                     color=viz.PALETTE[i], label=k)
    axes[1].set_title("SGD+Momentum 学习率对验证准确率的影响")
    axes[1].set_xlabel("训练轮次")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].legend(fontsize=8)
    viz.savefig(fig, "fig13_lr.png")

    # 图 13：批大小
    viz.plot_curves({k: v["history"] for k, v in out["batch"].items()},
                    ("val_acc",), "fig14_batch.png")
    # 图 14：权重衰减
    viz.plot_curves({k: v["history"] for k, v in out["weight_decay"].items()},
                    ("val_acc",), "fig15_weight_decay.png")
    return out


# ---------------------------------------------------------------------------
# 第 7 部分：数据增强与旋转鲁棒性
# ---------------------------------------------------------------------------


def eval_rotation_robustness(model: torch.nn.Module, D: DataBundle,
                             angles=(-30, -20, -10, 0, 10, 20, 30)) -> dict:
    """在旋转后的测试集上评估模型：模型自身没有旋转不变性，增强能不能补上？"""
    model.eval()
    accs = {}
    with torch.no_grad():
        for ang in angles:
            correct, total = 0, 0
            for i in range(0, len(D.Xte), 512):
                xb = rotate_batch(D.Xte[i:i + 512], ang).to(DEVICE)
                yb = D.yte[i:i + 512].to(DEVICE)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            accs[str(ang)] = correct / total
    return {"angles": list(angles), "acc": [accs[str(a)] for a in angles]}


def part7_augmentation(lib: dict) -> dict:
    print("\n[part 7] 数据增强与旋转鲁棒性（6,000 张训练图，20 轮）", flush=True)
    D: DataBundle = lib["mnist"]
    specs = [("无增强", "none"), ("旋转/平移 ±10°", "rotate10"),
             ("旋转/平移 ±20°", "rotate20"), ("±20° + 随机擦除", "rotate20_erase")]
    runs, models = {}, {}
    for label, aug in specs:
        loaders = D.loaders(n_train=6000, aug=aug)
        set_seed(SEED)
        model = LeNet5(activation="relu", pooling="max")
        runs[label] = run_experiment(label, model, loaders, epochs=20, seed=SEED,
                                     extra_config={"augmentation": aug})
        models[label] = model
        runs[label]["rotation"] = eval_rotation_robustness(model, D)

    viz.plot_curves({k: v["history"] for k, v in runs.items()},
                    ("val_acc", "train_loss"), "fig16_aug_curves.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (label, run) in enumerate(runs.items()):
        ax.plot(run["rotation"]["angles"], run["rotation"]["acc"], marker="o",
                color=viz.PALETTE[i], label=label)
    ax.set_xlabel("测试时把整张测试集旋转的角度（度）")
    ax.set_ylabel("测试准确率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("不同训练增强方式下，模型对旋转的鲁棒性")
    ax.legend(fontsize=8)
    viz.savefig(fig, "fig17_rotation.png")
    return {"order": [s[0] for s in specs], "runs": runs}


# ---------------------------------------------------------------------------
# 第 8 部分：最终模型与误差分析
# ---------------------------------------------------------------------------


def evaluate_predictions(model: torch.nn.Module, loader) -> dict:
    """在测试集上评估并整理误差分析所需的全部数据（混淆矩阵、每类指标、置信度等）。"""
    te = evaluate(model, loader, DEVICE, return_details=True)
    pred, label = te["pred"], te["label"]
    cm = confusion_matrix(label, pred, n_classes=10)
    stats = confidence_stats(te["prob"], pred, label)
    rows = per_class_report(cm)
    for r in rows:
        r["error_count"] = int(r["support"] - cm[r["class"], r["class"]])
    return {
        "test_acc": fnum(te["acc"]),
        "test_loss": fnum(te["loss"]),
        "n_errors": int((pred != label).sum()),
        "confusion": cm.tolist(),
        "per_class": rows,
        "top_confusions": top_confusions(cm, k=5),
        "confidence": {k: v for k, v in stats.items()
                       if k not in ("thresholds", "coverage", "precision_at_threshold")},
        "thresholds": stats["thresholds"],
        "coverage": stats["coverage"],
        "precision_at_threshold": stats["precision_at_threshold"],
    }


def part8_final(lib: dict) -> dict:
    print("\n[part 8] 最终模型：配方对比 + 3 个种子 + 误差分析", flush=True)
    D: DataBundle = lib["mnist"]

    # 8.1 候选配方：全量训练集、12 轮、单一种子，按验证集准确率选型
    recipes = [
        ("R1 LeNet-5 (ReLU+MaxPool)", "lenet",
         dict(activation="relu", pooling="max"), "none"),
        ("R2 R1 + Dropout 0.5", "lenet",
         dict(activation="relu", pooling="max", dropout_fc=0.5), "none"),
        ("R3 R1 + 增广 ±10°", "lenet",
         dict(activation="relu", pooling="max"), "rotate10"),
        ("R4 R2 + 增广 ±20°", "lenet",
         dict(activation="relu", pooling="max", dropout_fc=0.5), "rotate20"),
        ("R5 SmallVGG (更深+BN+Dropout)", "vgg", dict(dropout_fc=0.5), "none"),
    ]
    recipe_runs = {}
    for label, arch, kw, aug in recipes:
        loaders = D.loaders(n_train=None, aug=aug)
        set_seed(SEED)
        model = build_model(arch, **kw)
        recipe_runs[label] = run_experiment(label, model, loaders, epochs=12, seed=SEED,
                                            extra_config={"augmentation": aug, "arch": arch})
    best_label = max(recipe_runs, key=lambda k: recipe_runs[k]["best_val_acc"])
    best_arch, best_kw, best_aug = [
        (arch, kw, aug) for label, arch, kw, aug in recipes if label == best_label
    ][0]
    print(f"  >> 按验证集选择最终配方：{best_label}", flush=True)

    # 8.2 最终模型：同一配方训练 3 个随机种子，报告均值 ± 标准差
    seeds = [0, 1, 2]
    final_runs, final_models = [], {}
    for sd in seeds:
        loaders = D.loaders(n_train=None, aug=best_aug)
        set_seed(sd)
        model = build_model(best_arch, **best_kw)
        run = run_experiment(best_label, model, loaders, epochs=15, seed=sd,
                             extra_config={"augmentation": best_aug, "arch": best_arch})
        final_runs.append(run)
        final_models[sd] = model
    test_accs = [r["test_acc"] for r in final_runs]
    mean_acc = float(np.mean(test_accs))
    std_acc = float(np.std(test_accs, ddof=1))

    # 8.3 误差分析（用第 0 个种子的模型）
    model = final_models[seeds[0]]
    detail = evaluate_predictions(model, D.test_loader())
    detail.update({
        "recipe": best_label,
        "final_arch": best_arch,
        "final_aug": best_aug,
        "seeds": seeds,
        "per_seed_test_acc": test_accs,
        "mean_test_acc": fnum(mean_acc),
        "std_test_acc": fnum(std_acc),
        "recipe_runs": recipe_runs,
        "final_runs": final_runs,
        "params": int(count_parameters(model)),
    })

    # 图 18：最终模型 3 个种子的验证曲线
    viz.plot_curves({f"seed {sd}": r["history"] for sd, r in zip(seeds, final_runs)},
                    ("val_acc", "val_loss"), "fig18_final_curves.png")
    # 图 19：混淆矩阵
    viz.plot_confusion(np.array(detail["confusion"]), D.class_names, "fig19_confusion.png")
    # 图 20：每类 precision / recall / F1
    rows = detail["per_class"]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    x = np.arange(len(rows))
    w = 0.27
    for i, (key, name) in enumerate([("precision", "Precision"), ("recall", "Recall"),
                                     ("f1", "F1")]):
        ax.bar(x + (i - 1) * w, [r[key] for r in rows], width=w, label=name,
               color=viz.PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels(D.class_names)
    ax.set_xlabel("数字类别")
    vals = [r[k] for r in rows for k in ("precision", "recall", "f1")]
    ax.set_ylim(max(0.85, min(vals) - 0.02), 1.005)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("最终模型每类 Precision / Recall / F1（测试集）")
    ax.legend(fontsize=8)
    viz.savefig(fig, "fig20_per_class.png")

    # 图 21：错分样本；图 22：置信度分析
    te = evaluate(model, D.test_loader(), DEVICE, return_details=True)
    viz.plot_errors(D.Xte, te["label"], te["pred"], te["prob"].max(axis=1),
                    D.class_names, "fig21_errors.png")
    viz.plot_confidence({"conf_hist_correct": detail["confidence"]["conf_hist_correct"],
                         "conf_hist_wrong": detail["confidence"]["conf_hist_wrong"],
                         "coverage": detail["coverage"],
                         "precision_at_threshold": detail["precision_at_threshold"]},
                        "fig22_confidence.png")
    # 图 23 / 24：第一层卷积核 + 特征图
    conv1 = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    viz.plot_filters(conv1.weight, "fig23_filters.png")
    viz.plot_feature_maps(model, D.Xte[0:1], "fig24_feature_maps.png")

    # 8.4 最终模型对旋转的鲁棒性
    detail["rotation"] = eval_rotation_robustness(model, D)
    return detail


# ---------------------------------------------------------------------------
# 第 9 部分：Fashion-MNIST
# ---------------------------------------------------------------------------


def part9_fashion(lib: dict) -> dict:
    print("\n[part 9] Fashion-MNIST 验证（全量训练集，10 轮）", flush=True)
    Df: DataBundle = lib["fashion"]
    loaders = Df.loaders(n_train=None, aug="none")

    # 样本图
    fig, axes = plt.subplots(2, 10, figsize=(10.5, 2.8))
    g = torch.Generator().manual_seed(0)
    for c in range(10):
        idx = torch.nonzero(Df.ytr == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)[:2]]
        for r in range(2):
            axes[r, c].imshow(Df.Xtr[idx[r], 0], cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(Df.class_names[c], fontsize=9)
    fig.suptitle("Fashion-MNIST 服装样本（每类 2 张）", y=1.02)
    viz.savefig(fig, "fig25_fashion_samples.png")

    set_seed(SEED)
    cnn = run_experiment("CNN (LeNet-5)", LeNet5(activation="relu", pooling="max"),
                         loaders, epochs=10, seed=SEED)
    set_seed(SEED)
    mlp = run_experiment("MLP", MLP(hidden=(256, 128)), loaders, epochs=10, seed=SEED)
    viz.plot_grouped_final([{"label": "CNN (LeNet-5)", "value": cnn["test_acc"]},
                            {"label": "MLP", "value": mlp["test_acc"]}],
                           "fig26_fashion_compare.png", ylim=(0.8, 1.0))
    return {"dataset": describe("fashion"), "cnn": cnn, "mlp": mlp}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

PART_FUNCS = {
    1: part1_data_overview,
    2: part2_baseline,
    3: part3_activation,
    4: part4_dropout,
    5: part5_data_size,
    6: part6_hyper,
    7: part7_augmentation,
    8: part8_final,
    9: part9_fashion,
}
PART_NAMES = {
    1: "data_overview", 2: "baseline", 3: "activation", 4: "dropout", 5: "data_size",
    6: "hyper", 7: "augmentation", 8: "final", 9: "fashion",
}


def main():
    ap = argparse.ArgumentParser(description="实验五：卷积神经网络实验")
    ap.add_argument("--parts", type=int, nargs="+", default=list(PART_FUNCS),
                    help="要运行的部分编号，默认全部")
    ap.add_argument("--out", type=str, default=str(RESULTS_PATH))
    args = ap.parse_args()

    print("=" * 78)
    print(f"实验五：卷积神经网络（LeNet-5）  device={DEVICE}")
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print("=" * 78, flush=True)

    results_path = Path(args.out)
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results.setdefault("meta", {
        "environment": environment_report(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "dataset": "MNIST",
            "val_size": N_VAL,
            "batch_size": BATCH,
            "seed": SEED,
            "train_pool": 54001,
            "device": str(DEVICE),
        },
    })

    t_all = time.perf_counter()
    lib = {}
    if any(p in args.parts for p in (1, 2, 3, 4, 5, 6, 7, 8)):
        print("\n加载 MNIST ...", flush=True)
        lib["mnist"] = DataBundle("mnist")
    if 9 in args.parts:
        print("\n加载 Fashion-MNIST ...", flush=True)
        lib["fashion"] = DataBundle("fashion")

    for p in args.parts:
        t0 = time.perf_counter()
        part_result = PART_FUNCS[p](lib)
        results[PART_NAMES[p]] = to_py(part_result)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[part {p}] 完成，用时 {time.perf_counter() - t0:.1f}s，"
              f"已写入 {results_path.name}\n", flush=True)

    print("=" * 78)
    print(f"全部完成，总用时 {time.perf_counter() - t_all:.1f}s")
    print(f"结果文件：{results_path}")
    print(f"插图目录：{FIG_DIR}")


if __name__ == "__main__":
    main()
