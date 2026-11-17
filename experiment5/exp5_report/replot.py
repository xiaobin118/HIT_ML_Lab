#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 results.json 重新绘制少量需要调整版式的插图（不重新训练）。

用途：实验主脚本已经跑完后，如果某张图的范围/标注不合适，只需从 JSON 里的
曲线数据重画对应图片即可，避免为了改一张图重跑整个实验。

运行：cd exp5_report && python replot.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import viz  # noqa: E402

viz.set_fig_dir(HERE / "figures")
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def replot_baseline():
    runs = R["baseline"]["runs"]
    order = R["baseline"]["order"]
    values = [runs[k]["test_acc"] for k in order]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    bars = ax.bar(range(len(order)), values,
                  color=[viz.PALETTE[i % len(viz.PALETTE)] for i in range(len(order))],
                  edgecolor="white")
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0018, f"{v * 100:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=12, ha="right", fontsize=8.5)
    ax.set_ylim(0.92, 1.006)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("测试准确率")
    ax.set_title("基线模型测试准确率（全量训练集，10 轮）")
    viz.savefig(fig, "fig04_baseline_final.png")


def replot_activation():
    runs = R["activation"]["runs"]
    order = R["activation"]["order"]
    labels = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "GELU"]
    colors = [viz.PALETTE[i % len(viz.PALETTE)] for i in range(len(order))]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0))
    values = [runs[k]["test_acc"] for k in order]
    axes[0].bar(range(len(order)), values, color=colors)
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.001, f"{v * 100:.2f}%", ha="center", fontsize=8)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylim(0.93, 1.0)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("测试准确率（10,000 张训练图，8 轮）")

    for i, k in enumerate(order):
        g = runs[k]["grad_norm_first_conv"]
        axes[1].plot(range(1, len(g) + 1), g, marker="o", markersize=3,
                     color=colors[i], label=labels[i])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练轮次 (epoch)")
    axes[1].set_ylabel("第一层卷积权重梯度范数（对数尺度）")
    axes[1].set_title("第一层卷积的梯度大小（Sigmoid 起点低 3 个数量级）")
    axes[1].legend(fontsize=8, ncol=2)
    viz.savefig(fig, "fig06_activation_grad.png")


def replot_vanish():
    runs = R["activation"]["grad_diagnosis"]["runs"]
    order = R["activation"]["grad_diagnosis"]["order"]
    colors = [viz.PALETTE[i % len(viz.PALETTE)] for i in range(len(order))]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    values = [runs[k]["test_acc"] for k in order]
    axes[0].barh(range(len(order)), values, color=colors)
    for i, v in enumerate(values):
        axes[0].text(v + 0.005, i, f"{v * 100:.2f}%", va="center", fontsize=8)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=8)
    axes[0].set_xlim(0, 1.02)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("测试准确率（6 轮）")
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, k in enumerate(order):
        g = runs[k]["grad_norm_first_conv"]
        axes[1].plot(range(1, len(g) + 1), g, marker="o", markersize=3,
                     linestyle=styles[i % len(styles)], color=colors[i], label=k)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练轮次 (epoch)")
    axes[1].set_ylabel("第一层卷积权重梯度范数（对数尺度）")
    axes[1].set_title("梯度量级：Sigmoid 在 SGD 下比 ReLU 低约 3 个数量级")
    axes[1].legend(fontsize=6.5)
    viz.savefig(fig, "fig07_vanishing_grad.png")


if __name__ == "__main__":
    replot_baseline()
    replot_activation()
    replot_vanish()
    print("重新绘图完成")
