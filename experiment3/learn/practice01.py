#!/usr/bin/env python
# coding=utf-8
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import data_gen as dg


def calculate_distance(sample1, sample2, feature_indices):
    """两个样本在指定特征上的欧氏距离。

    feature_indices 传 [0] 只看第 0 列，传 [0, 1] 则两列一起算。
    """
    diff = sample1[feature_indices] - sample2[feature_indices]
    return float(np.sqrt(np.sum(diff ** 2)))


def extract_features(data, feature_indices):
    """只做取列，不做标准化。"""
    return data[:, feature_indices]


def standardize(data):
    """按列 z-score。常数列（标准差为 0）置 0，避免除零产生 nan。"""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    safe = std > 1e-12
    out = np.zeros_like(data)
    out[:, safe] = (data[:, safe] - mean[safe]) / std[safe]
    return out


def contributions(data):
    """每一维对全部两两距离平方和的贡献占比。不修改传入的 data。"""
    n_features = data.shape[1]
    total = 0.0
    contrib = np.zeros(n_features)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            diff = data[i] - data[j]
            total += np.sum(diff ** 2)
            contrib += diff ** 2
    return contrib / total


if __name__ == "__main__":
    X, y, feature_names = dg.load_dataset("iris_2d", standardize=False)
    print("特征:", feature_names, " 形状:", X.shape)

    # 练习 2：两种距离
    print("只看花瓣长 [0]  :", round(calculate_distance(X[0], X[60], [0]), 4))
    print("两列都用 [0, 1]:", round(calculate_distance(X[0], X[60], [0, 1]), 4))

    # 练习 3：标准化
    X_std = standardize(extract_features(X, [0, 1]))
    print("标准化后 均值:", np.abs(X_std.mean(axis=0)).round(10),
          " 标准差:", X_std.std(axis=0).round(10))

    # 练习 4：占比对比（注意 X.copy()，不要污染原始数据）
    print("原始占比        :", contributions(X).round(4))
    X_scaled = X.copy()
    X_scaled[:, 0] *= 1000
    print("第 0 列 x1000 后 :", contributions(X_scaled).round(4))
    print("标准化后占比    :", contributions(X_std).round(4))

    # 练习 1：画图（需要 matplotlib）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体：不设置的话图上的中文会变成方框（Glyph missing from font）
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 5))
    for j, c in enumerate(["setosa", "versicolor", "virginica"]):
        ax.scatter(X[y == j, 0], X[y == j, 1], s=30, alpha=0.75, label=c)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("练习 1：三个品种的二维分布")
    ax.legend()
    fig.savefig("figures/practice01_scatter.png", bbox_inches="tight")
    print("图已保存到 figures/practice01_scatter.png")

