"""聚类评价指标。全部手写，不依赖 sklearn。

由于聚类结果的簇编号与真实类别编号之间没有先验对应关系，直接比较
`labels == y_true` 是没有意义的（把三个簇编号循环移位一下，准确率可能
从 1.0 掉到 0.3）。因此需要先找最优的编号映射，或者使用对编号排列
不变的指标（ARI）。
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def best_match_accuracy(y_true, y_pred):
    """枚举全部标签排列，返回最高准确率及其映射。

    Parameters
    ----------
    y_true, y_pred : (N,) int array

    Returns
    -------
    acc : float
        最优排列下的准确率。
    mapping : dict[int, int]
        {预测簇编号: 真实类别编号}，即把 y_pred 映射到真实标签所用的字典。

    说明
    ----
    枚举 K! 种排列，K <= 8 时开销可忽略；K 很大时应换成匈牙利算法。
    每个样本只做一次比较，实现上用查找表把 y_pred 转成映射后的标签，
    避免 N 次字典查询。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    k_true = len(true_labels)
    k_pred = len(pred_labels)

    # 预测簇数多于真实类别数时（例如用 K=4 去分 3 类），多出来的簇
    # 统一映射到 -1，永远不匹配，准确率因此被正确地惩罚。
    pool = list(true_labels) + [-1] * max(0, k_pred - k_true)

    best_acc = -1.0
    best_mapping = None
    for perm in permutations(pool, k_pred):
        table = np.full(int(pred_labels.max()) + 1, -1, dtype=y_true.dtype)
        for pred_label, true_label in zip(pred_labels, perm):
            table[pred_label] = true_label
        acc = float(np.mean(table[y_pred] == y_true))
        if acc > best_acc:
            best_acc = acc
            best_mapping = {int(p): int(t) for p, t in zip(pred_labels, perm)}

    return best_acc, best_mapping


def confusion_matrix_from_match(y_true, y_pred, mapping=None):
    """按最优（或给定）映射对齐后计算混淆矩阵。

    返回 (C, C) 矩阵，C = 真实类别数。行是真实类别，列是映射后的预测类别。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if mapping is None:
        _, mapping = best_match_accuracy(y_true, y_pred)

    table = np.full(int(y_pred.max()) + 1, -1, dtype=y_true.dtype)
    for p, t in mapping.items():
        table[p] = t
    aligned = table[y_pred]

    classes = np.unique(y_true)
    c = len(classes)
    index = {int(v): i for i, v in enumerate(classes)}
    cm = np.zeros((c, c), dtype=np.int64)
    for t, p in zip(y_true, aligned):
        if p in index:
            cm[index[int(t)], index[int(p)]] += 1
    return cm


def adjusted_rand_index(y_true, y_pred):
    """调整兰德指数（ARI）。

    定义基于列联表 n_ij（同时属于真实类 i 和预测簇 j 的样本数）：

        RI   = (a + b) / C(N, 2)
        ARI  = (RI - E[RI]) / (max(RI) - E[RI])

    其中 a 为同类同簇的样本对数，b 为异类异簇的样本对数。ARI 对标签
    排列不变，取值上界为 1（完全一致），0 表示与随机划分的期望相当，
    可以为负。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    true_labels = {v: i for i, v in enumerate(np.unique(y_true))}
    pred_labels = {v: i for i, v in enumerate(np.unique(y_pred))}
    table = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        table[true_labels[t], pred_labels[p]] += 1

    def comb2(v):
        return v * (v - 1) / 2.0

    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    total = comb2(n)

    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    if abs(maximum - expected) < 1e-12:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def cluster_purity(y_true, y_pred):
    """簇纯度：每个预测簇取其中最多的真实类别，正确数占全部样本的比例。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        counts = np.bincount(y_true[y_pred == c])
        total += counts.max()
    return float(total / len(y_true))


def silhouette_score(X, labels, sample_size=None, seed=0, metric_chunk=None):
    """平均轮廓系数（内部指标，不需要真实标签）。

    对样本 i，记 a(i) 为它到同簇其他样本的平均距离，b(i) 为它到最近
    其他簇全部样本的平均距离，则

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    取值 [-1, 1]，越大表示簇内越紧凑、簇间越分离。单点簇的 s 定义为 0。

    Parameters
    ----------
    sample_size : int or None
        只用这么多随机抽样的点估计平均轮廓系数。距离矩阵是 $O(N^2)$ 的：
        $N=10992$ 时单个 (N,N) float64 数组就要 483 MB，$N=45312$ 时要
        16 GB，全量计算在大数据集上会直接耗尽内存。抽一两千个点估计均值
        已经足够稳定，因此在大数据集上**必须**传这个参数。注意抽样只针对
        样本，`labels` 仍然用全量，这样各簇的相对密度信息不会丢。
    seed : int
        抽样用的随机种子。
    metric_chunk : int or None
        距离矩阵分块计算的行数上限。默认一次算完（`None`）；$N$ 很大又不想
        抽样时（例如 $N$ 上万）可以设成 1024 这样的小值，把临时数组限制在
        `chunk * N * 8` 字节，用时间换内存。
    """
    x = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    if sample_size is not None and sample_size < len(x):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=sample_size, replace=False)
        x = x[idx]
        labels = labels[idx]

    n = len(x)
    if n < 2:
        return 0.0

    unique = np.unique(labels)
    chunk = n if (metric_chunk is None or metric_chunk <= 0) else min(metric_chunk, n)
    # 逐块算距离，避免一次性构造 (N, N)
    sum_other = {c: np.zeros(n) for c in unique}
    count_other = {c: np.zeros(n) for c in unique}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        diff = x[start:stop, None, :] - x[None, :, :]     # (chunk, N, D)
        dist = np.sqrt((diff ** 2).sum(axis=2))           # (chunk, N)
        for c in unique:
            mask = labels == c
            sum_other[c][start:stop] = dist[:, mask].sum(axis=1)
            count_other[c][start:stop] = mask.sum()

    scores = np.zeros(n)
    for i in range(n):
        c = labels[i]
        n_same = count_other[c][i] - 1
        if n_same <= 0:
            scores[i] = 0.0
            continue
        a = (sum_other[c][i] - 0.0) / n_same      # 自身距离为 0，直接除以同簇其他点数
        b = np.inf
        for other in unique:
            if other == c or count_other[other][i] == 0:
                continue
            b = min(b, sum_other[other][i] / count_other[other][i])
        # 只有一个簇时 b 保持 inf，此时轮廓系数没有定义，约定取 0
        if not np.isfinite(b) or max(a, b) <= 0:
            scores[i] = 0.0
        else:
            scores[i] = (b - a) / max(a, b)
    return float(scores.mean())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = np.repeat([0, 1, 2], 50)
    # 完美预测但编号错位：准确率应为 1.0，ARI 应为 1.0
    y_shift = (y_true + 1) % 3
    print("编号错位的完美划分:", best_match_accuracy(y_true, y_shift)[0], adjusted_rand_index(y_true, y_shift))
    # 随机划分：准确率约 1/3，ARI 约 0
    y_rand = rng.integers(0, 3, size=len(y_true))
    print("随机划分:", round(best_match_accuracy(y_true, y_rand)[0], 3), round(adjusted_rand_index(y_true, y_rand), 3))
