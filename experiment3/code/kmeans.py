"""k-means 聚类（Lloyd 算法）。

本文件的全部函数体需要你实现，标注了 `TODO` 的地方对应
`docs/02_实现任务与编码路线.md` 第 3 节的任务编号。

原理要点（详见 `docs/01_原理_kmeans与GMM的EM推导.md` 第 2 节）：
k-means 最小化簇内平方和 J = sum_n ||x_n - mu_{label_n}||^2。固定中心时
最优归属是最近邻，固定归属时最优中心是簇内均值，两块交替做，J 单调
不增且有下界，因此必然收敛到某个驻点（不保证全局最优）。
"""

from __future__ import annotations

import numpy as np

from metrics import best_match_accuracy


def assign_clusters(X, centroids):
    """任务 2.1：把每个样本分配给最近的中心。

    Parameters
    ----------
    X : (N, D) array
    centroids : (K, D) array

    Returns
    -------
    labels : (N,) int array

    提示：用广播构造 (N, K, D) 的差张量，平方求和后沿 axis=1 取 argmin。
    不要写 Python 双重循环。
    Example:
        X = np.array([[0, 0], [1, 1], [2, 2]])
        centroids = np.array([[0, 0], [2, 2]])
        # diff = X[:, None, :] - centroids[None, :, :]  # (3, 2, 2)
        # dist2 = np.sum(diff ** 2, axis=2)  # (3, 2)
        # labels = np.argmin(dist2, axis=1)  # (3,)
        return labels  # array([0, 0, 1])
    """
    diff = X[:, None, :] - centroids[None, :, :]  # (N, K, D)
    dist2 = np.sum(diff ** 2, axis=2)  # (N, K)
    labels = np.argmin(dist2, axis=1)  # (N,)
    return labels


def compute_inertia(X, labels, centroids):
    """任务 2.2：簇内平方和 J = sum_n ||x_n - mu_{label_n}||^2。

    提示：用中心矩阵按标签索引出每个样本对应的中心（centroids[labels]），
    再算差的平方和。
    """
    J = np.sum((X - centroids[labels]) ** 2)
    return J


def kmeans(X, k, max_iter=100, tol=1e-6, seed=0):
    """任务 2.3：Lloyd 算法主循环。

    Parameters
    ----------
    X : (N, D) array
    k : int
        簇数。
    max_iter : int
        最大迭代次数。
    tol : float
        中心位移的收敛阈值：所有中心的位移都小于 tol 时停止。
    seed : int
        随机种子，用于选择初始中心。

    Returns
    -------
    centroids : (K, D) array
    labels : (N,) int array
    inertia : float
        最终簇内平方和。
    history : list[dict]
        每轮记录 {'iter': int, 'inertia': float, 'shift': float}。
        注意 inertia 必须每轮都记，阶段四要画它的下降曲线。

    实现提示
    --------
    1. 初始化：rng = np.random.default_rng(seed)，
       idx = rng.choice(N, size=k, replace=False)，centroids = X[idx].copy()。
       要用 replace=False 保证初始中心互不重合，否则两个中心会永远同步移动。
    2. 循环：分配 -> 计算新中心 -> 算位移 -> 记录 -> 判断退出。
    3. 新中心 = 各簇样本均值。用 np.bincount(labels, minlength=k) 统计簇大小，
       某簇为 0 时（空簇）不要做除法，保留旧中心并在 history 里标记。
    4. 退出条件：shift < tol 或 iter >= max_iter。

    验收标准
    --------
    (a) inertia 序列单调不增；
    (b) seed=0 时与真实标签的最优匹配准确率为 1.0000（实测 5 轮收敛，J=3570.5）；
    (c) seed=0..9 的 inertia 有可观察差异：实测 10 次中有 1 次（seed=3）落到
        J=18063.7、准确率 0.539 的糟糕局部最优，其余 9 次均为 J=3570.5。
        这个 5 倍差距正说明单次运行不可靠、必须配合 kmeans_restarts 使用。
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.choice(N, size=k, replace=False)  # idx 是初始中心的样本索引
    centroids = X[idx].copy()
    history = []
    for i in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=k)
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] = X[labels == j].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        inertia = compute_inertia(X, labels, new_centroids)
        history.append({'iter': i, 'inertia': inertia, 'shift': shift})
        centroids = new_centroids
        if shift < tol:
            break
    return centroids, labels, inertia, history


def kmeans_restarts(X, k, n_init=10, **kwargs):
    """任务 2.4：多次随机重启，取 inertia 最小的结果。

    Returns
    -------
    best : dict
        含 'centroids'、'labels'、'inertia'、'history'、'seed'。
    all_inertias : list[float]
        每次重启的 inertia，用于报告里说明初始化对结果的影响。
    """
    best = None
    all_iterias = []
    for seed in range(n_init):
        centroids, labels, inertia, history = kmeans(X, k, seed=seed, **kwargs)
        all_iterias.append(inertia)
        if best is None or inertia < best['inertia']:
            best = {
                'centroids': centroids,
                'labels': labels,
                'inertia': inertia,
                'history': history,
                'seed': seed
            }
    return best, all_iterias


if __name__ == "__main__":
    # 自检脚本：实现完上面四个函数后直接运行本文件。
    # 运行方式：cd code && python kmeans.py
    from data_gen import TRUE_COVS, TRUE_MEANS, TRUE_WEIGHTS, make_synthetic_data

    X, y = make_synthetic_data(TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS, n_samples=1500, seed=0)

    print("=== 单次运行 (seed=0) ===")
    centroids, labels, inertia, history = kmeans(X, k=3, seed=0)
    acc, mapping = best_match_accuracy(y, labels)
    print(f"迭代次数: {len(history)}  最终 inertia: {inertia:.2f}")
    print(f"最优匹配准确率: {acc:.4f}  映射: {mapping}")
    print("inertia 单调不增:", bool(np.all(np.diff([h["inertia"] for h in history]) <= 1e-9)))
    print("估计中心:\n", centroids.round(3))
    print("真实均值:\n", TRUE_MEANS)

    print("\n=== 多次重启 ===")
    best, all_inertias = kmeans_restarts(X, k=3, n_init=10)
    print("各次 inertia:", [round(v, 1) for v in all_inertias])
    print(f"最好 {min(all_inertias):.1f}  最差 {max(all_inertias):.1f}  来自 seed={best['seed']}")
