#!/usr/bin/env python
# coding=utf-8
"""讲义 03 的练习：初始化、选 K、评价聚类。

这个文件里**已经写好的部分**你不需要改（数据处理、指标计算、画图），
**标了 TODO 的部分**需要你补。每个练习运行时会把该看的数字打印出来，
你的任务是解释这些数字，并把答案写在每个练习末尾的"你的结论"处。

运行方式：

    cd learn
    python practice03.py 1      # 只跑练习 1
    python practice03.py 3      # 只跑练习 3
    python practice03.py all    # 全部跑一遍

注意：练习 1、2、5 各有 100~200 次 k-means，跑到 1~3 分钟属正常。
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import data_gen as dg  # noqa: E402
from kmeans import assign_clusters, compute_inertia  # noqa: E402
from metrics import adjusted_rand_index, silhouette_score  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
SEP = "=" * 68


# ---------------------------------------------------------------------------
# 已提供：从指定初始中心跑 k-means
# ---------------------------------------------------------------------------
# 为什么需要这个函数？code/kmeans.py 里的 kmeans() 只能通过 seed 间接控制
# 初始化，没法直接指定初始中心。而练习 1、2 要对比不同初始化策略，
# 必须能自己给出初始中心。


def kmeans_from_init(X, k, init, tol=1e-6, max_iter=100):
    """从给定的初始中心跑 k-means。

    Returns
    -------
    centroids, labels, inertia, n_iter
    """
    C = np.array(init, dtype=float).copy()
    k = len(C)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        labels = assign_clusters(X, C)
        counts = np.bincount(labels, minlength=k)
        new_C = np.zeros_like(C)
        for j in range(k):
            new_C[j] = X[labels == j].mean(axis=0) if counts[j] > 0 else C[j]
        shift = np.linalg.norm(new_C - C, axis=1).max()
        C = new_C
        if shift < tol:
            break
    labels = assign_clusters(X, C)
    return C, labels, compute_inertia(X, labels, C), n_iter


# ---------------------------------------------------------------------------
# 已提供：三种初始化策略中的两种
# ---------------------------------------------------------------------------


def init_random_points(X, k, rng):
    """随机挑 k 个不同的数据点作初始中心。"""
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def init_uniform(rng, X, k):
    """在数据范围内均匀撒 k 个点（注意参数顺序与上面两个不同，看调用处）。"""
    lo, hi = X.min(axis=0), X.max(axis=0)
    return rng.uniform(lo, hi, size=(k, X.shape[1]))


# ---------------------------------------------------------------------------
# 已提供：配对设计用的辅助函数
# ---------------------------------------------------------------------------
# 为什么要"配对"？三种初始化策略消耗的随机数个数不同。如果让它们轮流从同一个
# rng 里取数（跑完一种接着跑下一种），轮到后面的策略时随机数流已经被推进过，
# 等于每种策略拿到的是不同的随机样本，比较就不公平。配对设计的做法是：每一轮
# 试验先取三组独立的随机数（都来自同一个 rng），分别喂给三种策略，使三者面对
# 的随机性对等。


def _pick_points_by_uniform(X, k, U):
    """用给定的 [0,1) 随机数 U（形状 (k,1)）挑 k 个不同的数据点。

    直接用 U 选行号可能撞车，这里用"撞车就顺延"的策略保证选出 k 个互不相同的点。
    """
    n = len(X)
    idx = []
    for r in U[:, 0]:
        j = min(int(r * n), n - 1)
        while j in idx:
            j = (j + 1) % n
        idx.append(j)
    return X[idx].copy()


def _pp_by_uniform(X, k, U, power=2.0):
    """用给定的 [0,1) 随机数 U（长度 k+1）执行 k-means++ 抽样。

    U[0] 用来选第一个中心，U[1..k-1] 作为累积概率的分位点。这样把随机性和
    算法逻辑分离开：随机数由外部提供，算法只负责把随机数映射成初始中心。

    power 控制抽样权重：power=2 时权重正比于 D^2（标准 k-means++），power=1
    时正比于 D。由于 d2 存的是平方距离，权重写成 d2 ** (power / 2)。
    """
    n = len(X)
    centers = [X[min(int(U[0] * n), n - 1)]]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for t in range(1, k):
        w = d2 ** (power / 2.0)
        total = w.sum()
        if total <= 0:
            break
        cum = np.cumsum(w / total)
        centers.append(X[min(int(np.searchsorted(cum, U[t])), n - 1)])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))
    return np.array(centers)


# ===========================================================================
# TODO(你) 练习 2 的核心：实现 k-means++
# ===========================================================================
def init_kmeanspp(X, k, rng, power=2.0):
    """k-means++ 初始化。

    步骤（讲义 03 第 1.1 节）：
      1. 第一个中心：从数据里均匀随机挑一个点。
      2. 之后每一步：对每个点算它到【最近】已有中心的距离，记为 D(x_n)。
         然后按"概率正比于 D(x_n) ** power"抽样选出一个点作为新中心。
         重复直到选够 k 个。

    Parameters
    ----------
    power : float
        概率的幂次。标准 k-means++ 用 2.0；练习 2 会让你改成 1.0 做对比，
        所以这里做成参数而不是写死。

    Returns
    -------
    (k, D) array

    实现提示
    --------
    - 维护一个数组 d2，表示"每个点到最近已有中心的距离平方"。
    - 每加入一个新中心 c，就更新 d2 = np.minimum(d2, 到 c 的距离平方)。
      用 np.minimum 逐元素取小，不要写循环。
    - 归一化成概率：p = d2 / d2.sum()，然后用 rng.choice(N, p=p) 抽样。
    - 边界情况：如果 d2 全为 0（所有点都和已有中心重合，真实数据里不该
      出现，但构造数据时可能），除法会得到 nan。要处理这种情况。
    """
    N, D = X.shape
    centroids = np.zeros((k, D), dtype=float)
    first_center = rng.choice(N)
    centroids[0] = X[first_center]
    d2 = np.full(N, np.inf)  # 初始化为无穷大，确保第一个中心后能正确计算距离

    for i in range(1, k):
        # 计算每个点到最近已有中心的距离平方
        distances = np.linalg.norm(X - centroids[i - 1], axis=1) ** power
        d2 = np.minimum(d2, distances)

        # 处理 d2 全为 0 的情况
        if np.all(d2 == 0):
            # d2 全为 0，说明所有点都和已有中心重合, 直接d2作为聚类中心
            break

        # 计算概率分布
        p = d2 / d2.sum()

        # 按概率抽样选出下一个中心
        next_center = rng.choice(N, p=p)
        centroids[i] = X[next_center]

    return centroids[:i + 1]  # 返回实际选出的中心数，可能少于 k


# ---------------------------------------------------------------------------
# 练习 1：复现讲义 03 第 1.2 节的表格
# ---------------------------------------------------------------------------
# 这个练习**全部写好了**，你只需要运行并核对数字。


def exercise1(n_trials=200, k=3):
    print(SEP)
    print("练习 1：复现讲义 03 第 1.2 节的初始化成败率表（配对设计）")
    print(SEP)
    X, y = dg.make_synthetic_data(dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, 1500, seed=0)
    print(f"数据 {X.shape}，全局最优 J = 3570.5，共 {n_trials} 次配对实验\n")

    # 关键：随机数【一次性预生成】。如果改成在循环里逐轮 rng.random(...)，
    # 那么试验里每多一种策略（每多取一组随机数），后面所有策略拿到的随机数
    # 都会整体偏移，结果随之改变——这会让"加一个对比方法"就推翻已有结论。
    # 预生成固定形状的随机数组可以彻底避免这个问题，跑多少次结果都一样。
    rng = np.random.default_rng(123)
    lo, hi = X.min(axis=0), X.max(axis=0)
    U_pts = rng.random((n_trials, k))            # 随机挑点：选行号
    U_unif = rng.random((n_trials, k, X.shape[1]))  # 均匀撒点：映射到数据范围
    U_pp = rng.random((n_trials, k + 1))         # k-means++：首点 + 累积概率分位点
    U_pp1 = rng.random((n_trials, k + 1))        # k-means++ 但 power=1（对照）

    Js_all = {"随机挑 K 个点": [], "范围内均匀撒点": [], "k-means++ (power=2)": [],
              "k-means++ (power=1)": []}
    its_all = {name: [] for name in Js_all}

    for t in range(n_trials):
        inits = {
            "随机挑 K 个点": _pick_points_by_uniform(X, k, U_pts[t][:, None]),
            "范围内均匀撒点": lo + U_unif[t] * (hi - lo),
            "k-means++ (power=2)": _pp_by_uniform(X, k, U_pp[t], power=2.0),
            "k-means++ (power=1)": _pp_by_uniform(X, k, U_pp1[t], power=1.0),
        }
        for name, init in inits.items():
            *_, J, it = kmeans_from_init(X, k, init)
            Js_all[name].append(J)
            its_all[name].append(it)

    print(f"{'初始化方法':<22}{'成功率':<9}{'平均 J':<11}{'最差 J':<11}{'平均迭代'}")
    for name in Js_all:
        Js = np.array(Js_all[name])
        print(f"{name:<22}{(Js < 4000).mean() * 100:<9.0f}{Js.mean():<11.1f}{Js.max():<11.1f}{np.mean(its_all[name]):.1f}")

    print("\n讲义 1.2 节：随机挑点 94% / 4511.5 / 18068.5 / 4.7 轮")
    print("            均匀撒点 98% / 3932.5 / 18068.5 / 4.3 轮")
    print("            k-means++ 98% / 3787.9 / 18068.5 / 2.9 轮")
    print("            k-means++(power=1) 99% / 3715.3 / 18058.1 / 3.2 轮")
    print("""
你的结论（写进报告）：
  1. 你跑出来的数字和讲义一致吗？如果不一致，差在哪一项？
  2. 三种方法的成功率差别大不大？差距主要体现在成功率还是迭代轮数上？
  3. 想一想：为什么在这份数据上三种方法都能成功 94% 以上？
     提示——回忆讲义 03 第 1.2 节对"数据难度"的讨论：这份数据符合 k-means 的
     球形假设吗？如果不符合，差别会不会更大？
""")


# ---------------------------------------------------------------------------
# 练习 2：实现 k-means++ 并验证"平方"的作用
# ---------------------------------------------------------------------------


def exercise2(n_trials=100, k=3):
    print(SEP)
    print("练习 2：实现 k-means++，对比 power=2 与 power=1")
    print(SEP)
    X, y = dg.make_synthetic_data(dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, 1500, seed=0)

    variants = [
        ("随机挑 K 个点", lambda rng: init_random_points(X, k, rng)),
        ("k-means++ (power=2)", lambda rng: init_kmeanspp(X, k, rng, power=2.0)),
        ("加权但 power=1", lambda rng: init_kmeanspp(X, k, rng, power=1.0)),
    ]

    rng = np.random.default_rng(123)
    print(f"{'初始化方法':<22}{'成功率':<9}{'平均 J':<10}{'最差 J':<10}{'平均迭代'}")
    results = {}
    for name, make_init in variants:
        Js, its = [], []
        for _ in range(n_trials):
            init = make_init(rng)
            *_, J, t = kmeans_from_init(X, k, init)
            Js.append(J)
            its.append(t)
        Js = np.array(Js)
        its = np.array(its)
        results[name] = (Js, its)
        print(f"{name:<22}{(Js < 4000).mean() * 100:<9.0f}{Js.mean():<10.0f}{Js.max():<10.0f}{its.mean():.1f}")

    # 同一批初始中心下，kmeans++ 的迭代轮数分布
    if "k-means++ (power=2)" in results:
        its_pp = results["k-means++ (power=2)"][1]
        its_rnd = results["随机挑 K 个点"][1]
        print(f"\n迭代轮数：k-means++ 中位数 {np.median(its_pp):.0f}，随机挑点中位数 {np.median(its_rnd):.0f}")
        print(f"           k-means++ 最多 {its_pp.max()} 轮，随机挑点最多 {its_rnd.max()} 轮")

    print("""
你的结论：
  1. power=2 与 power=1 的成功率有区别吗？差别有多大？
  2. k-means++ 相对随机挑点，收益主要体现在成功率上还是迭代轮数上？
  3. power 取 2 的理论理由是什么？（提示：讲义 1.1 节提到 k-means++ 有一个
     关于期望 J 的近似保证，那个保证正是基于 D^2 抽样。）
""")


# ---------------------------------------------------------------------------
# 练习 3：Iris 上复现 2.4 节的表，并画出 K=2 / K=3 的对比图
# ---------------------------------------------------------------------------


def sweep_k(X, y, kmax=6, n_init=10):
    """对 K=2..kmax 跑 k-means，返回 (K 列表, J 列表, 轮廓系数列表, ARI 列表)。"""
    from kmeans import kmeans_restarts

    ks, Js, sils, aris = [], [], [], []
    for K in range(2, kmax + 1):
        best, _ = kmeans_restarts(X, K, n_init=n_init)
        ks.append(K)
        Js.append(best["inertia"])
        sils.append(silhouette_score(X, best["labels"], sample_size=1500, seed=0))
        aris.append(adjusted_rand_index(y, best["labels"]))
    return np.array(ks), np.array(Js), np.array(sils), np.array(aris)


def plot_two_k(X, y, ks_to_plot):
    """并排画出若干个 K 的聚类结果，用于肉眼比较。"""
    from kmeans import kmeans_restarts

    n = len(ks_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8))
    axes = np.atleast_1d(axes)
    for ax, K in zip(axes, ks_to_plot):
        best, _ = kmeans_restarts(X, K, n_init=10)
        labels = best["labels"]
        for j in range(K):
            m = labels == j
            ax.scatter(X[m, 0], X[m, 1], s=34, alpha=0.75,
                       color=COLORS[j % len(COLORS)], edgecolor="white", linewidth=0.5)
        C = best["centroids"]
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=190, color="black", zorder=6)
        ari = adjusted_rand_index(y, labels)
        cnt = np.bincount(labels, minlength=K).tolist()
        ax.set_title(f"K={K}  J={best['inertia']:.2f}  ARI={ari:.4f}\n各簇点数 {cnt}", fontsize=10)
        ax.set_xlabel("花瓣长度")
        ax.set_ylabel("花瓣宽度")
    fig.suptitle("轮廓系数偏好 K=2，但真实类别数是 3 —— 用眼睛找出原因", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIG / "practice03_iris_K2_vs_K3.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {path.relative_to(FIG.parent)}")


def exercise3():
    print(SEP)
    print("练习 3：Iris 花瓣两维上的 K 扫描与内部指标陷阱")
    print(SEP)
    X, y, names = dg.load_dataset("iris_2d")
    ks, Js, sils, aris = sweep_k(X, y, kmax=6)

    print(f"{'K':<4}{'inertia':<11}{'轮廓系数':<11}{'ARI':<10}")
    for K, J, s, a in zip(ks, Js, sils, aris):
        print(f"{K:<4}{J:<11.3f}{s:<11.4f}{a:<10.4f}")

    print(f"\n轮廓系数最大处 K={ks[np.argmax(sils)]} (值 {sils.max():.4f})")
    print(f"ARI 最大处     K={ks[np.argmax(aris)]} (值 {aris.max():.4f})")
    print(f"真实类别数     K=3")
    print("\n讲义 2.4 节的数字：K=2 轮廓 0.7652 / ARI 0.5584；K=3 轮廓 0.6603 / ARI 0.8857")

    plot_two_k(X, y, [2, 3])
    print("""
你的结论：
  1. 轮廓系数在哪个 K 最大？和真实类别数一致吗？
  2. 看刚才生成的那张图：K=2 时哪两个品种被合到了一起？
  3. 关键问题：为什么把这两个品种合并之后，轮廓系数反而【更高】？
     提示——轮廓系数由 a(i)（到同簇的平均距离）和 b(i)（到最近其他簇的平均距离）
     决定。合并之后，"最近的其他簇"变成了谁？b(i) 变大了还是变小了？
""")


# ---------------------------------------------------------------------------
# 练习 4：一维情形
# ---------------------------------------------------------------------------


def exercise4():
    print(SEP)
    print("练习 4：单列特征（一维）时的 K 扫描")
    print(SEP)
    X4, y, names = dg.load_dataset("iris_4d")
    print(f"iris_4d 的四列: {names}\n")

    # 逐列做一维聚类，看看哪一列真正有用
    from kmeans import kmeans_restarts

    print("先用 K=3 把每一列单独跑一遍，看哪一列有区分力：")
    print(f"{'列':<16}{'ARI':<10}")
    for i, n in enumerate(names):
        best, _ = kmeans_restarts(X4[:, [i]], 3, n_init=10)
        print(f"{n:<16}{adjusted_rand_index(y, best['labels']):<10.4f}")

    # 对其中两列做完整的 K 扫描
    for col, tag in [(0, "花萼长度（区分力差）"), (2, "花瓣长度（区分力好）")]:
        X1 = X4[:, [col]]
        ks, Js, sils, aris = sweep_k(X1, y, kmax=6)
        print(f"\n=== {tag}：{names[col]} ===")
        print(f"{'K':<4}{'inertia':<11}{'轮廓系数':<11}{'ARI':<10}")
        for K, J, s, a in zip(ks, Js, sils, aris):
            print(f"{K:<4}{J:<11.3f}{s:<11.4f}{a:<10.4f}")
        print(f"轮廓系数最大处 K={ks[np.argmax(sils)]}，ARI 最大处 K={ks[np.argmax(aris)]}")

    print("""
你的结论：
  1. 四列里哪一列的 ARI 最高？哪一列最低？两者差多少倍？
  2. 一维时轮廓系数最大出现在哪个 K？和二维（练习 3）的结论一样吗？
  3. 关键对比：一维最好那一列的 ARI，和练习 3 里二维的 ARI（K=3 时 0.8857）
     相比如何？这说明"多加一个特征"在这份数据上带来了多少提升？
  4. 为什么换一列特征就会显著改变聚类结果？提示：回想讲义 01 里
     "每一维对距离的贡献占比"那个实验——区分力强的列本身就把样本拉开了，
     区分力弱的列几乎没有信息，但它在距离计算里同样占一维的权重。
""")


# ---------------------------------------------------------------------------
# 练习 5：重现 2.2 节现象 —— K 越大越难找到最优解
# ---------------------------------------------------------------------------


def exercise5():
    print(SEP)
    print("练习 5：K 越大，同样的重启次数越难找到最优解")
    print(SEP)
    from kmeans import kmeans_restarts

    X, y = dg.make_synthetic_data(dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, 1500, seed=0)

    print("第一步：对比 n_init=10 与 n_init=40")
    print(f"{'K':<5}{'n_init=10':<13}{'n_init=40':<13}{'相差':<10}{'40 次中命中最好的比例'}")
    for K in (3, 6):
        best10, _ = kmeans_restarts(X, K, n_init=10)
        best40, allJ40 = kmeans_restarts(X, K, n_init=40)
        J10, J40 = best10["inertia"], best40["inertia"]
        hit = sum(1 for v in allJ40 if v < J40 * 1.05)
        diff = (J10 - J40) / J40 * 100
        print(f"{K:<5}{J10:<13.1f}{J40:<13.1f}{diff:+.1f}%{'':<5}{hit}/40 = {hit / 40 * 100:.0f}%")

    print("\n第二步：固定 n_init=10，比较 K=3 与 K=6 时【单次运行】结果的分散程度")
    # 注意：这里直接调用 kmeans() 而不是 kmeans_restarts(n_init=1, seed=s)。
    # 原因是 kmeans_restarts 内部固定用 seed=seed 调 kmeans，从外面再传 seed
    # 会触发 "got multiple values for keyword argument 'seed'"。
    # 这正是 review_kmeans.md 第 3.4 节指出的那个问题。
    from kmeans import kmeans

    print(f"{'K':<5}{'最好':<11}{'最差':<11}{'标准差':<11}{'变异系数':<11}")
    spread = {}
    for K in (3, 6):
        Js = []
        for s in range(30):
            *_, J, _ = kmeans(X, K, seed=s)
            Js.append(J)
        Js = np.array(Js)
        spread[K] = Js
        print(f"{K:<5}{Js.min():<11.1f}{Js.max():<11.1f}{Js.std():<11.1f}{Js.std() / Js.mean() * 100:.1f}%")
    print(f"\nK=3 的 30 次 J 分布: {np.unique(np.round(spread[3], 1))}")
    print(f"K=6 的 30 次 J 分布(前 8 个不同的值): {np.unique(np.round(spread[6], 1))[:8]}")

    print("""
你的结论：
  1. K=6 时 n_init=10 和 n_init=40 得到的 J 一样吗？差多少？这个差距说明
     n_init=10 在 K=6 时发生了什么？
  2. 第二步的结果可能和你预期相反：K=3 的变异系数（86.5%）其实比 K=6
     （13.7%）还大。为什么？看一下打印出来的 J 分布——K=3 只出现了三个
     不同的值，而 K=6 出现了很多个。请分别解释这两组分布的形状，
     并说明"分散程度大"在两种情形下分别意味着什么。
  3. 综合第一、二步：判断"这个 K 能不能找到最优解"，看变异系数可靠吗？
     更可靠的判据应该是什么？
  4. 这对实验报告的写法有什么影响？如果你要在肘部图上比较 K=3 和 K=6 的 J，
     需要保证什么？
""")


# ---------------------------------------------------------------------------
# 练习 6：诊断练习 —— K 给大了会怎样
# ---------------------------------------------------------------------------


def exercise6():
    print(SEP)
    print("练习 6：K=6 时各簇的点数分布说明了什么")
    print(SEP)
    from kmeans import kmeans_restarts

    X, y = dg.make_synthetic_data(dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, 1500, seed=0)
    true_counts = np.bincount(y)
    print(f"真实三簇的点数: {true_counts.tolist()}")

    results = {}
    for K in (3, 6):
        best, _ = kmeans_restarts(X, K, n_init=40)
        cnt = np.bincount(best["labels"], minlength=K)
        results[K] = best
        print(f"\nK={K}: J = {best['inertia']:.1f}")
        print(f"  各簇点数 {cnt.tolist()}")
        print(f"  最大簇/最小簇 = {cnt.max()}/{cnt.min()}")
        print(f"  ARI（对真实标签）= {adjusted_rand_index(y, best['labels']):.4f}")

    # 画出两个结果并排
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax, K in zip(axes, (3, 6)):
        labels = results[K]["labels"]
        for j in range(K):
            m = labels == j
            ax.scatter(X[m, 0], X[m, 1], s=18, alpha=0.7,
                       color=COLORS[j % len(COLORS)], edgecolor="none")
        C = results[K]["centroids"]
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=200, color="black", zorder=6)
        ari = adjusted_rand_index(y, labels)
        ax.set_title(f"K={K}  J={results[K]['inertia']:.1f}  ARI={ari:.4f}", fontsize=11)
    fig.suptitle("K=6 的 J 更小，但它把真实簇拆开了 —— 更小的 J 不等于更好的聚类", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG / "practice03_K6_diagnosis.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {path.relative_to(FIG.parent)}")
    print("提示：重点看 K=6 时有没有哪个真实簇被多个中心瓜分。对比两边的黑叉位置。")

    print("""
你的结论：
  1. K=6 的 J 比 K=3 小还是大？（这是必然的，想一想为什么必然如此）
  2. K=6 的 ARI 比 K=3 高还是低？
  3. 核心问题：为什么"J 更小"不能说明"聚类更好"？请举出这份数据里的具体证据。
  4. 如果你在真实任务里（没有真实标签）看到某次聚类的点数分布是
     900 / 500 / 98 / 2，你会怀疑什么问题？
""")


# ---------------------------------------------------------------------------


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = {
        "1": exercise1,
        "2": exercise2,
        "3": exercise3,
        "4": exercise4,
        "5": exercise5,
        "6": exercise6,
    }
    if which == "all":
        for fn in todo.values():
            fn()
            print()
    elif which in todo:
        todo[which]()
    else:
        print(f"用法：python practice03.py [1|2|3|4|5|6|all]")


if __name__ == "__main__":
    main()
