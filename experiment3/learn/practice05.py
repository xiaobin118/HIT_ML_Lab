#!/usr/bin/env python
# coding=utf-8
"""讲义 05 的练习：混合高斯模型与责任度。

老规矩：**需要你写的只有练习 1**（责任度函数，而且要求对数域实现），
后面四个练习的计算和画图都已写好。

运行方式：

    cd learn
    python practice05.py 1      # 只跑练习 1
    python practice05.py all    # 全部（几秒）

讲义 05 用的两簇参数（方差相差 8 倍，这是 k-means 会分错的关键）：
    簇 0:  mu=0, var=0.25 （标准差 0.5，紧）
    簇 1:  mu=3, var=4.0  （标准差 2.0，散）
    权重各 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import data_gen as dg  # noqa: E402
from kmeans import kmeans  # noqa: E402
from metrics import adjusted_rand_index  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
SEP = "=" * 70

# 讲义 05 的两簇参数
MU0, VAR0 = 0.0, 0.25
MU1, VAR1 = 3.0, 4.0
W0 = W1 = 0.5


# ---------------------------------------------------------------------------
# 已提供的小工具
# ---------------------------------------------------------------------------


def log_normal_1d(x, mu, var):
    """一元高斯对数密度（讲义 04 练习 1 你已经写过，这里直接给出来复用）。"""
    return -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)


def make_two_cluster_data(n=2000, seed=3, var1=VAR1):
    """生成讲义 05 的两簇一维数据（var1 可以改，练习 5 会用到）。"""
    means = np.array([[MU0], [MU1]])
    covs = np.array([[[VAR0]], [[var1]]])
    X, y = dg.make_synthetic_data(means, covs, np.array([W0, W1]), n_samples=n, seed=seed)
    return X[:, 0], y


# ===========================================================================
# TODO(你) 练习 1：实现责任度（对数域）
# ===========================================================================
def responsibility(x, mu0=MU0, var0=VAR0, mu1=MU1, var1=VAR1, w0=W0, w1=W1):
    """返回 x 属于【第二个】分量的责任度 r1。

        r1(x) = w1*N1(x) / ( w0*N0(x) + w1*N1(x) )

    Parameters
    ----------
    x : float 或 ndarray
        可以是标量，也可以是数组（返回同形状）。

    Returns
    -------
    责任度，取值在 [0, 1]。

    要求：必须在对数域实现，不能直接按上面的公式算密度再除。

    为什么？当 x 离两个中心都很远时（例如 x=100），N0(x) 和 N1(x)
    都会下溢成 0，直接算就是 0/0 = nan。

    注意：要"两侧都下溢"才会 nan。像 x=30 这种地方只有 N0 下溢、
    N1 还是 5.3e-41，直接算碰巧也能得到 1.0——所以这个 bug 很隐蔽，
    它只在两个分量都够不着的位置才暴露出来。

    对数域的做法（log-sum-exp）：
        l0 = log(w0) + log_normal_1d(x, mu0, var0)
        l1 = log(w1) + log_normal_1d(x, mu1, var1)
        m  = max(l0, l1)                    # 逐元素取最大
        r1 = exp(l1 - m) / (exp(l0 - m) + exp(l1 - m))

    这一步的妙处：减去最大值之后，l0-m 和 l1-m 里至少有一个是 0，
    另一个不超过 0，所以两个指数都不会溢出；同时分子分母同乘 e^{-m}
    不改变比值，因此结果和直接算完全一致，但不会出现 0/0。

    实现提示
    --------
    - np.maximum 用于逐元素取最大（不用 np.max，那个会返回标量）。
    - 传进来的 x 可能是标量，np.exp 也能处理，返回 numpy 标量即可。
    - 可以顺便用 np.clip 把结果限制在 [0,1] 内，防止浮点误差导致的 1+1e-16。
    """
    l0 = np.log(w0) + log_normal_1d(x, mu0, var0)
    l1 = np.log(w1) + log_normal_1d(x, mu1, var1)
    m = np.maximum(l0, l1)
    r1 = np.exp(l1 - m) / (np.exp(l0 - m) + np.exp(l1 - m))
    return np.clip(r1, 0.0, 1.0)



# ---------------------------------------------------------------------------
# 练习 1
# ---------------------------------------------------------------------------


def exercise1():
    print(SEP)
    print("练习 1：实现责任度函数（对数域）")
    print(SEP)

    checks = [
        (1.0, 0.5284, "讲义 4.1 节的表"),
        (2.0, 0.9985, "讲义 4.1 节的表"),
        (0.0, 0.0751, "讲义 4.1 节的表"),
        (-2.0, 0.9704, "讲义 4.1 节的表（左侧尾巴，反直觉）"),
        (-1.3745, 0.5, "左分界"),
        (0.9745, 0.5, "右分界"),
    ]
    print(f"{'x':<12}{'你的结果':<14}{'期望值':<12}{'绝对误差':<12}来源")
    ok = True
    for x, want, src in checks:
        got = float(responsibility(np.array([x]))[0])
        err = abs(got - want)
        if err > 1e-3:
            ok = False
        print(f"{x:<12}{got:<14.6f}{want:<12.4f}{err:<12.2e}{src}")

    print("\n数值稳定性测试（x 远离两个中心，直接算密度会下溢成 0 -> nan）:")
    print(f"{'x':<12}{'责任度':<16}{'是否 nan':<10}")
    for x in (8.0, 10.0, 30.0, -30.0, -60.0, 100.0):
        got = float(responsibility(np.array([x]))[0])
        print(f"{x:<12}{got:<16.6f}{str(np.isnan(got)):<10}")
    print("  这些位置的责任度应该都是 1.000000：它们离簇 0 的中心(0)太远，")
    print("  只能来自簇 1 —— 即使 x=100 离簇 1 的中心(3)也有 97 个单位。")
    print("  这正是责任度'看相对大小'而非'看绝对距离'的体现。")

    # 数组输入与标量输入一致
    xs = np.array([-2.0, 0.0, 1.0, 2.0, 5.0])
    got_arr = responsibility(xs)
    print(f"\n数组输入 x={xs.tolist()}")
    print(f"  责任度 = {np.round(got_arr, 6).tolist()}")
    print(f"  形状 {np.shape(got_arr)}（应与输入同形状）")

    print(f"""
{'所有校验点通过' if ok else '有校验点未通过，检查公式或常数'}

你的结论：
  1. 责任度是否全部落在 [0,1] 且没有 nan？
  2. 如果去掉对数域那一步（直接算 w*N 再相除），哪些 x 会得到 nan？
     实测数据：x=10 时 p0=1.1e-87（还没下溢）直接算得 1.0（碰巧对）；
     x=30 时 p0 下溢但 p1=5.3e-41，直接算仍是 1.0；
     x=100 时两个密度**都**下溢，直接算才变成 nan。
     请解释：为什么这个 bug 只在"两侧都够不着"的位置暴露？
  3. 为什么 log-sum-exp 里要"先减去最大值"？不减会有什么问题？
     （提示：考虑 l0 和 l1 都很大的情况——本例中它们是负的，但换一组参数
     就可能出现正的、甚至很大的对数密度，那时 exp(l) 会溢出成 inf）
""")


# ---------------------------------------------------------------------------
# 练习 2：k-means 的分界 vs 责任度分界
# ---------------------------------------------------------------------------


def exercise2():
    print(SEP)
    print("练习 2：k-means 的分界错在哪")
    print(SEP)
    x, y = make_two_cluster_data()

    C, lab, J, hist = kmeans(x[:, None], 2, seed=0)
    km_bound = (C[0, 0] + C[1, 0]) / 2
    print(f"k-means 收敛中心: {np.round(C.ravel(), 4).tolist()}")
    print(f"它的分界点（两中心中点）: {km_bound:.4f}")
    print(f"各簇点数 {np.bincount(lab, minlength=2).tolist()}（真实 {np.bincount(y).tolist()}）")
    print(f"ARI = {adjusted_rand_index(y, lab):.4f}")

    # 用责任度找 0.5 分界：粗扫找变号区间，再二分
    def r_at(t):
        return float(responsibility(np.array([t]))[0])

    print("\n用责任度找 r1 = 0.5 的位置（注意：有两个解）:")
    grid = np.linspace(-40, 40, 20001)
    rv = responsibility(grid) - 0.5
    roots = []
    for i in range(len(grid) - 1):
        if rv[i] * rv[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(100):
                mid = (lo + hi) / 2
                if (r_at(mid) - 0.5) * (r_at(lo) - 0.5) > 0:
                    lo = mid
                else:
                    hi = mid
            roots.append((lo + hi) / 2)
    print(f"  找到 {len(roots)} 个分界: {[round(v, 4) for v in roots]}")
    for t in roots:
        print(f"    x={t:+.4f}  责任度={r_at(t):.6f}")

    if len(roots) == 2:
        left, right = roots
    else:
        left, right = roots[0], roots[-1]

    # 两种划分的 ARI
    bayes_lab = (responsibility(x) > 0.5).astype(int)
    print(f"\n用责任度划分（r1>0.5 归簇 1）:")
    print(f"  各簇点数 {np.bincount(bayes_lab, minlength=2).tolist()}")
    print(f"  ARI = {adjusted_rand_index(y, bayes_lab):.4f}")
    print(f"\n对比: k-means ARI = {adjusted_rand_index(y, lab):.4f}  "
          f"（差 {adjusted_rand_index(y, bayes_lab) - adjusted_rand_index(y, lab):+.4f}）")

    print(f"""
关键对照：
  k-means 分界 {km_bound:.4f}（一个点）
  责任度分界 {left:.4f} 与 {right:.4f}（两个点）

你的结论：
  1. k-means 的分界和责任度的分界差多少？k-means 多划走了多少个点？
  2. 为什么最优解是【两个】分界？（提示：讲义 05 第 1.1 节——两个分量的
     对数密度之差是开口向下的抛物线，方差不同时它有两个零点。方差相同时
     二次项系数为零，退化成直线，才只剩一个零点。）
  3. 左边那个分界（约 -1.37）的含义是什么？数轴最左端的一小截尾巴，
     为什么反而归簇 1？
""")


# ---------------------------------------------------------------------------
# 练习 3：软归属长什么样
# ---------------------------------------------------------------------------


def exercise3():
    print(SEP)
    print("练习 3：责任度曲线与『不确定』的样本")
    print(SEP)
    x, y = make_two_cluster_data()
    C, lab, J, _ = kmeans(x[:, None], 2, seed=0)
    km_bound = (C[0, 0] + C[1, 0]) / 2

    r = responsibility(x)
    mid = (r >= 0.4) & (r <= 0.6)
    print(f"责任度落在 [0.4, 0.6] 的样本（『不确定』区）:")
    print(f"  样本数 {int(mid.sum())} / {len(x)} = {mid.sum() / len(x) * 100:.2f}%")
    print(f"  其中真实属于簇 0 的 {int(((y == 0) & mid).sum())} 个，"
          f"属于簇 1 的 {int(((y == 1) & mid).sum())} 个")
    print(f"  这些点的 x 范围: [{x[mid].min():.3f}, {x[mid].max():.3f}]")
    print(f"\nk-means 怎么处理这些点：按 {km_bound:.4f} 一刀切，"
          f"落在左边的一律算簇 0")
    n_km_wrong = int(((x < km_bound) & (y == 1)).sum())
    print(f"  k-means 把 {n_km_wrong} 个真实属于簇 1 的点判成了簇 0")

    # 三个典型点
    print("\n三个典型点：")
    for t in (0.0, 1.0, 3.0):
        rx = float(responsibility(np.array([t]))[0])
        print(f"  x={t:<6.1f} 责任度={rx:.4f}  k-means 判为簇 "
              f"{'0' if t < km_bound else '1'}")

    # 画图
    xs = np.linspace(-8, 14, 1000)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.0, 1.25]})
    ax = axes[0]
    bins = np.linspace(-8, 14, 90)
    ax.hist(x[y == 0], bins=bins, color=COLORS[0], alpha=0.65, label="真实簇 0（紧）")
    ax.hist(x[y == 1], bins=bins, color=COLORS[1], alpha=0.65, label="真实簇 1（散）")
    ax.axvline(km_bound, color="black", ls="--", lw=2, label=f"k-means 分界 {km_bound:.2f}")
    ax.set_ylabel("样本数")
    ax.set_title("责任度曲线：k-means 只能切一刀，责任度是连续过渡", pad=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(xs, responsibility(xs), color=COLORS[2], lw=2.4, label="责任度 $r_1(x)$")
    ax.axhline(0.5, color="gray", ls=":", lw=1.4)
    ax.axvline(km_bound, color="black", ls="--", lw=2)
    ax.axhspan(0.4, 0.6, color=COLORS[2], alpha=0.12, label="[0.4, 0.6] 不确定区")
    ax.scatter(x[mid], r[mid], s=16, color="#C44E52", zorder=5,
               label=f"落在不确定区的 {int(mid.sum())} 个样本")
    ax.set_xlabel("x")
    ax.set_ylabel("责任度 $r_1(x)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="center right")

    fig.tight_layout()
    path = FIG / "practice05_responsibility.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {path.relative_to(FIG.parent)}")

    print("""
你的结论：
  1. 『不确定』（责任度在 0.4-0.6）的样本占比有多大？它们集中在 x 的哪个区间？
  2. k-means 对这些点做了什么？这样做合理吗？如果下游任务对这些点的误判
     代价很高，你会怎么用责任度来改进？
  3. 看图：责任度曲线在 x=0 附近有个低谷（约 0.075），往右升到 1，
     往左也升到 1。这说明曲线的形状是什么？（提示：不是单调的）
""")


# ---------------------------------------------------------------------------
# 练习 4：权重的作用
# ---------------------------------------------------------------------------


def exercise4():
    print(SEP)
    print("练习 4：混合权重 π 如何移动分界")
    print(SEP)
    print(f"固定 mu/var（簇0: {MU0}/{VAR0}，簇1: {MU1}/{VAR1}），只改权重\n")

    def boundaries(w0, w1):
        f = lambda t: (np.log(w0) + log_normal_1d(t, MU0, VAR0)) - \
                      (np.log(w1) + log_normal_1d(t, MU1, VAR1))
        grid = np.linspace(-60, 60, 24001)
        vals = f(grid)
        roots = []
        for i in range(len(grid) - 1):
            if vals[i] * vals[i + 1] < 0:
                lo, hi = grid[i], grid[i + 1]
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if f(mid) * f(lo) > 0:
                        lo = mid
                    else:
                        hi = mid
                roots.append((lo + hi) / 2)
        return sorted(roots)

    print(f"{'权重 (w0,w1)':<18}{'左分界':<14}{'右分界':<14}{'右分界相对平移'}")
    base_right = None
    for w0, w1 in [(0.5, 0.5), (0.9, 0.1), (0.1, 0.9)]:
        roots = boundaries(w0, w1)
        L = f"{roots[0]:.4f}" if roots else "无"
        R = f"{roots[-1]:.4f}" if roots else "无"
        if (w0, w1) == (0.5, 0.5):
            base_right = roots[-1]
        shift = f"{roots[-1] - base_right:+.4f}" if roots and base_right is not None else "—"
        print(f"({w0}, {w1}){'':<10}{L:<14}{R:<14}{shift}")

    # 画不同权重下的责任度曲线（需要练习 1 已实现）
    try:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        xs = np.linspace(-8, 14, 1000)
        for (w0, w1), col in zip([(0.5, 0.5), (0.9, 0.1), (0.1, 0.9)], COLORS):
            ax.plot(xs, responsibility(xs, w0=w0, w1=w1), color=col, lw=2.2,
                    label=f"$\\pi=({w0},{w1})$")
        ax.axhline(0.5, color="gray", ls=":", lw=1.4)
        ax.set_xlabel("x")
        ax.set_ylabel("责任度 $r_1(x)$")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("先验 $\\pi$ 越大，该分量的『势力范围』越宽", pad=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = FIG / "practice05_weights.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[fig] {path.relative_to(FIG.parent)}")
    except NotImplementedError:
        print("\n[跳过画图] 画责任度曲线需要先实现练习 1 的 responsibility()")

    print("""
你的结论：
  1. w=(0.9,0.1) 时右分界从 0.9745 移到了哪里？往哪边移？
  2. w=(0.1,0.9) 时呢？方向相反吗？
  3. 为什么"先验大的分量势力范围更宽"？从公式上看，π 出现在哪里？
     （提示：r1 的分子里有 w1、分母里有 w0 和 w1。w1 增大让分子变大、
      分母也变大，但分子的相对增幅更大。）
  4. 实务上这意味着什么？如果某个簇在数据里本来就占 90%，即便它的中心和
     另一个簇靠得很近，k-means 和 GMM 谁更容易把它认出来？
""")


# ---------------------------------------------------------------------------
# 练习 5：二维对照（方差相同 vs 相差 4 倍）
# ---------------------------------------------------------------------------


def exercise5():
    print(SEP)
    print("练习 5：k-means 的损失随两簇方差差异增大而增大")
    print(SEP)

    def log_normal_2d(X, mu, var):
        """二维各向同性高斯的对数密度（协方差 = var * I）。"""
        d2 = ((X - mu) ** 2).sum(axis=1)
        return -0.5 * 2 * np.log(2 * np.pi * var) - d2 / (2 * var)

    def run_2d(var1, tag):
        means = np.array([[0.0, 0.0], [3.0, 3.0]])
        covs = np.array([np.eye(2) * 0.5, np.eye(2) * var1])
        X2, y2 = dg.make_synthetic_data(means, covs, np.array([0.5, 0.5]),
                                        n_samples=2000, seed=5)
        # 用真实参数的 Bayes 划分
        l0 = np.log(0.5) + log_normal_2d(X2, means[0], 0.5)
        l1 = np.log(0.5) + log_normal_2d(X2, means[1], var1)
        bayes = (l1 > l0).astype(int)
        C2, lab2, J2, _ = kmeans(X2, 2, seed=0)
        ari_b = adjusted_rand_index(y2, bayes)
        ari_k = adjusted_rand_index(y2, lab2)
        print(f"  {tag}")
        print(f"    真实簇点数   {np.bincount(y2).tolist()}")
        print(f"    Bayes(真实参数) 点数 {np.bincount(bayes, minlength=2).tolist()}  ARI = {ari_b:.4f}")
        print(f"    k-means        点数 {np.bincount(lab2, minlength=2).tolist()}  ARI = {ari_k:.4f}")
        print(f"    k-means 相对 Bayes 差 {ari_k - ari_b:+.4f}")
        return ari_b, ari_k

    print("情景 A：两簇方差相差 4 倍（簇1 是 2.0，簇0 是 0.5）")
    ari_bA, ari_kA = run_2d(2.0, "方差比 4:1")

    print("\n情景 B：两簇方差相同（都是 0.5）")
    ari_bB, ari_kB = run_2d(0.5, "方差比 1:1")

    print(f"""
对照总结：
  方差比 4:1 时，k-means 比 Bayes 低 {ari_bA - ari_kA:.4f}
  方差比 1:1 时，k-means 比 Bayes 低 {ari_bB - ari_kB:.4f}

你的结论：
  1. 方差相同时，k-means 和 Bayes 的 ARI 一样吗？为什么？
     （提示：所有簇协方差相同且各向同性时，k-means 的分界恰好就是 Bayes 分界。）
  2. 方差相差 4 倍时，k-means 掉了多少？这个差距说明它的损失和什么有关？
  3. 把讲义 05 第 1 节那个一维例子（方差相差 8 倍，k-means ARI 0.4171 对
     Bayes 0.6871）和这里对比：方差差异越大，k-means 的损失越大吗？
     你可以把 run_2d 的 var1 换成 0.5、1.0、2.0、4.0、8.0 各跑一次，
     把 ARI 画成曲线验证这个猜想。
""")


# ---------------------------------------------------------------------------


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = {"1": exercise1, "2": exercise2, "3": exercise3,
            "4": exercise4, "5": exercise5}
    if which == "all":
        for fn in todo.values():
            try:
                fn()
            except NotImplementedError as e:
                print(f"\n[跳过] {e}")
            print()
    elif which in todo:
        todo[which]()
    else:
        print("用法：python practice05.py [1|2|3|4|5|all]")


if __name__ == "__main__":
    main()
