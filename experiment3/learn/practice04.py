#!/usr/bin/env python
# coding=utf-8
"""讲义 04 的练习：密度、似然、高斯与数值陷阱。

和练习 03 一样：**需要你写的只有练习 1**（手写高斯密度函数），其余练习的
计算和画图都已经写好，你的任务是运行、观察、回答每个练习末尾的问题。

运行方式：

    cd learn
    python practice04.py 1      # 只跑练习 1
    python practice04.py all    # 全部（很快，几秒）
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

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CLASSES = ["setosa", "versicolor", "virginica"]
SEP = "=" * 70


# ===========================================================================
# TODO(你) 练习 1：手写一元高斯密度
# ===========================================================================
def normal_pdf_1d(x, mu, sigma):
    """一元高斯密度

        p(x) = 1/sqrt(2*pi*sigma^2) * exp( -(x-mu)^2 / (2*sigma^2) )

    x 可以是标量，也可以是数组（返回同形状的结果）。

    实现提示：直接照公式写。np.sqrt、np.exp 都是逐元素的，所以传数组也能用。
    注意把 sigma^2 先算出来复用（写成 sigma ** 2 即可），别重复算三遍。
    """
    p = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return p


def normal_logpdf_1d(x, mu, sigma):
    """一元高斯对数密度

        log p(x) = -0.5*log(2*pi*sigma^2) - (x-mu)^2 / (2*sigma^2)

    实现提示：**不要**写成 np.log(normal_pdf_1d(...))。虽然数学上等价，但当
    密度小到 1e-300 以下时，先算密度会下溢成 0，取对数就变成 -inf；直接按
    上面的公式算则不会出问题。这正是讲义第 5 节强调"在对数域计算"的意思。
    """
    log_p = -0.5 * np.log(2 * np.pi * sigma ** 2) - ((x - mu) ** 2) / (2 * sigma ** 2)
    return log_p


# ---------------------------------------------------------------------------
# 练习 1
# ---------------------------------------------------------------------------


def exercise1():
    print(SEP)
    print("练习 1：手写一元高斯密度，并验证几个已知值")
    print(SEP)

    # 校验点 1：标准正态在 x=0 处的密度
    got = normal_pdf_1d(0.0, 0.0, 1.0)
    want = 1.0 / np.sqrt(2 * np.pi)
    print(f"标准正态 x=0 的密度:")
    print(f"  你的结果 {got:.10f}   理论值 {want:.10f}   相对误差 {abs(got-want)/want:.2e}")

    got_log = normal_logpdf_1d(0.0, 0.0, 1.0)
    want_log = -0.5 * np.log(2 * np.pi)
    print(f"标准正态 x=0 的对数密度:")
    print(f"  你的结果 {got_log:.10f}   理论值 {want_log:.10f}")
    print(f"  两者关系 e^(对数密度) = {np.exp(got_log):.10f}（应等于上面的密度）")

    # 校验点 2：数组输入与标量输入应一致
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\n数组输入 x={xs.tolist()}:")
    print(f"  normal_pdf_1d     = {np.round(normal_pdf_1d(xs, 0.0, 1.0), 6).tolist()}")
    print(f"  normal_logpdf_1d  = {np.round(normal_logpdf_1d(xs, 0.0, 1.0), 6).tolist()}")

    # 校验点 3：用 Iris setosa 的花瓣长度（mu=1.46, sd=0.17）
    X, y, _ = dg.load_dataset("iris_4d")
    v = X[y == 0, 2]
    mu, sd = v.mean(), v.std()
    print(f"\nIris setosa 花瓣长度: mu={mu:.4f}  sd={sd:.4f}")
    for x in (mu, mu + sd, 3.0):
        print(f"  x={x:.4f}  对数密度={normal_logpdf_1d(x, mu, sd):>10.4f}")

    print("""
说明：x=3.0 离均值有 (3.0-1.46)/0.17 ≈ 9.1 个标准差，是个极端的离群点。
这一点的对数密度很负，是合理的。把它的密度和 x=mu 处的密度相比，比值的
数量级有多大？算一下 np.exp(normal_logpdf_1d(3.0,mu,sd) - normal_logpdf_1d(mu,mu,sd))。

你的结论（回答两问）：
  1. 三个校验点都通过了吗？如果不通过，是公式写错还是常数写错？
  2. 为什么 normal_logpdf_1d 不能写成 np.log(normal_pdf_1d(...))？
     构造一个具体的数值例子说明（提示：试试 D 维标准正态，密度可以小到多少）。
""")


# ---------------------------------------------------------------------------
# 练习 2：一个高斯 vs 三个高斯
# ---------------------------------------------------------------------------


def exercise2():
    print(SEP)
    print("练习 2：用对数似然说明『一个高斯不够』")
    print(SEP)
    X, y, _ = dg.load_dataset("iris_4d")
    pl = X[:, 2]

    # (a) 一个高斯拟合全部数据
    mu1, sd1 = pl.mean(), pl.std()
    logL1 = normal_logpdf_1d(pl, mu1, sd1).sum()
    print(f"(a) 一个高斯: mu={mu1:.3f} sd={sd1:.3f}")
    print(f"    对数似然 = {logL1:.2f}")

    # (b) 三个高斯，每个品种一个（权重取样本比例）
    print(f"\n(b) 三个高斯（每品种一个）:")
    logL3 = 0.0
    for j, c in enumerate(CLASSES):
        v = pl[y == j]
        w = len(v) / len(pl)
        part = (np.log(w) + normal_logpdf_1d(v, v.mean(), v.std())).sum()
        logL3 += part
        print(f"    {c:<11} mu={v.mean():.3f} sd={v.std():.3f} 权重={w:.3f} 该组对数似然={part:.2f}")
    print(f"    合计对数似然 = {logL3:.2f}")
    print(f"\n    差值 = {logL3 - logL1:.1f}（越高越好）")

    # 画图对比
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    ax.hist(pl, bins=24, density=True, color="#BBBBBB", edgecolor="white")
    xs = np.linspace(0.5, 7.5, 500)
    ax.plot(xs, normal_pdf_1d(xs, mu1, sd1), color="#C44E52", lw=2.4)
    ax.axvline(mu1, color="#C44E52", ls=":", lw=1.2)
    ax.set_title(f"一个高斯（对数似然 {logL1:.1f}）", fontsize=10)
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("密度")
    ax.set_ylim(0, 0.7)
    ax.annotate("峰落在 2.0-3.0\n这段空白区", xy=(mu1, 0.22), xytext=(5.0, 0.52),
                fontsize=9, color="#C44E52", ha="center",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))

    ax = axes[1]
    for j, (c, col) in enumerate(zip(CLASSES, COLORS)):
        v = pl[y == j]
        ax.hist(v, bins=14, density=True, color=col, edgecolor="white", alpha=0.5)
        ax.plot(xs, normal_pdf_1d(xs, v.mean(), v.std()), color=col, lw=2.2,
                label=f"{c}: {v.mean():.2f}±{v.std():.2f}")
    ax.set_title(f"三个高斯（对数似然 {logL3:.1f}）", fontsize=10)
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylim(0, 6.0)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("同一个高斯 vs 每个品种一个高斯：对数似然差 %.0f" % (logL3 - logL1), y=1.03)
    fig.tight_layout()
    path = FIG / "practice04_one_vs_three.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {path.relative_to(FIG.parent)}")

    print("""
注意：三个高斯用了更多参数（3 组 mu/sd 而不是 1 组），所以对数似然更高是
"必然"的——参数多总能拟合得更好。所以在真实的模型选择里不能只看似然，还要
罚参数个数（这就是 AIC/BIC 干的事，讲义 07 会讲）。

但有两点是这份数据特有的、值得注意的：
  · setosa 那一组的标准差只有 0.17，而整体标准差是 1.76——差了 10 倍。
  · 一个高斯必须用一个 sigma 描述所有三个品种，它被迫取得很大。

你的结论：
  1. 三个高斯的对数似然比一个高斯高多少？
  2. 看左图：那个高斯在哪些位置高估/低估了数据？
  3. 为什么"一个高斯"必须把 sigma 取得很大？如果不取大，会有什么后果？
     提示——如果 sigma 只有 0.17（像 setosa 那样），那 versicolor 和 virginica
     的样本（离整体均值 3.76 有两三个单位远）的密度会变成多少？
""")


# ---------------------------------------------------------------------------
# 练习 3：亲手制造下溢
# ---------------------------------------------------------------------------


def exercise3():
    print(SEP)
    print("练习 3：密度乘积的溢出与下溢")
    print(SEP)
    tiny = np.finfo(float).tiny
    print(f"float64 能表示的最小正数 tiny = {tiny:.3e}\n")

    print("(1) D 维标准正态，距离原点 r 处的对数密度:")
    print("    公式: log p = -D/2*log(2*pi) - r^2*D/2")
    print(f"\n    {'D':<6}{'r=1 的对数密度':<18}{'r=1 的密度':<16}{'r=3 的对数密度':<18}{'r=3 的密度'}")
    for D in (1, 2, 10, 50, 100, 200, 400, 500, 600):
        lp1 = -0.5 * D * np.log(2 * np.pi) - 0.5 * 1 * D
        lp3 = -0.5 * D * np.log(2 * np.pi) - 0.5 * 9 * D
        p1 = np.exp(lp1) if lp1 > np.log(tiny) else 0.0
        p3 = np.exp(lp3) if lp3 > np.log(tiny) else 0.0
        print(f"    {D:<6}{lp1:<18.2f}{p1:<16.3e}{lp3:<18.2f}{p3:.3e}")

    crit = {}
    for r in (1.0, 3.0):
        for D in range(1, 2000):
            if np.exp(-0.5 * D * np.log(2 * np.pi) - 0.5 * r * r * D) < tiny:
                crit[r] = D
                break
    print(f"\n    临界维数: r=1 时 D={crit.get(1.0)}，r=3 时 D={crit.get(3.0)}")

    print("\n(2) 但真正的问题在【样本很多】的时候。假设 N 个样本，每个的密度都是 p:")
    N = 5000
    print(f"    {'D':<6}{'单点密度(半径1)':<20}{'N=5000 的对数似然':<22}{'N=5000 的连乘结果'}")
    for D in (2, 10, 20):
        p = np.exp(-0.5 * D * np.log(2 * np.pi) - 0.5 * D)
        logL = N * np.log(p)
        prod = "下溢为 0" if logL < np.log(tiny) else f"{np.exp(logL):.3e}"
        print(f"    {D:<6}{p:<20.4e}{logL:<22.0f}{prod}")

    print("""
你的结论：
  1. 单个点的密度要到多少维才会自己下溢？这个数字和你直觉一致吗？
  2. 但 5000 个样本连乘，在 D=2 时就已经下溢。为什么"样本多"比"维数高"更危险？
     提示——连乘 N 次相当于把量级放大 N 倍。用 log10(p)*N 算一下需要多少个
     数量级才到 -308。
  3. 如果代码里真的出现了连乘下溢为 0，下一步会发生什么？
     提示——回忆讲义 04 第 5 节末尾：似然变成 0 之后，算"这个点属于哪个簇"
     的时候会出现什么运算？
""")


# ---------------------------------------------------------------------------
# 练习 4：马氏距离与椭圆
# ---------------------------------------------------------------------------


def exercise4():
    print(SEP)
    print("练习 4：协方差矩阵、马氏距离与椭圆")
    print(SEP)

    def ellipse_from_cov(cov, mu=(0.0, 0.0), level=1.0, n=400):
        """返回满足 (x-mu)^T Sigma^{-1} (x-mu) = level 的点集。

        做法：对 Sigma 做特征分解 Sigma = V diag(lam) V^T。令
        x = mu + V diag(sqrt(lam*level)) u，其中 u 是单位圆上的点，
        则马氏距离恰好等于 level（下面会验证）。
        """
        lam, V = np.linalg.eigh(cov)
        t = np.linspace(0, 2 * np.pi, n)
        u = np.stack([np.cos(t), np.sin(t)], 1)
        return (u * np.sqrt(lam * level)) @ V.T + np.array(mu)

    cases = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "各向同性（圆）"),
        (np.array([[2.5, 0.0], [0.0, 0.3]]), "横躺细长"),
        (np.array([[1.2, 0.9], [0.9, 1.2]]), "倾斜"),
    ]

    print(f"{'协方差':<26}{'特征值':<22}{'长短轴比':<10}{'主轴方向':<12}")
    for cov, tag in cases:
        lam, V = np.linalg.eigh(cov)
        major = V[:, -1]
        ang = np.degrees(np.arctan2(major[1], major[0]))
        angstr = f"{ang:.1f} 度"
        print(f"{tag:<26}{str(np.round(lam,3)):<22}{lam[-1]/lam[0]:<10.2f}{angstr}")

    print("\n验证：椭圆上的点确实满足马氏距离 = 1")
    for cov, tag in cases:
        pts = ellipse_from_cov(cov, level=1.0)
        inv = np.linalg.inv(cov)
        d = np.einsum("ni,ij,nj->n", pts, inv, pts)
        print(f"  {tag:<14} 马氏距离平方: 最小 {d.min():.10f} 最大 {d.max():.10f}（应全为 1）")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    rng = np.random.default_rng(0)
    for ax, (cov, tag) in zip(axes, cases):
        pts = rng.multivariate_normal([0, 0], cov, size=400)
        ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.35, color=COLORS[0], edgecolor="none")
        for lv, ls in ((1.0, "-"), (4.0, "--")):
            e = ellipse_from_cov(cov, level=lv)
            ax.plot(e[:, 0], e[:, 1], color="#C44E52", lw=1.8, ls=ls,
                    label=f"马氏距离²={lv:g}" if ls == "-" else None)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("马氏距离为常数的点集是椭圆：半轴长 = sqrt(特征值)", y=1.03)
    fig.tight_layout()
    path = FIG / "practice04_ellipse.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {path.relative_to(FIG.parent)}")

    print("""
你的结论：
  1. 三种协方差的特征值和长短轴比对上了吗？（圆 1.00、横躺 8.33、倾斜 7.00）
  2. level=1 和 level=4 的两条椭圆，半轴长度差几倍？
     提示——半轴长正比于 sqrt(level)，所以 sqrt(4)/sqrt(1)=2 倍。
     二维时 level=1 的椭圆包含约 39% 的概率质量，level=4（即 2σ）约 86%。
  3. 第三个协方差的非对角元是 0.9（不为零），这说明 $x_1$ 和 $x_2$ 有什么关系？
     算一下 np.corrcoef 验证你的判断。
""")


# ---------------------------------------------------------------------------
# 练习 5：k-means 丢掉了什么（思考题）
# ---------------------------------------------------------------------------


def exercise5():
    print(SEP)
    print("练习 5（思考题）：k-means 丢掉了协方差的哪些信息")
    print(SEP)
    X, y, _ = dg.load_dataset("iris_4d")
    pl = X[:, 2]

    # K=1 时 k-means 的目标函数就是"所有点到均值的距离平方和"
    J = ((pl - pl.mean()) ** 2).sum()
    print(f"取 Iris 花瓣长度，K=1 的 k-means:")
    print(f"  唯一的中心 = 均值 = {pl.mean():.4f}")
    print(f"  J = 所有点到均值的距离平方和 = {J:.4f}")
    print(f"  J/N = 方差 = {J/len(pl):.4f}（即标准差 {np.sqrt(J/len(pl)):.4f} 的平方）")

    # 二维情形：把花瓣长度和花瓣宽度放一起
    X2 = X[:, 2:4]
    mu2 = X2.mean(axis=0)
    diff = X2 - mu2
    J2 = (diff ** 2).sum()
    cov = (diff.T @ diff) / len(X2)
    print(f"\n换成二维（花瓣长度 + 花瓣宽度），K=1:")
    print(f"  中心 = {np.round(mu2, 4).tolist()}")
    print(f"  J = {J2:.4f}  （= 协方差矩阵的迹 × N: {np.trace(cov)*len(X2):.4f}）")
    print(f"  协方差矩阵:\n{np.round(cov, 4)}")
    lam, V = np.linalg.eigh(cov)
    print(f"  特征值 {np.round(lam,4).tolist()}  长短轴比 {lam[-1]/lam[0]:.2f}")
    print(f"  最大特征值对应的方向（椭圆长轴）: {np.round(V[:,-1],4).tolist()}"
          f"  即 {np.degrees(np.arctan2(V[1,-1], V[0,-1])):.1f}°")

    # 关键实证：构造三组"J 完全相同、形状完全不同"的数据
    trace_cov = float(np.trace(cov))
    n = len(X2)
    covA = np.eye(2) * (trace_cov / 2)                       # 各向同性（圆）
    ratio = 10.0                                             # 方差比 10:1
    covB = np.diag([trace_cov / (1 + ratio) * ratio, trace_cov / (1 + ratio)])
    ang = np.radians(45.0)                                   # 倾斜 45 度
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    covC = R @ covB @ R.T

    print("\n构造三组二维数据，让它们的 J 完全相同（协方差的迹都一样）:")
    print(f"  {'形状':<20}{'协方差矩阵':<32}{'特征值':<22}{'trace':<9}{'J = N*trace'}")
    for tag, c in [("A 各向同性（圆）", covA), ("B 横躺细长", covB), ("C 倾斜细长", covC)]:
        lam2, _ = np.linalg.eigh(c)
        print(f"  {tag:<20}{str(np.round(c, 3).tolist()):<32}"
              f"{str(np.round(lam2, 3)):<22}{np.trace(c):<9.3f}{n * np.trace(c):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    rng = np.random.default_rng(1)
    for ax, (tag, c) in zip(axes, [("A 各向同性（圆）", covA),
                                   ("B 横躺细长", covB),
                                   ("C 倾斜细长", covC)]):
        pts = rng.multivariate_normal([0, 0], c, size=500)
        ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.4, color=COLORS[0], edgecolor="none")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(f"{tag}\nJ = {n * np.trace(c):.1f}", fontsize=9.5)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.suptitle(f"J 完全相同（都是 {n * trace_cov:.1f}），但数据形状完全不同", y=1.03)
    fig.tight_layout()
    path = FIG / "practice04_same_J_diff_shape.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {path.relative_to(FIG.parent)}")

    print(f"""
关键观察：上面 J 的数值（或协方差的迹）只告诉你"数据摊开了多大"，
它对应的是两个特征值之【和】。而椭圆的【朝向】和【长短轴比】由两个特征值
各自的大小决定——这个信息在求和/求迹的一瞬间就丢掉了。

刚才那三组数据的 J 完全相同（都是 {n * trace_cov:.1f}），但形状一个是圆、
一个是横躺的细长椭圆、一个是倾斜 45 度的细长椭圆。只看 J 无法区分它们。

你的结论：
  1. 第一问：J 是什么量？它和协方差矩阵有什么关系？
     （提示：K=1 时 J = N * trace(Sigma)，而 trace = 特征值之和）
  2. 第二问：为什么说 k-means "丢掉"了协方差的信息？
     具体地说，trace 能告诉你什么、不能告诉你什么？
  3. 第三问：如果一份二维数据的真实结构是一个"细长的斜椭圆"，用 k-means
     （K=1）描述它，会丢掉哪些信息？这些信息在 GMM 里靠什么恢复？
     （提示：讲义 04 第 4 节的三张椭圆图——它们的 trace 分别是 2.0、2.8、
      2.4，差别不大，但形状完全不同。）
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
    }
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
        print("用法：python practice04.py [1|2|3|4|5|all]")


if __name__ == "__main__":
    main()
