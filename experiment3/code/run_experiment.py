"""实验三主驱动脚本。

本文件把"流程编排、结果落盘、参数对照表"这类重复劳动写好了，你只需要在
标了 `TODO(你)` 的地方接上自己实现的算法函数。实验报告里要用的每个数字和
每张图都由本脚本产出，不需要手工抄终端输出。

运行方式（在工作目录 code/ 下）：

    python run_experiment.py

产出：
    results/run_summary.csv          每次运行一行的汇总表
    results/recovery_synthetic.json  合成数据上的完整参数、逐轮似然、协方差形态
    results/uci_results.json         UCI 数据集上的指标
    figures/*.png                    所有图片
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import viz
from data_gen import (
    TRUE_COVS,
    TRUE_MEANS,
    TRUE_WEIGHTS,
    load_dataset,
    make_synthetic_data,
)
from metrics import adjusted_rand_index, best_match_accuracy, silhouette_score
from viz import (
    plot_clusters,
    plot_convergence_compare,
    plot_curve,
    plot_gmm_ellipses,
    plot_parameter_recovery,
    plot_soft_assignment,
)

RESULT_DIR = Path(__file__).resolve().parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

SUMMARY_ROWS = []


# ---------------------------------------------------------------------------
# 结果记录
# ---------------------------------------------------------------------------


def add_summary(dataset, method, k, init, seed, log_lik, n_iter, ari, acc, extra=None):
    """追加一行汇总记录，最后统一写入 CSV。"""
    row = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "init": init,
        "seed": seed,
        "log_lik": None if log_lik is None else round(float(log_lik), 6),
        "n_iter": n_iter,
        "ari": None if ari is None else round(float(ari), 6),
        "acc": None if acc is None else round(float(acc), 6),
    }
    if extra:
        row.update(extra)
    SUMMARY_ROWS.append(row)
    return row


def flush_summary():
    """把汇总表写成 CSV。"""
    if not SUMMARY_ROWS:
        return None
    path = RESULT_DIR / "run_summary.csv"
    header = list(SUMMARY_ROWS[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in SUMMARY_ROWS:
            fh.write(",".join("" if row.get(h) is None else str(row.get(h)) for h in header) + "\n")
    print(f"[csv] {path}  ({len(SUMMARY_ROWS)} 行)")
    return path


def dump_json(name, payload):
    path = RESULT_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[json] {path}")
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"无法序列化的类型: {type(obj)}")


# ---------------------------------------------------------------------------
# 参数恢复分析（本部分已实现，直接用）
# ---------------------------------------------------------------------------


def empirical_stats(X, labels, k):
    """按给定标签分组计算经验统计量（比例、均值、协方差）。

    协方差用 `bias=True`（除以 N_k），与 EM 的 M 步公式一致。
    """
    X = np.asarray(X)
    n = len(X)
    weights, means, covs = [], [], []
    for j in range(k):
        mask = labels == j
        weights.append(mask.sum() / n)
        means.append(X[mask].mean(axis=0))
        covs.append(np.cov(X[mask].T, bias=True))
    return np.array(weights), np.array(means), np.array(covs)


def align_components(est_means, true_means):
    """枚举 K! 种排列，返回使均值偏差最小的排列。

    Returns
    -------
    perm : list[int]
        满足 `est_means[perm]` 与 `true_means` 逐分量对齐的排列。
    max_err : float
        该排列下的最大绝对偏差。
    """
    k = len(true_means)
    best_perm, best_err = list(range(k)), np.inf
    for perm in permutations(range(k)):
        err = np.abs(np.asarray(est_means)[list(perm)] - np.asarray(true_means)).max()
        if err < best_err:
            best_err, best_perm = err, list(perm)
    return best_perm, float(best_err)


def covariance_summary(cov):
    """用特征值比、行列式、主轴方向概括一个协方差矩阵，便于写入表格。

    特征向量有符号歧义：$v$ 与 $-v$ 表示同一条轴，`np.linalg.eigh` 返回哪个
    方向是任意的。若不规范化，水平的主轴可能被报成 $\\pm 179^\\circ$ 而不是
    $0^\\circ$，与设定值对照时看起来"完全错了"。这里把主轴统一到 $x$ 分量非负
    的朝向，角度落在 $(-90^\\circ, 90^\\circ]$ 内。
    """
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-300, None)
    major = vecs[:, -1]
    if major[0] < 0:
        major = -major
    return {
        "eigvals": vals.tolist(),
        "eigval_ratio": float(vals[-1] / vals[0]),
        "major_axis": major.tolist(),
        "angle_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
        "det": float(np.linalg.det(cov)),
        "trace": float(np.trace(cov)),
    }


def recovery_table(X, y, params, resp=None):
    """生成"估计值 vs 经验统计量 vs 设定值"的对照数据。

    这是第 4.4 节要求的两层验收的落地：第一层比较估计值与经验统计量
    （应当精确一致），第二层比较似然值（设定值应当更低）。
    """
    from gmm import compute_log_likelihood  # 你实现的函数

    X = np.asarray(X)
    k = len(params["weights"])
    labels = None if resp is None else resp.argmax(axis=1)

    emp_w, emp_mu, emp_cov = empirical_stats(X, y, k)
    perm, _ = align_components(params["means"], emp_mu)

    est_w = np.asarray(params["weights"])[perm]
    est_mu = np.asarray(params["means"])[perm]
    est_cov = np.asarray(params["covs"])[perm]

    ll_est = compute_log_likelihood(X, params["weights"], params["means"], params["covs"])
    ll_true = compute_log_likelihood(X, TRUE_WEIGHTS, TRUE_MEANS, TRUE_COVS)

    return {
        "permutation": perm,
        "weights": {
            "estimated": est_w.tolist(),
            "empirical": emp_w.tolist(),
            "true": TRUE_WEIGHTS.tolist(),
        },
        "means": {
            "estimated": est_mu.tolist(),
            "empirical": emp_mu.tolist(),
            "true": TRUE_MEANS.tolist(),
        },
        "max_err_vs_empirical": {
            "weights": float(np.abs(est_w - emp_w).max()),
            "means": float(np.abs(est_mu - emp_mu).max()),
            "covs": float(np.abs(est_cov - emp_cov).max()),
        },
        "max_err_vs_true": {
            "weights": float(np.abs(est_w - TRUE_WEIGHTS).max()),
            "means": float(np.abs(est_mu - TRUE_MEANS).max()),
            "covs": float(np.abs(est_cov - TRUE_COVS).max()),
        },
        "log_likelihood_compare": {
            "estimated_params": float(ll_est),
            "true_params": float(ll_true),
            "estimated_is_better": bool(ll_est >= ll_true),
        },
        "covariance_forms": [covariance_summary(c) for c in est_cov],
        "true_covariance_forms": [covariance_summary(c) for c in TRUE_COVS],
        "assignment_ari": None if labels is None else float(adjusted_rand_index(y, labels)),
    }


def print_recovery(table):
    """把对照表打印成可读形式。"""
    print("\n--- 参数恢复对照（已按最优排列对齐）---")
    print("估计均值:\n", np.round(table["means"]["estimated"], 4))
    print("经验均值:\n", np.round(table["means"]["empirical"], 4))
    print("设定均值:\n", np.round(table["means"]["true"], 4))
    print("估计比例:", np.round(table["weights"]["estimated"], 4))
    print("经验比例:", np.round(table["weights"]["empirical"], 4))
    print("设定比例:", np.round(table["weights"]["true"], 4))
    print(f"与经验统计量的最大偏差: {table['max_err_vs_empirical']}")
    print(f"与设定值的最大偏差:     {table['max_err_vs_true']}")
    cmp_ = table["log_likelihood_compare"]
    print(
        f"对数似然: 估计参数 {cmp_['estimated_params']:.4f}  vs  "
        f"设定参数 {cmp_['true_params']:.4f}  (估计更优: {cmp_['estimated_is_better']})"
    )
    for i, (est, tru) in enumerate(zip(table["covariance_forms"], table["true_covariance_forms"])):
        print(
            f"  协方差 {i}: 估计长短轴比 {est['eigval_ratio']:.2f} 主轴 {est['angle_deg']:.1f}°"
            f" | 设定比 {tru['eigval_ratio']:.2f} 主轴 {tru['angle_deg']:.1f}°"
        )


# ---------------------------------------------------------------------------
# 第一部分到第三部分：合成数据实验
# ---------------------------------------------------------------------------


def experiment_synthetic(n_samples=1500, seed=0, run_seed_sweep=True):
    """合成数据上的完整实验：数据生成 -> k-means -> GMM -> 图表 -> 落盘。"""
    from gmm import gmm_em
    from kmeans import kmeans_restarts

    X, y = make_synthetic_data(
        TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS, n_samples=n_samples, seed=seed
    )
    emp_w, emp_mu, emp_cov = empirical_stats(X, y, 3)
    print(f"\n=== 合成数据 {X.shape} ===")
    print(f"经验比例 {emp_w.round(4)}  与设定值的均值最大偏差 {np.abs(emp_mu - TRUE_MEANS).max():.4f}")

    # ---- 图 1：真实分布与真实协方差椭圆 ----
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_clusters(ax, X=X, labels=y, centroids=TRUE_MEANS, title="真实分布（按真实分量着色）")
    plot_gmm_ellipses(ax, TRUE_MEANS, TRUE_COVS, label_prefix="真实")
    viz.save_fig(fig, "fig01_true_distribution")

    # =====================================================================
    # TODO(你) 第 1 步：k-means 多次重启 + seed 扫描
    # =====================================================================
    # 要求：
    #   best, all_inertias = kmeans_restarts(X, k=3, n_init=10)
    #   再对 s in range(10) 各跑一次 kmeans(X, 3, seed=s)，收集每条的
    #   history 里的 inertia 序列、以及该次的准确率和 ARI。
    #   把每次的 seed / inertia / acc / ari 写进 add_summary("synthetic", "kmeans", ...)。
    raise NotImplementedError("TODO(你): 调用 kmeans_restarts 与 kmeans(seed=s)")

    # ---- 图 2：k-means 结果 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    plot_clusters(axes[0], X=X, labels=best["labels"], centroids=best["centroids"],
                  title=f"k-means（重启 10 次，J={best['inertia']:.1f}）")
    plot_curve(axes[1], y=inertia_history(best["history"]), label="重启后最优的一次",
               title="k-means 目标函数下降曲线", ylabel="簇内平方和 J", logy=True)
    viz.save_fig(fig, "fig02_kmeans_result")

    # =====================================================================
    # TODO(你) 第 2 步：GMM + EM
    # =====================================================================
    # 要求：
    #   params, resp, log_lik, history = gmm_em(X, k=3, init="kmeans", seed=seed)
    #   并断言 np.all(np.diff(log_lik) >= -1e-9)，把结果写进 add_summary。
    raise NotImplementedError("TODO(你): 调用 gmm_em")

    # ---- 图 3：GMM 结果（软归属 + 估计椭圆） ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    plot_soft_assignment(axes[0], X=X, resp=resp, means=params["means"], covs=params["covs"],
                         title="GMM 软归属（透明度 = 最大责任度）")
    plot_parameter_recovery(TRUE_MEANS, params["means"], TRUE_COVS, params["covs"])
    viz.save_fig(fig, "fig03_gmm_result")

    # ---- 图 4：对数似然曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    plot_curve(axes[0], y=log_lik, label="GMM 对数似然", title="EM 每轮对数似然", ylabel="log L")
    plot_curve(axes[1], y=inertia_history(best["history"]), label="k-means 的 J", title="k-means 每轮 J",
               ylabel="簇内平方和 J", logy=True)
    viz.save_fig(fig, "fig04_convergence")

    # ---- 参数恢复对照 ----
    table = recovery_table(X, y, params, resp)
    table["log_lik"] = [float(v) for v in log_lik]
    table["n_iter"] = len(log_lik)
    dump_json("recovery_synthetic.json", table)
    print_recovery(table)
    add_summary("synthetic", "gmm", 3, "kmeans", seed, log_lik[-1], len(log_lik),
                adjusted_rand_index(y, resp.argmax(1)), best_match_accuracy(y, resp.argmax(1))[0])

    # =====================================================================
    # TODO(你) 第 3 步：K = 2/3/4 的对数似然与 BIC 对比
    # =====================================================================
    # 要求：对 k in (2, 3, 4) 各跑一次 gmm_em，记录最终对数似然、迭代次数，
    #       按原理文档 6.2 节的公式算出 BIC，写入 add_summary 与 JSON。
    #       注意 k=4 时多出来的分量可能只分到极少数样本，这正是需要观察的现象。

    # =====================================================================
    # TODO(你) 第 4 步：初始化敏感性对照
    # =====================================================================
    # 要求：init 取 "kmeans" 与 "random" 各跑 seed=0..9，收集最终对数似然，
    #       用 plot_convergence_compare 画同一张图上的收敛曲线，
    #       并报告两组似然的均值与标准差。

    return X, y, best, params, resp, log_lik


# ---------------------------------------------------------------------------
# 第四部分：UCI 数据集实验
#
# 三个数据集各有明确分工，不是随便挑的大数据：
#   pendigits  主实验  样本量 10992、维度 16、10 类，聚得出结构且跑得动
#   optdigits  维度陷阱 64 维 10 类，每分量 562 样本却要估 2080 个协方差参数，
#                      用来展示高维下 GMM 反而不如 k-means，以及正则化的作用
#   electricity 反面案例 45312 个样本却几乎没有簇结构（ARI ≈ 0），
#                      说明样本量不等于可聚类性
#   seg_train/test 泛化实验 训练集 210、测试集 2100，用于检验学到的簇结构
#                      能否推广到没见过的数据
# ---------------------------------------------------------------------------


def inertia_history(history):
    """从 k-means 的 history 里取出每轮 inertia 序列。"""
    return [h["inertia"] for h in history]


def bic(log_lik, n_samples, k, d):
    """BIC = p*ln(N) - 2*ln(L)，p 为满协方差 GMM 的自由参数个数。

    p = K*(D + D(D+1)/2 + 1) - 1，最后减 1 是因为混合系数只有 K-1 个自由度。
    """
    p = k * (d + d * (d + 1) // 2 + 1) - 1
    return p * np.log(n_samples) - 2.0 * log_lik


def evaluate_clustering(X, y, labels, tag, sample_size=None):
    """算一组聚类结果的指标（ARI/准确率/轮廓系数），返回字典。"""
    ari = adjusted_rand_index(y, labels)
    acc, mapping = best_match_accuracy(y, labels)
    sil = silhouette_score(X, labels, sample_size=sample_size, seed=0)
    return {"tag": tag, "ari": float(ari), "acc": float(acc), "silhouette": float(sil), "mapping": mapping}


# ---------------------------------------------------------------------------
# 主实验：pendigits（10992 x 16，10 类）
# ---------------------------------------------------------------------------


def experiment_pendigits(seed=0):
    """最大且聚得出结构的数据集：k-means vs GMM、K 扫描、初始化敏感性。"""
    from gmm import gmm_em
    from kmeans import kmeans_restarts

    X, y, names = load_dataset("pendigits")
    k_true = len(np.unique(y))
    print(f"\n=== 主实验 UCI: pendigits {X.shape}  类别 {np.bincount(y)} ===")

    # =====================================================================
    # TODO(你) 第 5 步：k-means 与 GMM 的对比
    # =====================================================================
    # 要求：
    #   1. best, all_J = kmeans_restarts(X, k=k_true, n_init=10)，
    #      用 evaluate_clustering 得到指标，add_summary 记录一行；
    #   2. params, resp, log_lik, hist = gmm_em(X, k=k_true, init="kmeans", seed=seed)，
    #      检查 np.all(np.diff(log_lik) >= -|L|*1e-12)，add_summary 记录一行；
    #   3. 用 plot_convergence_compare 把两者的收敛曲线（对数似然与 inertia）画出来
    #      （注意两者量纲不同，要么画两张子图，要么各自归一化）；
    #   4. 把结果写入 results/uci_pendigits.json。
    raise NotImplementedError("TODO(你): pendigits 上调用 kmeans_restarts 与 gmm_em")

    # =====================================================================
    # TODO(你) 第 6 步：K 扫描与 BIC 模型选择
    # =====================================================================
    # 要求：对 k in (2, 3, 5, 10, 12) 各跑一次 gmm_em，记录 ARI、对数似然、
    #       迭代次数，用上面的 bic() 算 BIC（注意 bic 的 p 公式在 K 不是真实
    #       类别数时同样适用）。观察 BIC 选出的 K 是否等于真实的 10。
    #       10 类、16 维、每分量约 1100 样本，协方差是 136 个参数，条件数可能
    #       偏高，若对数似然不单调请用相对判据并记录条件数。


# ---------------------------------------------------------------------------
# 维度陷阱：optdigits（5620 x 64，10 类）
# ---------------------------------------------------------------------------


def experiment_optdigits_dimension(seed=0):
    """展示高维下 GMM 的反常行为，以及正则化强度对它的影响。"""
    from gmm import gmm_em
    from kmeans import kmeans_restarts

    X, y, names = load_dataset("optdigits")
    k_true = len(np.unique(y))
    n, d = X.shape
    print(f"\n=== 维度陷阱 UCI: optdigits {X.shape}  类别 {np.bincount(y)} ===")
    print(f"每分量约 {n // k_true} 个样本，协方差有 {d * (d + 1) // 2} 个自由参数")

    # =====================================================================
    # TODO(你) 第 7 步：维度陷阱与正则化扫描
    # =====================================================================
    # 要求：
    #   1. k-means 作为基线（k=k_true）；
    #   2. GMM 用默认 reg=1e-6 跑一次，报告 ARI 与最大条件数
    #      （np.linalg.cond 对每个协方差算，取最大）；
    #   3. 对 reg in (1e-6, 1e-4, 1e-3, 1e-2) 各跑一次，记录 ARI 与最终对数似然，
    #      做成表格。参考实现的结果是 ARI 从 0.238 升到 0.540，但对数似然从
    #      344055 掉到 -44301 —— 用这组数字说明"提升 ARI 的代价是放弃了
    #      GMM 的表达能力"。这里必须用 bic() 里的 d 参数注意 D=64。
    #   4. 用 plot_curve 画 reg 与 ARI、对数似然的关系（双 y 轴）。
    raise NotImplementedError("TODO(你): optdigits 的正则化扫描")


# ---------------------------------------------------------------------------
# 反面案例：electricity（45312 x 8，2 类）
# ---------------------------------------------------------------------------


def experiment_electricity_contrast(seed=0):
    """样本量最大但没有簇结构，用来否定"数据更大就更可聚类"。"""
    from gmm import gmm_em
    from kmeans import kmeans_restarts

    X, y, names = load_dataset("electricity")
    k_true = len(np.unique(y))
    majority = np.bincount(y).max() / len(y)
    print(f"\n=== 反面案例 UCI: electricity {X.shape}  类别 {np.bincount(y)} ===")
    print(f"参考基线：全猜多数类的准确率 = {majority:.4f}，随机划分的 ARI 期望 = 0")

    # =====================================================================
    # TODO(你) 第 8 步：确认它聚不出结构
    # =====================================================================
    # 要求：
    #   1. k-means 与 GMM（k=k_true）各跑一次，报告 ARI、准确率、迭代次数、耗时；
    #   2. 关键一步：把准确率与上面的 majority 基线对比，说明模型并没有学到
    #      任何东西 —— 这是本实验的结论所在，不要只报一个 ARI 数字；
    #   3. 时间序列诊断：算几个主要特征（nswdemand、transfer）的 lag-1 自相关，
    #      用 plot_curve 画出来，说明有效样本量远小于 45312；
    #   4. 把结论写进 JSON：这个数据集不适合做聚类，理由是类别由未来价格变化
    #      定义而非当前特征的簇结构。
    raise NotImplementedError("TODO(你): electricity 的反面案例")


# ---------------------------------------------------------------------------
# 泛化实验：segmentation 训练集 / 测试集
# ---------------------------------------------------------------------------


def experiment_seg_generalization(seed=0):
    """在训练集上估计簇结构，检验它能否推广到测试集。"""
    from gmm import gmm_em
    from kmeans import kmeans_restarts

    Xtr, ytr, names = load_dataset("seg_train")
    Xte, yte, _ = load_dataset("seg_test")
    print(f"\n=== 泛化实验 UCI: seg_train {Xtr.shape} -> seg_test {Xte.shape} ===")

    # =====================================================================
    # TODO(你) 第 9 步：训练/测试分离的聚类评估
    # =====================================================================
    # 要求：
    #   1. 在训练集上跑 kmeans_restarts 得到中心 C，用 assign_clusters(Xte, C)
    #      把测试集划分出来，算测试集的 ARI —— 这检验"学到的几何结构"是否泛化；
    #   2. 在训练集上跑 gmm_em 得到参数，用 estimate_log_resp(Xte, ...) 得到
    #      测试集的责任度，取 argmax 得到划分，同样算 ARI；
    #   3. 对照：直接在测试集上训练再评估（拟合+评估同一份数据），与第 1、2 步
    #      的结果比较。如果两者差距很大，说明簇结构只在训练集上成立；
    #   4. 注意测试集的类别分布是每类 300 个样本，非常均衡，因此大多数类基线
    #      是 1/7 ≈ 0.143，报告时要和它比较。
    raise NotImplementedError("TODO(你): segmentation 的泛化实验")


def main():
    """运行全部实验。可以用命令行参数只跑其中一部分：

        python run_experiment.py synthetic      只跑合成数据
        python run_experiment.py uci            只跑四个 UCI 实验
        python run_experiment.py all            全部（默认）
    """
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "synthetic"):
        experiment_synthetic()
    if which in ("all", "uci"):
        experiment_pendigits()
        experiment_optdigits_dimension()
        experiment_electricity_contrast()
        experiment_seg_generalization()

    flush_summary()
    print("\n完成。结果见 results/ 与 figures/")


if __name__ == "__main__":
    main()
