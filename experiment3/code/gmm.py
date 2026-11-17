"""混合高斯模型与 EM 算法。

本文件的全部函数体需要你实现，对应
`docs/02_实现任务与编码路线.md` 第 4 节的任务编号。

贯穿全文件的接口约定：**密度与似然一律在对数域计算**。高维高斯的密度
可能小到 1e-50 而直接下溢为 0，一旦出现 0/0，责任度矩阵会变成 nan 并
污染后续所有计算。因此 `estimate_log_resp` 返回 log 责任度，对数似然由
逐样本 log-sum-exp 累加得到，全程不出现"先算密度再取对数"的写法。

数学依据（详见 `docs/01_原理_kmeans与GMM的EM推导.md` 第 3、4 节）：

    E 步:  log gamma_nk = log pi_k + log N(x_n | mu_k, Sigma_k) - logsumexp_j(...)
    M 步:  N_k    = sum_n gamma_nk
           mu_k   = (1/N_k) sum_n gamma_nk x_n
           Sigma_k= (1/N_k) sum_n gamma_nk (x_n - mu_k)(x_n - mu_k)^T
           pi_k   = N_k / N
"""

from __future__ import annotations

import numpy as np

from data_gen import _regularize, normal_logpdf


def logsumexp(a, axis=None, keepdims=False):
    """数值稳定的 log(sum(exp(a)))。

    实现要点：先减去沿 axis 的最大值再取指数求和，最后加回最大值。
    不减最大值时，a 中若有较大的元素（例如 -1e-3 量级的对数密度），
    exp 会溢出为 inf。若某一行全是 -inf（该样本在所有分量下密度都下溢），
    返回 -inf 之前要避免 inf - inf = nan，可以给最大值加一个下限，
    例如用一个很负的常数（-1e300 或者 np.log(np.finfo(float).tiny) 的若干倍）。
    """
    raise NotImplementedError("任务 3.0：实现 logsumexp")


def estimate_log_gaussian_prob(X, means, covs):
    """任务 3.1：返回 (N, K) 的 log N(x_n | mu_k, Sigma_k)。

    对每个分量调用 data_gen.normal_logpdf，堆成列。
    """
    raise NotImplementedError("任务 3.1：实现 estimate_log_gaussian_prob")


def estimate_log_resp(X, weights, means, covs):
    """任务 3.2（E 步）：返回 log 责任度矩阵与逐样本对数似然。

    Parameters
    ----------
    X : (N, D)
    weights : (K,) 混合系数 pi_k
    means : (K, D)
    covs : (K, D, D)

    Returns
    -------
    log_resp : (N, K)
        每行 log 责任度。要求 exp(log_resp).sum(axis=1) 恒等于 1。
    log_prob_norm : (N,)
        每个样本的对数似然 log p(x_n)，即 log_resp 每行的 logsumexp。
        它的累加就是总对数似然，不要再单独算一遍密度。
    """
    raise NotImplementedError("任务 3.2：实现 estimate_log_resp")


def m_step(X, log_resp):
    """任务 3.3（M 步）：由 log 责任度更新参数。

    Returns
    -------
    weights : (K,)
    means : (K, D)
    covs : (K, D, D)

    关键要求
    --------
    1. 协方差必须用**更新后的**均值计算。用旧均值不会报错，但会让
       对数似然在若干轮后开始下降，破坏 EM 的单调性保证。
    2. 协方差是加权形式并除以 N_k（不是 N_k - 1），即
       np.cov(..., bias=True) 的约定。
    3. 不要写 (N, D, D) 的三重循环，用
       diff = X - mu_k; cov_k = (resp[:, k, None] * diff).T @ diff / nk[k]
       一次算完。
    4. 检查 nk 是否出现 0 或极小值，这类退化分量要与调用方约定好怎么
       处理（本实验里由 gmm_em 统一重置），不要在 m_step 内静默产生 nan。
    """
    raise NotImplementedError("任务 3.3：实现 m_step")


def compute_log_likelihood(X, weights, means, covs):
    """任务 3.5：用朴素定义独立重算对数似然，用于校验主循环。

        sum_n log sum_k pi_k N(x_n | mu_k, Sigma_k)

    这里可以"先算密度再取对数"（配合对数域求和），但必须与主循环的
    log-sum-exp 路径结果一致到 1e-6。这一步是为了排除"E 步写错但曲线
    依然好看"的情况。
    """
    raise NotImplementedError("任务 3.5：实现 compute_log_likelihood")


def init_params(X, k, init="kmeans", seed=0, reg=1e-6):
    """初始化 GMM 参数。

    Parameters
    ----------
    init : {"kmeans", "random"}
        "kmeans"：用 k-means 的中心作初始均值，协方差取各簇协方差
        （样本数不足时回退到全体协方差），权重取 1/K。这是实际最常用的
        做法，因为 k-means 已经提供了一个不错的粗划分。
        "random"：随机选 k 个样本作均值，协方差统一取全体数据的协方差，
        权重取 1/K。用于对照实验，量化 EM 对初始化的敏感性。

    Returns
    -------
    weights, means, covs

    注意：本函数依赖 kmeans.kmeans_restarts（任务 2.4），先完成 k-means
    再写这里。
    """
    raise NotImplementedError("任务 3.4 的一部分：实现 init_params")


def gmm_em(X, k, max_iter=200, tol=1e-6, reg=1e-6, init="kmeans", seed=0):
    """任务 3.4：EM 主循环。

    Parameters
    ----------
    X : (N, D)
    k : int
        分量个数。
    max_iter : int
        最大迭代次数。
    tol : float
        相对收敛阈值：|L_new - L_old| / |L_old| < tol 时停止。
    reg : float
        协方差正则化系数，防止奇异导致 Cholesky 失败。
    init : {"kmeans", "random"}
        初始化方式，见 init_params。
    seed : int
        随机种子。

    Returns
    -------
    params : dict
        {'weights': (K,), 'means': (K, D), 'covs': (K, D, D)}
    resp : (N, K)
        最终责任度（由最终 log_resp 取指数）。
    log_lik : list[float]
        每次迭代后的总对数似然，长度为实际迭代次数。
    history : list[dict]
        每轮记录 {'iter', 'log_lik', 'delta', 'n_k'}，n_k 为各分量有效样本数。

    实现提示
    --------
    1. 先 init_params。
    2. 每轮：E 步算 log_resp 与 log_prob_norm，累加得到总对数似然，
       **先把这一轮的似然记进 log_lik 和 history，再判断是否收敛**，
       这样 log_lik 的长度等于实际迭代次数，画图时横坐标不会错位。
    3. 收敛判据用相对变化量；第一轮没有前一轮可比较，直接进入 M 步。
    4. M 步更新后：给每个协方差加正则（用 data_gen._regularize），
       检查 n_k 是否有退化分量，若有则重置该分量的均值/协方差/权重。
    5. 退出后可以做一次收尾 E 步，保证返回的 resp 与 params 一致。

    验收标准
    --------
    先说清楚"正确结果"到底指什么。EM 求的是**当前这份样本**的最大似然解，
    不是设定值本身。有限样本下经验统计量与设定值必然有抽样偏差，例如
    seed=0、N=1500 时经验均值与 TRUE_MEANS 的最大偏差是 0.079，经验协方差
    与 TRUE_COVS 的最大偏差是 0.06，经验样本比例是 [0.391, 0.343, 0.266]
    而非设定的 [0.4, 0.35, 0.25]。拿估计值去和设定值比，只能得出"差距在
    抽样误差量级内"这种模糊结论，无法区分"算法写错了"和"样本恰好如此"。

    因此正确的验收方式是分两层。

    第一层，估计值应当**精确等于**按真实标签算出的经验统计量（这是 MLE 的
    定义，容差取 1e-6 量级）。分量编号与真实类别之间的关系是任意的，比较
    前要用 3! 种排列中偏差最小的那个对齐（`metrics.best_match_accuracy`
    返回的 mapping 可以复用，或自己枚举 permutations）：

        acc, mapping = best_match_accuracy(y, resp.argmax(1))
        # 对每个预测簇 j，取 X[y == mapping[j]] 的经验均值/协方差与之比较

    第二层，用对数似然做交叉验证：把设定值 (TRUE_WEIGHTS, TRUE_MEANS,
    TRUE_COVS) 直接代入似然函数，得到的值应当**低于**估计参数的值（seed=0
    的实测结果是 -5677.78 对 -5671.24）。这一条同时证明了 EM 确实提升了
    似然，以及它找到的解在这份样本上优于上帝视角的设定值。

    具体到本实验：
    (a) np.diff(log_lik) >= -1e-9 逐项成立（严格单调不降）；
    (b) 估计参数与经验统计量一致到 1e-6 量级（第一层，硬指标）；
    (c) 经验统计量与设定值的偏差在抽样误差量级内（均值偏差约 0.08、
        协方差偏差约 0.06、比例偏差约 0.016），据此说明算法恢复出了
        三种协方差形态：分量 1 的长短轴比约 10:1（横躺）、分量 2 约 7:1
        且主轴呈 45 度倾斜、分量 0 接近各向同性。注意形态结论要基于排列
        对齐后的分量，不能直接比较下标；
    (d) 与 compute_log_likelihood 的结果一致到 1e-6；
    (e) 设定值的对数似然低于估计参数的对数似然（第二层）。
    """
    raise NotImplementedError("任务 3.4：实现 gmm_em")


if __name__ == "__main__":
    # 自检脚本：实现完成后运行 cd code && python gmm.py
    from data_gen import TRUE_COVS, TRUE_MEANS, TRUE_WEIGHTS, make_synthetic_data

    X, y = make_synthetic_data(TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS, n_samples=1500, seed=0)

    params, resp, log_lik, history = gmm_em(X, k=3, seed=0)
    print("迭代次数:", len(log_lik))
    print("对数似然: 首轮 %.2f  末轮 %.2f" % (log_lik[0], log_lik[-1]))
    print("单调不降:", bool(np.all(np.diff(log_lik) >= -1e-9)))
    print("独立校验:", compute_log_likelihood(X, params["weights"], params["means"], params["covs"]))
    print("估计混合系数:", params["weights"].round(4), " 设定:", TRUE_WEIGHTS)
    print("估计均值:\n", params["means"].round(3), "\n设定:\n", TRUE_MEANS)
    for j in range(3):
        vals, vecs = np.linalg.eigh(params["covs"][j])
        print(f"分量 {j}: 特征值 {vals.round(4)}  主轴 {vecs[:, -1].round(3)}")
