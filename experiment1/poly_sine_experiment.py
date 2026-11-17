# -*- coding: utf-8 -*-
"""
实验一：多项式拟合正弦函数
内容：
1. 生成带高斯噪声的正弦数据；
2. 用多项式模型拟合；
3. 全批量梯度下降 + L2 正则化；
4. 比较不同阶数、样本量、学习率、正则化系数；
5. 画图并输出指标。
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

plt.rcParams["axes.unicode_minus"] = False
np.seterr(over="ignore", invalid="ignore")
RNG_SEED = 42


def generate_data(n, noise_std=0.2, seed=RNG_SEED):
    """在 [-1,1] 上生成 y=sin(2*pi*x)+高斯噪声"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = np.sin(2.0 * np.pi * x) + noise_std * rng.normal(size=n)
    return x, y


def polynomial_features(x, degree, mean=None, std=None):
    """
    构造 [1, x, x^2, ..., x^degree]。
    对非截距列做 z-score 标准化。
    训练集调用时 mean/std 为 None，会返回训练集统计量；测试集再传入同一组统计量。
    """
    x = np.asarray(x, dtype=float)
    Phi = np.vander(x, degree + 1, increasing=True)

    if degree >= 1:
        if mean is None:
            mean = Phi[:, 1:].mean(axis=0)
            std = Phi[:, 1:].std(axis=0)
            std = np.where(std < 1e-12, 1.0, std)
        Phi[:, 1:] = (Phi[:, 1:] - mean) / std
    else:
        mean = np.zeros(0)
        std = np.ones(0)
    return Phi, mean, std


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def train_gd(Phi, y, lr=0.1, epochs=60000, l2=0.0, reg_bias=False, tol=1e-12):
    """
    全批量梯度下降最小化:
        J(w) = 1/(2N) * ||Phi w - y||^2 + (lambda/2) * ||w_pen||^2
    其中除截距外都参与 L2 惩罚。
    返回 w 和 loss 历史。
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_samples, n_features = Phi.shape
    w = np.zeros(n_features, dtype=float)
    losses = []

    for epoch in range(epochs):
        y_pred = Phi @ w
        err = y_pred - y
        grad = Phi.T @ err / n_samples

        if l2 > 0.0:
            reg = l2 * w
            if not reg_bias:
                reg[0] = 0.0
            grad = grad + reg

        w = w - lr * grad
        if not np.all(np.isfinite(w)):
            raise FloatingPointError("梯度下降发散，请减小学习率或做特征标准化。")

        new_err = Phi @ w - y
        loss = 0.5 * np.mean(new_err ** 2)
        if l2 > 0.0:
            loss += 0.5 * l2 * np.sum(w[1:] ** 2)
        losses.append(float(loss))
        if epoch > 100 and abs(losses[-1] - losses[-2]) < tol:
            break

    return w, np.asarray(losses)


def fit_eval(x_train, y_train, x_test, y_test, degree, lr=0.1,
             epochs=60000, l2=0.0):
    """返回训练误差、测试误差、模型权重以及测试集预测"""
    Phi_train, mean, std = polynomial_features(x_train, degree)
    Phi_test, _, _ = polynomial_features(x_test, degree, mean=mean, std=std)
    w, losses = train_gd(Phi_train, y_train, lr=lr, epochs=epochs, l2=l2)
    train_mse = mse(y_train, Phi_train @ w)
    test_mse = mse(y_test, Phi_test @ w)
    return {
        "degree": degree, "train_mse": train_mse, "test_mse": test_mse,
        "w": w, "losses": losses, "mean": mean, "std": std,
    }


def main():
    noise_std = 0.2
    x_test = np.linspace(-1.0, 1.0, 600)
    y_test_clean = np.sin(2.0 * np.pi * x_test)

    # 固定同一个较大的数据池，取前 n 个作为不同样本量，保证比较尽量公平
    x_pool, y_pool = generate_data(200, noise_std=noise_std, seed=RNG_SEED)

    degrees = [1, 3, 5, 9, 15]
    sample_sizes = [15, 30, 100]
    base_lr = 0.1
    base_epochs = 60000

    print("=" * 88)
    print("实验1：不同数据量 N 和不同多项式阶数 M （无正则化, lr=%.2f, epochs=%d）" % (base_lr, base_epochs))
    print("=" * 88)
    print(f"{'N':>4s} {'M':>3s} {'训练MSE':>12s} {'测试MSE':>12s} {'||w||':>10s}")
    degree_results = {}
    for n in sample_sizes:
        x_train, y_train = x_pool[:n], y_pool[:n]
        for degree in degrees:
            res = fit_eval(x_train, y_train, x_test, y_test_clean,
                           degree=degree, lr=base_lr, epochs=base_epochs, l2=0.0)
            degree_results[(n, degree)] = res
            print(f"{n:4d} {degree:3d} {res['train_mse']:12.6f} "
                  f"{res['test_mse']:12.6f} {np.linalg.norm(res['w']):10.4f}")

    print("\n" + "=" * 88)
    print("实验2：L2 正则化系数影响（N=15, M=15, lr=%.2f, epochs=%d）" % (base_lr, base_epochs))
    print("=" * 88)
    n_small = 15
    x_train, y_train = x_pool[:n_small], y_pool[:n_small]
    l2_list = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    print(f"{'lambda':>10s} {'训练MSE':>12s} {'测试MSE':>12s} {'||w||':>10s}")
    l2_results = {}
    for l2 in l2_list:
        res = fit_eval(x_train, y_train, x_test, y_test_clean,
                       degree=15, lr=base_lr, epochs=base_epochs, l2=l2)
        l2_results[l2] = res
        print(f"{l2:10.4g} {res['train_mse']:12.6f} "
              f"{res['test_mse']:12.6f} {np.linalg.norm(res['w']):10.4f}")

    print("\n" + "=" * 88)
    print("实验3：学习率影响（N=30, M=15, 无正则化, epochs=60000）")
    print("=" * 88)
    n_mid = 30
    x_train, y_train = x_pool[:n_mid], y_pool[:n_mid]
    print(f"{'lr':>8s} {'训练MSE':>12s} {'测试MSE':>12s} {'状态':>8s}")
    lr_results = {}
    for lr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        try:
            res = fit_eval(x_train, y_train, x_test, y_test_clean,
                           degree=15, lr=lr, epochs=60000, l2=0.0)
            lr_results[lr] = res
            print(f"{lr:8.3f} {res['train_mse']:12.6f} {res['test_mse']:12.6f} {'正常':>8s}")
        except FloatingPointError:
            print(f"{lr:8.3f} {'---':>12s} {'---':>12s} {'发散':>8s}")

    # ----------------------------- 绘图 -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) 小样本高阶过拟合曲线
    ax = axes[0, 0]
    ax.scatter(x_pool[:n_small], y_pool[:n_small], s=30, c="k", label="Training data (N=15)")
    ax.plot(x_test, y_test_clean, "g--", linewidth=1.5, label="True sin(2*pi*x)")
    for degree, color in [(5, "tab:blue"), (15, "tab:red")]:
        res = degree_results[(n_small, degree)]
        Phi_test, _, _ = polynomial_features(x_test, degree, mean=res["mean"], std=res["std"])
        ax.plot(x_test, Phi_test @ res["w"], color=color, linewidth=1.5,
                label=f"M={degree}, test MSE={res['test_mse']:.3f}")
    res_reg = l2_results[1e-2]
    Phi_test, _, _ = polynomial_features(x_test, 15, mean=res_reg["mean"], std=res_reg["std"])
    ax.plot(x_test, Phi_test @ res_reg["w"], color="tab:orange", linestyle=":",
            linewidth=2.0, label=f"M=15, lambda=0.01, test MSE={res_reg['test_mse']:.3f}")
    ax.set_ylim(-2.0, 2.0)
    ax.set_title("(a) Overfitting: N=15, high-order polynomial")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 不同样本量/阶数的测试误差
    ax = axes[0, 1]
    for n in sample_sizes:
        test_errors = [degree_results[(n, d)]["test_mse"] for d in degrees]
        ax.semilogy(degrees, test_errors, "o-", label=f"N={n}")
    ax.set_title("(b) Test MSE vs polynomial degree")
    ax.set_xlabel("polynomial degree M")
    ax.set_ylabel("Test MSE (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) L2 正则化
    ax = axes[1, 0]
    train_errors = [l2_results[l2]["train_mse"] for l2 in l2_list]
    test_errors = [l2_results[l2]["test_mse"] for l2 in l2_list]
    xpos = np.arange(len(l2_list))
    ax.semilogy(xpos, train_errors, "o-", label="Train MSE")
    ax.semilogy(xpos, test_errors, "s-", label="Test MSE")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{v:.0e}" if 0 < v < 1e-3 else f"{v:g}" for v in l2_list])
    ax.set_title("(c) L2 regularization, N=15, M=15")
    ax.set_xlabel("lambda")
    ax.set_ylabel("MSE (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) 样本量影响（固定 M=15）
    ax = axes[1, 1]
    train_n = [degree_results[(n, 15)]["train_mse"] for n in sample_sizes]
    test_n = [degree_results[(n, 15)]["test_mse"] for n in sample_sizes]
    ax.semilogy(sample_sizes, train_n, "o-", label="Train MSE")
    ax.semilogy(sample_sizes, test_n, "s-", label="Test MSE")
    ax.set_title("(d) Effect of training size, M=15, lambda=0")
    ax.set_xlabel("N")
    ax.set_ylabel("MSE (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("poly_sine_results.png", dpi=150)
    print("\n图已保存: poly_sine_results.png")


if __name__ == "__main__":
    main()
