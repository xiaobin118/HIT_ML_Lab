#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二：逻辑回归完整实验代码
包含：
1. 无惩罚与 L2 惩罚逻辑回归
2. 梯度下降与牛顿法
3. 高斯生成数据实验
4. 与高斯朴素贝叶斯对比
5. UCI Bank Marketing 实际数据实验
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.linalg import solve

# 基础函数
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]


def standardize(X_train, X_test):
    """用训练集统计量标准化，避免测试集信息泄漏。"""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


# 逻辑回归：支持无惩罚与 L2 惩罚
class LogisticRegression:
    def __init__(self, lam=0.0):
        self.lam = lam
        self.theta = None
        self.history = []

    def _loss_grad(self, Xb, y):
        n = len(y)
        p = sigmoid(Xb @ self.theta)
        eps = 1e-12

        ce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * self.lam * np.sum(self.theta[1:] ** 2)
        loss = ce + reg

        error = p - y
        grad = Xb.T @ error / n
        if self.lam > 0:
            grad[1:] += self.lam * self.theta[1:]
        return loss, grad

    def fit_gd(self, X, y, lr=0.5, n_iter=20000, tol=1e-6, verbose=False):
        """批量梯度下降。"""
        Xb = add_bias(X)
        n, d1 = Xb.shape
        self.theta = np.zeros(d1)
        self.history = []

        for it in range(n_iter):
            loss, grad = self._loss_grad(Xb, y)
            self.theta -= lr * grad
            self.history.append(float(loss))

            if np.linalg.norm(grad) < tol:
                break

        if verbose:
            print("  GD iter=%d, loss=%.6f, |grad|=%.2e" % (it, loss, np.linalg.norm(grad)))
        return self

    def fit_newton(self, X, y, n_iter=50, tol=1e-8, verbose=False):
        """牛顿法（IRLS），H = 1/N X^T S X + lambda I。"""
        Xb = add_bias(X)
        n, d1 = Xb.shape
        self.theta = np.zeros(d1)
        self.history = []

        for it in range(n_iter):
            loss, grad = self._loss_grad(Xb, y)

            p = sigmoid(Xb @ self.theta)
            s = np.maximum(p * (1 - p), 1e-12)
            H = (Xb * s[:, None]).T @ Xb / n
            if self.lam > 0:
                H[1:, 1:] += self.lam * np.eye(d1 - 1)
            H += 1e-10 * np.eye(d1)

            try:
                delta = solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ grad

            self.theta -= delta
            self.history.append(float(loss))
            if np.linalg.norm(delta) < tol:
                break

        if verbose:
            print("  Newton iter=%d, loss=%.6f, |delta|=%.2e" % (it, loss, np.linalg.norm(delta)))
        return self

    def predict_proba(self, X):
        return sigmoid(add_bias(X) @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# 高斯朴素贝叶斯：忽略类内特征之间的相关性
class GaussianNB:
    def __init__(self):
        self.classes_ = None
        self.prior_ = {}
        self.mu_ = {}
        self.var_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            Xc = X[y == c]
            self.prior_[c] = len(Xc) / len(X)
            self.mu_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + 1e-9
        return self

    def predict_log_proba(self, X):
        logps = []
        for c in self.classes_:
            diff = X - self.mu_[c]
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[c]) + diff ** 2 / self.var_[c],
                axis=1
            )
            logps.append(np.log(self.prior_[c]) + log_lik)
        return np.column_stack(logps)

    def predict(self, X):
        idx = np.argmax(self.predict_log_proba(X), axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        # 二分类简化：classes_ = [0, 1]
        logp = self.predict_log_proba(X)
        score = logp[:, 1] - logp[:, 0]
        p1 = sigmoid(score)
        return np.c_[1 - p1, p1]


# ============================================================
# 人工数据与数据划分
# ============================================================
def make_gaussian_data(n_per_class=500, rho=0.8, seed=0):
    """
    类 0: mean = [-1, 0]
    类 1: mean = [ 1, 0]
    cov  = [[1, rho],
            [rho, 1]]
    """
    rng = np.random.default_rng(seed)
    mean0 = np.array([-1.0, 0.0])
    mean1 = np.array([1.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])

    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)

    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def split_dataset(X, y, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# 评价指标
# ============================================================
def classification_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def roc_curve_manual(y_true, y_score):
    """不依赖 sklearn 的 ROC 曲线计算。"""
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    # 只取分数发生变化的位置作为阈值
    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct, len(y_sorted) - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # 补上起点 (0,0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    auc = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0)
    return fpr, tpr, float(auc)


# 绘图工具
def plot_data(X, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               s=18, alpha=0.65, label="class 0", c="tab:orange")
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               s=18, alpha=0.65, label="class 1", c="tab:blue")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left")
    return ax


def make_meshgrid(X, resolution=200):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_boundary(proba1_func, X, ax=None, resolution=200,
                  fill=False, line_color="black", line_style="-",
                  label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    xx, yy, grid = make_meshgrid(X, resolution)
    p1 = proba1_func(grid).reshape(xx.shape)

    if fill:
        ax.contourf(xx, yy, p1, levels=[0, 0.5, 1],
                    alpha=0.15, colors=["tab:orange", "tab:blue"])

    ax.contour(xx, yy, p1, levels=[0.5],
               colors=line_color, linestyles=line_style, linewidths=1.8)
    return ax


# 实验一：相关高斯数据
def run_gaussian_experiment(outdir):
    result = {}

    X, y = make_gaussian_data(n_per_class=500, rho=0.8, seed=0)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3, seed=42)
    X_train_std, X_test_std, mu, sd = standardize(X_train, X_test)

    # 图 1：原始相关高斯数据散点图
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_data(X_train, y_train, ax)
    ax.set_title("Correlated Gaussian Data (rho=0.8)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_gaussian_data.png"), dpi=150)
    plt.close(fig)

    # 无惩罚逻辑回归：梯度下降
    lr_model = LogisticRegression(lam=0.0).fit_gd(
        X_train_std, y_train, lr=0.5, n_iter=20000, tol=1e-6, verbose=True
    )
    lr_train_pred = lr_model.predict(X_train_std)
    lr_test_pred = lr_model.predict(X_test_std)
    lr_train_acc = np.mean(lr_train_pred == y_train)
    lr_test_acc = np.mean(lr_test_pred == y_test)

    # 高斯朴素贝叶斯
    nb_model = GaussianNB().fit(X_train_std, y_train)
    nb_train_acc = np.mean(nb_model.predict(X_train_std) == y_train)
    nb_test_acc = np.mean(nb_model.predict(X_test_std) == y_test)

    # 图 2：决策边界对比
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_data(X_train_std, y_train, ax)

    lr_proba1 = lambda Z: lr_model.predict_proba(Z)
    nb_proba1 = lambda Z: nb_model.predict_proba(Z)[:, 1]

    plot_boundary(lr_proba1, X_train_std, ax=ax, fill=False,
                  line_color="black", line_style="-")
    plot_boundary(nb_proba1, X_train_std, ax=ax, fill=False,
                  line_color="red", line_style="--")

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="black", lw=1.8),
                Line2D([0], [0], color="red", lw=1.8, ls="--")]
    labels += ["Logistic Regression", "Gaussian NB"]
    ax.set_title("Logistic Regression vs Gaussian NB")
    ax.legend(handles, labels, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_boundary_compare.png"), dpi=150)
    plt.close(fig)

    # 图 3：loss 曲线
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lr_model.history)
    ax.set_xlabel("iteration")
    ax.set_ylabel("NLL loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_loss_curve.png"), dpi=150)
    plt.close(fig)

    # 正则化对比（这里 lambda 未除以样本数）
    lams = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    reg_results = []
    for lam in lams:
        model = LogisticRegression(lam=lam).fit_gd(
            X_train_std, y_train, lr=0.5, n_iter=20000, tol=1e-6
        )
        reg_results.append({
            "lambda": lam,
            "train_acc": float(np.mean(model.predict(X_train_std) == y_train)),
            "test_acc": float(np.mean(model.predict(X_test_std) == y_test)),
            "weight_norm": float(np.linalg.norm(model.theta[1:])),
            "bias": float(model.theta[0]),
            "weights": [float(v) for v in model.theta[1:]]
        })

    # 图 4：正则化对准确率和权重范数的影响
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xpos = np.arange(len(lams))
    axes[0].plot(xpos, [r["train_acc"] for r in reg_results], "o-", label="train")
    axes[0].plot(xpos, [r["test_acc"] for r in reg_results], "s--", label="test")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([str(l) for l in lams])
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy vs lambda")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xpos, [r["weight_norm"] for r in reg_results], "o-", color="tab:red")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(l) for l in lams])
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("||w||")
    axes[1].set_title("Weight norm vs lambda")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_regularization.png"), dpi=150)
    plt.close(fig)

    # 图 6：按 lambda 从 1.0 到 0.01 展示不同具体参数的 L2 收缩路径
    path_lams = [1.0, 0.1, 0.01]
    path_rows = [next(r for r in reg_results if r["lambda"] == lam) for lam in path_lams]
    xpos = np.arange(len(path_lams))
    w1_path = [r["weights"][0] for r in path_rows]
    w2_path = [r["weights"][1] for r in path_rows]
    bias_path = [r["bias"] for r in path_rows]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(xpos, w1_path, "o-", label="w1")
    ax.plot(xpos, w2_path, "s-", label="w2")
    ax.plot(xpos, bias_path, "^--", label="bias (not penalized)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["1.0", "0.1", "0.01"])
    ax.set_xlabel("lambda (from 1.0 to 0.01)")
    ax.set_ylabel("parameter value")
    ax.set_title("L2 parameter paths: lambda=1.0 -> 0.01")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_weight_paths.png"), dpi=150)
    plt.close(fig)

    result["data"] = {
        "n_per_class": 500,
        "rho": 0.8,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }
    result["logistic_regression"] = {
        "theta": [float(v) for v in lr_model.theta],
        "train_accuracy": float(lr_train_acc),
        "test_accuracy": float(lr_test_acc),
        "iterations": int(len(lr_model.history))
    }
    result["gaussian_nb"] = {
        "train_accuracy": float(nb_train_acc),
        "test_accuracy": float(nb_test_acc)
    }
    result["regularization"] = reg_results

    return result


# ============================================================
# 实验二：UCI Bank Marketing 实际数据
# ============================================================
def run_bank_experiment(outdir, data_path):
    df = pd.read_csv(data_path, sep=";")
    y = (df["y"] == "yes").astype(int).values
    X_df = pd.get_dummies(df.drop(columns=["y"]), drop_first=True)
    X = X_df.values.astype(float)
    feature_names = X_df.columns.tolist()

    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=0.3, seed=42
    )
    X_train_std, X_test_std, mu, sd = standardize(X_train, X_test)

    baseline_acc = float(np.mean(y_test == 0))

    # 牛顿法训练几个正则化强度
    lams = [0.0, 0.001, 0.01, 0.1, 1.0]
    bank_results = []
    best_model = None
    best_lam = None
    best_test_acc = -1.0

    for lam in lams:
        model = LogisticRegression(lam=lam).fit_newton(
            X_train_std, y_train, n_iter=50, tol=1e-8
        )
        pred = model.predict(X_test_std)
        metrics = classification_metrics(y_test, pred)
        metrics["lambda"] = lam
        metrics["test_accuracy"] = metrics["accuracy"]
        metrics["train_accuracy"] = float(np.mean(model.predict(X_train_std) == y_train))
        metrics["theta"] = [float(v) for v in model.theta]
        bank_results.append(metrics)

        # 先跑 lambda=0 的图，后面固定用它；这里记录最优用于结论
        if metrics["test_accuracy"] > best_test_acc:
            best_test_acc = metrics["test_accuracy"]
            best_model = model
            best_lam = lam

    # 用 lambda = 0 的模型画混淆矩阵，便于和前面结果对应
    # 保存最佳测试模型的全部参数（bias + 42 个特征权重）
    if best_model is not None:
        best_param_df = pd.DataFrame({
            "feature": ["bias"] + feature_names,
            "weight": [float(v) for v in best_model.theta]
        })
        best_param_df.to_csv(
            os.path.join(outdir, "best_bank_params.csv"),
            index=False, encoding="utf-8-sig"
        )

    model0 = LogisticRegression(lam=0.0).fit_newton(
        X_train_std, y_train, n_iter=50, tol=1e-8
    )
    pred0 = model0.predict(X_test_std)
    m0 = classification_metrics(y_test, pred0)
    m0["test_accuracy"] = m0["accuracy"]
    cm = np.array([[m0["tn"], m0["fp"]], [m0["fn"], m0["tp"]]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Bank Marketing Confusion Matrix (lambda=0)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_bank_confusion.png"), dpi=150)
    plt.close(fig)

    # 图 7：Bank Marketing 中绝对值最大的 5 个权重的正则化路径
    theta_paths = np.array([r["theta"] for r in bank_results])  # (n_lams, d+1)
    all_weights = theta_paths[:, 1:]                             # (n_lams, d)
    top_idx = np.argsort(np.abs(all_weights[0]))[::-1][:5]       # lambda=0 时 top 5

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for j in range(all_weights.shape[1]):
        ax.plot(lams, all_weights[:, j], color="gray", alpha=0.25, linewidth=1.0)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for k, j in enumerate(top_idx):
        ax.plot(lams, all_weights[:, j], "o-", color=colors[k],
                linewidth=1.8, label=feature_names[j])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("lambda (symlog scale)")
    ax.set_ylabel("weight value")
    ax.set_title("Bank Marketing: L2 regularization paths (top 5 weights)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig7_bank_weight_paths.png"), dpi=150)
    plt.close(fig)

    # 图 9：Bank Marketing 在不同 lambda 下的测试性能
    lam_list = [r["lambda"] for r in bank_results]
    xpos = np.arange(len(lam_list))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].plot(xpos, [r["train_accuracy"] for r in bank_results], "o-", label="train accuracy")
    axes[0].plot(xpos, [r["test_accuracy"] for r in bank_results], "s-", label="test accuracy")
    axes[0].axhline(baseline_acc, color="gray", linestyle="--", label="majority baseline")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([str(l) for l in lam_list])
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Bank Marketing: accuracy vs lambda")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xpos, [r["precision"] for r in bank_results], "o-", label="precision")
    axes[1].plot(xpos, [r["recall"] for r in bank_results], "s-", label="recall")
    axes[1].plot(xpos, [r["f1"] for r in bank_results], "^-", label="F1")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(l) for l in lam_list])
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("score")
    axes[1].set_title("Bank Marketing: P / R / F1 vs lambda")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig9_bank_metrics.png"), dpi=150)
    plt.close(fig)

    # 图 10：最佳测试模型的 ROC 曲线
    best_scores = best_model.predict_proba(X_test_std)
    fpr, tpr, auc = roc_curve_manual(y_test, best_scores)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot(fpr, tpr, linewidth=2,
            label="best test model (lambda=%.4g, AUC=%.4f)" % (best_lam, auc))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Bank Marketing: ROC curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig10_bank_roc.png"), dpi=150)
    plt.close(fig)

    return {
        "dataset": "UCI Bank Marketing (bank.csv)",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "positive_ratio": float(np.mean(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "baseline_accuracy": baseline_acc,
        "feature_names": feature_names,
        "best_test_lambda": float(best_lam),
        "best_test_auc": float(auc),
        "lambda_results": bank_results,
        "confusion_lambda0": m0
    }


def plot_best_test_weights(results, outdir):
    """按测试集准确率选出两个实验中的最佳模型，并画出具体权重。"""
    g = results["gaussian_experiment"]
    b = results["bank_experiment"]

    best_g = max(g["regularization"], key=lambda r: r["test_acc"])
    best_b = max(b["lambda_results"], key=lambda r: r["test_accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # 左图：人工高斯数据最佳测试权重
    labels_g = ["bias", "w1", "w2"]
    values_g = [best_g["bias"]] + best_g["weights"]
    axes[0].bar(labels_g, values_g, color=["gray", "tab:blue", "tab:orange"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("weight value")
    axes[0].set_title("Synthetic: best test weights (lambda=%.4g)" % best_g["lambda"])
    for i, v in enumerate(values_g):
        axes[0].text(i, v, "%.3f" % v, ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.3)

    # 右图：Bank Marketing 最佳测试权重（绝对值最大的 12 个）
    theta = np.array(best_b["theta"])
    names = b["feature_names"]
    order = np.argsort(np.abs(theta[1:]))[::-1][:12]
    axes[1].barh([names[j] for j in order], theta[1:][order], color="tab:green")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("weight value")
    axes[1].set_title("Bank: top 12 weights (best test lambda=%.4g)" % best_b["lambda"])
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig8_best_test_weights.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    results = {}

    print("Running synthetic Gaussian experiment...")
    results["gaussian_experiment"] = run_gaussian_experiment(base)

    print("Running UCI Bank Marketing experiment...")
    data_path = os.path.join(base, "..", "data", "bank.csv")
    results["bank_experiment"] = run_bank_experiment(base, data_path)

    plot_best_test_weights(results, base)

    out_json = os.path.join(base, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nResults saved to:", out_json)
    print(json.dumps(results, ensure_ascii=False, indent=2))
