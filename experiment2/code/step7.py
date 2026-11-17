import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from step1_forward import predict_proba, sigmoid
from step3 import train_logreg_gd
from step4 import (
    accuracy,
    make_gaussian_data,
    split_dataset,
    standardize,
)
from step5_plot import (
    make_meshgrid,
    plot_data,
    plot_decision_boundary,
)


class GaussianNB:
    def __init__(self):
        self.classes_ = None
        self.prior_ = {}
        self.mu_ = {}
        self.var_ = {}

    def fit(self, X, y):
        """
        X: (N, d)
        y: (N,)
        """
        # TODO 1: self.classes_ = np.unique(y)
        # unique: 返回数组中唯一值的排序副本
        # Examples:
        # >>> np.unique([1, 2, 2, 3])
        # array([1, 2, 3])
        self.classes_ = np.unique(y)
        # TODO 2: 对每个类 c:
        #     Xc = X[y == c]
        #     self.prior_[c] = len(Xc) / len(X)
        #     self.mu_[c] = Xc.mean(axis=0)
        #     self.var_[c] = Xc.var(axis=0) + 1e-9
        for c in self.classes_:
            Xc = X[y == c]
            self.prior_[c] = len(Xc) / len(X)
            self.mu_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + 1e-9
        return self

    def predict_log_proba(self, X):
        """
        返回:
            (N, C) 的 log posterior
        """
        # TODO 1: logps = []
        # TODO 2: 遍历 self.classes_
        #     diff = X - self.mu_[c]
        #     log_lik = -0.5 * np.sum(
        #         np.log(2 * np.pi * self.var_[c]) + diff ** 2 / self.var_[c],
        #         axis=1
        #     )
        #     logps.append(np.log(self.prior_[c]) + log_lik)
        # TODO 3: return np.column_stack(logps)
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
        # TODO:
        # logp = self.predict_log_proba(X)
        # idx = np.argmax(logp, axis=1)
        # return self.classes_[idx]
        logp = self.predict_log_proba(X)
        idx = np.argmax(logp, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        # TODO:
        # logp = self.predict_log_proba(X)
        # score = logp[:, 1] - logp[:, 0]
        # p1 = sigmoid(score)
        # return np.c_[1 - p1, p1]
        logp = self.predict_log_proba(X)
        score = logp[:, 1] - logp[:, 0]
        p1 = sigmoid(score)
        return np.c_[1 - p1, p1]


def plot_nb_boundary(X, nb, ax=None, resolution=200):
    """在二维特征平面画高斯朴素贝叶斯的 0.5 决策边界。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # TODO 1: xx, yy, grid = make_meshgrid(X, resolution)
    xx, yy, grid = make_meshgrid(X, resolution)
    # TODO 2: p1 = nb.predict_proba(grid)[:, 1]
    p1 = nb.predict_proba(grid)[:, 1]
    # TODO 3: zz = p1.reshape(xx.shape)
    zz = p1.reshape(xx.shape)
    # TODO 4: ax.contour(xx, yy, zz, levels=[0.5],
    #                    colors="red", linestyles="--", linewidths=1.8)
    ax.contour(xx, yy, zz, levels=[0.5],
               colors="red", linestyles="--", linewidths=1.8)
    return ax


if __name__ == "__main__":
    # 1. 数据
    X, y = make_gaussian_data(n_per_class=500, rho=0.8, seed=0)

    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=0.3, seed=42
    )
    X_train_std, X_test_std, mu, sigma = standardize(X_train, X_test)

    # 2. 逻辑回归
    lr_w, lr_b, lr_history = train_logreg_gd(
        X_train_std, y_train,
        lr=0.5, n_iter=20000, tol=1e-6, lam=0.0
    )

    lr_train_acc = accuracy(X_train_std, y_train, lr_w, lr_b)
    lr_test_acc = accuracy(X_test_std, y_test, lr_w, lr_b)

    # 3. 高斯朴素贝叶斯
    nb = GaussianNB().fit(X_train_std, y_train)

    # TODO 5: 计算 nb_train_acc 和 nb_test_acc

    print("Logistic Regression: train=%.4f test=%.4f"
          % (lr_train_acc, lr_test_acc))
    # TODO 6: 打印 NB 的准确率

    # 4. 对比类内完整协方差与 NB 使用的对角方差
    Xc0 = X_train_std[y_train == 0]

    print("class 0 full covariance:\n", np.cov(Xc0, rowvar=False))
    print("class 0 diagonal used by NB:\n",
          np.diag(np.var(Xc0, axis=0)))

    # 5. 画决策边界对比
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    plot_data(X_train_std, y_train, ax=ax)

    # TODO 7: 画逻辑回归边界
    # plot_decision_boundary(X_train_std, lr_w, lr_b, ax=ax, resolution=200)
    plot_decision_boundary(X_train_std, lr_w, lr_b, ax=ax, resolution=200)
    # TODO 8: 画朴素贝叶斯边界
    # plot_nb_boundary(X_train_std, nb, ax=ax, resolution=200)
    plot_nb_boundary(X_train_std, nb, ax=ax, resolution=200)

    # 6. 补充图例
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="black", lw=1.8),
        Line2D([0], [0], color="red", lw=1.8, ls="--"),
    ]
    labels += ["Logistic Regression", "Gaussian NB"]
    ax.legend(handles, labels, loc="upper left")
    ax.set_title("Logistic Regression vs Gaussian NB")

    fig.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "step7_nb_compare.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("figure saved to", out_path)
    plt.close(fig)
