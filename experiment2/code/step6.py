import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 必须在 import matplotlib.pyplot 之前设置
import matplotlib.pyplot as plt

from step3 import train_logreg_gd
from step4 import (
    accuracy,
    make_gaussian_data,
    split_dataset,
    standardize,
)
from step5_plot import plot_data, plot_decision_boundary


def run_lambda_experiment(X_train, y_train, X_test, y_test, lams,
                          lr=0.5, n_iter=20000, tol=1e-6):
    """
    对每个 lam 训练一次，返回结果列表。
    每个元素是 dict:
        {
            "lam": ...,
            "train_acc": ...,
            "test_acc": ...,
            "weight_norm": ...,
            "w": ...,
            "b": ...,
            "history": ...,
        }
    """
    results = []

    for lam in lams:
        w, b, history = train_logreg_gd(
            X_train, y_train,
            lr=lr, n_iter=n_iter, tol=tol, lam=lam
        )

        train_acc = accuracy(X_train, y_train, w, b)
        test_acc = accuracy(X_test, y_test, w, b)
        weight_norm = np.linalg.norm(w)

        results.append({
            "lam": lam,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "weight_norm": weight_norm,
            "w": w,
            "b": b,
            "history": history,
        })

    return results


if __name__ == "__main__":
    # 1. 准备数据
    X, y = make_gaussian_data(n_per_class=500, rho=0.8, seed=0)

    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=0.3, seed=42
    )

    X_train_std, X_test_std, mu, sigma = standardize(X_train, X_test)

    # 2. 设置要比较的 lam
    lams = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    # 3. 训练并收集结果
    results = run_lambda_experiment(
        X_train_std, y_train,
        X_test_std, y_test,
        lams,
        lr=0.5, n_iter=20000, tol=1e-6,
    )

    # 4. 打印表格
    print(f"{'lam':<10}{'train_acc':<12}{'test_acc':<12}{'||w||':<12}")
    for res in results:
        print(f"{res['lam']:<10g}"
              f"{res['train_acc']:<12.4f}"
              f"{res['test_acc']:<12.4f}"
              f"{res['weight_norm']:<12.3f}")

    # 5. 画准确率与权重范数
    xpos = np.arange(len(lams))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(xpos, [res["train_acc"] for res in results],
                 "o-", label="Train Acc")
    axes[0].plot(xpos, [res["test_acc"] for res in results],
                 "s--", label="Test Acc")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([str(lam) for lam in lams])
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train/Test Accuracy vs Lambda")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xpos, [res["weight_norm"] for res in results],
                 "o-", color="tab:orange")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(lam) for lam in lams])
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel(r"$||w||_2$")
    axes[1].set_title("Weight Norm vs Lambda")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    first_path = os.path.join(out_dir, "step6_regularization.png")
    fig.savefig(first_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figure saved to", first_path)

    # 6. 画三个不同 lam 的决策边界
    show_lams = [0.0, 1e-2, 1.0]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, lam in zip(axes2, show_lams):
        res = next(r for r in results if r["lam"] == lam)
        plot_data(X_train_std, y_train, ax=ax)
        plot_decision_boundary(
            X_train_std, res["w"], res["b"], ax=ax, resolution=200
        )
        ax.set_title(
            "lambda=%g\n"
            "train acc=%.4f, test acc=%.4f"
            % (lam, res["train_acc"], res["test_acc"])
        )

    fig2.tight_layout()
    second_path = os.path.join(out_dir, "step6_boundaries.png")
    fig2.savefig(second_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("figure saved to", second_path)

    # 7. 基本检查
    assert results[0]["train_acc"] > 0.90
    assert results[0]["test_acc"] > 0.90
    assert all(
        results[i]["weight_norm"] > results[i + 1]["weight_norm"]
        for i in range(len(results) - 1)
    )
    print("step6.py 测试通过")
