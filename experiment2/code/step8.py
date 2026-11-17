import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step3 import train_logreg_gd
from step4 import accuracy, predict_class, split_dataset, standardize


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "bank.csv",
)


def load_bank_data(path=DATA_PATH):
    # Bank Marketing 使用分号分隔
    df = pd.read_csv(path, sep=";")

    y = (df["y"] == "yes").astype(int).values
    X_df = pd.get_dummies(df.drop(columns=["y"]), drop_first=True)
    X = X_df.values.astype(float)

    return X, y, X_df.columns.tolist()


def classification_metrics(y_true, y_pred):
    """
    计算分类指标: 准确率、精确率、召回率、F1 分数
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy_ = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return accuracy_, precision, recall, f1


def run_bank_experiment(X_train, y_train, X_test, y_test, lams,
                        lr=0.5, n_iter=10000, tol=1e-6):
    results = []

    for lam in lams:
        w, b, history = train_logreg_gd(
            X_train, y_train,
            lr=lr, n_iter=n_iter, tol=tol, lam=lam
        )

        y_pred = predict_class(X_test, w, b)
        accuracy_, precision, recall, f1 = classification_metrics(y_test, y_pred)
        train_acc = accuracy(X_train, y_train, w, b)
        weight_norm = np.linalg.norm(w)

        results.append({
            "lam": lam,
            "train_acc": train_acc,
            "test_acc": accuracy_,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "weight_norm": weight_norm,
            "w": w,
            "b": b,
            "history": history,
        })

    return results


if __name__ == "__main__":
    # 1. 读取与预处理
    X, y, feature_names = load_bank_data(DATA_PATH)

    print("X.shape =", X.shape)
    print("y.shape =", y.shape)
    print("y.mean() =", y.mean())

    # 2. 划分与标准化
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=0.3, seed=42
    )

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print("test baseline_acc =", baseline_acc)
    X_train_std, X_test_std, mu, sigma = standardize(X_train, X_test)

    # 3. lambda 扫描
    lams = [0.0, 0.001, 0.01, 0.1, 1.0]
    results = run_bank_experiment(
        X_train_std, y_train,
        X_test_std, y_test,
        lams, lr=0.5, n_iter=10000, tol=1e-6,
    )

    # 4. 打印表格
    print(f"{'lam':<10} {'accuracy':<10} {'precision':<10} "
          f"{'recall':<10} {'f1':<10} {'train_acc':<10} {'||w||':<10}")
    for res in results:
        print(f"{res['lam']:<10.4g} {res['test_acc']:<10.4f} "
              f"{res['precision']:<10.4f} {res['recall']:<10.4f} "
              f"{res['f1']:<10.4f} {res['train_acc']:<10.4f} "
              f"{res['weight_norm']:<10.4f}")

    # 5. 画准确率与 precision / recall / f1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xpos = np.arange(len(lams))

    axes[0].plot(xpos, [r["test_acc"] for r in results],
                 "o-", label="test accuracy")
    axes[0].plot(xpos, [r["train_acc"] for r in results],
                 "s--", label="train accuracy")
    axes[0].axhline(baseline_acc, color="gray", ls="--",
                    label="majority baseline")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([str(lam) for lam in lams])
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Bank Marketing: accuracy vs lambda")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xpos, [r["precision"] for r in results],
                 "o-", label="precision")
    axes[1].plot(xpos, [r["recall"] for r in results],
                 "s-", label="recall")
    axes[1].plot(xpos, [r["f1"] for r in results],
                 "^-", label="f1")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(lam) for lam in lams])
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("score")
    axes[1].set_title("Bank Marketing: precision / recall / F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "step8_bank_metrics.png",
    )
    fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("metrics figure saved to", metrics_path)

    # 6. TODO 10: 画 best lam 的混淆矩阵
    # 按测试准确率选一个用于可视化；严格实验应改用验证集选择。
    best = max(results, key=lambda r: r["test_acc"])
    y_pred_best = predict_class(X_test_std, best["w"], best["b"])

    tp = int(np.sum((y_pred_best == 1) & (y_test == 1)))
    tn = int(np.sum((y_pred_best == 0) & (y_test == 0)))
    fp = int(np.sum((y_pred_best == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred_best == 0) & (y_test == 1)))

    cm = np.array([[tn, fp],
                   [fn, tp]])

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
    im = ax_cm.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm.set_yticklabels(["True 0", "True 1"])
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title("Confusion matrix (lambda=%.4g)" % best["lam"])
    fig_cm.colorbar(im, ax=ax_cm)

    fig_cm.tight_layout()
    cm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "step8_bank_confusion.png",
    )
    fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cm)
    print("confusion matrix saved to", cm_path)
    print("confusion matrix:\n", cm)
