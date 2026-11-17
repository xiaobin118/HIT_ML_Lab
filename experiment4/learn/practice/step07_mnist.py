"""第 7 步：MNIST 全量实验与误差分析。

目标：
    1. 在 60000 张训练图、10000 张测试图上训练线性分类器和 MLP；
    2. 自己实现混淆矩阵与每类 precision / recall / F1；
    3. 画训练曲线、混淆矩阵、错分样本、第一层权重、隐层 PCA；
    4. 为最终实验报告准备数字和图。

本步的训练代码已经给出；你只需要写两个评价指标函数。

运行：python step07_mnist.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step01_linear import LinearClassifier
from step04_train import MLP, accuracy, fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from datasets import load_mnist  # noqa: E402

Array = np.ndarray
N_CLASSES = 10


def get_data() -> tuple[Array, Array, Array, Array]:
    """加载 MNIST 全量数据。已提供。"""
    return load_mnist(ROOT / "data")


class LinearAsList(LinearClassifier):
    """把线性分类器适配成 fit 需要的接口。

    fit 需要：
        model.Ws / model.bs      参数列表
        loss_and_grads 返回 (loss, grads 列表)
    线性分类器只有 W、b，且返回 (loss, dW, db)，这里包一层。
    """

    @property
    def Ws(self) -> list[Array]:
        return [self.W]

    @property
    def bs(self) -> list[Array]:
        return [self.b]

    def loss_and_grads(
        self, X: Array, y: Array, l2: float = 0.0
    ) -> tuple[float, list[Array]]:
        loss, dW, db = super().loss_and_grads(X, y, l2=l2)
        return float(loss), [dW, db]


# ---------------------------------------------------------------------------
# TODO 1：混淆矩阵
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: Array, y_pred: Array, n_classes: int) -> Array:
    """统计真实类别 vs 预测类别的计数矩阵。

    Parameters
    ----------
    y_true : Array
        真实类别，形状 (N,)，整数 0..C-1。
    y_pred : Array
        预测类别，形状 (N,)。
    n_classes : int
        类别数 C。

    Returns
    -------
    Array
        形状 (C, C) 的整数矩阵，cm[i, j] = 真实为 i、预测为 j 的样本数。

    提示：
        cm = np.zeros((C, C), dtype=int)
        np.add.at(cm, (y_true, y_pred), 1)
    np.add.at 是"带重复下标的累加"：同一个 (i,j) 出现多次时，每次都会加 1。
    普通 cm[i, j] += 1 在重复索引时只加一次。
    """
    # TODO 1：实现。
    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


# ---------------------------------------------------------------------------
# TODO 2：每类指标
# ---------------------------------------------------------------------------
def per_class_metrics(cm: Array) -> list[dict]:
    """从混淆矩阵计算每类的 precision、recall、F1、support。

    Parameters
    ----------
    cm : Array
        形状 (C, C) 的混淆矩阵。

    Returns
    -------
    list[dict]
        每个类别一个字典：
            {"class": i, "precision": float, "recall": float,
             "f1": float, "support": int}

    公式：
        precision_i = cm[i, i] / sum_j cm[j, i]     # 预测为 i 的样本里有多少是真的
        recall_i    = cm[i, i] / sum_j cm[i, j]     # 真实为 i 的样本里有多少被找到
        f1_i        = 2 * P * R / (P + R)
    注意分母可能为 0；用 np.divide(..., where=..., out=...) 或先判断。
    """
    # TODO 2：实现。
    metrics = []
    for i in range(cm.shape[0]):
        true_positive = cm[i, i]
        predicted_positive = cm[:, i].sum()
        actual_positive = cm[i, :].sum()

        precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.append({
            "class": i,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(actual_positive)
        })
    return metrics


# ---------------------------------------------------------------------------
# 训练（已提供）
# ---------------------------------------------------------------------------
def train_models() -> dict:
    """训练线性分类器与 MLP，返回模型、历史、预测和混淆矩阵。"""
    Xtr, ytr, Xte, yte = get_data()

    print("训练线性分类器（Adam，20 轮）...")
    t0 = time.perf_counter()
    lin = LinearAsList(784, 10, seed=0)
    hist_lin = fit(lin, Xtr, ytr, Xte, yte, epochs=20, batch_size=128,
                   lr=1e-3, optimizer="adam", record_test_loss=True, seed=0)
    t_lin = time.perf_counter() - t0

    print("训练 MLP 784-256-128-10（Adam，20 轮）...")
    t0 = time.perf_counter()
    mlp = MLP([784, 256, 128, 10], activation="relu", init="he", seed=0)
    hist_mlp = fit(mlp, Xtr, ytr, Xte, yte, epochs=20, batch_size=128,
                   lr=1e-3, optimizer="adam", record_test_loss=True, seed=0)
    t_mlp = time.perf_counter() - t0

    pred_lin = lin.predict(Xte)
    pred_mlp = mlp.predict(Xte)
    return {
        "linear": {"model": lin, "history": hist_lin, "pred": pred_lin,
                   "acc": accuracy(lin, Xte, yte), "time": t_lin},
        "mlp": {"model": mlp, "history": hist_mlp, "pred": pred_mlp,
                "acc": accuracy(mlp, Xte, yte), "time": t_mlp},
        "Xte": Xte, "yte": yte,
        "cm_linear": confusion_matrix(yte, pred_lin, N_CLASSES),
        "cm_mlp": confusion_matrix(yte, pred_mlp, N_CLASSES),
    }


def top_confusions(cm: Array, k: int = 6) -> list[dict]:
    """按错误数排序，返回最容易混淆的 k 个类别对。已提供。"""
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true": i, "pred": j, "count": int(cm[i, j]),
                    "of_class": float(cm[i, j] / max(cm[i].sum(), 1)),
                })
    rows.sort(key=lambda r: -r["count"])
    return rows[:k]


# ---------------------------------------------------------------------------
# 绘图（已提供）
# ---------------------------------------------------------------------------
def plot_curves(lin_hist: list[dict], mlp_hist: list[dict], path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for hist, tag, col in [(lin_hist, "线性分类器", "#1f77b4"),
                           (mlp_hist, "MLP(256-128)", "#d62728")]:
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=col, label=f"{tag} 训练")
        axes[0].plot(ep, [h["test_loss"] for h in hist], color=col,
                     linestyle="--", alpha=0.7, label=f"{tag} 测试")
        axes[1].plot(ep, [h["train_acc"] for h in hist], color=col, label=f"{tag} 训练")
        axes[1].plot(ep, [h["test_acc"] for h in hist], color=col,
                     linestyle="--", alpha=0.7, label=f"{tag} 测试")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("交叉熵")
    axes[0].set_title("损失（实线训练 / 虚线测试）")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_confusions(cm_lin: Array, cm_mlp: Array, acc_lin: float, acc_mlp: float,
                    path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, cm, title in [
        (axes[0], cm_lin, f"线性分类器（{acc_lin:.4f}）"),
        (axes[1], cm_mlp, f"MLP（{acc_mlp:.4f}）"),
    ]:
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(title)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if norm[i, j] > 0.005:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_errors(Xte: Array, yte: Array, pred: Array, path: str, n: int = 16) -> None:
    wrong = np.flatnonzero(pred != yte)
    rng = np.random.default_rng(0)
    rng.shuffle(wrong)
    sel = wrong[:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(11.0, 3.2))
    for ax, i in zip(axes.ravel(), sel):
        ax.imshow(Xte[i].reshape(28, 28), cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"{yte[i]}→{pred[i]}", fontsize=9, color="crimson")
    fig.suptitle("MLP 分错的样本（真实→预测）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_filters(W1: Array, path: str, n: int = 32) -> None:
    norms = np.linalg.norm(W1, axis=0)
    top = np.argsort(norms)[::-1][:n]
    vmax = np.abs(W1[:, top]).max()
    fig, axes = plt.subplots(4, n // 4, figsize=(11.0, 6.0))
    for ax, j in zip(axes.ravel(), top):
        ax.imshow(W1[:, j].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"#{j}", fontsize=7)
    fig.suptitle("第一层权重（按范数取前 32 个；红正蓝负）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def pca2(X: Array) -> Array:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def plot_hidden_pca(model: MLP, Xte: Array, yte: Array, path: str) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xte), 3000, replace=False)
    Xs, ys = Xte[idx], yte[idx]
    _, layers = model.forward(Xs, cache=True)
    hidden = layers[len(model.Ws) - 2][2]   # 最后一个隐藏层的输出

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, Xrep, title in [
        (axes[0], pca2(Xs), "原始像素的 PCA 前两维"),
        (axes[1], pca2(hidden), "第二隐藏层（128 维）的 PCA 前两维"),
    ]:
        for c in range(10):
            m = ys == c
            ax.scatter(Xrep[m, 0], Xrep[m, 1], s=4, color=colors[c], alpha=0.7, label=str(c))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        ax.set_aspect("equal")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   markerscale=2, title="数字", title_fontsize=7)
    fig.suptitle("隐藏层把像素映射到更易线性分开的表示", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(passed)


def self_check() -> dict:
    print("=" * 72)
    print("第 7 步自检")
    print("=" * 72)
    ok = True

    # 1. 玩具混淆矩阵
    yt = np.array([0, 1, 1, 2])
    yp = np.array([0, 2, 1, 2])
    cm = confusion_matrix(yt, yp, 3)
    expected = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    ok &= check("1. 混淆矩阵", np.array_equal(cm, expected), f"\n{cm}")

    # 2. 每类指标
    rows = per_class_metrics(cm)
    m0, m1, m2 = rows
    ok2 = (abs(m0["precision"] - 1.0) < 1e-12 and abs(m0["recall"] - 1.0) < 1e-12
           and abs(m1["precision"] - 1.0) < 1e-12 and abs(m1["recall"] - 0.5) < 1e-12
           and abs(m2["precision"] - 0.5) < 1e-12 and abs(m2["recall"] - 1.0) < 1e-12)
    ok &= check("2. 每类指标", ok2,
                f"class1 P={m1['precision']:.2f} R={m1['recall']:.2f}, "
                f"class2 P={m2['precision']:.2f} R={m2['recall']:.2f}")

    # 3. 训练两个模型
    print("\n开始训练（约 2 分钟）...")
    t0 = time.perf_counter()
    R = train_models()
    print(f"训练完成，耗时 {time.perf_counter() - t0:.1f}s")
    lin_acc = R["linear"]["acc"]
    mlp_acc = R["mlp"]["acc"]
    ok &= check("3. 线性分类器 > 90%", lin_acc > 0.90, f"{lin_acc:.4f}")
    ok &= check("4. MLP > 97%", mlp_acc > 0.97, f"{mlp_acc:.4f}")
    ok &= check("5. MLP 明显优于线性", mlp_acc - lin_acc > 0.03,
                f"gap = {(mlp_acc - lin_acc) * 100:.2f} 个百分点")

    # 6. 矩阵与指标自洽
    total = R["cm_mlp"].sum()
    support = sum(r["support"] for r in per_class_metrics(R["cm_mlp"]))
    ok &= check("6. 混淆矩阵自洽", total == len(R["yte"]) and support == total,
                f"总数 {total}，support 之和 {support}")

    print("-" * 72)
    print("全部通过。" if ok else "还有检查未通过。")
    R["ok"] = ok
    return R


# ---------------------------------------------------------------------------
# 演示：误差分析与可视化
# ---------------------------------------------------------------------------
def demo(R: dict) -> None:
    print("\n生成报告用图：")
    plot_curves(R["linear"]["history"], R["mlp"]["history"], "step07_curves.png")
    plot_confusions(R["cm_linear"], R["cm_mlp"],
                    R["linear"]["acc"], R["mlp"]["acc"], "step07_confusion.png")
    plot_errors(R["Xte"], R["yte"], R["mlp"]["pred"], "step07_errors.png")
    plot_filters(R["mlp"]["model"].Ws[0], "step07_filters.png")
    plot_hidden_pca(R["mlp"]["model"], R["Xte"], R["yte"], "step07_hidden_pca.png")

    print("\nMLP 每类指标：")
    print("  类别   precision  recall    F1     support")
    for r in per_class_metrics(R["cm_mlp"]):
        print(f"    {r['class']:2d}     {r['precision']:.3f}     {r['recall']:.3f}"
              f"   {r['f1']:.3f}     {r['support']}")

    print("\n最易混淆的类别对：")
    for c in top_confusions(R["cm_mlp"], 6):
        print(f"  {c['true']} -> {c['pred']}: {c['count']} 例"
              f"（占该类 {c['of_class'] * 100:.1f}%）")

    print(f"\n最终测试准确率：线性 {R['linear']['acc']:.4f}，"
          f"MLP {R['mlp']['acc']:.4f}")


if __name__ == "__main__":
    try:
        results = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if results["ok"]:
        demo(results)
