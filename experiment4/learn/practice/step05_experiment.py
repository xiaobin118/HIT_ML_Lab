"""第 5 步：二维数据上的线性分类器 vs MLP。

目标：用第 1、3、4 步写好的模型，做一个规范的对照实验：
    3 份数据 × 2 个模型 × 3 个随机种子，
    报告测试准确率的均值和标准差，并画出决策边界。

数据与绘图工具在 common.py，训练函数已经给出；
你只需要写 run_seed（跑一次实验）和 compare（汇总 3 个种子）。

运行：python step05_experiment.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    Array,
    accuracy,
    make_blobs,
    make_rings,
    make_spiral,
    plot_decision_boundary,
    train_test_split,
)
from step01_linear import LinearClassifier, train_full_batch
from step04_train import MLP, fit

DATASETS = {"blobs": make_blobs, "rings": make_rings, "spiral": make_spiral}
TITLES = {
    "blobs": "四团高斯（线性可分）",
    "rings": "四同心圆环",
    "spiral": "四臂螺旋",
}
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# 两个训练函数（已提供）
# ---------------------------------------------------------------------------
def train_linear(Xtr: Array, ytr: Array, seed: int) -> LinearClassifier:
    """训练线性分类器：全批量梯度下降 1500 步，学习率 0.1。

    Parameters
    ----------
    Xtr, ytr : Array
        训练数据。
    seed : int
        初始化种子（线性模型全零初始化，实际不影响结果）。
    """
    model = LinearClassifier(2, 4, seed=seed)
    train_full_batch(model, Xtr, ytr, steps=1500, lr=0.1)
    return model


def train_mlp(
    Xtr: Array, ytr: Array, Xte: Array, yte: Array, seed: int
) -> MLP:
    """训练 MLP：2-32-32-4，Adam，600 个 epoch，批大小 64，学习率 0.01。

    Parameters
    ----------
    Xtr, ytr : Array
        训练数据。
    Xte, yte : Array
        测试数据，只用于每个 epoch 记录曲线，不参与参数更新。
    seed : int
        初始化与打乱数据的种子。
    """
    model = MLP([2, 32, 32, 4], activation="relu", init="he", seed=seed)
    fit(
        model, Xtr, ytr, Xte, yte,
        epochs=600, batch_size=64, lr=1e-2,
        optimizer="adam", seed=seed,
    )
    return model


# ---------------------------------------------------------------------------
# TODO 1：跑一次实验
# ---------------------------------------------------------------------------
def run_seed(name: str, kind: str, seed: int) -> dict:
    """在指定数据、指定模型、指定种子上训练一次，返回结果。

    Parameters
    ----------
    name : str
        "blobs" / "rings" / "spiral"。
    kind : str
        "linear" 或 "mlp"。
    seed : int
        同时用于数据生成、划分、模型初始化。

    Returns
    -------
    dict
        至少包含：
            "acc"   : float，测试准确率
            "model" : 训练好的模型
            "Xtr", "ytr" : 训练集（画决策边界用）

    步骤：
        1. X, y = DATASETS[name](seed=seed)          # 生成数据
        2. Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=seed)
        3. kind == "linear" -> train_linear(Xtr, ytr, seed)
           kind == "mlp"    -> train_mlp(Xtr, ytr, Xte, yte, seed)
        4. acc = accuracy(model.predict, Xte, yte)
        5. return {"acc": acc, "model": model, "Xtr": Xtr, "ytr": ytr}
    """
    # TODO 1：按上面的 5 步实现。
    X, y = DATASETS[name](seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=seed)
    if kind == "linear":
        model = train_linear(Xtr, ytr, seed)
    elif kind == "mlp":
        model = train_mlp(Xtr, ytr, Xte, yte, seed)
    else:
        raise ValueError(f"未知模型类型：{kind}")

    acc = accuracy(model.predict, Xte, yte)
    return {"acc": acc, "model": model, "Xtr": Xtr, "ytr": ytr}



# ---------------------------------------------------------------------------
# TODO 2：汇总多个种子
# ---------------------------------------------------------------------------
def compare(name: str, seeds: tuple[int, ...] = SEEDS) -> dict:
    """对同一份数据，分别跑线性模型和 MLP，汇总均值与标准差。

    Parameters
    ----------
    name : str
        数据集名字。
    seeds : tuple[int, ...]
        重复实验用的随机种子。

    Returns
    -------
    dict
        {
          "linear": {"mean": float, "std": float, "accs": list[float]},
          "mlp":    {"mean": float, "std": float, "accs": list[float]},
          "gap":    float,     # mlp.mean - linear.mean
        }

    提示：
        - 对每种模型，用 [run_seed(name, kind, s)["acc"] for s in seeds] 收集准确率；
        - 均值 np.mean，标准差 np.std（总体标准差即可）；
        - gap = mlp 的均值 - linear 的均值。
    """
    # TODO 2：实现汇总。
    linear_accs = [run_seed(name, "linear", s)["acc"] for s in seeds]
    mlp_accs = [run_seed(name, "mlp", s)["acc"] for s in seeds]

    linear_mean = np.mean(linear_accs)
    linear_std = np.std(linear_accs)
    mlp_mean = np.mean(mlp_accs)
    mlp_std = np.std(mlp_accs)

    gap = mlp_mean - linear_mean
    return {
        "linear": {"mean": linear_mean, "std": linear_std, "accs": linear_accs},
        "mlp": {"mean": mlp_mean, "std": mlp_std, "accs": mlp_accs},
        "gap": gap,
    }

# ---------------------------------------------------------------------------
# 画一组对比图（已提供）
# ---------------------------------------------------------------------------
def plot_pair(name: str, lin: LinearClassifier, mlp: MLP,
              Xtr: Array, ytr: Array, acc_lin: float, acc_mlp: float,
              path: str | Path) -> None:
    """左边线性、右边 MLP，画在同一张图的两个子图上。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    plot_decision_boundary(lin.predict, Xtr, ytr,
                           title=f"线性分类器：{acc_lin:.3f}", ax=axes[0])
    plot_decision_boundary(mlp.predict, Xtr, ytr,
                           title=f"MLP(32-32)：{acc_mlp:.3f}", ax=axes[1])
    fig.suptitle(TITLES[name], fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存到 {path}")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(passed)


def self_check() -> dict:
    print("=" * 72)
    print("第 5 步自检")
    print("=" * 72)
    results = {}
    ok = True
    t0 = time.perf_counter()

    # 先单独跑一次，确保 run_seed（TODO 1）已经实现；
    # 否则 compare 会先报 TODO 2，掩盖真正的第一个 TODO。
    _ = run_seed("blobs", "linear", 0)

    for name in ["blobs", "rings", "spiral"]:
        r = compare(name)
        results[name] = r
        print(f"\n[{TITLES[name]}]")
        for kind in ["linear", "mlp"]:
            accs = " / ".join(f"{a:.3f}" for a in r[kind]["accs"])
            print(f"  {kind:6s}: {r[kind]['mean']:.3f} ± {r[kind]['std']:.3f}"
                  f"   （{accs}）")
        print(f"  gap   : {r['gap']:+.3f}")

    bl, rg, sp = results["blobs"], results["rings"], results["spiral"]
    ok &= check("1. 四团高斯：两种模型都很好",
                bl["linear"]["mean"] > 0.93 and bl["mlp"]["mean"] > 0.92,
                f"{bl['linear']['mean']:.3f} vs {bl['mlp']['mean']:.3f}")
    ok &= check("2. 四团高斯：MLP 没有明显优势",
                abs(bl["gap"]) < 0.05, f"gap = {bl['gap']:+.3f}")
    ok &= check("3. 圆环：线性接近随机水平",
                rg["linear"]["mean"] < 0.35, f"{rg['linear']['mean']:.3f}")
    ok &= check("4. 圆环：MLP 明显更好",
                rg["mlp"]["mean"] > 0.90 and rg["gap"] > 0.40,
                f"{rg['mlp']['mean']:.3f}, gap = {rg['gap']:+.3f}")
    ok &= check("5. 螺旋：线性失败、MLP 成功",
                sp["linear"]["mean"] < 0.40 and sp["mlp"]["mean"] > 0.90
                and sp["gap"] > 0.40,
                f"{sp['linear']['mean']:.3f} -> {sp['mlp']['mean']:.3f}")
    ok &= check("6. 标准差合理", all(
        r[k]["std"] < 0.08 for r in results.values() for k in ["linear", "mlp"]),
        "每个配置 3 个种子的标准差都 < 0.08")

    print("\n" + "-" * 72)
    print(f"全部通过（{time.perf_counter() - t0:.1f}s）。" if ok else "还有检查未通过。")
    return {"results": results, "ok": ok}


# ---------------------------------------------------------------------------
# 演示：画三组决策边界
# ---------------------------------------------------------------------------
def demo(results: dict) -> None:
    print("\n画决策边界（每个数据集用 seed 0 的一个代表模型）：")
    for name in ["blobs", "rings", "spiral"]:
        r = results[name]
        # 重新训练一次 seed 0，拿到模型和训练点
        lin = run_seed(name, "linear", 0)
        mlp = run_seed(name, "mlp", 0)
        plot_pair(name, lin["model"], mlp["model"], lin["Xtr"], lin["ytr"],
                  lin["acc"], mlp["acc"], f"step05_{name}.png")


if __name__ == "__main__":
    try:
        out = self_check()
    except NotImplementedError as exc:
        print(f"\n还有 TODO 没实现：{exc}")
        raise SystemExit(1)
    if out["ok"]:
        demo(out["results"])
