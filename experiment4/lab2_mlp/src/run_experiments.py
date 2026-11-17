"""Lab2：Ionosphere 二分类的完整实验。

运行：
    cd lab2_mlp
    python src/run_experiments.py

输出：
    results/results.json    全部指标
    results/model_best.pt   最优配置在全部数据上训练的模型
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data import load_ionosphere, standardize, stratified_kfold
from train import Config, binary_metrics, cross_validate, predict_proba, train_fold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ionosphere.data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    torch.set_num_threads(4)
    t_start = time.perf_counter()
    X, y = load_ionosphere(DATA)
    folds = stratified_kfold(y, n_splits=5, seed=42)

    print("=" * 78)
    print("Lab2：Ionosphere + PyTorch MLP")
    print("=" * 78)
    print(f"数据: X={X.shape}, 正类 g={int(y.sum())}, 负类 b={int((1 - y).sum())}")
    print("五折划分（分层）:")
    for k, (tr, te) in enumerate(folds):
        print(f"  Fold {k + 1}: train={len(tr)}, test={len(te)}, "
              f"test 中 g={int(y[te].sum())} / b={int((1 - y[te]).sum())}")

    # ------------------------------------------------------------------
    # 表 1：模板指定的 5 组超参数
    # ------------------------------------------------------------------
    table1_cfgs = [
        Config((10,), "relu", 0.0, False, 0.01, "sgd", 16, 100),
        Config((20,), "relu", 0.0, False, 0.01, "sgd", 16, 100),
        Config((30,), "relu", 0.0, False, 0.01, "sgd", 16, 100),
        Config((32, 32), "relu", 0.2, False, 0.001, "adam", 32, 100),
        Config((64, 64, 64), "tanh", 0.3, False, 0.001, "adam", 64, 200),
    ]

    print("\n" + "=" * 78)
    print("表 1：5 组超参数的五折结果（封闭=训练集，开放=验证集）")
    print("=" * 78)
    print(f"{'#':<3}{'配置':<52}{'参数量':>9}{'封闭':>9}{'F1':>8}{'开放':>9}{'耗时':>8}")
    table1 = []
    for i, cfg in enumerate(table1_cfgs, 1):
        res = cross_validate(X, y, cfg, folds=folds)
        res["index"] = i
        table1.append(res)
        print(f"{i:<3}{res['config']:<52}{res['n_params']:>9}"
              f"{pct(res['closed_acc_mean']):>9}{res['closed_f1_mean']:>8.3f}"
              f"{pct(res['open_acc_mean']):>9}{res['time_sec']:>8.1f}s")

    # ------------------------------------------------------------------
    # 表 2：三组模型配置，逐折开放精度
    # ------------------------------------------------------------------
    table2_cfgs = [
        Config((16,), "relu", 0.0, False, 0.01, "sgd", 16, 100),
        Config((64, 64), "relu", 0.0, False, 0.01, "sgd", 16, 100),
        Config((64, 64), "relu", 0.3, False, 0.01, "sgd", 16, 100),
    ]
    table2_names = ["1 隐藏层, 16", "2 隐藏层, 64", "2 隐藏层, 64 + Dropout 0.3"]

    print("\n" + "=" * 78)
    print("表 2：三组配置的五折开放精度")
    print("=" * 78)
    print(f"{'模型配置':<32}" +
          "".join(f"{'Fold' + str(k + 1):>8}" for k in range(5)) +
          f"{'均值':>9}{'标准差':>9}")
    table2 = []
    for name, cfg in zip(table2_names, table2_cfgs):
        res = cross_validate(X, y, cfg, folds=folds)
        res["name"] = name
        table2.append(res)
        print(f"{name:<32}" +
              "".join(f"{pct(v):>8}" for v in res["fold_open_accs"]) +
              f"{pct(res['open_acc_mean']):>9}{res['open_acc_std'] * 100:>8.2f}%")

    # ------------------------------------------------------------------
    # 超参数扫描
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("超参数扫描（每组均为五折开放精度均值）")
    print("=" * 78)

    def sweep(name: str, configs: list[Config]) -> list[dict]:
        rows = []
        print(f"\n[{name}]")
        for cfg in configs:
            res = cross_validate(X, y, cfg, folds=folds)
            rows.append({
                "label": cfg.describe(),
                "hidden_dims": list(cfg.hidden_dims),
                "activation": cfg.activation,
                "dropout": cfg.dropout,
                "batchnorm": cfg.batchnorm,
                "lr": cfg.lr,
                "optimizer": cfg.optimizer,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "weight_decay": cfg.weight_decay,
                "open_acc_mean": res["open_acc_mean"],
                "open_acc_std": res["open_acc_std"],
                "open_f1_mean": res["open_f1_mean"],
                "n_params": res["n_params"],
                "time_sec": res["time_sec"],
            })
            print(f"  {res['config']:<52}{pct(res['open_acc_mean']):>9}"
                  f" ± {res['open_acc_std'] * 100:>4.2f}%")
        return rows

    sweeps: dict[str, list[dict]] = {}
    sweeps["width"] = sweep("隐层宽度（1 层，Adam）", [
        Config((w,), "relu", 0.0, False, 0.001, "adam", 32, 100)
        for w in (8, 16, 32, 64, 128)
    ])
    sweeps["depth"] = sweep("隐层层数（宽度 32，Adam）", [
        Config((32,) * d, "relu", 0.0, False, 0.001, "adam", 32, 100)
        for d in (1, 2, 3, 4)
    ])
    sweeps["lr"] = sweep("学习率（SGD，1×32）", [
        Config((32,), "relu", 0.0, False, lr, "sgd", 16, 100)
        for lr in (0.1, 0.01, 0.001)
    ])
    sweeps["batch"] = sweep("Batch 大小（Adam，2×64）", [
        Config((64, 64), "relu", 0.0, False, 0.001, "adam", bs, 100)
        for bs in (8, 16, 32, 64)
    ])
    sweeps["optimizer"] = sweep("优化器（2×64）", [
        Config((64, 64), "relu", 0.0, False, 0.01, "sgd", 32, 100),
        Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 100),
    ])
    sweeps["activation"] = sweep("激活函数（Adam，2×64）", [
        Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 100),
        Config((64, 64), "tanh", 0.0, False, 0.001, "adam", 32, 100),
    ])
    sweeps["dropout"] = sweep("Dropout（Adam，2×64）", [
        Config((64, 64), "relu", d, False, 0.001, "adam", 32, 100)
        for d in (0.0, 0.2, 0.4)
    ])
    sweeps["weight_decay"] = sweep("L2 权重衰减（Adam，2×64）", [
        Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 100, wd)
        for wd in (0.0, 1e-4, 1e-3, 1e-2)
    ])
    sweeps["batchnorm"] = sweep("BatchNorm（Adam，2×64）", [
        Config((64, 64), "relu", 0.0, bn, 0.001, "adam", 32, 100)
        for bn in (False, True)
    ])
    sweeps["early_stopping"] = sweep("早停（Adam，2×64）", [
        Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 150, 0.0, 0),
        Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 150, 0.0, 10),
    ])

    # ------------------------------------------------------------------
    # 基准训练曲线 + 正则化对比
    # ------------------------------------------------------------------
    base_cfg = Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 150)
    base_res = cross_validate(X, y, base_cfg, folds=folds, keep_history=True)
    print("\n" + "=" * 78)
    print("基准配置训练曲线（2×64, Adam, 150 epochs）")
    print("=" * 78)
    for k, hist in enumerate(base_res["histories"]):
        print(f"  Fold {k + 1}: 最终 train_acc={hist[-1]['train_acc']:.4f}, "
              f"val_acc={hist[-1]['val_acc']:.4f}, val_loss={hist[-1]['val_loss']:.4f}")

    reg_cfgs = [
        ("Baseline", base_cfg),
        ("+ L2 1e-3", Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 150, 1e-3)),
        ("+ Dropout 0.3", Config((64, 64), "relu", 0.3, False, 0.001, "adam", 32, 150)),
        ("+ BatchNorm", Config((64, 64), "relu", 0.0, True, 0.001, "adam", 32, 150)),
        ("+ EarlyStop 10", Config((64, 64), "relu", 0.0, False, 0.001, "adam", 32, 150, 0.0, 10)),
    ]
    print("\n" + "=" * 78)
    print("正则化对比（五折开放精度）")
    print("=" * 78)
    regularization = []
    for name, cfg in reg_cfgs:
        res = cross_validate(X, y, cfg, folds=folds)
        regularization.append({
            "name": name,
            "config": res["config"],
            "open_acc_mean": res["open_acc_mean"],
            "open_acc_std": res["open_acc_std"],
            "open_f1_mean": res["open_f1_mean"],
            "time_sec": res["time_sec"],
        })
        print(f"  {name:<18}{pct(res['open_acc_mean'])} ± "
              f"{res['open_acc_std'] * 100:.2f}%  (F1 {res['open_f1_mean']:.3f})")

    # ------------------------------------------------------------------
    # 最优配置：全量训练 + OOF 误差分析
    # ------------------------------------------------------------------
    best_idx = int(np.argmax([r["open_acc_mean"] for r in table1]))
    best_cfg = table1_cfgs[best_idx]
    best_cv = cross_validate(X, y, best_cfg, folds=folds, collect_oof=True)
    oof_prob = np.array(best_cv["oof_prob"])
    best_metrics = binary_metrics(y, oof_prob)

    print("\n" + "=" * 78)
    print(f"最优配置（表 1 第 {best_idx + 1} 行）：{best_cfg.describe()}")
    print("=" * 78)
    print(f"  开放精度 = {pct(best_cv['open_acc_mean'])} ± "
          f"{best_cv['open_acc_std'] * 100:.2f}%")
    print(f"  OOF 合并指标: acc={best_metrics['acc']:.4f}, "
          f"precision={best_metrics['precision']:.4f}, "
          f"recall={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}")
    print(f"  混淆矩阵: TN={best_metrics['tn']}, FP={best_metrics['fp']}, "
          f"FN={best_metrics['fn']}, TP={best_metrics['tp']}")

    pred = (oof_prob >= 0.5).astype(np.int64)
    err_idx = np.flatnonzero(pred != y)
    error_examples = []
    for i in err_idx:
        error_examples.append({
            "index": int(i),
            "true": int(y[i]),
            "pred": int(pred[i]),
            "prob_g": float(oof_prob[i]),
            "margin": float(abs(oof_prob[i] - 0.5)),
        })
    error_examples.sort(key=lambda r: r["margin"])

    confidence_bins = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = (oof_prob >= lo) & (oof_prob < hi if b < 9 else oof_prob <= hi)
        confidence_bins.append({
            "range": f"[{lo:.1f},{hi:.1f})",
            "n": int(mask.sum()),
            "n_error": int((pred[mask] != y[mask]).sum()),
        })

    # 用全部 351 个样本训练最终模型，保存权重
    X_all, _, mu, sd = standardize(X, X)
    final_model, _, best_epoch = train_fold(
        X_all, y, X_all, y, best_cfg, seed=best_cfg.seed,
    )
    final_prob = predict_proba(final_model, X_all)
    final_metrics = binary_metrics(y, final_prob)
    torch.save({
        "state_dict": final_model.state_dict(),
        "config": asdict(best_cfg),
        "mu": mu,
        "sd": sd,
        "input_dim": 34,
        "class_map": {"b": 0, "g": 1},
    }, OUT / "model_best.pt")
    print(f"  全量模型封闭精度 = {pct(final_metrics['acc'])}，"
          f"最佳 epoch={best_epoch}")
    print(f"  权重已保存: {OUT / 'model_best.pt'}")

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    meta = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": int(os.cpu_count() or 0),
        "seed": 42,
        "n_splits": 5,
        "dataset": "UCI Ionosphere",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_counts": {"b": int((1 - y).sum()), "g": int(y.sum())},
        "total_time_sec": time.perf_counter() - t_start,
    }
    results = {
        "meta": meta,
        "data": {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "class_counts": meta["class_counts"],
            "fold_sizes": [
                {"fold": k + 1, "n_train": int(len(tr)), "n_test": int(len(te))}
                for k, (tr, te) in enumerate(folds)
            ],
        },
        "table1": table1,
        "table2": table2,
        "sweeps": sweeps,
        "base": {
            "config": base_res["config"],
            "open_acc_mean": base_res["open_acc_mean"],
            "open_acc_std": base_res["open_acc_std"],
            "histories": base_res["histories"],
        },
        "regularization": regularization,
        "best": {
            "index": best_idx + 1,
            "config": best_cfg.describe(),
            "hidden_dims": list(best_cfg.hidden_dims),
            "activation": best_cfg.activation,
            "dropout": best_cfg.dropout,
            "lr": best_cfg.lr,
            "optimizer": best_cfg.optimizer,
            "batch_size": best_cfg.batch_size,
            "epochs": best_cfg.epochs,
            "n_params": best_cv["n_params"],
            "open_acc_mean": best_cv["open_acc_mean"],
            "open_acc_std": best_cv["open_acc_std"],
            "open_f1_mean": best_cv["open_f1_mean"],
            "oof_metrics": best_metrics,
            "final_closed_acc": final_metrics["acc"],
            "final_closed_f1": final_metrics["f1"],
            "best_epoch": int(best_epoch),
            "oof_prob": oof_prob.tolist(),
        },
        "error_analysis": {
            "n_errors": int(len(error_examples)),
            "error_examples": error_examples[:12],
            "confidence_bins": confidence_bins,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
        },
    }
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print(f"\n结果已写入 {OUT / 'results.json'}")
    print(f"总耗时 {results['meta']['total_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
