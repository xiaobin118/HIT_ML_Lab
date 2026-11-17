"""评估指标：混淆矩阵、每类 precision / recall / F1、最易混淆类别对、置信度统计。"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """返回 (n_classes, n_classes) 的混淆矩阵，行是真实类别，列是预测类别。"""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report(cm: np.ndarray) -> list[dict]:
    """由混淆矩阵算每类的 precision / recall / F1 / 支持数。"""
    rows = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        support = cm[c].sum()
        predicted = cm[:, c].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class": c,
            "support": int(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    return rows


def top_confusions(cm: np.ndarray, k: int = 5) -> list[dict]:
    """找出混淆矩阵里非对角线上最大的 k 个元素，即最常被搞混的类别对。"""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": i, "pred": j, "count": int(cm[i, j])})
    pairs.sort(key=lambda r: r["count"], reverse=True)
    return pairs[:k]


def confidence_stats(prob: np.ndarray, pred: np.ndarray, label: np.ndarray) -> dict:
    """把预测置信度（softmax 最大概率）分成正确/错误两组统计。"""
    conf = prob.max(axis=1)
    correct = pred == label
    out = {
        "conf_correct_mean": float(conf[correct].mean()) if correct.any() else 0.0,
        "conf_wrong_mean": float(conf[~correct].mean()) if (~correct).any() else 0.0,
        "conf_hist_correct": np.histogram(conf[correct], bins=10, range=(0, 1))[0].tolist(),
        "conf_hist_wrong": np.histogram(conf[~correct], bins=10, range=(0, 1))[0].tolist(),
    }
    # 按置信度阈值给出"拒识率 vs 覆盖率"曲线：超过阈值的样本里正确率是多少
    thresholds = np.linspace(0.0, 0.99, 100)
    coverage, precision = [], []
    for t in thresholds:
        mask = conf >= t
        coverage.append(float(mask.mean()))
        precision.append(float(correct[mask].mean()) if mask.any() else 1.0)
    out["thresholds"] = thresholds.tolist()
    out["coverage"] = coverage
    out["precision_at_threshold"] = precision
    return out
