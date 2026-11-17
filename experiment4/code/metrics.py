"""分类评价指标：准确率、混淆矩阵、每类 P/R/F1、最易混淆类别对。"""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_report(y_true, y_pred, n_classes, class_names=None):
    """返回每类的 precision / recall / f1 / support。"""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    rows = []
    for c in range(n_classes):
        rows.append(
            {
                "class": int(c),
                "name": class_names[c] if class_names else str(c),
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
        )
    return rows


def top_confusions(y_true, y_pred, n_classes, k=8):
    """按「真实类 -> 预测类」的错误数量排序，返回 top-k 混淆对。"""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": int(i),
                        "pred": int(j),
                        "count": int(cm[i, j]),
                        "rate_of_class": float(cm[i, j] / max(cm[i].sum(), 1)),
                        "rate_of_pred": float(cm[i, j] / max(cm[:, j].sum(), 1)),
                    }
                )
    pairs.sort(key=lambda d: -d["count"])
    return pairs[:k]
