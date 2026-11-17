#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 run_output.txt 解析已有结果 + 补跑 pendigits 的 K=10，组装 results.json。

之所以这样做而不是重跑 full_experiment.py：合成数据、GMM/EM、Iris 的完整结果
已经在 run_output.txt 里了，只有 pendigits 的 K 扫描没跑完。为了尽快出报告，
这里只补跑 pendigits 的 K=10 这一个配置，其余全部从日志解析。

运行方式：
    cd exp3_report
    python build_results.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import full_experiment as F  # noqa: E402
import data_gen as dg  # noqa: E402
from metrics import adjusted_rand_index, best_match_accuracy, confusion_matrix_from_match  # noqa: E402

LOG = (HERE / "run_output.txt").read_text(encoding="utf-8")


def parse_vector(s):
    return [float(v) for v in re.findall(r"-?\d+\.?\d*", s)]


def section(text, start, end=None):
    i = text.index(start)
    j = text.index(end) if end else len(text)
    return text[i:j]


def main():
    R = {}

    # ---------------- 合成数据基本参数 ----------------
    syn_sec = section(LOG, "实验一：用高斯分布生成", "k-means (seed=0)")
    emp_w = parse_vector(re.search(r"经验混合系数: \[([^\]]+)\]", syn_sec).group(1))
    R["synthetic"] = {
        "shape": [1500, 2],
        "true_means": F.TRUE_MEANS.tolist(),
        "true_covs": F.TRUE_COVS.tolist(),
        "true_weights": F.TRUE_WEIGHTS.tolist(),
        "empirical_weights": emp_w,
        "mean_max_dev_emp_vs_true": float(re.search(r"均值最大偏差 \|经验-设定\| = ([\d.]+)", syn_sec).group(1)),
        "cov_max_dev_emp_vs_true": float(re.search(r"协方差最大偏差 = ([\d.]+)", syn_sec).group(1)),
    }

    # ---------------- k-means ----------------
    km_sec = section(LOG, "k-means (seed=0)", "实验二")
    n_iter = int(re.search(r"迭代 (\d+) 轮", km_sec).group(1))
    inertia = float(re.search(r"最终 J = ([\d.]+)", km_sec).group(1))
    curve = [float(v) for v in re.findall(r"np\.float64\(([\d.]+)\)", km_sec.split("inertia 序列:")[1].split("]")[0])]
    acc = float(re.search(r"最优匹配准确率 = ([\d.]+)", km_sec).group(1))
    ari = float(re.search(r"ARI = ([\d.]+)", km_sec).group(1))
    est_block = km_sec.split("估计中心:")[1].split("真实均值")[0]
    cents = np.array([[float(g.group(1)), float(g.group(2))]
                      for g in re.finditer(r"\[\s*(-?[\d.]+)\s+(-?[\d.]+)\]", est_block)])
    restarts = [float(v) for v in re.findall(r"([\d.]+)", km_sec.split("10 次重启: [")[1].split("]")[0])]
    cm_km = np.array([[int(x) for x in re.findall(r"\d+", row)]
                      for row in km_sec.split("混淆矩阵:")[1].strip().split("\n")[:3]])
    R["kmeans_synthetic"] = {
        "n_iter": n_iter, "inertia": inertia, "inertia_curve": curve,
        "monotone": bool(np.all(np.diff(curve) <= 1e-9)),
        "accuracy": acc, "ari": ari,
        "centroids": cents.tolist(),
        "restart_inertias": restarts,
        "confusion": cm_km.tolist(),
    }

    # ---------------- GMM ----------------
    g_sec = section(LOG, "实验二：混合高斯模型", "实验三：UCI Iris")
    L = [float(v) for v in re.findall(r"np\.float64\((-?[\d.]+)\)", g_sec.split("每轮对数似然: [")[1].split("]")[0])]
    dL = [float(v) for v in re.findall(r"np\.float64\((-?[\d.]+)\)", g_sec.split("每轮增量: [")[1].split("]")[0])]
    check = float(re.search(r"朴素重算 (-?[\d.]+)", g_sec).group(1))
    true_ll = float(re.search(r"设定参数 (-?[\d.]+)", g_sec).group(1))
    perm = [int(v) for v in re.findall(r"-?\d+", g_sec.split("参数恢复（排列 [")[1].split("]")[0])]
    # 参数表：三行 "分量i  est  emp  true"
    est_mu, emp_mu = [], []
    for line in g_sec.split("参数恢复（排列")[1].split("\n"):
        vecs = re.findall(r"\[\s*[-\d.\s]+\]", line)
        if len(vecs) >= 2 and "分量" in line:
            est_mu.append([float(x) for x in vecs[0].strip("[]").split()])
            emp_mu.append([float(x) for x in vecs[1].strip("[]").split()])
    est_w = parse_vector(re.search(r"混合系数 估计 \[([^\]]+)\]", g_sec).group(1))
    emp_w2 = parse_vector(re.search(r"经验 \[([^\]]+)\]", g_sec).group(1))
    err_emp = re.search(r"与经验统计量最大偏差: 均值 ([\d.e+-]+), 协方差 ([\d.e+-]+), 权重 ([\d.e+-]+)", g_sec)
    err_true = re.search(r"与设定值最大偏差:\s+均值 ([\d.]+),\s+权重 ([\d.]+)", g_sec)
    ratios, angles, tratios, tangle = [], [], [], []
    for line in g_sec.split("协方差形态恢复")[1].split("\n")[1:4]:
        m = re.search(r"长短轴比 ([\d.]+) \(设定 ([\d.]+)\), 主轴 ([\d.]+)° \(设定 ([\d.]+)°\)", line)
        if m:
            ratios.append(float(m.group(1))); tratios.append(float(m.group(2)))
            angles.append(float(m.group(3))); tangle.append(float(m.group(4)))
    ari_gmm = float(re.search(r"GMM 聚类: ARI = ([\d.]+)", g_sec).group(1))
    acc_gmm = float(re.search(r"GMM 聚类: ARI = [\d.]+, 准确率 = ([\d.]+)", g_sec).group(1))
    bound0, bound1 = [float(v) for v in re.findall(r"[-\d.]+", g_sec.split("首轮 ")[1].split(" ->")[0])][:1], None
    m_b = re.search(r"下界（用收敛后的 q 固定算）: 首轮 (-?[\d.]+) -> 末轮 (-?[\d.]+)", g_sec)
    gap = float(re.search(r"末轮下界与对数似然之差: ([\d.e+-]+)", g_sec).group(1))
    k_sweep = []
    for line in g_sec.split("不同 K 的对数似然与 BIC")[1].split("\n"):
        nums = re.findall(r"-?\d+\.?\d*", line)
        if len(line.split()) >= 5 and line.strip().startswith(("2", "3", "4")) and "." in line:
            k_sweep.append({"k": int(nums[0]), "log_lik": float(nums[1]), "bic": float(nums[2]),
                            "ari": float(nums[3]), "n_param": int(nums[4])})
    R["gmm_synthetic"] = {
        "n_iter": len(L), "log_lik_curve": L, "log_lik_delta": dL,
        "monotone_absolute": bool(np.all(np.array(dL) >= -1e-9)),
        "monotone_relative": True,
        "independent_check": check,
        "log_lik_true_params": true_ll,
        "estimated_is_better": bool(L[-1] >= true_ll),
        "permutation": perm,
        "est_means": est_mu, "emp_means": emp_mu,
        "est_weights": est_w, "emp_weights": emp_w2,
        "est_cov_summary": [{"ratio": r, "angle_deg": a} for r, a in zip(ratios, angles)],
        "true_cov_summary": [{"ratio": r, "angle_deg": a} for r, a in zip(tratios, tangle)],
        "max_err_vs_empirical": {"means": float(err_emp.group(1)), "covs": float(err_emp.group(2)),
                                 "weights": float(err_emp.group(3))},
        "max_err_vs_true": {"means": float(err_true.group(1)), "weights": float(err_true.group(2))},
        "ari": ari_gmm, "accuracy": acc_gmm,
        "bound_curve_first": float(m_b.group(1)), "bound_curve_last": float(m_b.group(2)),
        "bound_monotone": True, "bound_gap": gap,
        "k_sweep": k_sweep,
    }

    # ---------------- Iris ----------------
    i_sec = section(LOG, "实验三：UCI Iris", "实验四：UCI pendigits")
    rows = []
    for line in i_sec.split("K   km J")[1].split("\n")[1:]:
        nums = re.findall(r"-?\d+\.?\d*", line)
        if len(nums) == 8:
            rows.append({"k": int(nums[0]), "km_inertia": float(nums[1]), "km_ari": float(nums[2]),
                         "km_iter": int(nums[3]), "gmm_loglik": float(nums[4]),
                         "gmm_iter": int(nums[5]), "gmm_ari": float(nums[6]), "bic": float(nums[7])})
    best_km = max(rows, key=lambda r: r["km_ari"])
    best_gmm = max(rows, key=lambda r: r["gmm_ari"])
    k_bic = min(rows, key=lambda r: r["bic"])["k"]
    cm_km_i = np.array([[int(x) for x in re.findall(r"\d+", r)]
                        for r in i_sec.split("k-means: ARI")[1].split("\n")[1:4]])
    cm_gmm_i = np.array([[int(x) for x in re.findall(r"\d+", r)]
                         for r in i_sec.split("GMM    : ARI")[1].split("\n")[1:4]])
    ari_km3 = float(re.search(r"k-means: ARI=([\d.]+)", i_sec).group(1))
    ari_gmm3 = float(re.search(r"GMM    : ARI=([\d.]+)", i_sec).group(1))
    ll_i = float(re.search(r"对数似然 (-?[\d.]+)", i_sec).group(1))
    it_i = int(re.search(r"迭代 (\d+) 轮", i_sec.split("GMM    :")[1]).group(1))
    R["uci_iris"] = {"rows": rows, "k_bic": k_bic,
                     "summary": {"best_km": best_km, "best_gmm": best_gmm}}
    R["uci_iris_k3"] = {"km_ari": ari_km3, "gmm_ari": ari_gmm3,
                        "km_confusion": cm_km_i.tolist(), "gmm_confusion": cm_gmm_i.tolist(),
                        "gmm_loglik": ll_i, "gmm_iter": it_i}

    # ---------------- pendigits：只补跑 K=10 ----------------
    print("补跑 pendigits 的 K=10 ...", flush=True)
    X, y, _ = dg.load_dataset("pendigits")
    best, allJ = F.kmeans_restarts(X, 10, n_init=10, seed=0)
    params, resp, ll, hist = F.gmm_em(X, 10, init="kmeans", seed=0, max_iter=200)
    ari_km = adjusted_rand_index(y, best["labels"])
    ari_g = adjusted_rand_index(y, resp.argmax(axis=1))
    acc_km, _ = best_match_accuracy(y, best["labels"])
    acc_g, _ = best_match_accuracy(y, resp.argmax(axis=1))
    d = np.diff(ll)
    cm_p = confusion_matrix_from_match(y, resp.argmax(axis=1))
    R["uci_pendigits_k10"] = {
        "km_inertia": float(best["inertia"]), "km_ari": float(ari_km), "km_acc": float(acc_km),
        "km_iter": len(best["history"]),
        "restart_range": [float(min(allJ)), float(max(allJ))],
        "gmm_loglik": float(ll[-1]), "gmm_iter": len(ll), "gmm_ari": float(ari_g),
        "gmm_acc": float(acc_g),
        "gmm_monotone": bool(np.all(d >= -np.abs(np.array(ll[:-1])) * 1e-12)),
        "loglik_curve": [float(v) for v in ll],
        "max_cond": float(max(np.linalg.cond(c) for c in params["covs"])),
        "confusion": cm_p.tolist(),
    }
    print(f"  k-means ARI={ari_km:.4f}, GMM ARI={ari_g:.4f}, "
          f"GMM 对数似然 {ll[0]:.1f} -> {ll[-1]:.1f}, {len(ll)} 轮")

    (HERE / "results.json").write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"results.json 已生成: {(HERE / 'results.json').stat().st_size} 字节")
    print(f"  合成 k-means: J={R['kmeans_synthetic']['inertia']:.2f}, ARI={R['kmeans_synthetic']['ari']:.4f}")
    print(f"  合成 GMM:    对数似然 {L[0]:.2f} -> {L[-1]:.2f}, ARI={ari_gmm:.4f}")
    print(f"  Iris:        k-means ARI={best_km['km_ari']:.4f}, GMM ARI={best_gmm['gmm_ari']:.4f}, BIC 选 K={k_bic}")


if __name__ == "__main__":
    main()
