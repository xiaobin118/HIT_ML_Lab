# -*- coding: utf-8 -*-
"""只补跑 pendigits 的 K=10，产出报告需要的数字。"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_experiment as F
import data_gen as dg
from metrics import adjusted_rand_index, best_match_accuracy, confusion_matrix_from_match

X, y, _ = dg.load_dataset("pendigits")
print(f"pendigits: {X.shape}, 类别 {np.bincount(y).tolist()}", flush=True)
best, allJ = F.kmeans_restarts(X, 10, n_init=10, seed=0)
print(f"k-means K=10: J={best['inertia']:.1f}, ARI={adjusted_rand_index(y,best['labels']):.4f}, "
      f"acc={best_match_accuracy(y,best['labels'])[0]:.4f}, 重启J范围[{min(allJ):.1f},{max(allJ):.1f}]", flush=True)
params, resp, ll, hist = F.gmm_em(X, 10, init="kmeans", seed=0, max_iter=200)
d = np.diff(ll)
out = {
  "km_inertia": float(best["inertia"]), "km_ari": float(adjusted_rand_index(y,best["labels"])),
  "km_acc": float(best_match_accuracy(y,best["labels"])[0]), "km_iter": len(best["history"]),
  "restart_range": [float(min(allJ)), float(max(allJ))],
  "gmm_loglik": float(ll[-1]), "gmm_iter": len(ll),
  "gmm_ari": float(adjusted_rand_index(y,resp.argmax(1))),
  "gmm_acc": float(best_match_accuracy(y,resp.argmax(1))[0]),
  "gmm_monotone": bool(np.all(d >= -np.abs(np.array(ll[:-1]))*1e-12)),
  "loglik_first": float(ll[0]), "loglik_curve": [float(v) for v in ll],
  "max_cond": float(max(np.linalg.cond(c) for c in params["covs"])),
  "confusion": confusion_matrix_from_match(y, resp.argmax(1)).tolist(),
}
print(f"GMM K=10: 对数似然 {ll[0]:.1f} -> {ll[-1]:.1f}, {len(ll)} 轮, "
      f"ARI={out['gmm_ari']:.4f}, acc={out['gmm_acc']:.4f}, 单调={out['gmm_monotone']}, "
      f"条件数={out['max_cond']:.2e}", flush=True)
Path(HERE/"pendigits_k10.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写入 pendigits_k10.json")
