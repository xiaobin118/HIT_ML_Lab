import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
P = json.loads((HERE/"results.json").read_text(encoding="utf-8"))["uci_pendigits_k10"]
C = ["#4C72B0", "#DD8452", "#55A868"]
L = P["loglik_curve"]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax = axes[0]
ax.plot(range(1, len(L)+1), L, "-", color=C[0], lw=2)
ax.set_xlabel("EM 迭代轮数"); ax.set_ylabel("对数似然")
ax.set_title(f"pendigits (K=10)：EM 共 {len(L)} 轮，对数似然 {L[0]:.0f} → {L[-1]:.0f}")
ax = axes[1]
labels = ["k-means", "GMM"]
ari = [P["km_ari"], P["gmm_ari"]]
acc = [P["km_acc"], P["gmm_acc"]]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, ari, w, color=C[0], label="ARI")
b2 = ax.bar(x + w/2, acc, w, color=C[1], label="准确率")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.4f}",
                ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.85); ax.set_ylabel("指标值")
ax.set_title(f"pendigits (K=10)：GMM 优于 k-means\n（协方差最大条件数 {P['max_cond']:.1e}）")
ax.legend(fontsize=9)
fig.tight_layout()
out = HERE/"figures"/"fig9_pendigits.png"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print("已生成", out.name)
