"""生成第 1 步讲义用的示意图（数字全部由本脚本算出，不手填）。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent

# 一个 5x5 的"竖边缘"图像：左半 0，右半 1
x = np.array([
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
], dtype=np.float32)
# 竖直边缘检测核（Sobel 的竖直版）
k = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)
# 手算 3x3 输出（cross-correlation，无 padding）
y = np.zeros((3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        y[i, j] = float((x[i:i+3, j:j+3] * k).sum())

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, mat, title, cmap, vmin, vmax in [
    (axes[0], x, "输入 5×5（左黑右白，一条竖直边缘）", "gray", 0, 1),
    (axes[1], k, "卷积核 3×3（竖直边缘检测器）", "RdBu_r", -2, 2),
    (axes[2], y, "输出 3×3（每格是一个加权和）", "RdBu_r", -6, 6),
]:
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                color="white" if abs(v - (vmin + vmax) / 2) > abs(vmax - vmin) / 4 else "black",
                fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("卷积直觉：一个 3×3 的小窗口在 5×5 图像上滑动，每个位置得到一个加权和", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "01_conv_intuition.png", bbox_inches="tight", dpi=140)
plt.close(fig)

# 最大池化示意：4x4 -> 2x2
x2 = np.arange(16, dtype=np.float32).reshape(4, 4)
p = np.array([[x2[i:i+2, j:j+2].max() for j in (0, 2)] for i in (0, 2)])
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
for ax, mat, title in [(axes[0], x2, "输入 4×4"), (axes[1], p, "最大池化 2×2，步长 2 → 2×2")]:
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=15)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                color="white" if v >= 8 else "black", fontsize=12)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
for (i, j) in [(0, 0), (2, 2)]:
    axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 2, 2, fill=False, edgecolor="red", linewidth=2))
    axes[1].add_patch(plt.Rectangle((j/2 - 0.5, i/2 - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
fig.suptitle("池化：每个 2×2 窗口只保留最大值", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "01_maxpool.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print("figures written to", OUT)
