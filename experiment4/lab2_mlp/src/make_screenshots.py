"""把实验终端输出渲染成报告用截图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "run_logs" / "experiment.log"
FIG = ROOT / "figures"
FONT = "Noto Sans Mono CJK SC"


def render(lines: list[str], out: Path, title: str, fontsize: float = 7.2) -> None:
    width = 7.4
    line_h = fontsize * 1.38 / 72
    height = max(1.2, len(lines) * line_h + 0.55)
    fig = plt.figure(figsize=(width, height), dpi=170)
    bg, fg = "#1e1e1e", "#d4d4d4"
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.006, 0.996, title, va="top", ha="left", fontsize=8.2,
            color="#8b949e", family=FONT, transform=ax.transAxes)
    y = 1.0 - 0.42 / height
    for line in lines:
        color = fg
        if "PASS" in line:
            color = "#3fb950"
        elif line.startswith("==") or line.startswith("--"):
            color = "#8b949e"
        elif line.startswith("[图]"):
            color = "#58a6ff"
        ax.text(0.006, y, line, va="top", ha="left", family=FONT,
                fontsize=fontsize, color=color, transform=ax.transAxes)
        y -= line_h / height
    fig.savefig(out, dpi=170, facecolor=bg)
    plt.close(fig)
    print(f"[截图] {out}")


def main() -> None:
    lines = LOG.read_text(encoding="utf-8").splitlines()
    clean = [ln for ln in lines if ln.strip()]

    def find(prefix: str) -> int:
        return next(i for i, ln in enumerate(clean) if ln.startswith(prefix))

    data_line = next(ln for ln in clean if ln.startswith("数据: X="))
    i1 = find("表 1：")
    i2 = find("超参数扫描")
    render([data_line, ""] + clean[i1:i2], FIG / "fig_run_main.png",
           "$ python src/run_experiments.py  (表 1 / 表 2)", fontsize=8.0)

    i2 = find("超参数扫描")
    i3 = find("基准配置训练曲线")
    render(clean[i2:i3], FIG / "fig_run_sweeps.png",
           "$ python src/run_experiments.py  (超参数扫描)")

    i4 = find("最优配置")
    render(clean[i4:], FIG / "fig_run_final.png",
           "$ python src/run_experiments.py  (最优配置与误差分析)")


if __name__ == "__main__":
    main()
