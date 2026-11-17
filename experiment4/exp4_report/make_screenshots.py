"""生成报告用的代码截图与运行结果截图。

输出到 exp4_report/figures/：
    fig19_code_backprop.png    mlp.py 反向传播核心
    fig20_code_fit.png         train.py 训练循环
    fig21_run_gradcheck.png    梯度校验运行输出
    fig22_run_2d.png           二维实验运行输出
    fig23_run_mnist.png        MNIST 全量运行输出
    fig24_run_extra.png        额外数据集运行输出
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
LOGS = HERE / "run_logs"

FONT = "Noto Sans Mono CJK SC"


def render(
    lines: list[str],
    out: Path,
    title: str,
    dark: bool = True,
    fontsize: float = 7.0,
    width: float = 7.2,
) -> None:
    """把若干行文本渲染成一张类似终端/编辑器的截图。"""
    line_h_in = fontsize * 1.38 / 72.0
    height = max(1.2, len(lines) * line_h_in + 0.55)
    fig = plt.figure(figsize=(width, height), dpi=160)
    bg, fg = ("#1e1e1e", "#d4d4d4") if dark else ("#f6f8fa", "#24292f")
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title_color = "#8b949e" if dark else "#57606a"
    ax.text(0.006, 0.996, title, va="top", ha="left", fontsize=8.5,
            color=title_color, family=FONT, transform=ax.transAxes)

    y = 1.0 - 0.42 / height
    for line in lines:
        color = fg
        if "PASS" in line:
            color = "#3fb950"
        elif "FAIL" in line:
            color = "#f85149"
        elif line.startswith("[fig]") or line.startswith("[图]") or line.startswith("[结果]"):
            color = "#58a6ff"
        elif line.startswith("==") or line.startswith("--"):
            color = "#8b949e"
        ax.text(0.006, y, line, va="top", ha="left", family=FONT,
                fontsize=fontsize, color=color, transform=ax.transAxes)
        y -= line_h_in / height

    fig.savefig(out, dpi=160, facecolor=bg)
    plt.close(fig)
    print(f"[截图] {out}")


def numbered_slice(path: Path, marker: str, n: int, offset: int = 0) -> list[str]:
    """从 marker 所在行开始截取 n 行，并加上真实行号。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line) + offset
    selected = lines[start:start + n]
    return [f"{start + i + 1:4d}  {line}" for i, line in enumerate(selected)]


def read_log(name: str, max_lines: int = 40) -> list[str]:
    text = (LOGS / name).read_text(encoding="utf-8").splitlines()
    text = [line for line in text if line.strip()]
    return text[:max_lines]


def main() -> None:
    FIG.mkdir(exist_ok=True)

    # 代码截图
    code1 = numbered_slice(ROOT / "code" / "mlp.py", "def loss_and_grads", 38)
    render(code1, FIG / "fig19_code_backprop.png",
           "code/mlp.py  —  loss_and_grads（反向传播）", dark=False)

    code2 = numbered_slice(ROOT / "code" / "train.py", "for epoch in range(1, epochs + 1):", 26, offset=-3)
    render(code2, FIG / "fig20_code_fit.png",
           "code/train.py  —  fit 的小批量训练循环（节选）", dark=False)

    # 运行结果截图
    render(read_log("gradcheck.txt", 20), FIG / "fig21_run_gradcheck.png",
           "$ python exp4_report/full_experiment.py --parts 0")
    render(read_log("part1.txt", 24), FIG / "fig22_run_2d.png",
           "$ python exp4_report/full_experiment.py --parts 1")
    render(read_log("part5.txt", 24), FIG / "fig23_run_mnist.png",
           "$ python exp4_report/full_experiment.py --parts 5")

    extra_lines = read_log("extra.txt", 40)
    start = next((i for i, line in enumerate(extra_lines) if line.startswith("====")), 0)
    render(extra_lines[start:start + 22], FIG / "fig24_run_extra.png",
           "$ python learn/practice/extra_datasets.py")


if __name__ == "__main__":
    main()
