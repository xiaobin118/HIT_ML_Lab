"""把已生成的报告重排成学校要求的 7 部分，并追加代码截图、运行截图和完整代码。

流程：
    1. 调用 gen_report.py 重新生成基础报告（Markdown + docx）；
    2. 读取基础 Markdown，按章节切分；
    3. 重排为：
         1 实验目的
         2 实验内容
         3 实验环境
         4 实验过程、结果及分析（原理 / 实现 / 四组实验 / 扩展 / 讨论 / 复现）
         5 实验总体结论
         6 完整实验代码
         7 参考文献
    4. 在 4.2 插入代码截图与运行截图，在 4.7 插入额外数据集运行截图；
    5. 写出提交版 Markdown，并调用 pandoc 生成 docx。

运行：python make_submission_report.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_MD = HERE / "实验四_多层感知机_实验报告.md"
BASE_DOCX = HERE / "实验四_多层感知机_实验报告.docx"

# 先重新生成基础报告，保证内容与 results.json / extra_results.json 一致
print("重新生成基础报告 ...")
subprocess.run([sys.executable, str(HERE / "gen_report.py")], check=True)

text = BASE_MD.read_text(encoding="utf-8")

chunks = re.split(r"(?m)^## ", text)
front = chunks[0].rstrip() + "\n\n"
front = front.replace(
    "再由 `python gen_report.py` 生成本报告",
    "再由 `python gen_report.py` 与 `python make_submission_report.py` 生成提交版报告",
)
sections: dict[str, str] = {}
for chunk in chunks[1:]:
    head, _, body = chunk.partition("\n")
    sections[head.strip()] = body


def body(prefix: str) -> str:
    for head, content in sections.items():
        if head.startswith(prefix):
            return content.strip("\n")
    raise KeyError(f"找不到章节: {prefix}\n现有: {list(sections)}")


def subsections(content: str, old_major: int, new_major: str) -> str:
    """把 ### old.x 改成 #### new.x。"""
    return re.sub(
        rf"(?m)^### {old_major}\.(\d+) ",
        lambda m: f"#### {new_major}.{m.group(1)} ",
        content,
    )


def fix_refs(content: str) -> str:
    """修正章节号与表格编号的交叉引用。"""
    content = content.replace("第 13 节", "第 4.7 节")
    content = content.replace("第 8 节", "第 4.5 节")
    content = content.replace("第 9.4 节", "第 4.6.4 节")
    content = content.replace("第 6～9 节", "第 4.3～4.6 节")
    content = content.replace("4.1 节", "4.1.1 节")
    content = content.replace("结果见表 2、图 4", "结果见图 4 与下表")
    content = content.replace("结果见表 3、图 5", "结果见图 5 与下表")
    return content


# ---------------------------------------------------------------------------
# 4.2 后面的代码 / 运行截图
# ---------------------------------------------------------------------------

SHOTS = r"""
#### 4.2.5 关键代码截图

下面两张截图直接截取自仓库源码，行号与 `code/mlp.py`、`code/train.py` 对应。

![代码截图 1　MLP 反向传播核心（code/mlp.py，节选）](figures/fig19_code_backprop.png)

![代码截图 2　小批量训练循环（code/train.py，节选）](figures/fig20_code_fit.png)

#### 4.2.6 运行结果截图

以下为实验脚本的实际终端输出（`full_experiment.py` 的梯度校验、二维实验、MNIST 全量实验三部分）。

![运行结果截图 1　数值梯度校验](figures/fig21_run_gradcheck.png)

![运行结果截图 2　二维对照实验](figures/fig22_run_2d.png)

![运行结果截图 3　MNIST 全量实验](figures/fig23_run_mnist.png)
"""

EXTRA_SHOT = r"""
![运行结果截图 4　额外数据集对照实验](figures/fig24_run_extra.png)
"""

# ---------------------------------------------------------------------------
# 第 6 部分：完整代码
# ---------------------------------------------------------------------------

CODE_FILES = [
    ("code/mlp.py", "MLP 前向传播、反向传播、初始化、数值梯度校验"),
    ("code/linear.py", "线性 softmax 分类器"),
    ("code/train.py", "SGD / Momentum / Adam 与训练循环"),
    ("code/datasets.py", "二维合成数据、MNIST 加载、分层抽样"),
    ("code/metrics.py", "混淆矩阵、每类指标、易混淆类别对"),
    ("code/viz.py", "决策边界、曲线、混淆矩阵等绘图"),
    ("exp4_report/full_experiment.py", "四组实验的主脚本"),
]


def build_code_section() -> str:
    total = sum(len((ROOT / p).read_text(encoding="utf-8").splitlines()) for p, _ in CODE_FILES)
    out = [
        "## 6. 完整实验代码",
        "",
        f"> 以下源码与仓库文件逐字一致，共 {total} 行。运行入口是 `exp4_report/full_experiment.py`；"
        "报告生成脚本 `gen_report.py`、截图脚本 `make_screenshots.py` 不列入正文。",
        "",
        "| 文件 | 行数 | 作用 |",
        "|---|---|---|",
    ]
    for path, desc in CODE_FILES:
        n = len((ROOT / path).read_text(encoding="utf-8").splitlines())
        out.append(f"| `{path}` | {n} | {desc} |")
    out.append("")

    for idx, (path, desc) in enumerate(CODE_FILES, start=1):
        code = (ROOT / path).read_text(encoding="utf-8").rstrip()
        out += [
            f"### 6.{idx} `{path}`",
            "",
            desc + "。",
            "",
            "```python",
            code,
            "```",
            "",
        ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

parts = [
    front,
    "## 摘要\n" + fix_refs(body("摘要")),
    "## 1. 实验目的\n" + fix_refs(body("1. 实验目的")),
    "## 2. 实验内容\n" + fix_refs(body("2. 实验内容"))
    + "\n\n**扩展部分。** 为了检验上述结论是否只对二维合成数据和 MNIST 成立，"
    "又在 Fashion-MNIST、Pendigits、Optdigits、Letter、Iris、WDBC 六个公开数据集上"
    "重复了线性分类器与 MLP 的对照（详见 4.7 节）。",
    "## 3. 实验环境\n" + fix_refs(body("3. 实验环境")),
    "## 4. 实验过程、结果及分析",
    "### 4.1 实验原理\n" + fix_refs(subsections(body("4. 实验原理"), 4, "4.1")),
    "### 4.2 实现与正确性验证\n"
    + fix_refs(subsections(body("5. 实现与正确性验证"), 5, "4.2"))
    + "\n"
    + SHOTS,
    "### 4.3 实验一：二维四类数据上的决策边界\n"
    + fix_refs(subsections(body("6. 实验一"), 6, "4.3")),
    "### 4.4 实验二：不同数据量下的表现\n"
    + fix_refs(subsections(body("7. 实验二"), 7, "4.4")),
    "### 4.5 实验三：超参数对结果的影响\n"
    + fix_refs(subsections(body("8. 实验三"), 8, "4.5")),
    "### 4.6 实验四：MNIST 手写数字分类\n"
    + fix_refs(subsections(body("9. 实验四"), 9, "4.6")),
    "### 4.7 扩展实验：更多数据集上的验证\n"
    + fix_refs(body("13. 扩展实验"))
    + "\n"
    + EXTRA_SHOT,
    "### 4.8 讨论\n" + fix_refs(subsections(body("10. 讨论"), 10, "4.8")),
    "### 4.9 复现说明\n" + fix_refs(subsections(body("12. 复现说明"), 12, "4.9")),
    "## 5. 实验总体结论\n" + fix_refs(body("11. 结论")),
    build_code_section(),
    "## 7. 参考文献\n" + fix_refs(body("参考文献")),
]

final_md = "\n\n".join(part.strip("\n") for part in parts if part.strip()) + "\n"

# 基本自检：7 个一级标题齐全
for required in [
    "## 1. 实验目的",
    "## 2. 实验内容",
    "## 3. 实验环境",
    "## 4. 实验过程、结果及分析",
    "## 5. 实验总体结论",
    "## 6. 完整实验代码",
    "## 7. 参考文献",
]:
    if required not in final_md:
        raise RuntimeError(f"缺少标题: {required}")

BASE_MD.write_text(final_md, encoding="utf-8")
print(f"提交版 Markdown 已写入 {BASE_MD}（{len(final_md)} 字符）")

subprocess.run(
    ["pandoc", str(BASE_MD), "-o", str(BASE_DOCX), "--resource-path", str(HERE)],
    check=True,
)
print(f"提交版 DOCX 已写入 {BASE_DOCX}")
