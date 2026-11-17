#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把实验六报告重排成课程模板要求的 7 部分结构（不重写正文内容）。

输入：实验六_基于生成模型的图像生成_实验报告.md（原 17 节版本）
输出：实验六_基于生成模型的图像生成_实验报告_7部分.md

章节映射（模板：摘要 + 7 个编号部分）：
    摘要
    1. 实验目的
    2. 实验内容
    3. 实验环境
    4. 实验过程、结果及分析（4.1 实验原理 … 4.15 运行时间与复现方式）
    5. 实验总体结论
    6. 完整实验代码
    7. 参考文献
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "实验六_基于生成模型的图像生成_实验报告.md"
OUT = HERE / "实验六_基于生成模型的图像生成_实验报告_7部分.md"

# 原 ## 节标题 -> 新标题（None 表示丢弃）
SECTION_MAP = {
    "## ——用 DCGAN、变分自编码器与扩散模型生成 64×64 二次元动漫头像": None,
    "## 摘要": "## 摘要",
    "## 1. 实验目的": "## 1. 实验目的",
    "## 2. 实验内容": "## 2. 实验内容",
    "## 3. 实验环境": "## 3. 实验环境",
    "## 4. 实验原理": "## 4. 实验过程、结果及分析\n\n### 4.1 实验原理",
    "## 5. 数据：Anime Face Dataset": "### 4.2 数据与预处理",
    "## 6. 模型实现": "### 4.3 模型实现",
    "## 7. 正确性验证": "### 4.4 正确性验证",
    "## 8. 主实验：DCGAN 全量训练": "### 4.5 DCGAN 主实验",
    "## 9. 激活函数实验": "### 4.6 激活函数实验",
    "## 10. Dropout 实验": "### 4.7 Dropout 实验",
    "## 11. 数据量实验": "### 4.8 数据量实验",
    "## 12. 超参数实验": "### 4.9 超参数实验",
    "## 13. VAE 对照实验": "### 4.10 VAE 对照实验",
    "## 14. 最终模型与 DCGAN vs VAE 对比": "### 4.11 最终 DCGAN 模型",
    "## 15. 追加实验：扩散模型对照": "### 4.12 扩散模型对照实验",
    "## 16. 讨论": "### 4.13 讨论",
    "## 17. 结论": "## 5. 实验总体结论",
    "## 参考文献": "## 7. 参考文献",
    "## 附录 A：代码结构": None,
    "## 附录 B：全部生成模型配置的 FID 汇总": "### 4.14 全部生成模型配置的 FID 汇总",
    "## 附录 C：运行时间与复现方式": "### 4.15 运行时间与复现方式",
}

# 需要重排子标题的源节：新节号 -> 源 ## 标题
SUBSECTION_PARENTS = {
    "4.1": "## 4. 实验原理",
    "4.3": "## 6. 模型实现",
    "4.9": "## 12. 超参数实验",
    "4.11": "## 14. 最终模型与 DCGAN vs VAE 对比",
    "4.12": "## 15. 追加实验：扩散模型对照",
    "4.13": "## 16. 讨论",
}

CROSSREF = [
    ("第 4.4 节", "第 4.1.4 节"),
    ("第 9 节", "第 4.6 节"),
    ("第 10 节", "第 4.7 节"),
    ("第 11 节", "第 4.8 节"),
    ("第 13 节", "第 4.10 节"),
    ("第 14 节", "第 4.11 节"),
    ("第 15 节", "第 4.12 节"),
]

CODE_FILES = [
    ("6.1", "code/data.py", "数据：去重、缓存、划分与 DataLoader"),
    ("6.2", "code/models.py", "模型：DCGAN 生成器/判别器与卷积 VAE"),
    ("6.3", "code/engine.py", "训练引擎：GAN/VAE 训练循环、采样与插值"),
    ("6.4", "code/diffusion.py", "扩散模型：DDPM U-Net、余弦调度、EMA、DDIM 采样"),
    ("6.5", "code/metrics.py", "评估：FID、最近邻、C2ST、重建指标"),
    ("6.6", "code/viz.py", "绘图：全部插图"),
    ("6.7", "exp6_report/full_experiment.py", "DCGAN + VAE 十部分实验主脚本"),
    ("6.8", "exp6_report/diffusion_train.py", "扩散模型训练脚本"),
    ("6.9", "exp6_report/diffusion_eval.py", "扩散模型评估脚本"),
    ("6.10", "exp6_report/diffusion_report.py", "扩散模型插图与结果写回"),
]


def split_sections(lines: list[str]):
    """按 ## 标题切分（跳过代码块里的行），返回 [(heading, body_lines)]。"""
    sections = []
    cur_head, cur_body = None, []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## "):
            if cur_head is not None:
                sections.append((cur_head, cur_body))
            cur_head, cur_body = line.rstrip(), []
        else:
            if cur_head is None:
                continue  # 丢弃封面/标题区
            cur_body.append(line)
    if cur_head is not None:
        sections.append((cur_head, cur_body))
    return sections


def renumber_subsections(body: list[str], new_parent: str) -> list[str]:
    """把 ### X.Y 重排为 #### 4.parent.k，#### X.Y.Z 重排为 ##### 4.parent.k.z。"""
    out = []
    sub = 0
    in_code = False
    sub_sub = 0
    for line in body:
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("### "):
            sub += 1
            sub_sub = 0
            title = line[4:].strip()
            # 去掉原编号（如 "4.1 "、"12.3 "）
            title = re.sub(r"^\d+(\.\d+)*\s*", "", title)
            out.append(f"#### {new_parent}.{sub} {title}")
            continue
        if not in_code and line.startswith("#### "):
            sub_sub += 1
            title = re.sub(r"^\d+(\.\d+)*\s*", "", line[5:].strip())
            out.append(f"##### {new_parent}.{sub}.{sub_sub} {title}")
            continue
        out.append(line)
    return out


def code_section() -> list[str]:
    lines = ["## 6. 完整实验代码\n",
             "本实验的全部代码均由本人编写（PyTorch 部分只使用基础张量/网络层 API，"
             "GAN、VAE、DDPM 的训练与采样逻辑均从零实现）。下列代码与 `exp6_report/` 中的实际文件一致，"
             "可直接运行复现正文中的全部结果。\n"]
    for num, rel, desc in CODE_FILES:
        path = HERE.parent / rel
        src = path.read_text(encoding="utf-8")
        lines.append(f"### {num} `{rel}`\n")
        lines.append(f"{desc}。\n")
        lines.append("````python")
        lines.append(src.rstrip("\n"))
        lines.append("````\n")
    return lines


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    lines = md.splitlines()
    sections = split_sections(lines)

    parts: list[str] = []
    appendix: list[str] = []
    for head, body in sections:
        if head not in SECTION_MAP:
            print(f"[warn] 未映射的节，原样保留：{head}")
            parts.extend([head, ""] + body)
            continue
        new = SECTION_MAP[head]
        if new is None:
            continue
        # 子标题重排
        for parent, src_head in SUBSECTION_PARENTS.items():
            if src_head == head:
                body = renumber_subsections(body, parent)
                break
        if head.startswith("## 附录 B") or head.startswith("## 附录 C"):
            appendix.extend([new, ""] + body)
            continue
        parts.extend([new, ""] + body)

    body_text = "\n".join(parts)
    idx5 = body_text.find("## 5. 实验总体结论")
    if idx5 < 0:
        raise SystemExit("找不到第 5 部分标题")
    body_text = body_text[:idx5] + "\n".join(appendix).rstrip() + "\n\n" + body_text[idx5:]
    text = body_text
    # 交叉引用改成新编号
    for old, new in CROSSREF:
        text = text.replace(old, new)
    # 数学公式：\tag{n} -> \qquad (n)
    text = re.sub(r"\\tag\{(\d+)\}", r"\\qquad (\1)", text)
    # 在参考文献前插入完整代码
    code_lines = code_section()
    idx = text.find("## 7. 参考文献")
    if idx < 0:
        raise SystemExit("找不到参考文献标题")
    text = text[:idx] + "\n".join(code_lines) + "\n\n" + text[idx:]

    OUT.write_text(text, encoding="utf-8")
    print(f"已写出 {OUT.name}: {len(text):,} 字符，{text.count(chr(10)):,} 行")


if __name__ == "__main__":
    main()
