#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 pandoc 生成的 7 部分正文 docx 加上课程模板封面，输出最终报告 docx。

- 封面（标题/第 六 次/实验题目/学号/姓名）直接复制自课程模板，只替换次号与题目；
- 正文样式来自模板的 styles.xml（pandoc --reference-doc），因此标题/正文/图注风格与模板一致；
- 正文第一段（摘要）设置分页，封面单独成页。
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import docx
from docx.enum.text import WD_BREAK

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = ROOT / "template_requirements.docx"
RAW = Path("/tmp/report7_raw.docx")
OUT = HERE / "实验六_基于生成模型的图像生成_实验报告_模板版.docx"


def main() -> None:
    content = docx.Document(str(RAW))
    tpl = docx.Document(str(TEMPLATE))

    # 复制封面的前 9 个段落，替换“次号”和“实验题目”
    cover_ps = tpl.paragraphs[:9]
    for p in cover_ps:
        for run in p.runs:
            run.text = run.text.replace("四", "六").replace(
                "多层感知机", "基于生成模型的图像生成"
            )

    body = content.element.body
    for i, p in enumerate(cover_ps):
        body.insert(i, copy.deepcopy(p._p))

    # 正文第一段（摘要）另起一页
    for para in content.paragraphs:
        if para.text.strip() == "摘要":
            para.paragraph_format.page_break_before = True
            break

    # 模板里的显示公式是居中的：把含 m:oMathPara 的段落设为居中
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    n_math = 0
    for para in content.paragraphs:
        if "m:oMathPara" in para._p.xml:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            n_math += 1
    print(f"居中显示公式段落：{n_math}")

    content.save(str(OUT))
    size = OUT.stat().st_size / 1e6
    print(f"已生成 {OUT.name}（{size:.1f} MB，{len(content.paragraphs)} 段）")


if __name__ == "__main__":
    main()
