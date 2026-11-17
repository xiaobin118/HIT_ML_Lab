#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付前的报告自检：核对报告中的关键数字、插图链接、章节完整性。

运行：cd exp6_report && python audit_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "实验六_基于生成模型的图像生成_实验报告.md"
DOCX = HERE / "实验六_基于生成模型的图像生成_实验报告.docx"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

def main() -> int:
    md = MD.read_text(encoding="utf-8")
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1) 关键数字是否出现在报告里
    key_numbers = {
        "主实验 FID": f"{R['main']['final_fid']:.2f}",
        "最终 DCGAN FID": f"{R['final']['final_fid']:.2f}",
        "DCGAN C2ST": f"{R['final']['c2st']['dcgan']['accuracy']:.4f}",
        "VAE C2ST": f"{R['final']['c2st']['vae']['accuracy']:.4f}",
    }
    if "diffusion" in R:
        key_numbers["扩散模型 FID"] = f"{R['diffusion']['eval']['final']['fid']:.2f}"
        key_numbers["扩散模型 C2ST"] = f"{R['diffusion']['eval']['final']['c2st']['accuracy']:.4f}"
        key_numbers["扩散训练步数"] = f"{R['diffusion']['train']['steps']:,}"
    for name, val in key_numbers.items():
        check(f"数字出现：{name}={val}", val in md)

    # 2) 插图链接全部存在，且数量符合预期
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    missing = [l for l in links if not (HERE / l).exists()]
    check("插图链接无缺失", not missing, f"缺失 {missing}")
    n_expected = 31 + (8 if "diffusion" in R else 0)
    check(f"插图数量={n_expected}", len(links) == n_expected, f"实际 {len(links)}")

    # 3) 章节完整性
    headings = ["摘要", "实验目的", "实验内容", "实验环境", "实验原理", "数据", "模型实现",
                "正确性验证", "主实验", "激活函数", "Dropout", "数据量", "超参数",
                "VAE 对照", "最终模型", "讨论", "结论", "参考文献", "附录"]
    for h in headings:
        check(f"章节存在：{h}", h in md)
    if "diffusion" in R:
        check("章节存在：扩散模型", "扩散模型" in md and "## 15." in md)

    # 4) 没有坏占位符
    for bad in ("nan", "PRESENT", "TODO", "FIXME", "None"):
        check(f"无异常字符串 {bad}", bad not in md)

    # 5) docx / figure 文件存在
    check("docx 存在", DOCX.exists() and DOCX.stat().st_size > 1_000_000,
          f"{DOCX.stat().st_size/1e6:.1f} MB" if DOCX.exists() else "缺失")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\n结果：{len(checks)-n_fail}/{len(checks)} 通过")
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
