#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results.json 生成实验三报告（Markdown + docx）。所有数字从 JSON 读取。"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
OUT_MD = HERE / "实验三_kmeans与GMM_实验报告.md"
OUT_DOCX = HERE / "实验三_kmeans与GMM_实验报告.docx"

syn, km, g = R["synthetic"], R["kmeans_synthetic"], R["gmm_synthetic"]
iris, i3, p10 = R["uci_iris"], R["uci_iris_k3"], R["uci_pendigits_k10"]
L, dL, perm = g["log_lik_curve"], g["log_lik_delta"], g["permutation"]
est_mu, emp_mu = g["est_means"], g["emp_means"]
est_w, emp_w = g["est_weights"], g["emp_weights"]
ecs, tcs = g["est_cov_summary"], g["true_cov_summary"]
rows = iris["rows"]
bkm, bgm = iris["summary"]["best_km"], iris["summary"]["best_gmm"]
r3 = next(r for r in rows if r["k"] == 3)
r6 = next(r for r in rows if r["k"] == 6)

def vec(v, nd=3): return "(" + ", ".join(f"{x:.{nd}f}" for x in v) + ")"
def table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)
def curve(vals, nd=1): return " → ".join(f"{v:.{nd}f}" for v in vals)

P = []
P.append("""# 实验三：实现 k-means 聚类方法与混合高斯模型

> 姓名：________　学号：________　班级：________　日期：________
>
> 主实验代码：`exp3_report/full_experiment.py`
>
> 复用模块：`code/kmeans.py`（自实现的 k-means）、`code/data_gen.py`、`code/metrics.py`
>
> 结果文件：`exp3_report/run_output.txt`、`results.json`、`figures/`
>
> 运行方式：`cd exp3_report && python full_experiment.py`""")

P.append("""## 1. 实验目的

1. 理解 k-means 聚类的原理，掌握"分配—更新"两步交替迭代，并能从簇内平方和目标函数推导两步的闭式解。
2. 理解 k-means 的收敛性（目标函数单调不增、有限步停止）与局限（只能到局部最优、决策边界只能是直线、$K$ 需事先给定）。
3. 理解混合高斯模型的生成过程与似然函数，理解隐变量造成的"参数估计与归属推断互相依赖"这一困难。
4. 掌握 EM 算法的推导：用 Jensen 不等式构造对数似然的下界，E 步计算责任度、M 步做加权统计，并证明对数似然单调不降。
5. 掌握用高斯分布生成多个不同均值与方差的数据集的方法，能在已知真实参数的前提下定量评估参数恢复效果。
6. 区分聚类评价的外部指标（ARI、准确率、混淆矩阵）与内部指标（对数似然、BIC），理解各自的适用条件与陷阱。
7. 把实现的算法应用到 UCI 真实数据集（Iris、pendigits），完成从加载、预处理、模型选择到结果分析的完整流程。""")

P.append(f"""## 2. 实验内容

用纯 NumPy 实现 k-means 与混合高斯模型（EM 估计参数），不使用 scipy/sklearn 等提供现成聚类实现的库，也不使用任何自动微分工具。实验分四部分：

1. **合成数据上的 k-means**：用三个不同均值、不同协方差的高斯分布生成 {syn['shape'][0]} 个二维样本，用自实现的 k-means 聚类，考察聚类效果、目标函数的单调性与初始化的影响。
2. **合成数据上的 GMM 与 EM**：用同一份数据，用自实现的 EM 估计 $K=3$ 个分量的参数，记录每轮迭代后的对数似然、检验单调性，并把估计参数与设定值逐项对比。
3. **UCI Iris**：150 样本、4 维、3 类，验证真实数据上两种算法的差异并考察 $K$ 的选取。
4. **UCI pendigits**：10992 样本、16 维、10 类，验证算法在更大规模数据上的可扩展性与收敛行为。""")

P.append("""## 3. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | WSL2 Ubuntu（Linux 6.6 x86_64） |
| 编程语言 | Python 3.13.12（Anaconda） |
| 核心库 | NumPy 2.5.2、Matplotlib 3.11.0 |
| 未使用 | scipy、scikit-learn（无现成聚类实现）、任何自动微分工具 |
| 运行方式 | `python full_experiment.py` |
| 硬件 | x86_64 CPU，无需 GPU |
| 随机性 | 合成数据 `seed=0`；k-means 重启用 `seed=0..9`；EM 用 k-means 结果初始化 |""")

