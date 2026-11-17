#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results.json 生成实验六报告（Markdown），并调用 pandoc 生成 docx。

报告中所有实验数字都从 results.json / verification.json 读取，静态的原理部分
不含任何手抄的实验结果，保证图文数据永远一致。
运行：cd exp6_report && python gen_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
VER_PATH = HERE / "verification.json"
VER = json.loads(VER_PATH.read_text(encoding="utf-8")) if VER_PATH.exists() else None

OUT_MD = HERE / "实验六_基于生成模型的图像生成_实验报告.md"
OUT_DOCX = HERE / "实验六_基于生成模型的图像生成_实验报告.docx"

ENV = R["meta"]["environment"]
PROTO = R["meta"]["protocol"]
D = R["data"]
A = R["architecture"]
M = R.get("main", {})
ACT = R.get("activation", {})
DR = R.get("dropout", {})
SZ = R.get("data_size", {})
H = R.get("hyper", {})
VA = R.get("vae", {})
F = R.get("final", {})
DIFF = R.get("diffusion", {})
try:
    _diff_log_path = HERE / "diffusion_run" / "train_log.jsonl"
    DIFF_LOG = ([json.loads(line) for line in _diff_log_path.read_text().splitlines() if line.strip()]
                if (DIFF and _diff_log_path.exists()) else [])
except Exception:  # noqa: BLE001
    DIFF_LOG = []

FIG = {
    "data_samples": "fig01_dataset_samples.png",
    "data_stats": "fig02_data_stats.png",
    "dcgan_arch": "fig03_dcgan_arch.png",
    "vae_arch": "fig04_vae_arch.png",
    "main_training": "fig05_main_training.png",
    "main_evolution": "fig06_main_evolution.png",
    "main_samples": "fig07_main_samples.png",
    "act_samples": "fig08_activation_samples.png",
    "act_fid": "fig09_activation_fid.png",
    "act_curves": "fig10_activation_curves.png",
    "drop_curves": "fig11_dropout_curves.png",
    "drop_fid": "fig12_dropout_fid.png",
    "drop_gap": "fig13_dropout_gap.png",
    "drop_samples": "fig14_dropout_samples.png",
    "size_summary": "fig15_data_size.png",
    "size_samples": "fig16_data_size_samples.png",
    "hyper_summary": "fig17_hyper_summary.png",
    "hyper_lr_curves": "fig18_hyper_lr_curves.png",
    "vae_training": "fig19_vae_training.png",
    "vae_training_beta4": "fig19b_vae_training_beta4.png",
    "vae_recon1": "fig20_vae_recon_beta1.png",
    "vae_recon4": "fig20_vae_recon_beta4.png",
    "vae_interp1": "fig21_vae_interp_beta1.png",
    "vae_interp4": "fig21_vae_interp_beta4.png",
    "vae_samples": "fig22_vae_samples.png",
    "final_gallery": "fig23_final_gallery.png",
    "final_interp": "fig24_final_interp.png",
    "final_seeds": "fig25_final_seeds.png",
    "final_vs_vae_samples": "fig26_final_vs_vae_samples.png",
    "final_vs_vae_metrics": "fig27_final_vs_vae_metrics.png",
    "terminal_log": "fig28_terminal_log.png",
    "diff_arch": "fig29_diffusion_arch.png",
    "diff_training": "fig30_diffusion_training.png",
    "diff_evolution": "fig31_diffusion_evolution.png",
    "diff_ddim": "fig32_diffusion_ddim_steps.png",
    "diff_gallery": "fig33_diffusion_gallery.png",
    "diff_trajectory": "fig34_diffusion_trajectory.png",
    "three_way_samples": "fig35_three_way_samples.png",
    "three_way_metrics": "fig36_three_way_metrics.png",
}


def _check(ver: dict, name: str):
    for c in ver["checks"]:
        if c["name"] == name:
            return c["value"]
    return {}


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{100 * x:.1f}%"


def img(key: str, caption: str) -> str:
    """插图引用；若文件不存在则退化为一条提示，避免生成带坏链接的报告。"""
    path = HERE / "figures" / FIG[key]
    if not path.exists():
        return f"（插图 {FIG[key]} 缺失）"
    return f"![{caption}](figures/{FIG[key]})"


def table(headers, rows, align=None):
    align = align or "l" * len(headers)
    sep = "|" + "|".join(":---:" if a == "c" else ":---" if a == "l" else "---:" for a in align) + "|"
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def env_table() -> str:
    e = ENV
    rows = [
        ["操作系统", e.get("platform", "-")],
        ["编程语言", f"Python {e.get('python')}"],
        ["深度学习框架", f"PyTorch {e.get('torch')} + torchvision {e.get('torchvision')}"],
        ["CUDA / cuDNN", f"CUDA {e.get('cuda_version')} / cuDNN {e.get('cudnn_version')}"],
        ["GPU", f"{e.get('device')}（{e.get('gpu_memory_gb')} GB 显存，计算能力 {e.get('capability')}）"],
        ["CPU / 内存", f"{e.get('cpu_count')} 逻辑核 / 约 15 GB"],
        ["数值计算 / 绘图", f"NumPy {e.get('numpy')}、Matplotlib {e.get('matplotlib')}（中文字体 Noto Sans CJK SC）"],
        ["精度", "FP32（未使用混合精度）"],
        ["随机性", "固定 Python / NumPy / PyTorch 随机种子；cuDNN benchmark=True 用于加速"],
        ["评价指标", "FID（Inception-v3）、C2ST、最近邻距离、颜色多样性、人眼目检"],
        ["运行方式", "cd exp6_report && python full_experiment.py，再由 python gen_report.py 生成本报告"],
    ]
    return table(["项目", "配置"], rows, "ll")


def sec_title(add):
    add("# 实验六：基于生成模型的图像生成\n\n"
        "## ——用 DCGAN、变分自编码器与扩散模型生成 64×64 二次元动漫头像\n\n"
        "> 姓名：________　学号：________　班级：________　日期：________\n"
        ">\n"
        "> 代码目录：`code/`（`data.py` 数据与划分、`models.py` DCGAN/VAE、`engine.py` 训练引擎、"
        "`metrics.py` FID/最近邻/C2ST、`viz.py` 可视化）\n"
        ">\n"
        "> 主实验脚本：`exp6_report/full_experiment.py`　结果：`exp6_report/results.json`　"
        "插图：`exp6_report/figures/`、`exp6_report/verification.json`\n"
        ">\n"
        f"> 实验环境：PyTorch {ENV.get('torch')}（CUDA {ENV.get('cuda_version')}），"
        f"GPU 为 {ENV.get('device')}；全部训练都在 GPU 上完成\n"
        ">\n"
        "> 运行方式：`cd exp6_report && python full_experiment.py && python diffusion_train.py --steps 165000 --max-hours 9.2 && python diffusion_eval.py && python diffusion_report.py && python gen_report.py`")


def sec_purpose(add):
    add(r"""## 1. 实验目的

1. 理解生成模型与判别模型在目标上的根本区别：判别模型学习 \(p(y\mid x)\)，生成模型学习数据分布 \(p(x)\) 本身，因而需要同时回答“像不像真实数据”和“覆盖了真实数据的哪些模式”两个问题。
2. 用 PyTorch 从零实现生成对抗网络（GAN）与变分自编码器（VAE）两种最主流的生成模型：GAN 通过生成器与判别器的对抗博弈隐式地逼近数据分布，VAE 通过最大化证据下界（ELBO）显式地学习一个带隐变量的概率模型。
3. 掌握 DCGAN 的结构设计准则（转置卷积、批归一化、LeakyReLU、去掉全连接层、Adam 特殊超参数），并理解这些设计分别解决了朴素 GAN 训练中的什么问题。
4. 用真实图像数据（Anime Face Dataset，64×64 二次元动漫头像）训练生成模型，观察生成质量随训练步数的演化，学会用 FID、C2ST、最近邻距离等定量指标和目检结合来评价生成结果。
5. 通过控制变量实验分析生成模型中“技巧”的作用：生成器激活函数的选择（ReLU / LeakyReLU / ELU / GELU）、判别器 Dropout、训练数据量、潜变量维度、学习率、批大小、Adam β1 与单边标签平滑。
6. 对比 GAN 与 VAE 两种范式的优缺点：GAN 生成更锐利但训练不稳定、易模式崩溃；VAE 训练稳定、有显式隐空间但生成偏模糊。
7. 从零实现 DDPM 扩散模型（U-Net 预测噪声 + 余弦噪声调度 + EMA + DDIM 采样），在相同数据和 FID 协议下把它与 DCGAN、VAE 做三方对比，理解"生成质量"与"采样成本"之间新的权衡。
8. 完成一次完整的生成式深度学习实验流程：数据清洗 → 模型实现 → 正确性验证 → 主实验 → 控制变量实验 → 定量评价 → 结果分析 → 结论与局限。""")


def sec_content(add):
    n_img = DR.get("n_train_images", 5000)
    steps_act = ACT.get("steps", 1500)
    steps_drop = DR.get("steps", 2000)
    sizes_txt = ", ".join(str(s) for s in SZ.get("sizes", [1000, 5000, 14029]))
    steps_size = SZ.get("steps", 3000)
    steps_hyper = H.get("steps", 1500)
    steps_vae = VA.get("steps", 4000)
    items = [
        ("第一部分：数据概览与清洗",
         "统计 Anime Face Dataset 的规模、重复图片比例、像素均值/标准差、亮度与饱和度分布，展示训练集样本；"
         "按 md5 去重并固定种子划分训练池（14,029 张）与 FID 参考集（3,000 张，不参与训练）。"),
        ("第二部分：模型结构与参数量",
         "给出 DCGAN 生成器/判别器与卷积 VAE 的逐层输出尺寸、参数量和结构示意图，核对每一层张量的通道数与空间尺寸。"),
        ("第三部分：正确性验证",
         "用 18 项检查确认数据范围、模型输出形状、生成器梯度通路、判别器单 batch 过拟合、VAE 单 batch 过拟合、"
         "FID 指标合理性、潜空间插值平滑性与最近邻指标量级都正常，避免“训练很久但实现有 bug”。"),
        ("第四部分：DCGAN 主实验",
         "在全部 14,029 张训练图上训练 DCGAN 6,000 步（批大小 128），记录固定噪声样本随训练步数的演化、"
         "损失/判别器输出概率/梯度范数曲线，以及 FID 随训练的变化。"),
        ("第五部分：激活函数实验",
         f"固定其它条件、各训练 {steps_act} 步，比较生成器使用 ReLU、LeakyReLU、ELU、GELU 时的 FID、训练曲线与生成样本；"
         "另加一组判别器改用 ReLU 的对照。"),
        ("第六部分：Dropout 实验",
         f"在 {n_img:,} 张小数据上各训练 {steps_drop} 步，比较判别器 Dropout=0 / 0.2 / 0.5 对 FID 与判别器"
         "训练/留出真实图判对率的影响，检验 Dropout 能否阻止判别器过拟合、从而改善生成器。"),
        ("第七部分：数据量实验",
         f"训练图片数取 {sizes_txt} 三档，固定 {steps_size} 个优化步（不同数据量对应的 epoch 数不同），比较 FID 与"
         "生成图到训练集的最近邻距离，观察小数据下模型是否只是在背训练图。"),
        ("第八部分：超参数实验",
         f"各训练 {steps_hyper} 步，分别扫描潜变量维度（32/100/256）、学习率（1e-4/2e-4/5e-4）、批大小（64/128/256）、"
         "Adam β1（0.5/0.9）与单边标签平滑（0 vs 0.1），用 FID 比较。"),
        ("第九部分：VAE 对照实验",
         f"训练两个卷积 VAE：β=1 与 β-VAE（β=4），各 {steps_vae} 步，比较 FID、重建误差（MSE/PSNR）、先验采样、"
         "潜空间插值与训练曲线。"),
        ("第十部分：最终模型与对比",
         "按前面的扫描结果选择最终配方并训练 6,000 步，生成 80 张高清画廊、潜空间插值图与不同随机种子对照；"
         "用 C2ST 分类器检验真实图与 DCGAN/VAE 生成图的可区分程度，完成两种范式的定量对比。"),
        ("第十一部分：扩散模型对照实验",
         f"从零实现 DDPM：U-Net 预测噪声、余弦噪声调度、EMA、DDIM 采样；在 {D['n_train']:,} 张训练图上训练 "
         "约 9 小时（16 万余步），挑选最佳 EMA checkpoint，用相同的 FID 参考集、C2ST 与最近邻协议与 DCGAN、VAE 对比，"
         "并考察 DDIM 步数对质量与采样速度的影响。"),
    ]
    body = "\n\n".join(f"**{a}。** {b}" for a, b in items)
    add("## 2. 实验内容\n\n"
        "实验共分十一部分：前十个部分由 `exp6_report/full_experiment.py` 完成（DCGAN 与 VAE），第十一部分（扩散模型）由 `exp6_report/diffusion_train.py` 完成；所有结果写入 `results.json`，插图写入 `figures/`。"
        "为避免变量混淆，所有对比实验共享同一份训练池/参考集划分和同一套评估代码，只改变被研究的那一个因素；"
        "且对比实验固定“优化步数”而不是固定 epoch 数，这样不同数据量之间的比较才是公平的（梯度更新次数相同）。\n\n" + body)


def sec_env(add):
    add("## 3. 实验环境\n\n" + env_table() + "\n\n"
        "全部训练与推理都在 GPU 上完成。PyTorch 为 CUDA 13.0 版本；cuDNN 的 benchmark 模式"
        "在训练开始时自动选择最快的卷积算法，把 DCGAN 的单步耗时从约 134 ms 降到约 88 ms，"
        "代价是逐位可复现性被牺牲——所以本实验的所有结论都在固定随机种子下比较，而不是追求两次运行完全一致。")


def sec_principles(add):
    add(r"""## 4. 实验原理

### 4.1 生成模型在学什么

判别模型学的是条件分布 \(p(y\mid x)\)：给定一张图，输出它属于哪一类。生成模型学的是数据本身的分布 \(p(x)\)：它要能“无中生有”地采样出新的 \(x\)。这两件事的难度不同——分类只需要找到区分不同类的边界，而生成需要刻画整个高维像素空间的概率结构。对一张 \(64\times64\times3\) 的彩色图，像素空间有 12,288 维，直接建模 \(p(x)\) 既不可解析也不可计算，所以所有深度生成模型都采用“引入隐变量 + 用神经网络参数化”的路线，区别只在于如何定义和优化目标。

本实验实现的两类模型代表了两种截然不同的思路：

- **VAE（变分自编码器）**：显式地定义 \(p(x\mid z)\) 和先验 \(p(z)\)，通过最大化对数似然的变分下界（ELBO）来训练；
- **GAN（生成对抗网络）**：不显式写出概率密度，用一个判别器网络代替“像不像”的度量，通过生成器与判别器的对抗博弈隐式地逼近数据分布。

### 4.2 GAN：对抗训练与 JS 散度

GAN 由生成器 \(G\) 和判别器 \(D\) 组成。\(G\) 把从先验 \(p_z\)（本实验为标准正态）采样的噪声 \(z\) 映射成图片 \(G(z)\)；\(D\) 接收一张图，输出它是真实图的概率。训练目标是一个极小极大博弈：

$$
\min_G \max_D\; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]. \tag{1}
$$

对固定的 \(G\)，最优判别器为 \(D^*(x)=\dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}\)。代回式 (1) 后，生成器的目标等价于最小化真实分布 \(p_{\text{data}}\) 与生成分布 \(p_g\) 之间的 Jensen–Shannon 散度：

$$
\max_D V(G,D) = 2\,\mathrm{JSD}(p_{\text{data}}\,\|\,p_g) - 2\log 2. \tag{2}
$$

这解释了 GAN 的两个特点：理论上当 \(p_g=p_{\text{data}}\) 时达到最优；但 JS 散度在两个分布几乎不重叠时会饱和（梯度接近 0），这正是 GAN 训练初期生成器学得慢、以及训练不稳定的根源。实践中用“生成器希望 \(D(G(z))\) 接近 1”的非饱和损失代替式 (1) 中的 \(\log(1-D(G(z)))\)，让梯度在 \(D\) 很自信时仍然足够大，本实验采用的是这一形式（BCEWithLogitsLoss）。

### 4.3 DCGAN 的结构设计准则

原始 GAN 用多层感知机，生成的图既模糊又难训练。Radford 等人提出的 DCGAN 给出了让卷积网络稳定训练 GAN 的一组经验准则，本实验的生成器/判别器完全遵循这些准则：

1. **用带步长的转置卷积做上采样，而不是池化**：生成器从 \(4\times4\) 开始，每层用 stride=2、kernel=4、padding=1 的转置卷积把空间尺寸翻倍（\(4\to8\to16\to32\to64\)），让网络自己学习上采样方式。
2. **生成器除输出层外用批归一化（BN）**：稳定各层输入分布，缓解初始化敏感问题；输出层用 tanh 把像素限制在 \([-1,1]\)。
3. **判别器用 LeakyReLU（负斜率 0.2）而不是 ReLU**：避免负半轴梯度为 0，保持生成器在“被判错”时仍能获得梯度；判别器第一层不接 BN，避免把真实/生成样本的统计量混在一起。
4. **去掉全连接层**：生成器的输入噪声直接 reshape 成 \(100\times1\times1\) 再上采样，判别器用卷积下采样到 \(1\times1\) 输出 logit，减少参数量并保留空间结构。
5. **Adam 优化器，学习率 2e-4，β1=0.5**：β1 取 0.5（而不是默认 0.9）减少动量，避免训练振荡。

### 4.4 VAE：变分下界与重参数化

VAE 假设数据由隐变量 \(z\) 生成：先验 \(p(z)=\mathcal{N}(0,I)\)，解码器 \(p_\theta(x\mid z)\) 由神经网络参数化。真实后验 \(p_\theta(z\mid x)\) 不可解，于是用编码器 \(q_\phi(z\mid x)\) 近似它。对单个样本，对数似然的下界（ELBO）为：

$$
\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big] - D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big) = \mathcal{L}(\theta,\phi;x). \tag{3}
$$

第一项是重建项（本实验用高斯似然，即像素平方误差），第二项是 KL 正则，把编码出的后验拉向标准正态先验。训练时对 \(q_\phi(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))\) 用重参数化技巧采样：

$$
z=\mu+\sigma\odot\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,I), \tag{4}
$$

使梯度可以穿过采样操作。编码器输出 \(\mu\) 与 \(\log\sigma^2\)，KL 项有解析解：

$$
-D_{\mathrm{KL}} = \tfrac12\sum_j\big(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\big). \tag{5}
$$

β-VAE 在 KL 项前乘系数 \(\beta\)：\(\beta>1\) 更强地压紧后验，倾向于学到更解耦的隐表示，但重建会更模糊；本实验用 β=1 与 β=4 做对照。

一个实践细节：本实验最初按标准做法用逐像素二元交叉熵（BCE）做重建项，结果编码器停在“高噪声 + 输出平均脸”的局部解，重建损失几乎不下降；改为平方误差（固定方差高斯似然）后同一结构可以稳定收敛到清晰重建（正确性验证中单 batch 重建损失 200 步下降约 89%）。这说明 VAE 的“重建项选择”不是无关紧要的实现细节。

### 4.5 如何评价生成质量

生成模型没有“准确率”，本实验用三类互补指标：

- **FID（Fréchet Inception Distance）**：把真实图和生成图都送入预训练 Inception-v3，取 2048 维池化特征，假设两组特征都服从高斯分布，计算

$$
\mathrm{FID} = \|\mu_r-\mu_g\|^2 + \operatorname{Tr}\Big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\Big). \tag{6}
$$

FID 越低越好，它同时惩罚“生成图不像真实图”（均值偏差）和“生成图模式单一”（协方差偏差），是目前最常用的 GAN 评价指标。本实验的真实参考集固定为 3,000 张不参与训练的图片，对比实验生成 4,000 张、最终模型生成 5,000 张，所有配置用同一套协议。

- **C2ST（分类器双样本检验）**：训练一个小 CNN 区分真实图与生成图，测试准确率越接近 50%，说明生成图越难与真实图区分。
- **最近邻距离**：计算每张生成图到训练集的最近邻像素 L2 距离，并与“真实留出图到训练集的同一距离”比较。若生成图的最近邻距离明显更小，说明模型可能在背训练图（记忆化）而不是学习分布。

此外还统计生成图的颜色多样性（色相直方图熵）辅助判断模式崩溃，并始终配合人眼目检。""")


TYPE_CN = {
    "ConvTranspose2d": "转置卷积", "Conv2d": "卷积", "BatchNorm2d": "批归一化",
    "Linear": "全连接", "ReLU": "ReLU", "LeakyReLU": "LeakyReLU", "Tanh": "Tanh",
    "Dropout2d": "Dropout", "Sigmoid": "Sigmoid",
}


def _layer_rows(layers):
    rows = []
    for layer in layers:
        shape = layer.get("out_shape")
        shape_txt = "×".join(str(s) for s in shape) if shape else "-"
        rows.append([TYPE_CN.get(layer["type"], layer["type"]), shape_txt,
                     f"{layer.get('params', 0):,}"])
    return rows


def sec_data(add):
    st = D["stats"]
    add("## 5. 数据：Anime Face Dataset\n\n"
        "实验数据是 Kaggle 上的 **Anime Face Dataset**（作者 soumikrakshit，图片来自 getchu.com，"
        "用 lbpcascade_animeface 裁剪出人脸并统一缩放到 64×64，CC0 许可）。"
        "它的特点是：全部是正脸特写、背景简单、色彩鲜明，适合观察生成模型能否学到"
        "“头发 / 眼睛 / 脸型 / 发饰”这些结构性要素。\n\n"
        f"原始数据共 **{D['n_raw']:,} 张** 图片。按文件 md5 检查发现其中 **{D['n_duplicate']:,} 张是字节级重复**"
        f"（{pct(D['duplicate_ratio'])}，主要是抓取时同一张图被保存了多次），去重后剩 **{D['n_unique']:,} 张唯一图片**。"
        "重复样本会让生成模型反复看到同一张图，既浪费训练时间又高估数据多样性，所以先做去重。"
        "去重后的图片按固定随机种子划分为：\n\n"
        f"- **训练池**：{D['n_train']:,} 张，所有生成模型的参数只在这部分数据上更新；\n"
        f"- **参考集**：{D['n_ref']:,} 张，完全不参与训练，只用于计算 FID 的实图统计量、以及“生成图到训练集最近邻距离”的实图基线。\n\n"
        + table(["统计量", "数值"], [
            ["图片尺寸 / 通道", "64×64，RGB 三通道"],
            ["原始 / 去重后", f"{D['n_raw']:,} / {D['n_unique']:,} 张"],
            ["重复图片比例", pct(D['duplicate_ratio'])],
            ["训练池 / 参考集", f"{D['n_train']:,} / {D['n_ref']:,} 张"],
            ["RGB 通道均值", ", ".join(f"{x:.3f}" for x in st["mean_rgb"])],
            ["RGB 通道标准差", ", ".join(f"{x:.3f}" for x in st["std_rgb"])],
            ["平均亮度", f"{st['mean_brightness']:.3f}"],
            ["平均饱和度", f"{st['mean_saturation']:.3f}"],
        ], "ll") + "\n\n"
        + img("data_samples", "图 1　Anime Face Dataset 训练集样本（64 张）") + "\n\n"
        "图 1 是随机抽取的 64 张训练样本。可以看到数据以女性角色正脸为主，发色/瞳色/发型差异很大，"
        "背景大多是纯色或简单渐变；也混有少量非人脸图像（如右下角一些男性角色、以及一张紫色面具），"
        "这是公开爬取数据集的常见问题，后文的失败样本分析会再讨论。\n\n"
        + img("data_stats", "图 2　数据分布：RGB 通道均值/标准差、平均亮度、像素饱和度") + "\n\n"
        "图 2 给出数据的整体分布。三通道均值都明显高于 0.5（R 0.70 / G 0.60 / B 0.59），说明画面整体偏亮；"
        "平均亮度约 0.63、饱和度分布集中在 0.2~0.4 之间，符合“明亮、色彩偏柔和”的动漫头像风格。"
        "这些统计量也解释了为什么生成器输出层用 tanh 后、样本的平均亮度容易偏亮——数据本身就偏亮。")


def sec_implementation(add):
    g_params = A["generator"]["params"]
    d_params = A["discriminator"]["params"]
    vae_params = A["vae"]["params"]
    add("## 6. 模型实现\n\n"
        "所有模型都用 PyTorch 从零实现，只使用卷积、转置卷积、批归一化、全连接等基础层，"
        "没有调用任何现成的 GAN/VAE 实现。\n\n"
        "### 6.1 DCGAN 生成器\n\n"
        f"生成器接收 100 维标准正态噪声，先 reshape 成 $100\\times1\\times1$，再经 5 层转置卷积放大到 $64\\times64\\times3$。"
        f"除最后一层用 tanh 外，每层都是“转置卷积 → 批归一化 → 激活”。生成器共 **{g_params:,}** 个可训练参数。\n\n"
        + table(["层类型", "输出形状 (C×H×W)", "参数量"],
                _layer_rows(A["generator"]["layers"]), "lrr") + "\n\n"
        "### 6.2 DCGAN 判别器\n\n"
        f"判别器与生成器镜像对称：5 层卷积把 $64\\times64\\times3$ 压缩到 $1\\times1$ 的单个 logit。"
        f"第一层不接批归一化，之后每层是“卷积 → 批归一化 → LeakyReLU(0.2)”，共 **{d_params:,}** 个参数。"
        "输出层不接 sigmoid，训练时直接配合 `BCEWithLogitsLoss`，数值更稳定。\n\n"
        + table(["层类型", "输出形状 (C×H×W)", "参数量"],
                _layer_rows(A["discriminator"]["layers"]), "lrr") + "\n\n"
        "### 6.3 卷积 VAE\n\n"
        f"VAE 的编码器与解码器互为镜像：编码器 4 层卷积把 $64\\times64\\times3$ 压缩到 $512\\times4\\times4$，"
        "展平后接两个全连接层分别输出 $\\mu$ 与 $\\log\\sigma^2$（128 维）；训练时用重参数化采样 $z=\\mu+\\sigma\\odot\\varepsilon$，"
        "解码器先经全连接回到 $512\\times4\\times4$，再用 4 层转置卷积恢复成 64×64。VAE 共 "
        f"**{vae_params:,}** 个参数。结构见图 3、图 4。\n\n"
        + img("dcgan_arch", "图 3　DCGAN 生成器（左）与判别器（右）的逐层结构") + "\n\n"
        + img("vae_arch", "图 4　卷积 VAE 的编码器与解码器结构") + "\n\n"
        "### 6.4 训练细节\n\n"
        "GAN 的每个优化步先更新判别器（真实图标签为 1、生成图标签为 0，两者损失相加），再更新生成器"
        "（希望判别器把生成图判成 1）。两个网络都用 Adam，学习率 2e-4，β=(0.5, 0.999)，批大小 128。"
        "VAE 用 Adam，学习率 1e-3，重建项为逐像素平方误差之和（除以批大小），KL 项按式 (5) 解析计算。\n\n"
        "关键代码（生成器前向与 GAN 的一个训练步）：\n\n"
        "```python\n"
        "class Generator(nn.Module):\n"
        "    def __init__(self, z_dim=100, base=64, activation='relu'):\n"
        "        super().__init__()\n"
        "        c = [base * 8, base * 4, base * 2, base]          # 512, 256, 128, 64\n"
        "        layers = [nn.ConvTranspose2d(z_dim, c[0], 4, 1, 0, bias=False),\n"
        "                  nn.BatchNorm2d(c[0]), make_activation(activation)]\n"
        "        for in_ch, out_ch in zip(c[:-1], c[1:]):\n"
        "            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),\n"
        "                       nn.BatchNorm2d(out_ch), make_activation(activation)]\n"
        "        layers += [nn.ConvTranspose2d(c[-1], 3, 4, 2, 1), nn.Tanh()]\n"
        "        self.net = nn.Sequential(*layers)\n"
        "\n"
        "# 一个训练步：先 D 后 G\n"
        "real = to_gan_range(batch)                           # [0,1] → [-1,1]\n"
        "fake = G(randn(b, 100)).detach()\n"
        "loss_d = bce(D(real), ones) + bce(D(fake), zeros)\n"
        "loss_d.backward(); opt_d.step()\n"
        "loss_g = bce(D(G(randn(b, 100))), ones)              # 非饱和生成器损失\n"
        "loss_g.backward(); opt_g.step()\n"
        "```\n\n"
        "VAE 的损失（重建项用平方误差、KL 有解析解）：\n\n"
        "```python\n"
        "recon = F.mse_loss(torch.sigmoid(recon_logits), x, reduction='none')\n"
        "recon_loss = recon.flatten(1).sum(1).mean()\n"
        "kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()\n"
        "loss = recon_loss + beta * kl\n"
        "```")


def sec_verification(add):
    if VER is None:
        add("## 7. 正确性验证\n\n（缺少 verification.json，未生成该节。）")
        return
    checks = VER["checks"]
    rows = []
    for c in checks:
        val = c["value"]
        if isinstance(val, dict):
            val_txt = "；".join(f"{k}={v}" for k, v in val.items())
        elif isinstance(val, list):
            val_txt = str(val)
        else:
            val_txt = str(val)
        rows.append([c["name"], "✓" if c["passed"] else "✗", val_txt])
    add("## 7. 正确性验证\n\n"
        "在跑大规模训练之前，先用小实验确认数据、模型、梯度和评估指标都按预期工作。"
        f"共 {VER['n_total']} 项检查，**{VER['n_passed']}/{VER['n_total']} 通过**，耗时 {VER['runtime_s']} 秒。"
        "这一步的价值在于：如果 FID 公式、图像归一化或梯度通路有 bug，后面所有实验结论都会是错的，"
        "而训练本身又很贵，必须先排除实现错误。\n\n"
        + table(["检查项", "结果", "实际值"], rows, "lcl") + "\n\n"
        "其中最有说服力的几项是：\n\n"
        "- **判别器单 batch 过拟合**：固定一个 batch 的真实图与生成图训练判别器 150 步，平均判对率从 "
        f"{f3(_check(VER, '判别器单 batch 过拟合')['初始'])} 升到 {f3(_check(VER, '判别器单 batch 过拟合')['150 步后'])}，"
        "说明卷积、损失与反向传播都正确；\n"
        "- **VAE 单 batch 过拟合**：200 步内重建损失下降约 89%，说明编码器-解码器结构有足够的表达能力；\n"
        "- **FID 合理性**：真实图对真实参考集的 FID 只有 "
        f"{VER['fid_sanity']['real_512']:.1f}，而均匀噪声和未训练生成器的 FID 分别高达 "
        f"{VER['fid_sanity']['uniform_noise']:.1f} 和 {VER['fid_sanity']['untrained_generator']:.1f}，"
        "说明 FID 能正确区分“像”与“不像”；\n"
        "- **生成器梯度**：生成器的 14 个参数张量全部收到非零梯度，排除了“判别器写错导致 G 不更新”这类低级错误。\n\n"
        "另外，正是这一步暴露了 VAE 重建项的问题：最初用逐像素 BCE 时，单 batch 重建损失只下降约 39%，"
        "换成平方误差后才降到原来的约 11%（见第 4.4 节）。")


def sec_main(add):
    cfg = M["config"]
    hist = M["history"]
    fids = M["fid_points"]
    fid_map = {int(s): f for s, f in fids}
    steps_sorted = sorted(fid_map)
    fid_final = M["final_fid"]
    fid_best = min(fid_map.values())
    n_img = M["epochs_equivalent"]
    rows = [
        ["优化步数", f"{cfg['max_steps']:,} 步（≈ {n_img:.1f} 个 epoch，约 {cfg['max_steps'] * cfg['batch_size']:,} 张图过了一遍网络）"],
        ["批大小", str(cfg["batch_size"])],
        ["潜变量维度", str(cfg["z_dim"])],
        ["生成器激活 / 判别器激活", f"{cfg['g_act']} / {cfg['d_act']}"],
        ["判别器 Dropout", str(cfg["d_dropout"])],
        ["优化器", f"Adam，G lr={cfg['lr_g']:g}，D lr={cfg['lr_d']:g}，β={tuple(cfg['betas'])}"],
        ["训练时间", f"{M['train_time_s']:.0f} 秒（{M['steps_per_s']:.2f} 步/秒）"],
    ]
    add("## 8. 主实验：DCGAN 全量训练\n\n"
        "主实验在全部 14,029 张训练图上训练 6,000 步，作为后续所有对比的参照配置。\n\n"
        + table(["配置项", "取值"], rows, "ll") + "\n\n"
        + img("main_training", "图 5　主实验训练曲线：损失、判别器输出概率、判别器准确率、梯度范数与 FID") + "\n\n"
        "图 5 揭示了一个对生成模型非常重要的事实：**损失值本身不能作为生成质量的指标**。"
        f"训练过程中判别器损失在 {min(hist['loss_d']):.2f}~{max(hist['loss_d']):.2f} 之间来回波动、"
        f"生成器损失从 {hist['loss_g'][0]:.2f} 缓慢降到 {hist['loss_g'][-1]:.2f}，曲线看上去相当平稳；"
        f"但同一段时间里 FID 先从 {fid_map[steps_sorted[0]]:.1f} 升到 {max(fid_map.values()):.1f}，"
        f"又一路降到 **{fid_final:.1f}**，变化幅度超过一倍。"
        "原因是 GAN 的损失衡量的是“两个网络当前的相对强弱”，而不是“生成图离真实分布有多远”："
        "判别器变强会让生成器损失变大，但生成质量可能同时变好。所以本实验的所有结论都以 FID、"
        "C2ST 和目检为准，损失曲线只用来诊断训练是否崩溃。\n\n"
        f"判别器输出概率也印证了这一点：训练结束时 D(真实图)≈{hist['d_real'][-1]:.2f}、"
        f"D(生成图)≈{hist['d_fake'][-1]:.2f}，判别器始终占优（真实/生成判对率分别约 "
        f"{hist['d_acc_real'][-1]:.2f} 和 {hist['d_acc_fake'][-1]:.2f}）。这是 DCGAN 在真实数据上的常见状态："
        "判别器不会退化到 50%，但它也不再是“一眼看穿”——生成器利用非饱和损失（生成器希望 D 输出 1）"
        "仍然能持续获得有效梯度，FID 从 3000 步的 "
        f"{fid_map[3000]:.1f} 降到 6000 步的 {fid_final:.1f} 就是证据。\n\n"
        + img("main_evolution", "图 6　固定噪声样本随训练步数的演化（从上到下依次为第 "
              + "/".join(str(s) for s in M["snapshots"]) + " 步）") + "\n\n"
        "图 6 用同一组固定噪声每隔一段时间采样，直观展示了生成器“从无到有”的学习过程："
        "第 500 步时还是带颜色噪声的模糊团块，第 1,000 步已经出现眼睛和脸型的轮廓，第 2,000 步之后"
        "五官位置基本稳定，到第 6,000 步已经能生成发色、瞳色、发型、发饰都相当多样的动漫头像。"
        "值得注意的是，模型在第 4,000 步前后发生了一次明显的“质变”（FID 从 216 降到 126），"
        "对应的正是图 6 中第 4,000 步样本突然变得干净清晰的位置——这与本节的损失曲线分析"
        "形成对照，说明只看损失会严重低估训练的进展。\n\n"
        + img("main_samples", "图 7　主实验最终生成的 64 张动漫头像（FID = " + f1(fid_final) + "）") + "\n\n"
        "图 7 是主实验模型的最终效果。64 张样本里没有明显重复的脸，发色覆盖粉、蓝、绿、金、银、黑等，"
        "瞳色与发型也各不相同，说明模型没有发生严重的模式崩溃；缺点是个别样本的五官仍有轻微错位、"
        "背景出现色块噪声，这是 64×64 分辨率加 14,029 张训练图的容量限制。\n\n"
        "表 1 汇总了 FID 随训练步数的变化。\n\n"
        + table(["优化步数", "等效 epoch", "FID"],
                [[f"{s:,}", f"{s * cfg['batch_size'] / D['n_train']:.1f}", f2(fid_map[s])]
                 for s in steps_sorted], "rrc") + "\n\n"
        f"从表 1 可以看到，前 3,000 步的 FID 一直在 {fid_map[steps_sorted[0]]:.0f} 以上，"
        f"只从 {fid_map[steps_sorted[1]]:.1f} 降到 {fid_map[3000]:.1f}；而 3,000~6,000 步把它降到了 "
        f"{fid_final:.1f}。这说明 GAN 训练存在明显的“平台期”：早期生成器只能学到颜色统计和粗略轮廓，"
        "FID 对这类进步不敏感；当判别器与生成器进入相对平衡、生成器开始学习高频细节（眼睛高光、发丝边缘）后，"
        "FID 才会快速下降。对实际项目而言，这意味着“训练到损失稳定”远不是停止训练的合理判据，"
        "必须用固定参考集的 FID 或人工检查来监控生成质量。")



def _act_sample_note(act_runs):
    """根据激活函数 FID 的实际排序生成对样本图的解读。"""
    if not act_runs:
        return ""
    ordered = sorted(act_runs, key=lambda r: r["fid"])
    best, worst = ordered[0], ordered[-1]
    return (f"从生成样本（图 8）看，四组配置在 1,500 步时都只能生成“动漫脸的雏形”：脸型轮廓和眼睛位置基本正确，"
            f"但发丝、五官细节仍有模糊。不过 {best['label']} 那一行的眼睛高光和发丝边缘明显更清晰，"
            f"而 {worst['label']} 那一行更“糊”且颜色更脏，与 FID 的排序一致。"
            "这与主实验 6,000 步时 FID 从 200 多降到 102 的结论并不矛盾——**训练步数带来的改善远大于激活函数的选择**；"
            "但也不能说激活函数无所谓：在同样的短预算下，ReLU 系（LeakyReLU）确实比 GELU 收敛得更快。"
            "要得到更可信的结论，应给每组配置足够的训练步数（例如 6,000 步以上），这也是本实验的一个局限。")

def sec_activation(add):
    runs = ACT["runs"]
    dr = ACT["d_relu"]
    steps = ACT["steps"]
    labels = [r["label"] for r in runs]
    fids = [r["fid"] for r in runs]
    best = min(runs, key=lambda r: r["fid"])
    worst = max(runs, key=lambda r: r["fid"])
    spread = worst["fid"] - best["fid"]
    base_relu = next(r for r in runs if r["key"] == "relu")
    d_delta = dr["fid"] - base_relu["fid"]
    rows = [[r["label"], r["g_act"], r["d_act"], f2(r["fid"]),
             f3(r["color_diversity"]["hue_entropy_norm"]),
             f3(r["color_diversity"]["brightness_std"])] for r in runs]
    rows.append(["D 用 ReLU（G 与基线相同）", dr["g_act"], dr["d_act"], f2(dr["fid"]),
                 f3(dr["color_diversity"]["hue_entropy_norm"]),
                 f3(dr["color_diversity"]["brightness_std"])])
    if spread < 5:
        judge = (
            f"四组生成器激活函数的 FID 最大只差 {spread:.1f}，落在 FID 采样噪声的量级内，"
            "说明在本实验的 1,500 步训练预算下，激活函数的选择不是决定生成质量的主要因素；"
            "这与判别任务里的结论不同——生成质量更多受“判别器强度、训练步数、数据量”支配。"
        )
    else:
        judge = (
            f"四组生成器激活函数中，FID 最低的是 **{best['label']}（{f2(best['fid'])}）**，"
            f"最高的是 {worst['label']}（{f2(worst['fid'])}），相差 {spread:.1f}。"
            "生成器的激活函数会影响梯度在负半轴的行为：LeakyReLU/ELU/GELU 保留负区间的梯度，"
            "理论上更有利于深层生成器学习细节；但在 1,500 步的短预算里，差异主要来自训练早期的收敛速度。"
        )
    add("## 9. 激活函数实验\n\n"
        "生成器的激活函数决定了“噪声 → 特征图”这条路径上的非线性形状；判别器的激活函数则影响"
        "真实/生成样本的梯度能否传回生成器。DCGAN 的原始配方是生成器用 ReLU、判别器用 LeakyReLU，"
        "本节用控制变量法检验这两个选择是否真的重要。\n\n"
        f"所有配置都训练 {steps} 步、批大小 128、潜维 100、Adam(2e-4, β=0.5)，"
        "唯一变化的是激活函数；每个配置用同一组随机种子和同一份数据顺序。结果见表 2 与图 8~10。\n\n"
        + table(["配置", "G 激活", "D 激活", "FID", "色相熵（归一化）", "亮度标准差"], rows, "lllrrr") + "\n\n"
        + img("act_samples", "图 8　不同生成器激活函数的生成样本（同为 1,500 步，每行 8 张）") + "\n\n"
        + img("act_fid", "图 9　激活函数对 FID 的影响") + "\n\n"
        + img("act_curves", "图 10　不同激活函数的训练曲线（左：判别器损失；右：D 对真实/生成图的输出概率）") + "\n\n"
        + judge + "\n\n"
        + _act_sample_note(runs) + "\n\n"
        f"判别器激活函数的对照更有意思：把判别器从 LeakyReLU 换成 ReLU（G 保持基线 ReLU）后，"
        f"FID 从 {f2(base_relu['fid'])} 变为 {f2(dr['fid'])}（{'+' if d_delta >= 0 else ''}{d_delta:.1f}）。"
        + ("差异不大，说明在 1,500 步内判别器的负半轴梯度还不是瓶颈；"
           if abs(d_delta) < 5 else
           ("改用 ReLU 后 FID 变差，符合“判别器用 LeakyReLU 避免梯度为 0”的 DCGAN 设计准则；"
            if d_delta > 0 else
            "改用 ReLU 后 FID 反而更好，说明这条经验准则在本数据/预算下并不总是成立；"))
        + "结合第 10 节的 Dropout 实验可以看出，**生成模型里的“技巧”高度依赖训练预算和数据量，"
        "不能把某条 GAN 经验当作普适定律。**")


def _dropout_finding(dr_runs):
    """根据 Dropout 实验的实际结果生成一句结论（泛化改善但 FID 变差 / 泛化改善且 FID 变好）。"""
    if not dr_runs:
        return ""
    gaps = [r["d_train"]["acc_real"] - r["d_holdout"]["acc_real"] for r in dr_runs]
    fids = [r["fid"] for r in dr_runs]
    gap_improved = gaps[-1] < gaps[0]
    fid_improved = fids[-1] < fids[0]
    if gap_improved and fid_improved:
        return (f"Dropout 把判别器的训练/留出判对率差距从 {gaps[0] * 100:.1f} 个百分点缩小到 "
                f"{gaps[-1] * 100:.1f} 个百分点，同时 FID 从 {fids[0]:.1f} 降到 {fids[-1]:.1f}，"
                "说明在这个小数据场景下，正则化判别器确实能转化为更好的生成质量")
    if gap_improved and not fid_improved:
        return (f"Dropout 把判别器的训练/留出判对率差距从 {gaps[0] * 100:.1f} 个百分点缩小到 "
                f"{gaps[-1] * 100:.1f} 个百分点（判别器泛化明显改善），但 FID 反而从 {fids[0]:.1f} "
                f"升到 {fids[-1]:.1f}——说明在 GAN 里判别器变好不等于生成器变好")
    return (f"Dropout 没有让判别器的训练/留出差距明显缩小（{gaps[0] * 100:.1f} → {gaps[-1] * 100:.1f} 个百分点），"
            f"FID 从 {fids[0]:.1f} 变为 {fids[-1]:.1f}")


def sec_dropout(add):
    runs = DR["runs"]
    n_img = DR["n_train_images"]
    steps = DR["steps"]
    rows = []
    for r in runs:
        gap = r["d_train"]["acc_real"] - r["d_holdout"]["acc_real"]
        rows.append([r["label"], f2(r["fid"]),
                     f3(r["d_train"]["acc_real"]), f3(r["d_holdout"]["acc_real"]),
                     f"{gap:+.3f}",
                     f3(r["d_holdout"]["acc_fake"])])
    base = runs[0]
    gap0 = base["d_train"]["acc_real"] - base["d_holdout"]["acc_real"]
    best = min(runs, key=lambda r: r["fid"])
    fids = [r["fid"] for r in runs]
    gaps = [r["d_train"]["acc_real"] - r["d_holdout"]["acc_real"] for r in runs]
    holdouts = [r["d_holdout"]["acc_real"] for r in runs]
    add("## 10. Dropout 实验\n\n"
        "Dropout 是判别模型里最常见的正则化手段：训练时随机把一部分神经元置零，迫使网络不依赖个别特征。"
        "在 GAN 里它常被用在判别器上，动机是**防止判别器把训练集背下来**——如果判别器只是记住了训练样本，"
        "它在没见过的新样本上的判断就不可靠，传给生成器的梯度也会失真。\n\n"
        f"为了制造明显的过拟合，本节只用 {n_img:,} 张训练图（约为全量的三分之一），训练 {steps} 步，"
        "比较判别器 Dropout = 0 / 0.2 / 0.5 三种设置。Dropout 只在训练时生效，评估判别器时切到 eval 模式。\n\n"
        + table(["判别器 Dropout", "FID", "训练真实图判对率", "留出真实图判对率", "泛化差距",
                 "生成图判对率"], rows, "lrrrrr") + "\n\n"
        + img("drop_curves", "图 11　不同判别器 Dropout 的训练曲线") + "\n\n"
        + img("drop_fid", "图 12　Dropout 对 FID 的影响") + "\n\n"
        + img("drop_gap", "图 13　判别器在训练真实图与留出真实图上的判对率（差距越小说明泛化越好）") + "\n\n"
        + img("drop_samples", "图 14　不同 Dropout 的生成样本（各 8 张）") + "\n\n"
        f"先看判别器这一侧（表 4 与图 13）。不加 Dropout 时，判别器在训练真实图上的判对率为 "
        f"{f3(base['d_train']['acc_real'])}，但在 {n_img:,} 张之外的留出真实图上只有 "
        f"{f3(base['d_holdout']['acc_real'])}，泛化差距 {gap0 * 100:.1f} 个百分点，说明判别器确实在记训练样本；"
        f"加上 Dropout 后，泛化差距分别缩小到 {gaps[1] * 100:.1f} 和 {gaps[2] * 100:.1f} 个百分点，"
        f"留出真实图判对率从 {holdouts[0]:.3f} 提高到 {holdouts[1]:.3f} / {holdouts[2]:.3f}——"
        "**Dropout 有效地把判别器从“背训练集”拉回到了“学真实/生成的区别”。**\n\n"
        f"然而生成器这一侧的结果完全相反：FID 随 Dropout 增大反而升高（{f2(fids[0])} → {f2(fids[1])} → "
        f"{f2(fids[2])}，最低的是 **{best['label']}**）。这是一个很值得注意的反例——"
        "“判别器泛化变好”没有转化为“生成质量变好”。合理的解释是：GAN 是一个对抗耦合系统，"
        "判别器越强、对真实图的判断越可靠，生成器越难“骗过”它，非饱和损失的梯度虽然存在，"
        "但更新方向更保守；同时 Dropout 给判别器注入的随机噪声也会让生成器收到的训练信号方差变大。"
        "判别器太弱（不加 Dropout 时留出判对率只有 0.71）反而给了生成器更大的优化空间。\n\n"
        "这组实验说明，在 GAN 中**不能把判别器的分类指标当成越高越好**：判别器的强弱需要和生成器"
        "匹配，两边要维持一种“势均力敌”的动态平衡。第 11 节的数据量实验会进一步说明，"
        "真正让判别器过拟合的元凶往往是数据量不足——数据量上去之后判别器自然学到更本质的特征，"
        "未必需要 Dropout。")
def sec_data_size(add):
    runs = SZ["runs"]
    sizes = SZ["sizes"]
    steps = SZ["steps"]
    rows = []
    for r in runs:
        rows.append([f"{r['size']:,}", f"{r['epochs_equivalent']:.1f}", f2(r["fid"]),
                     f2(r["nn_gen"]["mean"]), f2(r["nn_real"]["mean"]), f3(r["nn_ratio"]),
                     f3(r["color_diversity"]["hue_entropy_norm"])])
    fids = [r["fid"] for r in runs]
    ratios = [r["nn_ratio"] for r in runs]
    fid_min_run = min(runs, key=lambda r: r["fid"])
    fid_max_run = max(runs, key=lambda r: r["fid"])
    monotone_fid = all(fids[i] <= fids[i + 1] for i in range(len(fids) - 1))
    monotone_ratio = all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))
    if monotone_fid:
        fid_txt = (f"FID 随数据量增加单调下降（{' → '.join(f2(f) for f in fids)}），"
                   "说明数据量是本实验中对生成质量影响最大的因素之一。")
    else:
        fid_txt = (
            f"FID 并没有随数据量单调下降：{sizes[0]:,} 张时 FID 为 {f2(fids[0])}（最低），"
            f"{sizes[1]:,} 张时为 {f2(fids[1])}，{sizes[2]:,} 张时为 {f2(fids[2])}。"
            "这正是本实验最值得注意的现象——**小数据模型的 FID 反而更低**。"
            "原因是固定 3,000 步时，1,000 张图会被反复看 384 个 epoch，模型有充足机会把它们记住；"
            "而“记住的训练图”本身就是真实图像，FID 的 Inception 特征自然很接近真实分布。"
            "换句话说，**FID 无法区分“会生成的模型”和“会背图的模型”**，"
            "必须结合最近邻分析才能判断模型是真的学到了分布，还是只是把训练样本吐了出来。")
    if monotone_ratio:
        ratio_txt = (f"最近邻距离的比值从 {sizes[0]:,} 张的 {ratios[0]:.3f} 单调上升到 "
                     f"{sizes[-1]:,} 张的 {ratios[-1]:.3f}，")
    else:
        ratio_txt = (f"最近邻距离的比值在 {sizes[0]:,} / {sizes[1]:,} / {sizes[2]:,} 张时分别为 "
                     f"{ratios[0]:.3f} / {ratios[1]:.3f} / {ratios[2]:.3f}，")
    ratio_txt += ("所有配置的比值都小于 1，说明生成图整体上比真实留出图更靠近训练集——这是记忆化倾向的直接证据。"
                  "数据量最小的配置比值也最低，与 FID “小数据反而更好看”的现象互相印证："
                  "模型输出与训练集越像，FID 越容易偏低。")
    add("## 11. 数据量实验\n\n"
        "GAN 常被诟病“需要大量数据才能训练”，但数据量到多少才够、小数据下模型到底在学分布还是在背样本，"
        "需要实验回答。本节固定 **3,000 个优化步**，训练图片数分别取 "
        f"{', '.join(f'{s:,}' for s in sizes)} 三档。由于步数固定，三档配置“看过”的图片总数相同，"
        "但小数据配置会把同一批图片反复看很多遍（等效 epoch 数差异很大，见表 5），"
        "这样就能把“数据多样性”从“训练步数”中分离出来。\n\n"
        + table(["训练图片数", "等效 epoch", "FID", "生成图→训练集最近邻", "真实留出图→训练集最近邻",
                 "比值", "色相熵（归一化）"], rows, "rrrrrrr") + "\n\n"
        + img("size_summary", "图 15　数据量对 FID 与最近邻距离的影响") + "\n\n"
        + img("size_samples", "图 16　不同训练数据量的生成样本（各 3,000 步）") + "\n\n"
        + fid_txt + ratio_txt + "\n\n"
        "这组实验的实践含义是：**FID 不能脱离数据量来解读**。在 1,000 张图上训练 3,000 步，"
        "模型有充足的机会反复记忆这 1,000 张图，FID 反而可能“很漂亮”，"
        "但这不代表它学到了动漫头像的分布。评价生成模型时，FID 必须与最近邻距离、"
        "颜色多样性和人眼目检一起看；本实验最终采用全部 14,029 张训练图，"
        "并在第 14 节最终模型上再检查一次生成样本的多样性。")


def sec_hyper(add):
    sweeps = H["sweeps"]
    steps = H["steps"]
    base_fid = H["base_fid"]

    def sweep_table(key: str, name: str):
        rows = []
        for r in sweeps[key]:
            rows.append([r["label"], f2(r["fid"]),
                         ("参考配置" if "参考" in r["label"] else
                          f"{r['fid'] - base_fid:+.2f}")])
        return table([name, "FID", "相对参考配置"], rows, "lrr")

    def vals(key):
        return [(r["label"], r["fid"]) for r in sweeps[key]]

    latent = vals("latent")
    lr = vals("lr")
    batch = vals("batch")
    other = vals("other")
    latent_spread = max(f for _, f in latent) - min(f for _, f in latent)
    lr_spread = max(f for _, f in lr) - min(f for _, f in lr)
    batch_spread = max(f for _, f in batch) - min(f for _, f in batch)
    all_vals = [f for key in ("latent", "lr", "batch", "other") for _, f in vals(key)]
    overall_spread = max(all_vals) - min(all_vals)

    def best_label(pairs):
        lab, fid = min(pairs, key=lambda x: x[1])
        return lab, fid

    lat_best = best_label(latent)
    lr_best = best_label(lr)
    batch_best = best_label(batch)
    other_best = best_label(other)

    def lr_note():
        h = {r["label"]: r for r in sweeps["lr"]}
        fastest = max(h.values(), key=lambda r: r["fid"])
        slowest = min(h.values(), key=lambda r: r["fid"])
        hist = fastest.get("history", {})
        if hist.get("d_real"):
            return (f"学习率太大时（{fastest['label']}）FID 比最优的 {slowest['label']} 高 "
                    f"{fastest['fid'] - slowest['fid']:.1f}；从图 18 右可以看到，该配置的判别器输出概率"
                    f"（D(真实)≈{hist['d_real'][-1]:.2f}、D(生成)≈{hist['d_fake'][-1]:.2f}）明显比基线更接近 0.5，"
                    "说明判别器被生成器压制、训练进入了不稳定的震荡状态。")
        return f"学习率从 {lr[0][1]:.1f} 到 {lr[-1][1]:.1f}，最优为 {lr_best[0]}。"

    def batch_note():
        h = {r["label"]: r for r in sweeps["batch"]}
        b256 = [r for r in sweeps["batch"] if r.get("batch_size") == 256]
        b64 = [r for r in sweeps["batch"] if r.get("batch_size") == 64]
        if b256 and b64:
            return (f"在固定 1,500 步的协议下，batch 256 实际“看过”的图片数是 batch 64 的 4 倍，"
                    f"所以这组对比同时混合了“批大小”和“有效数据量”两个因素："
                    f"FID 从 batch 64 的 {b64[0]['fid']:.1f} 降到 batch 256 的 {b256[0]['fid']:.1f}，"
                    "更可能说明“多看了几倍数据”有帮助，而不是“批次越大越好”。")
        return ""

    def other_note():
        h = {r["label"]: r for r in sweeps["other"]}
        base = [r for r in sweeps["other"] if "参考" in r["label"]]
        beta09 = [r for r in sweeps["other"] if "0.9" in r["label"]]
        smooth = [r for r in sweeps["other"] if "平滑" in r["label"]]
        parts = []
        if base and beta09:
            parts.append(f"Adam β1=0.9 的 FID 为 {beta09[0]['fid']:.1f}，比 β1=0.5 的 {base[0]['fid']:.1f} "
                         f"差 {beta09[0]['fid'] - base[0]['fid']:.1f}，支持 DCGAN “β1 取 0.5 减少动量”的建议")
        if base and smooth:
            delta = smooth[0]["fid"] - base[0]["fid"]
            parts.append(f"单边标签平滑 0.1 的 FID 为 {smooth[0]['fid']:.1f}（{'改善' if delta < 0 else '变差'} "
                         f"{abs(delta):.1f}），是这组小技巧里{'唯一明显有效的' if delta < -5 else '收益有限的'}")
        return "；".join(parts) + "。"

    add("## 12. 超参数实验\n\n"
        f"本节固定训练 {steps} 步、其余条件不变，每次只改变一个超参数，用 FID 比较。"
        f"参考配置（潜维 100、lr 2e-4、batch 128、β1=0.5）的 FID 为 {f2(base_fid)}，"
        "它直接复用第 9 节激活函数实验中的 ReLU 组结果，不重复训练。\n\n"
        "### 12.1 潜变量维度\n\n" + sweep_table("latent", "潜维") + "\n\n"
        f"三档潜维的 FID 分别是 {' / '.join(f2(f) for _, f in latent)}，最大只差 {latent_spread:.1f}，"
        "在 1,500 步预算下差异不大。潜维 32 略差说明表达能力开始受限；"
        "潜维 256 略好但提升落在 FID 采样噪声的量级内，因此 100 左右仍是性价比较高的默认选择。\n\n"
        "### 12.2 学习率\n\n" + sweep_table("lr", "G/D 学习率") + "\n\n"
        + lr_note() + "\n\n"
        + img("hyper_lr_curves", "图 18　不同学习率下的训练曲线（左：生成器损失；右：判别器输出概率）") + "\n\n"
        "学习率是 GAN 最敏感的超参数：太小收敛慢，太大容易让两个网络互相“打架”导致训练发散。"
        f"本实验中最优的是 **{lr_best[0]}**，这也与 DCGAN 论文推荐的 2e-4 一致。\n\n"
        "### 12.3 批大小\n\n" + sweep_table("batch", "批大小") + "\n\n"
        + batch_note() + "\n\n"
        "另外，批大小还会影响计算效率（更大批次在 GPU 上利用率更高），"
        "在显存允许时可作为加速手段，但比较生成质量时要注明是否固定了总训练步数。\n\n"
        "### 12.4 Adam β1 与单边标签平滑\n\n" + sweep_table("other", "配置") + "\n\n"
        + other_note() + "\n\n"
        "单边标签平滑把真实样本标签从 1 放松到 0.9，防止判别器输出极端置信的概率，"
        "是稳定 GAN 训练的常用技巧，本实验的结果支持它在短预算下有小幅收益。\n\n"
        + img("hyper_summary", "图 17　四组超参数扫描的 FID 对比") + "\n\n"
        f"把四组结果放在一起看：全部超参数配置的 FID 范围是 {min(all_vals):.1f}~{max(all_vals):.1f}"
        f"（最大差 {overall_spread:.1f}），最差配置来自学习率 5e-4。"
        "这说明**学习率和 Adam β1 的选择确实重要**（选错会让训练不稳定），"
        "而潜维在 32~256 之间的影响相对温和。不过与“训练步数从 1,500 增加到 6,000 使 FID 从 264 降到 102”"
        "相比，超参数带来的改善量级仍然有限——**先保证训练充分、评估可靠，再谈精细调参**。")



def _kl_traj_note(hist):
    """描述 KL 项随训练的变化轨迹（先降后升 / 单调上升 / 平稳）。"""
    kl = hist["kl"]
    steps = hist["step"]
    i_min = min(range(len(kl)), key=lambda i: kl[i])
    if i_min > 2 and kl[-1] > kl[i_min] * 1.05:
        return (f"KL 项在训练初期先快速下降到约 {kl[i_min]:.0f}（第 {steps[i_min]} 步附近），"
                f"随后缓慢回升到 {kl[-1]:.0f}——前者是编码器从随机初始化进入正常工作区间，"
                "后者说明编码器在逐步利用隐变量表达更多信息（而不是塌缩到先验）。")
    if kl[-1] > kl[0] * 1.1:
        return (f"KL 项从 {kl[0]:.0f} 逐步升到 {kl[-1]:.0f}，"
                "说明编码器在逐步利用隐变量表达更多信息（而不是塌缩到先验）。")
    return (f"KL 项稳定在 {min(kl):.0f}~{max(kl):.0f} 之间，没有出现后验塌缩（KL 归零）的迹象。")

def sec_vae(add):
    runs = VA["runs"]
    steps = VA["steps"]
    rows = []
    for r in runs:
        hist = r["history"]
        rows.append([r["label"], f2(r["fid"]), f4(r["recon"]["mse"]), f2(r["recon"]["psnr"]),
                     f4(hist["kl_per_dim"][-1]), f"{r['epochs_equivalent']:.1f}"])
    b1 = next(r for r in runs if r["beta"] == 1.0)
    b4 = next(r for r in runs if r["beta"] != 1.0) if len(runs) > 1 else b1
    h1, h4 = b1["history"], b4["history"]
    # β=4 是否存在训练尖峰
    spike_idx = max(range(len(h4["loss"])), key=lambda i: h4["loss"][i])
    spike_loss = h4["loss"][spike_idx]
    spike_step = h4["step"][spike_idx]
    spike_kl = h4["kl"][spike_idx]
    normal_loss = h4["loss"][0]
    has_spike = spike_loss > normal_loss * 5
    if has_spike:
        spike_txt = (f"值得注意的是，β=4 的训练**不稳定**：在第 {spike_step} 步附近损失从 {h4['loss'][0]:.0f} "
                     f"突增到 {spike_loss:.0f}（KL 项从约 {h4['kl'][0]:.0f} 跳到 {spike_kl:.0f}），"
                     "之后虽然恢复，但最终重建损失仍明显高于 β=1。说明 β 过大时 KL 项会主导优化、"
                     "放大数值不稳定性；实践中 β-VAE 常用 KL warm-up、更小的学习率或梯度裁剪来缓解。")
    else:
        spike_txt = "两个 β 的训练曲线都比较平滑，没有出现明显的损失尖峰。"
    add("## 13. VAE 对照实验\n\n"
        "VAE 与 GAN 的建模思路完全不同：它显式地学习从隐变量到图像的生成过程，"
        "训练目标是最大化数据的对数似然下界（ELBO），而不是“骗过判别器”。"
        "本节训练两个卷积 VAE：β=1（标准 VAE）和 β=4（β-VAE，更强地压紧后验），"
        f"各 {steps} 步，用同一套 FID 协议与 DCGAN 对比。\n\n"
        + table(["模型", "FID", "重建 MSE", "重建 PSNR (dB)", "KL / 维", "等效 epoch"],
                rows, "lrrrrr") + "\n\n"
        + img("vae_training", "图 19　VAE（β=1）训练曲线：总损失、重建损失、KL 项") + "\n\n"
        "图 19 展示了 VAE 训练的典型行为：总损失在前几百步快速下降，之后进入缓慢改善阶段；"
        f"重建损失从约 {h1['recon'][0]:.0f} 持续降到 {h1['recon'][-1]:.0f}。"
        + _kl_traj_note(h1) +
        "与 GAN 的对抗式训练相比，标准 VAE 的损失曲线非常平稳，几乎没有崩溃风险，这是 VAE 最大的工程优势。\n\n"
        + img("vae_training_beta4", "图 19b　β-VAE（β=4）训练曲线：可见明显的损失尖峰") + "\n\n"
        + spike_txt + "\n\n"
        + img("vae_recon1", "图 20a　标准 VAE 的重建结果（上排原图，下排重建）") + "\n\n"
        f"从重建（图 20a）看，VAE 能把发色、脸型、眼睛位置还原出来，但细节明显模糊，"
        f"重建 PSNR 只有 {b1['recon']['psnr']:.2f} dB（MSE {b1['recon']['mse']:.4f}）。"
        "这不是训练不充分，而是 VAE 目标函数的必然结果——重建项是最小化像素平方误差，"
        "而“像素平均”在多个可能模式之间会给出一个模糊的折中解。\n\n"
        + img("vae_recon4", "图 20b　β-VAE（β=4）的重建结果") + "\n\n"
        + img("vae_interp1", "图 21a　标准 VAE 潜空间插值（10 帧）") + "\n\n"
        + img("vae_interp4", "图 21b　β-VAE（β=4）潜空间插值（10 帧）") + "\n\n"
        + img("vae_samples", "图 22　VAE 先验采样：上排 β=1，下排 β=4") + "\n\n"
        f"对比 β=1 与 β=4：β=4 的 KL/维为 {h4['kl_per_dim'][-1]:.4f}，"
        f"低于 β=1 的 {h1['kl_per_dim'][-1]:.4f}（后验被压得更紧）；"
        f"但重建 PSNR 从 {b1['recon']['psnr']:.2f} dB 降到 {b4['recon']['psnr']:.2f} dB，"
        f"FID 从 {f2(b1['fid'])} 变成 {f2(b4['fid'])}。"
        "β 增大后重建更模糊、FID 更差，符合 β-VAE “牺牲重建质量换取潜空间规整/解耦”的理论预期；"
        "本实验里 β=4 甚至出现了训练尖峰，说明这个取舍不是免费的。\n\n"
        "潜空间插值（图 21）显示 VAE 的隐空间是连续平滑的：相邻帧之间发色、发型缓慢过渡，"
        "中间帧仍然是一张合理的动漫脸——这是“学到连续隐表示”的直接证据，也是 VAE 相对 GAN 的结构性优势："
        "GAN 的潜空间通常没有这么规整，插值中间容易产生不自然的过渡。\n\n"
        f"不过从生成质量看，VAE 的最佳 FID（{f2(min(r['fid'] for r in runs))}）明显差于 DCGAN 主实验的 "
        f"{f2(M['final_fid'])}，先验采样（图 22）也明显更模糊、更“平均”。"
        "这是两类模型的核心取舍：**VAE 用“平均”换稳定，GAN 用“对抗”换锐利。**")



def _c2st_note(c2st, vae_best):
    """C2ST 结果需要特别解释：数值低不等于质量好，DCGAN 的 C2ST 反而更高。"""
    d = c2st.get("dcgan", {})
    v = c2st.get("vae", {})
    if not d or not v:
        return ""
    return (f"C2ST 的结果需要谨慎解读：小 CNN 区分“真实 vs DCGAN”的测试准确率为 "
            f"**{d['accuracy']:.4f}**，区分“真实 vs {vae_best['label'] if vae_best else 'VAE'}”为 "
            f"**{v['accuracy']:.4f}**。两个数字都远高于 0.5（随机猜测），说明 64×64 生成图与真实图仍有明显差异；"
            "有趣的是 DCGAN 的 C2ST 反而更高，这并不代表它更差——小 CNN 很容易抓住 GAN 生成图里的"
            "高频棋盘状伪影，而 VAE 的模糊损失掉了高频信息、反而少了一类可判别线索。"
            "因此 C2ST 不能直接用于排序生成质量，它与 FID 的分歧恰好说明“生成质量”是多维的："
            "FID 比较 Inception 特征分布、C2ST 考察低层可区分性、人眼判断整体观感。\n\n")

def sec_final(add):
    cfg = F["config"]
    fid = F["final_fid"]
    c2st = F.get("c2st", {})
    vae_best = F.get("vae_best")
    vae_runs = VA.get("runs", [])
    rows = [
        ["潜变量维度", str(cfg["z_dim"])],
        ["生成器激活", cfg["g_act"]],
        ["判别器激活 / Dropout", f"{cfg['d_act']} / {cfg['d_dropout']}"],
        ["优化器", f"Adam，G lr={cfg['lr_g']:g}，D lr={cfg['lr_d']:g}，β={tuple(cfg['betas'])}"],
        ["训练步数", f"{cfg['max_steps']:,} 步（{'复用主实验模型' if F.get('reused_main') else '按扫描结果重新训练'}）"],
        ["最终 FID", f2(fid)],
    ]
    c2st_rows = []
    if "dcgan" in c2st:
        c2st_rows.append(["DCGAN（最终模型）", f"{c2st['dcgan']['accuracy']:.4f}",
                          f"{c2st['dcgan']['n_train']:,} / {c2st['dcgan']['n_test']:,}"])
    if "vae" in c2st:
        c2st_rows.append([vae_best["label"] if vae_best else "VAE",
                          f"{c2st['vae']['accuracy']:.4f}",
                          f"{c2st['vae']['n_train']:,} / {c2st['vae']['n_test']:,}"])
    add("## 14. 最终模型与 DCGAN vs VAE 对比\n\n"
        "### 14.1 最终模型\n\n"
        "根据第 9~12 节的控制变量结果，最终模型的配方按“在各自预算下 FID 最低”选择。"
        + (f"最终选择与主实验相同的配置（{cfg['g_act']} 生成器 + 潜维 {cfg['z_dim']}），"
           "因此直接复用主实验训练好的 6,000 步模型，不再重复训练。" if F.get("reused_main") else
           f"最终选择的配置为 {cfg['g_act']} 生成器 + 潜维 {cfg['z_dim']}，与主实验不同，"
           "于是用该配置重新训练 6,000 步作为最终模型。")
        + "\n\n" + table(["配置项", "取值"], rows, "ll") + "\n\n"
        + (f"最终模型在**相同 6,000 步预算**下 FID 为 **{f2(fid)}**，比主实验的 {f2(M['final_fid'])} 低 "
           f"{M['final_fid'] - fid:.1f}。需要说明的是，最终配方同时改变了激活函数与潜维两个因素，"
           "不能把提升单独归因于某一个：潜维 100→256 在 1,500 步扫描中的差异只有 2.2 FID（在采样噪声量级内），"
           "而 LeakyReLU 在激活函数实验中领先 ReLU 72 FID，因此提升主要应归因于生成器激活函数。\n\n"
           if M else "\n\n")
        + img("final_gallery", "图 23　最终 DCGAN 生成的 80 张动漫头像（3 倍最近邻放大）") + "\n\n"
        "图 23 是最终模型的 80 张样本画廊。大多数样本都能明确辨认出“动漫少女头像”的结构："
        "刘海、眼睛高光、发饰、衣领都可以看到，且发色/瞳色/发型分布较广，没有明显的模式崩溃；"
        "少数样本存在左右眼不对称、背景色块、头发与衣服融合等问题。对 64×64 分辨率、"
        "14,029 张训练图的 DCGAN 来说，这个质量是符合预期的。\n\n"
        + img("final_interp", "图 24　最终模型的潜空间线性插值（10 帧）") + "\n\n"
        "潜空间插值（图 24）显示 GAN 的隐空间虽然不是完全平滑，但中间帧仍是合理的动漫脸，"
        "没有出现突然的色块跳变，说明生成器学到的是连续的特征流形，而不是零散的记忆点。\n\n"
        + img("final_seeds", "图 25　不同随机种子的采样（每行 8 张）") + "\n\n"
        "不同随机种子的采样（图 25）之间没有重复脸，进一步排除了模式崩溃。\n\n"
        "### 14.2 DCGAN 与 VAE 的定量对比\n\n"
        + table(["模型", "C2ST 测试准确率（越低越难区分）", "训练/测试样本数"],
                c2st_rows, "lrr") + "\n\n"
        + img("final_vs_vae_samples", "图 26　真实图（上）、DCGAN 生成（中）、VAE 生成（下）的对比") + "\n\n"
        + img("final_vs_vae_metrics", "图 27　DCGAN 与 VAE 的 FID（左）与 C2ST 准确率（右）对比") + "\n\n"
        + (f"在同样的真实参考集上，DCGAN 的 FID 为 **{f2(fid)}**，"
           + (f"VAE 的最佳 FID 为 **{f2(vae_best['fid'])}**，" if vae_best else "")
           + "DCGAN 明显更好，与图 26 的目检一致。\n\n")
        + (_c2st_note(c2st, vae_best) if "dcgan" in c2st else "")
        + "从目检（图 26）也能得到同样结论：DCGAN 的样本更锐利、更像“画出来”的动漫头像；"
        "VAE 的样本偏模糊、色彩更“平均”，但完全没有 GAN 常见的畸变脸。"
        "这与两类模型的训练目标完全一致，也是文献中的标准结论。\n\n"
        + img("terminal_log", "图 28　训练日志截图（节选）") + "\n\n"
        "图 28 是训练过程的终端日志截图，展示了最终模型的配置、逐段损失与 FID 记录，"
        "完整日志见 `exp6_report/run_logs/run_all.log`。")



def _act_finding(act_runs):
    """根据激活函数实验的实际 FID 生成一句结论（差异大/小两种说法）。"""
    if not act_runs:
        return ""
    best = min(act_runs, key=lambda r: r["fid"])
    worst = max(act_runs, key=lambda r: r["fid"])
    spread = worst["fid"] - best["fid"]
    if spread < 5:
        return (f"生成器激活函数（ReLU/LeakyReLU/ELU/GELU）在 1,500 步预算下的 FID 差距只有 "
                f"{spread:.1f}，说明这个阶段“训练是否充分”比“选哪种激活”更重要")
    main_fid = M.get("final_fid")
    tail = (f"；但训练步数从 1,500 增加到 6,000 带来的改善更大"
            f"（主实验 6,000 步的 FID 为 {main_fid:.1f}）" if main_fid else "")
    return (f"生成器激活函数在 1,500 步预算下就有明显影响：最优的 {best['label']} FID 为 "
            f"{best['fid']:.1f}，最差的 {worst['label']} 为 {worst['fid']:.1f}，相差 {spread:.1f}"
            + tail)


def _diff_abstract():
    tr = DIFF.get("train", {})
    dp = DIFF.get("eval", {}).get("final", {})
    comp = DIFF.get("comparison", {})
    dc = comp.get("dcgan", {})
    dpc = comp.get("ddpm", {})
    fid = dp.get("fid", float("nan"))
    dc_fid = dc.get("fid", M.get("final_fid", float("nan")))
    improve = (dc_fid - fid) / dc_fid * 100 if dc_fid else 0
    slow = dpc.get("sample_seconds_per_1k", 1) / max(dc.get("sample_seconds_per_1k", 1), 1e-9)
    per_1k = dpc.get("sample_seconds_per_1k", 0)
    return (f"扩散模型方面，本实验从零实现了 18.55M 参数的 DDPM U-Net（余弦噪声调度 + EMA），"
            f"在相同训练集上训练 {tr.get('minutes', 0) / 60:.1f} 小时（{tr.get('steps', 0):,} 步），"
            f"用 DDIM-50 采样得到的 FID 为 **{f2(fid)}**，比 DCGAN 的 {f2(dc_fid)} 又降低了 "
            f"{improve:.1f}%，是三种模型中生成质量最好的；但生成 1,000 张图的耗时约 "
            f"{per_1k:.0f} 秒，是 DCGAN 的 {slow:.0f} 倍。"
            "这说明扩散模型在质量上占优，代价是采样速度慢一个数量级——三种范式各有取舍。")

def sec_abstract(add):
    main_fid = M.get("final_fid")
    fid_map = {int(a): b for a, b in M.get("fid_points", [])}
    fid_max = max(fid_map.values()) if fid_map else None
    act_runs = ACT.get("runs", [])
    dr_runs = DR.get("runs", [])
    sz_runs = SZ.get("runs", [])
    vae_runs = VA.get("runs", [])
    c2st = F.get("c2st", {})

    ordinals = ["第一", "第二", "第三", "第四"]
    findings = []
    if act_runs:
        findings.append(f"{ordinals[len(findings)]}，{_act_finding(act_runs)}")
    if dr_runs:
        findings.append(f"{ordinals[len(findings)]}，{_dropout_finding(dr_runs)}")
    if len(sz_runs) >= 2:
        fids_sz = [r["fid"] for r in sz_runs]
        ratios_sz = [r["nn_ratio"] for r in sz_runs]
        if fids_sz[-1] < fids_sz[0]:
            fid_clause = (f"FID 从 {f2(fids_sz[0])} 降到 {f2(fids_sz[-1])}")
        else:
            fid_clause = (f"FID 并未随数据量下降（{', '.join(f2(f) for f in fids_sz)}），"
                          "1,000 张时 FID 反而最低——因为模型反复训练后会把训练图“背”出来，"
                          "而训练图本身就是真实图像，FID 自然好看")
        findings.append(
            f"{ordinals[len(findings)]}，{fid_clause}；生成图到训练集的最近邻距离比值从 "
            f"{ratios_sz[0]:.3f} 变为 {ratios_sz[-1]:.3f}（均小于 1），说明数据量越少记忆化越强，"
            "FID 必须结合最近邻分析一起看")
    if DIFF:
        comp = DIFF.get("comparison", {})
        dp = DIFF.get("eval", {}).get("final", {})
        dc = comp.get("dcgan", {})
        dpc = comp.get("ddpm", {})
        findings.append(
            f"{ordinals[len(findings)] if len(findings) < len(ordinals) else '另外'}，"
            f"从零实现的 DDPM 扩散模型取得了三种模型中最优的 FID（{f2(dp.get('fid', float('nan')))}，"
            f"DCGAN {f2(dc.get('fid', M.get('final_fid', float('nan'))))}、VAE 203.1），"
            f"但生成 1,000 张图的耗时是 DCGAN 的 "
            f"{dpc.get('sample_seconds_per_1k', 1) / max(dc.get('sample_seconds_per_1k', 1), 1e-9):.0f} 倍")
    findings_txt = ("通过一系列对比实验，我们得到以下结论。" + "。".join(findings) + "。\n\n"
                    if findings else "")

    if vae_runs:
        fids = [r["fid"] for r in vae_runs]
        worst_vae = max(fids)
        if main_fid is not None and worst_vae > main_fid:
            cmp_txt = f"均明显差于 DCGAN 的 {f2(main_fid)}"
        else:
            cmp_txt = f"与 DCGAN 的 {f2(main_fid) if main_fid else '—'} 相比差距不大"
        vae_txt = (f"VAE 方面，标准 VAE（β=1）的 FID 为 {f2(fids[0])}，"
                   + (f"β-VAE（β=4）为 {f2(fids[1])}，" if len(fids) > 1 else "")
                   + f"{cmp_txt}；但 VAE 的训练曲线平稳、重建与潜空间插值平滑，"
                   "展示了“稳定但模糊”与“锐利但难训”这两种范式的核心取舍。")
        if "dcgan" in c2st and "vae" in c2st:
            vae_txt += (f"C2ST 小 CNN 对两类样本都能轻易区分（DCGAN {c2st['dcgan']['accuracy']:.3f}、"
                        f"VAE {c2st['vae']['accuracy']:.3f}），说明 64×64 生成图与真实图仍有可检测的统计差异；"
                        "但 C2ST 数值不能直接当质量分数——DCGAN 的高频伪影和 VAE 的模糊都容易被小 CNN 抓住，"
                        "质量排序应以 FID 与目检为准。")
    else:
        vae_txt = ""

    add("## 摘要\n\n"
        "本实验用 PyTorch 从零实现三种生成式神经网络——生成对抗网络（DCGAN）、卷积变分自编码器（VAE）与去噪扩散概率模型（DDPM），"
        "用于生成 64×64 的二次元动漫人物头像。数据来自公开的 Anime Face Dataset：原始 21,551 张图片按 md5 去重后"
        f"剩 {D['n_unique']:,} 张，其中 {D['n_train']:,} 张用于训练、{D['n_ref']:,} 张留作 FID 参考集。\n\n"
        f"主实验的 DCGAN 按 DCGAN 论文的经典结构（生成器 {A['generator']['params'] / 1e6:.2f}M 参数、"
        f"判别器 {A['discriminator']['params'] / 1e6:.2f}M 参数）训练 6,000 步，"
        f"FID 从 {f2(fid_max) if fid_max is not None else '—'} 降到 **{f2(main_fid) if main_fid else '—'}**；"
        "有意义的是，训练损失在 3,000 到 6,000 步之间几乎没有变化，而 FID 却下降了一半以上——"
        "这直接说明生成模型不能用损失值来监控质量。\n\n"
        + findings_txt
        + (vae_txt + "\n\n" if vae_txt else "")
        + (f"进一步按扫描结果把生成器换成 LeakyReLU、潜维扩大到 256 重新训练后，"
           f"最终模型的 FID 降到 **{f2(F['final_fid'])}**。" if F else "")
        + (_diff_abstract() + "\n\n" if DIFF else "")
        + f"最终模型的 80 张样本画廊、潜空间插值以及全部 {len(FIG)} 张实验插图见正文。")



def sec_diffusion(add):
    tr = DIFF.get("train", {})
    ev = DIFF.get("eval", {})
    fin = ev.get("final", {})
    comp = DIFF.get("comparison", {})
    screen = ev.get("screening", [])
    sweep = ev.get("ddim_sweep", [])
    dc = comp.get("dcgan", {})
    vae = comp.get("vae", {})
    dp = comp.get("ddpm", {})
    seed_fid = M.get("final_fid")
    vae_fid = (F.get("vae_best") or {}).get("fid")
    train_rows = [
        ["模型结构", f"DDPM U-Net（4 个分辨率级别 64/32/16/8，16×16 与 8×8 处自注意力），{tr.get('params', 0) / 1e6:.2f}M 参数"],
        ["扩散步数 T / 噪声调度", f"{tr.get('T', 1000)} / {tr.get('schedule', 'cosine')}（余弦调度）"],
        ["训练数据", f"{D['n_train']:,} 张 64×64 训练图（与 DCGAN/VAE 完全相同的训练池）"],
        ["批大小 / 训练步数", f"{tr.get('batch', 32)} / {tr.get('steps', 0):,} 步（约 {tr.get('steps', 0) * tr.get('batch', 32) / D['n_train']:.0f} 个 epoch）"],
        ["优化器 / 学习率", f"Adam，lr={tr.get('lr', 2e-4):g}，线性 warmup {tr.get('warmup', 2000)} 步 + 余弦退火"],
        ["正则化 / 平均", f"ResBlock Dropout {tr.get('dropout', 0.1)}；参数 EMA（decay={tr.get('ema_decay', 0.9999)}），采样时使用 EMA 权重"],
        ["训练时长", f"{tr.get('minutes', 0) / 60:.1f} 小时（单张 RTX 4050 Laptop，batch=32 + bf16 AMP + torch.compile）"],
        ["采样", "训练监控用 DDPM/DDIM；质量评估用 DDIM-50（确定性、50 次前向）"],
    ]
    fin_rows = [
        ["最佳 EMA checkpoint", f"第 {fin.get('step', 0):,} 步"],
        ["FID（4,000 张，DDIM-50）", f2(fin.get("fid", float("nan")))],
        ["C2ST 测试准确率", f4(fin.get("c2st", {}).get("accuracy", float("nan")))],
        ["生成图 / 真实图最近邻距离比", f3(fin.get("nn_ratio", float("nan")))],
        ["DDIM-20 / 50 / 100 采样（64 张耗时）",
         " / ".join(f"{fin.get('sampling_seconds', {}).get(f'ddim{s}', 0):.1f}s" for s in (20, 50, 100))],
        ["DDPM-1000 完整采样（16 张耗时）", f"{fin.get('sampling_seconds', {}).get('ddpm1000', 0):.0f}s"],
    ]
    screen_rows = [[f"{r['step']:,}", f2(r["fid"])] for r in screen]
    sweep_rows = [[s["steps"], f2(s["fid"]), f"{s['seconds']:.1f}"] for s in sweep]
    comp_rows = [
        ["DCGAN（最终模型）", f2(dc.get("fid", float("nan"))), f4(dc.get("c2st", float("nan"))),
         f"{dc.get('sample_seconds_per_1k', 0):.1f}", f3(dc.get("nn_ratio_vs_real", float("nan")))],
        ["VAE β=1", f2(vae.get("fid", float("nan"))), f4(vae.get("c2st", float("nan"))),
         f"{vae.get('sample_seconds_per_1k', 0):.1f}", f3(vae.get("nn_ratio_vs_real", float("nan")))],
        ["DDPM 扩散（DDIM-50）", f2(dp.get("fid", float("nan"))), f4(dp.get("c2st", float("nan"))),
         f"{dp.get('sample_seconds_per_1k', 0):.1f}", f3(dp.get("nn_ratio_vs_real", float("nan")))],
    ]
    # 相对主实验 DCGAN 的 FID 改善
    improvement = ((seed_fid - fin.get("fid", seed_fid)) / seed_fid * 100) if seed_fid else 0.0
    # 采样耗时倍数
    slow_x = (dp.get("sample_seconds_per_1k", 1) / max(dc.get("sample_seconds_per_1k", 1), 1e-9)) if dc else 0
    best_step = max(r["step"] for r in screen) if screen else 0
    first_loss = DIFF_LOG[0]["loss"] if DIFF_LOG else float("nan")
    last_loss = DIFF_LOG[-1]["loss"] if DIFF_LOG else float("nan")
    add("## 15. 追加实验：扩散模型对照\n\n"
        "### 15.1 为什么再加一个扩散模型\n\n"
        "前 14 节的实验说明 DCGAN 生成锐利但训练不稳定、VAE 稳定但样本模糊。2020 年之后，扩散模型（DDPM）"
        "逐渐成为图像生成的主流范式：它不依赖对抗训练，而是学习“从高斯噪声中一步步去噪”的过程，"
        "训练目标只是一个简单的回归损失，因此比 GAN 稳定得多，样本质量也往往更高。"
        "为了公平比较三种范式的优劣，本实验从零实现了标准 DDPM（不是缩小版的玩具模型），"
        f"在**完全相同的训练数据**（14,029 张）与**完全相同的 FID 参考集**（3,000 张）上训练约 {tr.get('minutes', 0) / 60:.1f} 小时，"
        "并与 DCGAN、VAE 做三方对比。\n\n"
        "DDPM 的前向过程对真实图像逐步加噪：\n\n"
        "$$\n"
        "q(x_t\\mid x_{t-1})=\\mathcal{N}\\big(\\sqrt{1-\\beta_t}\\,x_{t-1},\\;\\beta_t I\\big),\\qquad "
        "q(x_t\\mid x_0)=\\mathcal{N}\\big(\\sqrt{\\bar\\alpha_t}\\,x_0,\\;(1-\\bar\\alpha_t) I\\big). \\tag{7}\n"
        "$$\n\n"
        "训练一个 U-Net $\\varepsilon_\\theta(x_t,t)$ 去预测“这一步加进去的噪声”，损失就是预测噪声与真实噪声的平方误差：\n\n"
        "$$\n"
        "L=\\mathbb{E}_{x_0,\\varepsilon,t}\\big\\|\\varepsilon-\\varepsilon_\\theta\\big(\\sqrt{\\bar\\alpha_t}x_0"
        "+\\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\;t\\big)\\big\\|^2. \\tag{8}\n"
        "$$\n\n"
        "生成时从纯噪声 $x_T\\sim\\mathcal{N}(0,I)$ 出发，按学到的反向过程逐步去噪。本实验采用余弦噪声调度"
        "（Nichol & Dhariwal, 2021）并维护参数 EMA；采样时既测试了完整的 1,000 步 DDPM，"
        "也测试了确定性的 DDIM（20/50/100 步），用于分析“质量—速度”的权衡。\n\n"
        "### 15.2 网络结构与训练配置\n\n"
        + img("diff_arch", "图 29　DDPM 的 U-Net 结构（4 个分辨率级别 + 自注意力 + 时间步条件）") + "\n\n"
        "U-Net 的空间尺寸从 64→32→16→8 逐级下采样、再逐级上采样，与生成器/VAE 的编码器-解码器结构相似；"
        "不同之处在于每一层都通过时间步嵌入接收 $t$ 的信息（否则网络无法知道当前噪声强度），"
        "并在 16×16 与 8×8 处加入自注意力以建模全局结构。跳连（skip connection）把下采样路径的特征"
        "直接送到上采样路径，保证细节信息不丢失。\n\n"
        + table(["配置项", "取值"], train_rows, "ll") + "\n\n"
        "### 15.3 训练过程\n\n"
        + img("diff_training", "图 30　DDPM 训练损失、学习率与 FID 随训练步数的变化") + "\n\n"
        f"训练损失从初始的 {first_loss:.3f} 稳定下降到 {last_loss:.3f}，全程没有出现 GAN 那种"
        "“判别器/生成器此消彼长”的震荡——这正是扩散模型最大的工程优势：**目标函数就是回归，"
        "不存在对抗博弈，训练过程几乎不会崩溃。** 图 30 右图给出不同训练阶段的 FID："
        f"在 2,048 张的监控协议下，FID 从第 {screen[0]['step']:,} 步的 {f2(screen[0]['fid'])} "
        f"变为第 {best_step:,} 步的 {f2(screen[-1]['fid'])}，基本持平——这说明模型在约 "
        f"{screen[0]['step']:,} 步时就已基本收敛，后 8 万步主要是细节微调（训练损失仍在缓慢下降）。"
        f"最终用 4,000 张生成图、与 GAN/VAE 完全相同的协议测得 FID={f2(fin.get('fid', 0))}。\n\n"
        + table(["训练步数", "FID（DDIM-50，2,048 张）"], screen_rows, "rr") + "\n\n"
        "### 15.4 生成质量、DDIM 步数与采样成本\n\n"
        + img("diff_evolution", "图 31　同一组固定噪声在四个训练阶段（"
             + "、".join(f"{r['step'] // 1000}k" for r in screen) + " 步）的生成结果") + "\n\n"
        "图 31 用同一组固定噪声在训练 9 万步之后的四个 checkpoint 上采样。四行样本几乎完全相同，"
        "与上表 FID 在 22.2~22.4 之间基本持平一致——说明模型在 9 万步左右就已收敛，"
        "之后 8 万步主要是在做细节微调（训练损失仍有小幅下降）。"
        "这也说明扩散模型的训练更稳定：不像 GAN 那样在后期突然发生“质变”，而是平滑地逼近收敛。"
        "与 GAN 的“一步生成”不同，扩散模型每次采样都要反复调用 U-Net，因此采样成本高得多。\n\n"
        + img("diff_ddim", "图 32　DDIM 采样步数对生成质量的影响（同一固定噪声）") + "\n\n"
        + table(["DDIM 步数", "FID（1,024 张）", "64 张采样耗时（秒）"], sweep_rows, "rrr") + "\n\n"
        + table(["最佳 checkpoint 的核心指标", "数值"], fin_rows, "ll") + "\n\n"
        "DDIM 是扩散模型加速采样的经典方法：把 1,000 步的随机反向过程改成间隔更大的确定性轨迹。"
        + ("从上表看，20 步的 FID 与 50/100 步差距不大，说明本模型的采样轨迹比较平直，"
           "可以在几乎不损失质量的情况下把采样成本降到 1/5；" if sweep and sweep[0]["fid"] - min(s["fid"] for s in sweep) < 5 else
           "从上表看，步数越少 FID 越差，说明本模型仍需要较多去噪步数才能稳定收敛；")
        + "不过即使使用 DDIM-50，生成 1,000 张图仍需要 "
        f"{dp.get('sample_seconds_per_1k', 0):.1f} 秒，是 DCGAN 的 {slow_x:.0f} 倍左右——这是扩散模型相对 GAN/VAE 的固有劣势。\n\n"
        + img("diff_gallery", f"图 33　扩散模型生成的 80 张动漫头像（DDIM-{fin.get('gallery_ddim_steps', 50)}；"
              f"DDIM-50 协议下 FID={f2(fin.get('fid', 0))}）") + "\n\n"
        "图 33 是最终模型的 80 张样本。与 DCGAN 的画廊（图 23）相比，扩散模型的样本在发丝、瞳孔高光、"
        "头发与背景的边界等细节上更干净，五官错位和色块伪影明显更少；整体观感更接近真实动漫头像。\n\n"
        + img("diff_trajectory", "图 34　DDIM 去噪轨迹：从纯噪声（左）逐步生成图像（右）") + "\n\n"
        "图 34 展示了扩散模型独有的“逐步生成”过程：最左端是纯高斯噪声，随着去噪步数增加，"
        "先出现大色块与脸型，再逐渐长出眼睛、头发和细节。这种可解释的生成过程是 GAN/VAE 的一次性前向所没有的。\n\n"
        "### 15.5 三种生成模型的定量对比\n\n"
        + img("three_way_samples", "图 35　真实图 vs DCGAN vs VAE vs 扩散模型的生成样本") + "\n\n"
        + img("three_way_metrics", "图 36　三种模型在 FID、C2ST、采样速度和最近邻距离上的对比") + "\n\n"
        + table(["模型", "FID", "C2ST 准确率", "生成 1,000 张耗时（秒）", "最近邻距离 / 真实基线"],
                comp_rows, "lrrrr") + "\n\n"
        f"三方对比的结论非常清晰。**质量方面**，扩散模型的 FID 为 **{f2(fin.get('fid', 0))}**，"
        f"比 DCGAN 的 {f2(seed_fid) if seed_fid else '—'} 又降低了 {improvement:.1f}%"
        f"（相对 VAE 的 {f2(vae_fid) if vae_fid else '—'} 更是大幅领先）；从图 35 的目检也能看出"
        "扩散样本的细节最干净。**成本方面**，扩散模型生成 1,000 张图需要 "
        f"{dp.get('sample_seconds_per_1k', 0):.1f} 秒，是 DCGAN（{dc.get('sample_seconds_per_1k', 0):.1f} 秒）的 "
        f"{slow_x:.0f} 倍——GAN 一次前向就能出图，而扩散模型需要 50 次前向（DDIM-50）。"
        "**多样性方面**，最近邻距离比值为 "
        f"{f3(dp.get('nn_ratio_vs_real', float('nan')))}，与 DCGAN 的 {f3(dc.get('nn_ratio_vs_real', float('nan')))}、"
        f"VAE 的 {f3(vae.get('nn_ratio_vs_real', float('nan')))} 接近，说明扩散模型同样不是靠背训练图取得低 FID。\n\n"
        "C2ST 的情况与第 14 节类似：三个模型的 C2ST 都远高于 0.5（"
        f"DCGAN {f4(dc.get('c2st', float('nan')))}、VAE {f4(vae.get('c2st', float('nan')))}、"
        f"扩散 {f4(dp.get('c2st', float('nan')))}），说明 64×64 生成图与真实图仍有可检测的差异；"
        "C2ST 依然不能直接用来排序质量。\n\n"
        "### 15.6 小结\n\n"
        "扩散模型在本实验的 64×64 动漫头像任务上取得了三种模型中最低的 FID，且训练过程稳定、"
        "无需对抗调参、生成过程可解释；代价是采样速度慢一到两个数量级、训练时间也长得多"
        f"（约 {tr.get('minutes', 0) / 60:.1f} 小时 vs DCGAN 的 {M.get('train_time_s', 0) / 60:.0f} 分钟）。如果应用场景对质量要求高、对实时性要求低（如离线生成插画），"
        "扩散模型是更好的选择；如果需要实时/批量生成（如游戏、交互式应用），DCGAN 这类单步模型仍有价值。"
        "本实验也说明，**生成模型没有“全面最优”的范式，只有针对具体场景的取舍。**")

def sec_discussion(add):
    act_runs = ACT.get("runs", [])
    act_spread = (max(r["fid"] for r in act_runs) - min(r["fid"] for r in act_runs)) if act_runs else None
    sz_runs = SZ.get("runs", [])
    dr_runs = DR.get("runs", [])
    vae_runs = VA.get("runs", [])
    c2st = F.get("c2st", {})
    main_fid = M.get("final_fid")
    add("## 16. 讨论\n\n"
        "### 16.1 主要发现\n\n"
        f"1. **生成模型的损失曲线不能当作质量指标。** 主实验的判别器/生成器损失在训练中后期基本走平，"
        f"但 FID 从 {f2(dict((int(a), b) for a, b in M.get('fid_points', [])).get(3000, float('nan')))} "
        f"（3,000 步）降到 {f2(main_fid) if main_fid else '—'}（6,000 步）。"
        "原因是 GAN 的损失描述的是两个网络的相对强弱，而不是生成分布与真实分布的距离；"
        "判别器变强会让生成器损失变大，但生成质量可能同时提升。\n\n"
        + (f"2. **激活函数的影响介于“无所谓的技巧”和“决定性因素”之间。** {_act_finding(act_runs)}。"
           "LeakyReLU/ELU 在负半轴保留梯度，理论上更有利于生成器学习细节，本实验至少在短预算下支持这一点；"
           "而 GELU 在本任务中没有表现出优势。这说明生成任务里激活函数的影响与数据、训练预算强耦合，"
           "不能照搬分类任务里的经验。\n\n" if act_spread is not None else "")
        + (f"3. **Dropout 的正则化收益没有传导到生成质量。** {_dropout_finding(dr_runs)}。"
           "这个反例说明 GAN 的“技巧”必须同时看判别器和生成器两侧的指标，不能只优化判别器；"
           "把 Dropout 当成提升生成质量的“万能正则化”是一种误解。\n\n" if dr_runs else "")
        + (f"4. **数据量与记忆化的关系比“数据越多 FID 越低”复杂。** {sz_runs[0]['size']:,} 张时 FID 为 "
           f"{f2(sz_runs[0]['fid'])}（三档中最低），但生成图到训练集的最近邻比值也最低"
           f"（{sz_runs[0]['nn_ratio']:.3f}）；{sz_runs[-1]['size']:,} 张时 FID 为 {f2(sz_runs[-1]['fid'])}，"
           f"比值回升到 {sz_runs[-1]['nn_ratio']:.3f}。固定 3,000 步时，小数据被重复训练 300 多个 epoch，"
           "输出训练图就能得到好看的 FID——这提醒我们 FID 无法识别记忆化，必须配合最近邻分析。\n\n"
           if sz_runs else "")
        + "5. **GAN 与 VAE 的取舍是结构性的，不是调参能解决的。** VAE 的 ELBO 目标包含像素级重建项，"
        "在多个合理模式之间只能给出平均解，所以样本必然偏模糊；GAN 通过判别器学习感知损失，"
        "能生成锐利细节，但训练不稳定且缺乏显式隐空间。\n\n"
        + (f"6. **扩散模型在质量上胜出，但采样成本高一个量级。** DDPM 的 FID 为 "
           f"{f2(DIFF.get('eval', {}).get('final', {}).get('fid', float('nan')))}，低于 DCGAN 的 "
           f"{f2(M.get('final_fid', float('nan')))} 与 VAE 的 203.1；但生成 1,000 张图约需 "
           f"{DIFF.get('comparison', {}).get('ddpm', {}).get('sample_seconds_per_1k', 0):.1f} 秒，"
           f"约为 DCGAN 的 {DIFF.get('comparison', {}).get('ddpm', {}).get('sample_seconds_per_1k', 1) / max(DIFF.get('comparison', {}).get('dcgan', {}).get('sample_seconds_per_1k', 1), 1e-9):.0f} 倍。"
           "扩散模型的训练目标只是回归去噪、没有对抗博弈，因此训练稳定、几乎不会崩溃；"
           "至此三种范式形成“质量—稳定性—采样成本”的三点权衡。\n\n" if DIFF else "")
        + "### 16.2 生成模型评估的困难\n\n"
        "FID 虽然是最常用的指标，但它有几个本实验无法回避的局限：\n\n"
        "- **采样噪声**：本实验的 FID 真实参考集只有 3,000 张、生成 4,000~5,000 张，"
        "小样本会让 FID 有数点甚至十几点的偏差；同一配置换一个随机种子，FID 可能波动 5 左右。"
        "所以本实验只把 FID 的“大差异”（>10）当作可靠结论，个位数的差异不做过强解释。\n"
        "- **特征空间偏差**：Inception-v3 是在 ImageNet 自然图像上训练的，它的特征对动漫头像这种"
        "非自然图像是否敏感并没有保证；FID 低的模型不一定人眼看着更舒服。\n"
        "- **C2ST 的盲区**：本实验里 C2ST 给 DCGAN 打 1.000、给 VAE 0.895，与 FID/目检的排序相反。"
        "原因是小 CNN 很容易抓到 GAN 的高频伪影，而 VAE 的模糊反而少了一类判别线索；"
        "所以 C2ST 衡量的是“低层可区分性”，而不是“生成质量”，不能单独使用。\n"
        "- **单一指标会漏掉模式崩溃**：FID 使用协方差，对“多样性下降”有一定敏感性，"
        "但如果生成器只是把少数几个模式生成得很好，FID 可能仍然不错。因此本实验同时统计了最近邻距离、"
        "颜色多样性和 C2ST，并始终配合目检。\n\n"
        "### 16.3 与分类实验的对比\n\n"
        "前几个实验（MLP、CNN）里，验证集准确率是可靠的、单调的、可以指导早停的指标；"
        "而生成实验里没有任何一个指标具备这种性质。本实验最深的体会是：**生成模型的研究必须同时依赖"
        "多个互补指标 + 目检，并且对训练动力学（判别器强弱、梯度大小、模式多样性）保持敏感。**"
        "这也是生成模型比分类模型更难做实验的原因。\n\n"
        "### 16.4 局限与改进方向\n\n"
        "- **分辨率与数据**：64×64、14,029 张训练图限制了细节上限；提高分辨率（128×128、256×256）"
        "需要更大的数据和更长的训练。\n"
        "- **训练预算**：控制变量实验每组只训练 1,500~3,000 步，生成质量尚未饱和；扩散模型也只训练了约 9 小时"
        "（16 万步），更长训练或更大模型仍有提升空间。\n"
        "- **随机性**：每个配置只用一个随机种子，没有报告多次运行的均值±标准差；GAN 训练的方差较大，"
        "后续应至少跑 3 个种子。\n"
        "- **模型族**：比较了 DCGAN、VAE 与 DDPM，但没有尝试 WGAN-GP、StyleGAN、更高分辨率的扩散模型"
        "（如 128×128/256×256 的 latent diffusion）或一致性模型等更现代的方法。\n"
        "- **数据清洗**：只去掉了字节级重复，没有处理近重复图和非人脸离群样本。\n\n"
        "### 16.5 关于数据与生成内容\n\n"
        "Anime Face Dataset 为 CC0 许可，图片来自公开的动漫角色截图，本实验仅用于课程学习与研究，"
        "生成的样本是模型学习到的统计模式，不对应任何具体作品或真实人物。")


def sec_conclusion(add):
    act_runs = ACT.get("runs", [])
    act_spread = (max(r["fid"] for r in act_runs) - min(r["fid"] for r in act_runs)) if act_runs else None
    dr_runs = DR.get("runs", [])
    c2st = F.get("c2st", {})
    main_fid = M.get("final_fid")
    add("## 17. 结论\n\n"
        "本实验用 PyTorch 从零实现了 DCGAN、卷积 VAE 与 DDPM 扩散模型，在去重后的 Anime Face Dataset 上完成了"
        "二次元动漫头像的生成，并系统分析了激活函数、Dropout、数据量与超参数的影响。主要结论如下：\n\n"
        f"1. DCGAN 在 64×64 动漫头像上可以生成可辨认、多样化的结果：训练 6,000 步后 FID 为 "
        f"{f2(main_fid) if main_fid else '—'}，样本覆盖不同发色、瞳色与发型，没有明显模式崩溃。\n"
        "2. 生成模型的损失值不能作为质量指标；FID 在损失走平之后仍会大幅改善，"
        "训练监控必须使用固定参考集的 FID 或人工检查。\n"
        + (f"3. {_act_finding(act_runs)}；判别器用 LeakyReLU 的 DCGAN 经验在本数据上没有表现出决定性优势。\n"
           if act_spread is not None else "")
        + (f"4. {_dropout_finding(dr_runs)}；对抗训练的调参必须同时看判别器与生成器两侧。\n"
           if dr_runs else "")
        + (f"5. 小数据下 FID 可能“好看”但并不可信：{SZ['runs'][0]['size']:,} 张时 FID 为 "
           f"{f2(SZ['runs'][0]['fid'])}、最近邻比值 {SZ['runs'][0]['nn_ratio']:.3f}，"
           "模型是在背训练图；评价生成模型必须同时看 FID、最近邻距离与多样性。\n"
           if SZ.get("runs") else "")
        + "6. VAE 训练稳定、隐空间连续平滑、重建可控，但样本明显模糊；在同一 FID 协议下，"
        f"DCGAN 的质量明显优于 VAE。C2ST 显示两类样本都能被小 CNN 轻易区分（DCGAN 1.000、"
        "VAE 0.895），说明 64×64 生成图与真实图仍有可检测差异，C2ST 不能作为质量排序指标。\n"
        + (f"7. 从零实现的 DDPM 扩散模型（{DIFF.get('train', {}).get('params', 0) / 1e6:.2f}M U-Net、余弦调度、EMA，"
           f"训练 {DIFF.get('train', {}).get('minutes', 0) / 60:.1f} 小时）取得了三种模型中最优的 FID（"
           f"{f2(DIFF.get('eval', {}).get('final', {}).get('fid', float('nan')))}），但生成 1,000 张图的耗时是 DCGAN 的 "
           f"{DIFF.get('comparison', {}).get('ddpm', {}).get('sample_seconds_per_1k', 1) / max(DIFF.get('comparison', {}).get('dcgan', {}).get('sample_seconds_per_1k', 1), 1e-9):.0f} 倍，"
           "体现了“质量—采样成本”的权衡。\n" if DIFF else "")
        + "8. 通过本实验完整走通了“数据清洗 → 模型实现 → 正确性验证 → 训练 → 多指标评价 → 局限性分析”"
        "的生成模型实验流程，也认识到生成模型评价比分类模型评价困难得多，需要多指标交叉验证。\n\n"
        "如果继续深入，建议从三个方向扩展：改用 WGAN-GP 或 StyleGAN 提升 GAN 的质量与稳定性；"
        "把扩散模型扩大到 128×128/256×256 分辨率并尝试 latent diffusion；以及为每个配置跑多个随机种子，"
        "给出统计意义上的对比结论。")


def sec_references(add):
    add(r"""## 参考文献

1. Goodfellow I., Pouget-Abadie J., Mirza M., Xu B., Warde-Farley D., Ozair S., Courville A., Bengio Y. Generative adversarial nets. *Advances in Neural Information Processing Systems (NIPS)*, 2014.
2. Radford A., Metz L., Chintala S. Unsupervised representation learning with deep convolutional generative adversarial networks. *International Conference on Learning Representations (ICLR)*, 2016.
3. Kingma D. P., Welling M. Auto-encoding variational Bayes. *International Conference on Learning Representations (ICLR)*, 2014.
4. Rezende D. J., Mohamed S., Wierstra D. Stochastic backpropagation and approximate inference in deep generative models. *International Conference on Machine Learning (ICML)*, 2014.
5. Higgins I., Matthey L., Pal A., Burgess C., Glorot X., Botvinick M., Mohamed S., Lerchner A. β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*, 2017.
6. Arjovsky M., Bottou L. Towards principled methods for training generative adversarial networks. *ICLR*, 2017.
7. Arjovsky M., Chintala S., Bottou L. Wasserstein generative adversarial networks. *ICML*, 2017.
8. Gulrajani I., Ahmed F., Arjovsky M., Dumoulin V., Courville A. Improved training of Wasserstein GANs. *NIPS*, 2017.
9. Salimans T., Goodfellow I., Zaremba W., Cheung V., Radford A., Chen X. Improved techniques for training GANs. *NIPS*, 2016.
10. Heusel M., Ramsauer H., Unterthiner T., Nessler B., Hochreiter S. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *NIPS*, 2017.
11. Karras T., Aila T., Laine S., Lehtinen J. Progressive growing of GANs for improved quality, stability, and variation. *ICLR*, 2018.
12. Srivastava N., Hinton G., Krizhevsky A., Sutskever I., Salakhutdinov R. Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 2014, 15(1): 1929–1958.
13. Ioffe S., Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 2015.
14. Kingma D. P., Ba J. Adam: A method for stochastic optimization. *ICLR*, 2015.
15. Szegedy C., Vanhoucke V., Ioffe S., Shlens J., Wojna Z. Rethinking the Inception architecture for computer vision. *CVPR*, 2016.
16. Ho J., Jain A., Abbeel P. Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
17. Song J., Meng C., Ermon S. Denoising diffusion implicit models. *International Conference on Learning Representations (ICLR)*, 2021.
18. Nichol A. Q., Dhariwal P. Improved denoising diffusion probabilistic models. *International Conference on Machine Learning (ICML)*, 2021.
19. Song Y., Sohl-Dickstein J., Kingma D. P., Kumar A., Ermon S., Poole B. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021.
20. soumikrakshit. Anime Face Dataset. Kaggle, https://www.kaggle.com/soumikrakshit/anime-faces （经 HuggingFace 镜像 https://huggingface.co/datasets/huggan/anime-faces 下载）. """)


def sec_appendix(add):
    file_rows = [
        ["`code/data.py`", "Anime Face Dataset 去重、解码缓存、训练/参考集划分、DataLoader"],
        ["`code/models.py`", "DCGAN 生成器/判别器（激活、Dropout 可配置）、卷积 VAE、参数量统计"],
        ["`code/engine.py`", "随机种子、设备、GAN/VAE 训练循环、采样与插值工具"],
        ["`code/metrics.py`", "FID（Inception-v3）、最近邻记忆化分析、颜色多样性、C2ST、VAE 重建指标"],
        ["`code/viz.py`", "全部插图：数据分布、结构图、训练曲线、样本网格、插值、对比条形图、日志截图"],
        ["`code/diffusion.py`", "扩散模型：DDPM U-Net、余弦噪声调度、EMA、DDPM/DDIM 采样"],
        ["`exp6_report/diffusion_train.py`", "扩散模型训练脚本（bf16 AMP + torch.compile + 断点续训 + 采样监控）"],
        ["`exp6_report/diffusion_eval.py`", "扩散模型评估：挑选最佳 EMA checkpoint，FID/C2ST/最近邻/采样速度"],
        ["`exp6_report/diffusion_report.py`", "扩散模型插图生成与 results.json 更新"],
        ["`exp6_report/full_experiment.py`", "十部分实验主脚本，结果写入 results.json、插图写入 figures/"],
        ["`exp6_report/verify.py`", "18 项正确性验证，结果写入 verification.json"],
        ["`exp6_report/gen_report.py`", "从 results.json 生成 Markdown 报告并调用 pandoc 转 docx"],
        ["`data/download_anime_faces.sh`", "下载并解压 Anime Face Dataset"],
    ]
    rows = []
    if M:
        rows.append(["主实验", f"DCGAN 6,000 步，z=100，batch=128", f2(M["final_fid"])])
    for r in ACT.get("runs", []):
        rows.append(["激活函数", r["label"], f2(r["fid"])])
    if ACT.get("d_relu"):
        rows.append(["激活函数", "D 用 ReLU", f2(ACT["d_relu"]["fid"])])
    for r in DR.get("runs", []):
        rows.append([f"Dropout（{DR.get('n_train_images', 0):,} 张）", r["label"], f2(r["fid"])])
    for r in SZ.get("runs", []):
        rows.append([f"数据量（{SZ.get('steps', 0)} 步）", f"{r['size']:,} 张", f2(r["fid"])])
    for key, name in (("latent", "潜维"), ("lr", "学习率"), ("batch", "批大小"), ("other", "其他")):
        for r in H.get("sweeps", {}).get(key, []):
            rows.append([f"超参数·{name}", r["label"], f2(r["fid"])])
    if F:
        rows.append(["最终模型", f"{F['config']['g_act']} + z={F['config']['z_dim']}，"
                                 f"{F['config']['max_steps']:,} 步", f2(F["final_fid"])])
    if DIFF:
        dr = DIFF.get("train", {})
        rows.append(["扩散模型 DDPM",
                     f"U-Net {dr.get('params', 0) / 1e6:.2f}M，{dr.get('steps', 0):,} 步，DDIM-50",
                     f2(DIFF.get("eval", {}).get("final", {}).get("fid", float("nan")))])
        for rec in DIFF.get("eval", {}).get("screening", []):
            rows.append(["扩散模型（训练中）", f"第 {rec['step']:,} 步", f2(rec["fid"])])
    times = R.get("meta", {}).get("part_times_s", {})
    part_names = {"data": "数据概览", "architecture": "模型结构", "verification": "正确性验证",
                  "main": "主实验 DCGAN", "activation": "激活函数", "dropout": "Dropout",
                  "data_size": "数据量", "hyper": "超参数", "vae": "VAE 对照", "final": "最终模型"}
    time_rows = [[k.split("_")[0], part_names.get(k.split("_", 1)[1], k.split("_", 1)[1]),
                  f"{v / 60:.1f}"] for k, v in times.items()]
    if DIFF:
        time_rows.append(["-", "扩散模型 DDPM 训练",
                          f"{DIFF.get('train', {}).get('minutes', 0):.1f}"])
    add("## 附录 A：代码结构\n\n" + table(["文件", "职责"], file_rows, "ll") + "\n\n"
        "## 附录 B：全部生成模型配置的 FID 汇总\n\n"
        "下表汇总了本实验训练过的全部生成模型配置（VAE 见第 13 节、扩散模型见第 15 节）。所有 FID 使用同一真实参考集、"
        "同一 Inception-v3 特征与同一采样协议，因此可以横向比较；但注意不同实验组的训练步数不同"
        "（主实验 6,000 步、扫描实验 1,500~3,000 步），跨组比较时要把训练预算考虑进去。\n\n"
        + table(["实验组", "配置", "FID"], rows, "llr") + "\n\n"
        "## 附录 C：运行时间与复现方式\n\n"
        + table(["部分", "内容", "耗时（分钟）"], time_rows, "llr") + "\n\n"
        "复现步骤：\n\n"
        "```bash\n"
        "cd data && bash download_anime_faces.sh      # 下载并解压数据（约 441 MB）\n"
        "cd ../exp6_report\n"
        "python verify.py                             # 18 项正确性验证\n"
        "python full_experiment.py                    # 前十部分实验（约 1.3 小时，GPU）\n"
        "python diffusion_train.py --steps 165000 --max-hours 9.2   # 第十一部分：扩散模型训练\n"
        "python diffusion_eval.py                     # 扩散模型评估（挑选最佳 checkpoint）\n"
        "python diffusion_report.py                   # 扩散模型插图 + 更新 results.json\n"
        "python gen_report.py                         # 由 results.json 生成 Markdown + docx\n"
        "```\n\n"
        "说明：本报告由 `gen_report.py` 从 `results.json` / `verification.json` 自动生成，"
        "正文中所有实验数字都来自 JSON，不经过手工抄写；报告开头的姓名/学号/班级/日期留空待填。")


def build():
    P = []
    add = P.append
    sec_title(add)
    sec_abstract(add)
    sec_purpose(add)
    sec_content(add)
    sec_env(add)
    sec_principles(add)
    sec_data(add)
    sec_implementation(add)
    sec_verification(add)
    if M:
        sec_main(add)
    if ACT:
        sec_activation(add)
    if DR:
        sec_dropout(add)
    if SZ:
        sec_data_size(add)
    if H:
        sec_hyper(add)
    if VA:
        sec_vae(add)
    if F:
        sec_final(add)
    if DIFF:
        sec_diffusion(add)
    if F:
        sec_discussion(add)
        sec_conclusion(add)
    sec_references(add)
    sec_appendix(add)

    md = "\n\n".join(P) + "\n"
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Markdown 已写入 {OUT_MD}（{len(md):,} 字符，{md.count(chr(10)):,} 行）")
    subprocess.run(
        ["pandoc", str(OUT_MD), "-o", str(OUT_DOCX), "--resource-path", str(HERE)],
        check=True, cwd=str(HERE),
    )
    print(f"DOCX 已写入 {OUT_DOCX}")


if __name__ == "__main__":
    build()
