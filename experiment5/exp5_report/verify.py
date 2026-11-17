#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正确性验证：在正式实验开始前，确认数据、模型、反向传播与训练链路没有实现错误。

六项检查：
    1. 参数量核对：PyTorch 统计值与手算值一致
    2. 输出尺寸核对：每层输出形状与卷积输出公式一致
    3. 初始损失：随机初始化下平均交叉熵应接近 ln(10)
    4. 数值梯度校验：中心差分与 PyTorch 自动求导的梯度对比
    5. 单批过拟合：一个小批次反复训练，损失应能降到接近 0
    6. 旋转函数校验：0° 旋转不改变图像 + 其它角度形状/范围正确

结果写入 exp5_report/verification.json，供报告第 5.5 节引用。
运行：cd exp5_report && python verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

from data import load_raw_dataset, rotate_batch  # noqa: E402
from models import LeNet5, count_parameters  # noqa: E402

OUT = HERE / "verification.json"
torch.manual_seed(0)
np.random.seed(0)


def check_parameters(model: nn.Module) -> dict:
    """手算参数量并与 PyTorch 统计值核对。"""
    # conv1: 5*5*1*6+6=156; conv2: 5*5*6*16+16=2416
    # fc1: 256*120+120=30840; fc2: 120*84+84=10164; fc3: 84*10+10=850
    hand = 156 + 2416 + 30840 + 10164 + 850
    real = count_parameters(model)
    return {"hand_computed": hand, "pytorch_count": real, "match": hand == real}


def check_shapes(model: nn.Module) -> dict:
    """用 forward hook 抓取每层输出形状，与公式核对。"""
    x = torch.randn(2, 1, 28, 28)
    acts = []

    def hook(mod, inp, out):
        acts.append((type(mod).__name__, tuple(out.shape)))

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    expected = [
        ("Conv2d", (2, 6, 24, 24)),
        ("MaxPool2d", (2, 6, 12, 12)),
        ("Conv2d", (2, 16, 8, 8)),
        ("MaxPool2d", (2, 16, 4, 4)),
        ("Linear", (2, 120)),
        ("Linear", (2, 84)),
        ("Linear", (2, 10)),
    ]
    mismatch = [f"{a} != {e}" for a, e in zip(acts, expected) if a[1] != e[1]]
    return {
        "actual": [f"{name}{shape}" for name, shape in acts],
        "expected": [f"{name}{shape}" for name, shape in expected],
        "match": len(mismatch) == 0,
        "mismatch": mismatch,
    }


def check_initial_loss() -> dict:
    """未训练模型在 512 张训练图上的平均交叉熵应接近 ln(10)=2.302585。"""
    torch.manual_seed(1)
    model = LeNet5()
    Xtr, ytr, _, _ = load_raw_dataset("mnist")
    x, y = Xtr[:512], ytr[:512]
    with torch.no_grad():
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y).item()
    return {"measured": loss, "expected": float(np.log(10.0)),
            "abs_error": abs(loss - float(np.log(10.0)))}


def check_numeric_gradient(eps: float = 1e-5) -> dict:
    """中心差分数值梯度 vs 自动求导。

    为了精度，把模型和数据都转成 float64，并在 CPU 上计算。
    随机挑 8 个参数元素（卷积层与全连接层各若干），逐个做 ±eps 扰动。
    """
    torch.manual_seed(2)
    model = LeNet5().double()
    Xtr, ytr, _, _ = load_raw_dataset("mnist")
    x = Xtr[:4].double()
    y = ytr[:4]
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    rng = np.random.default_rng(0)
    rows = []
    picked = 0
    for name, p in params:  # 依次从 conv1, conv2, fc1, fc2, fc3 的权重/偏置里取参数
        if picked >= 10:
            break
        for _ in range(2):
            if picked >= 10:
                break
            idx = tuple(int(rng.integers(0, s)) for s in p.shape)
            with torch.no_grad():
                orig = p[idx].item()
                p[idx] = orig + eps
                lp = criterion(model(x), y).item()
                p[idx] = orig - eps
                lm = criterion(model(x), y).item()
                p[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = p.grad[idx].item()
            if max(abs(numeric), abs(analytic)) < 1e-8:
                continue  # 梯度接近 0 的参数相对误差没有意义
            picked += 1
            denom = max(abs(numeric), abs(analytic), 1e-12)
            rows.append({
                "param": name, "index": list(idx),
                "numeric": numeric, "analytic": analytic,
                "abs_error": abs(numeric - analytic),
                "rel_error": abs(numeric - analytic) / denom,
            })
    worst = max(rows, key=lambda r: r["rel_error"])
    return {"eps": eps, "n_checked": len(rows), "max_rel_error": worst["rel_error"],
            "rows": rows}


def check_single_batch_overfit(steps: int = 400) -> dict:
    """固定一个 64 张图的小批次，反复训练，损失应降到接近 0。"""
    torch.manual_seed(3)
    model = LeNet5()
    Xtr, ytr, _, _ = load_raw_dataset("mnist")
    g = torch.Generator().manual_seed(3)
    idx = torch.randperm(len(Xtr), generator=g)[:64]
    x, y = Xtr[idx], ytr[idx]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    losses = []
    for step in range(steps):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (model(x).argmax(dim=1) == y).float().mean().item()
    return {"initial_loss": losses[0], "final_loss": losses[-1],
            "final_acc": acc, "steps": steps}


def check_rotation() -> dict:
    """旋转 0° 应与原图完全一致；±15° 后形状与取值范围不变、内容发生变化。"""
    Xtr, _, _, _ = load_raw_dataset("mnist")
    x = Xtr[:16]
    x0 = rotate_batch(x, 0)
    x15 = rotate_batch(x, 15)
    return {
        "zero_angle_identical": bool(torch.allclose(x, x0)),
        "shape_preserved": tuple(x15.shape) == tuple(x.shape),
        "range_ok": bool((x15 >= 0).all() and (x15 <= 1).all()),
        "changed_at_15": bool(not torch.allclose(x, x15)),
        "max_abs_diff_at_0": float((x - x0).abs().max()),
    }


def main():
    model = LeNet5()
    results = {
        "parameters": check_parameters(model),
        "shapes": check_shapes(model),
        "initial_loss": check_initial_loss(),
        "numeric_gradient": check_numeric_gradient(),
        "single_batch_overfit": check_single_batch_overfit(),
        "rotation": check_rotation(),
    }
    results["all_passed"] = bool(
        results["parameters"]["match"]
        and results["shapes"]["match"]
        and results["initial_loss"]["abs_error"] < 0.02
        and results["numeric_gradient"]["max_rel_error"] < 1e-4
        and results["single_batch_overfit"]["final_loss"] < 0.05
        and results["rotation"]["zero_angle_identical"]
        and results["rotation"]["range_ok"]
    )
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 70)
    print("正确性验证结果")
    print("=" * 70)
    print(f"1. 参数量：手算 {results['parameters']['hand_computed']:,} / "
          f"PyTorch {results['parameters']['pytorch_count']:,}  "
          f"{'✓' if results['parameters']['match'] else '✗'}")
    print(f"2. 输出尺寸：{'✓' if results['shapes']['match'] else '✗'}  "
          f"{results['shapes']['actual']}")
    print(f"3. 初始损失：{results['initial_loss']['measured']:.6f}（期望 ln10 = "
          f"{results['initial_loss']['expected']:.6f}，误差 {results['initial_loss']['abs_error']:.2e}）")
    print(f"4. 数值梯度：检查 {results['numeric_gradient']['n_checked']} 个参数，"
          f"最大相对误差 {results['numeric_gradient']['max_rel_error']:.2e}")
    print(f"5. 单批过拟合：初始损失 {results['single_batch_overfit']['initial_loss']:.4f} → "
          f"最终 {results['single_batch_overfit']['final_loss']:.2e}，"
          f"准确率 {results['single_batch_overfit']['final_acc'] * 100:.1f}%")
    print(f"6. 旋转函数：0° 不变 {'✓' if results['rotation']['zero_angle_identical'] else '✗'}，"
          f"形状保持 {'✓' if results['rotation']['shape_preserved'] else '✗'}，"
          f"取值范围 {'✓' if results['rotation']['range_ok'] else '✗'}")
    print("=" * 70)
    print(f"全部通过：{'是' if results['all_passed'] else '否'}；结果写入 {OUT}")
    if not results["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
