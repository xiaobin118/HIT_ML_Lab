"""第 4 步：正式训练循环。

目标：把前 3 步的模型和数据接起来，写出 train_one_epoch / evaluate / fit，
在 GPU 上训练 LeNet-5，并记录每轮的训练/验证损失与准确率。

规则
----
1. 只修改标了 `TODO` 的地方，其它代码不要动。
2. 写完后运行 `python step04_train.py`，全部 6 项自检必须 PASS。
3. 不要打开 `../code/engine.py`。先自己写，卡住 20 分钟以上再来问。
4. 讲义在 `../docs/04_训练循环.md`。
"""

from __future__ import annotations

import copy
import random
import traceback

import numpy as np
import torch
import torch.nn as nn

from step02_lenet import LeNet5
from step03_data import load_mnist, make_loader


# ===========================================================================
# TODO 1：固定随机种子
# ===========================================================================
def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch / CUDA 的随机种子。

    要求：连续两次 `set_seed(0)` 之后各自新建的模型，参数应完全相同；
          用 seed=1 新建的模型应与 seed=0 的不同。
    提示：random.seed、np.random.seed、torch.manual_seed、torch.cuda.manual_seed_all。
    """
    # TODO 1：写 4~6 行
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# TODO 2：选择设备
# ===========================================================================
def get_device() -> torch.device:
    """有 CUDA 时返回 torch.device("cuda")，否则返回 torch.device("cpu")。

    提示：torch.cuda.is_available()。
    """
    # TODO 2：写 2~3 行
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ===========================================================================
# TODO 3：优化器工厂
# ===========================================================================
def make_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """按名字构造优化器。

    支持的名字：
        "sgd"      -> torch.optim.SGD(..., lr=lr, weight_decay=weight_decay)
        "momentum" -> SGD(..., lr=lr, momentum=0.9, weight_decay=weight_decay)
        "adam"     -> torch.optim.Adam(...)
        "adamw"    -> torch.optim.AdamW(...)
    其它名字抛 ValueError。
    """
    # TODO 3：写 8~12 行
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知优化器名字：{name!r}")


# ===========================================================================
# TODO 4：训练一轮
# ===========================================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在 loader 上训练一个 epoch，返回 (平均损失, 准确率)。

    要求
    ----
    1. 先 `model.train()`；
    2. 每个 batch：数据搬到 device → `optimizer.zero_grad(set_to_none=True)`
       → 前向 → 算损失 → `loss.backward()` → `optimizer.step()`；
    3. 用加权平均累计损失（`loss.item() * batch_size` 再除以总样本数），
       准确率用 `logits.argmax(dim=1) == y` 统计；
    4. 返回两个 Python float，准确率在 [0,1]。

    注意：不要把每个 batch 的 loss 张量直接存进 list——那会保留计算图、吃显存；
          累计前先 `.item()`。
    """
    # TODO 4：写 20~30 行
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ===========================================================================
# TODO 5：评估
# ===========================================================================
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (平均损失, 准确率)；不更新参数。

    要求
    ----
    1. `model.eval()`；
    2. 用 `with torch.no_grad():` 包住整个循环；
    3. 不调用 backward、不调用 optimizer.step；
    4. 返回两个 Python float。

    验收：调用前后 model 的参数完全不变，且所有参数的 `.grad` 仍为 None。
    """
    # TODO 5：写 15~20 行
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
    return avg_loss, accuracy


# ===========================================================================
# TODO 6：完整训练流程
# ===========================================================================
def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 2,
    lr: float = 1e-3,
    optimizer: str = "adam",
    device: torch.device | None = None,
) -> dict:
    """训练 epochs 轮，返回 history 字典。

    要求
    ----
    1. device 为 None 时调用 get_device()；
    2. 用 make_optimizer 构造优化器，交叉熵用 nn.CrossEntropyLoss()；
    3. 每轮：train_one_epoch → evaluate → 把四个指标追加到 history；
    4. 记录验证准确率最高的轮次：history["best_val_acc"]、history["best_epoch"]；
    5. 训练结束后把模型参数**恢复到验证准确率最高的那一轮**
       （提示：`copy.deepcopy(model.state_dict())` 保存，训练完 `model.load_state_dict(...)`）；
    6. history 的 key 必须是：
       "train_loss", "train_acc", "val_loss", "val_acc"（四个长度 = epochs 的列表），
       以及 "best_val_acc"、"best_epoch"。

    验收：训练后 `evaluate(model, val_loader, ...)` 的准确率应等于 history["best_val_acc"]。
    """
    # TODO 6：写 25~35 行
    if device is None:
        device = get_device()

    model.to(device)
    optimizer = make_optimizer(model, optimizer, lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }

    best_model_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0


    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch

    model.load_state_dict(best_model_state)
    return history


# ===========================================================================
# 自检代码：不要修改下面的内容
# ===========================================================================


CHECKS: list[tuple[str, callable]] = []
_CACHE: dict = {}


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


def _small_data():
    """取 2000 张训练 + 1000 张验证的随机子集（CPU、几秒内能跑完）。"""
    if "small" not in _CACHE:
        Xtr, ytr, _, _ = load_mnist()
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ytr), generator=g)
        tr_idx, va_idx = idx[:2000], idx[2000:3000]
        _CACHE["small"] = (
            Xtr[tr_idx], ytr[tr_idx],
            Xtr[va_idx], ytr[va_idx],
        )
    Xtr, ytr, Xva, yva = _CACHE["small"]
    return (
        make_loader(Xtr, ytr, batch_size=64, shuffle=True, seed=0),
        make_loader(Xva, yva, batch_size=256, shuffle=False),
    )


@check("TODO 1  set_seed：同 seed 参数相同、不同 seed 不同")
def _test_set_seed() -> None:
    set_seed(0)
    m1 = LeNet5()
    set_seed(0)
    m2 = LeNet5()
    same = all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    assert same, "同一个 seed 两次初始化应得到完全相同的参数"
    set_seed(1)
    m3 = LeNet5()
    diff = any(not torch.equal(a, b) for a, b in zip(m1.parameters(), m3.parameters()))
    assert diff, "不同 seed 初始化的参数应该不同"


@check("TODO 2  get_device：cuda 可用时返回 cuda，否则 cpu")
def _test_get_device() -> None:
    dev = get_device()
    assert isinstance(dev, torch.device), f"应返回 torch.device，实际 {type(dev)}"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected, f"device 应为 {expected}，实际 {dev}"


@check("TODO 3  make_optimizer：四种优化器、学习率、未知名字报错")
def _test_make_optimizer() -> None:
    model = nn.Linear(4, 2)
    cases = [
        ("sgd", torch.optim.SGD),
        ("momentum", torch.optim.SGD),
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
    ]
    for name, cls in cases:
        opt = make_optimizer(model, name, lr=0.05, weight_decay=1e-2)
        assert isinstance(opt, cls), f"{name} 应返回 {cls.__name__}，实际 {type(opt).__name__}"
        assert abs(opt.param_groups[0]["lr"] - 0.05) < 1e-12, f"{name} 的 lr 没设对"
        assert abs(opt.param_groups[0]["weight_decay"] - 1e-2) < 1e-12, f"{name} 的 weight_decay 没设对"
    mom = make_optimizer(model, "momentum", lr=0.1)
    assert abs(mom.param_groups[0]["momentum"] - 0.9) < 1e-12, "momentum 优化器的 momentum 应为 0.9"
    try:
        make_optimizer(model, "lamb", lr=1e-3)
    except ValueError:
        pass
    else:
        raise AssertionError("未知优化器名字应抛 ValueError")


@check("TODO 4  train_one_epoch：返回 float、损失下降、参数被更新")
def _test_train_epoch() -> None:
    set_seed(0)
    train_loader, _ = _small_data()
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]

    loss1, acc1 = train_one_epoch(model, train_loader, opt, criterion, torch.device("cpu"))
    loss2, acc2 = train_one_epoch(model, train_loader, opt, criterion, torch.device("cpu"))
    for v, name in [(loss1, "loss1"), (acc1, "acc1"), (loss2, "loss2"), (acc2, "acc2")]:
        assert isinstance(v, float), f"{name} 应返回 Python float，实际 {type(v)}"
    assert 0.0 <= acc1 <= 1.0 and 0.0 <= acc2 <= 1.0, "准确率应在 [0,1]"
    assert loss2 < loss1, f"训练两轮后损失应下降：{loss1:.4f} -> {loss2:.4f}"
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert changed, "训练后模型参数应发生变化"


@check("TODO 5  evaluate：返回 float、参数不变、不产生梯度")
def _test_evaluate() -> None:
    set_seed(0)
    train_loader, val_loader = _small_data()
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_one_epoch(model, train_loader, opt, criterion, torch.device("cpu"))
    model.zero_grad(set_to_none=True)   # 清掉训练留下的梯度，专门检查 evaluate 不会产生新梯度

    before = [p.detach().clone() for p in model.parameters()]
    val_loss, val_acc = evaluate(model, val_loader, criterion, torch.device("cpu"))
    assert isinstance(val_loss, float) and isinstance(val_acc, float), "应返回两个 Python float"
    assert val_loss > 0 and 0.0 <= val_acc <= 1.0, f"损失/准确率不合理：{val_loss}, {val_acc}"
    unchanged = all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert unchanged, "评估不应该修改模型参数"
    assert all(p.grad is None for p in model.parameters()), "评估不应该产生梯度（参数 .grad 应为 None）"

    # 手动用同一模型算一遍准确率，验证 evaluate 的统计没错
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.numel()
    assert abs(correct / total - val_acc) < 1e-6, \
        f"evaluate 的准确率 {val_acc:.4f} 与手动统计 {correct / total:.4f} 不一致"


@check("TODO 6  fit：history 完整、损失下降、best 参数被恢复")
def _test_fit() -> None:
    set_seed(0)
    train_loader, val_loader = _small_data()
    model = LeNet5()
    history = fit(model, train_loader, val_loader, epochs=2, lr=1e-3,
                  optimizer="adam", device=torch.device("cpu"))
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert key in history, f"history 缺少 key: {key}"
        assert isinstance(history[key], list) and len(history[key]) == 2, \
            f"history[{key!r}] 应为长度 2 的列表，实际 {history.get(key)!r}"
    assert history["train_loss"][1] < history["train_loss"][0], "训练损失应下降"
    best = max(history["val_acc"])
    assert abs(history["best_val_acc"] - best) < 1e-12, "best_val_acc 应等于 val_acc 的最大值"
    best_epoch = history["val_acc"].index(best) + 1
    assert history["best_epoch"] == best_epoch, f"best_epoch 应为 {best_epoch}，实际 {history['best_epoch']}"

    criterion = nn.CrossEntropyLoss()
    _, final_acc = evaluate(model, val_loader, criterion, torch.device("cpu"))
    assert abs(final_acc - history["best_val_acc"]) < 1e-6, \
        f"训练后模型应恢复到验证最优轮：evaluate={final_acc:.4f}, best={history['best_val_acc']:.4f}"


def main() -> None:
    print("=" * 74)
    print("第 4 步自检：正式训练循环")
    print("=" * 74)
    n_pass = 0
    failures = []
    for name, fn in CHECKS:
        try:
            fn()
        except NotImplementedError as e:
            print(f"❌ {name}\n     {e}")
            failures.append(name)
        except AssertionError as e:
            print(f"❌ {name}\n     {e}")
            failures.append(name)
        except Exception:
            print(f"❌ {name}\n     {traceback.format_exc(limit=2).strip()}")
            failures.append(name)
        else:
            print(f"✅ {name}")
            n_pass += 1
    print("-" * 74)
    print(f"结果：{n_pass}/{len(CHECKS)} PASS")
    if failures:
        print("未通过：")
        for name in failures:
            print("  -", name)
        print("\n按上面的报错逐个修改 TODO。改完重新运行本脚本。")
        raise SystemExit(1)
    print("\n全部通过！把输出、fit 的代码、思考题答案发给老师，")
    print("就可以进入第 5 步：正确性验证与训练技巧。")


if __name__ == "__main__":
    main()
