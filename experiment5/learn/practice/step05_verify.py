"""第 5 步：正确性验证与训练技巧。

目标：
1. 学会在"相信实验结果"之前先做 4 个 sanity check：初始损失、单批过拟合、数值梯度、梯度量级；
2. 把 BatchNorm、Dropout 正确接进 LeNet-5，并理解它们的参数量和 train/eval 行为；
3. 用第一层卷积梯度范数量化"为什么 Sigmoid + SGD 学不动"。

规则
----
1. 只修改标了 `TODO` 的地方，其它代码不要动。
2. 写完后运行 `python step05_verify.py`，全部 6 项自检必须 PASS。
3. 不要打开 `../code/` 或 `../exp5_report/` 里的成品实现。
4. 讲义在 `../docs/05_正确性验证与训练技巧.md`。

注意：本练习全部在 CPU 上跑（数据量很小），真实训练在第 6 步用 GPU。
"""

from __future__ import annotations

import copy
import traceback

import torch
import torch.nn as nn

from step02_lenet import make_activation
from step03_data import load_mnist


# ===========================================================================
# TODO 1：带 BN / Dropout 的 LeNet-5
# ===========================================================================
class LeNetReg(nn.Module):
    """LeNet-5 + 可选的 BatchNorm 和 Dropout。

    结构（和实验五的 LeNet5 完全一致，只是多了两个可选部件）：
        features:
            Conv2d(1, 6, 5)  -> [BatchNorm2d(6)] -> 激活 -> MaxPool2d(2)
            Conv2d(6, 16, 5) -> [BatchNorm2d(16)] -> 激活 -> MaxPool2d(2)
        classifier:
            Flatten
            Linear(256, 120) -> 激活 -> [Dropout(p)]
            Linear(120, 84)  -> 激活 -> [Dropout(p)]
            Linear(84, 10)

    要求
    ----
    1. use_bn=True 时，在每个卷积层之后、激活之前插入 `nn.BatchNorm2d(out_ch)`；
    2. dropout_fc>0 时，在分类器每个激活之后插入 `nn.Dropout(dropout_fc)`；
    3. activation 参数交给 step02 的 `make_activation`；
    4. 默认配置（无 BN、无 Dropout）参数量必须等于 44,426；
       加 BN 后应为 44,470（BN 每个通道 2 个参数：weight + bias）；
       Dropout 没有参数，参数量不变。

    提示：用 `nn.Sequential` 拼 features 和 classifier；可以用一个内部函数
          生成每个卷积块，避免重复代码。
    """

    def __init__(self, use_bn: bool = False, dropout_fc: float = 0.0,
                 activation: str = "relu") -> None:
        # TODO 1：写 25~35 行
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1：写 1 行
        raise NotImplementedError("TODO 1：还没有实现 LeNetReg.forward")


# ===========================================================================
# TODO 2：初始损失
# ===========================================================================
def initial_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """返回模型**未训练**时在 (X, y) 上的平均交叉熵（Python float）。

    要求：用 `with torch.no_grad()`；不要调用 backward。
    验收：10 类均衡数据、随机初始化下应接近 ln(10) ≈ 2.302585（误差 < 0.05）。
    提示：nn.CrossEntropyLoss()(model(X), y).item()。
    """
    # TODO 2：写 3~5 行
    raise NotImplementedError("TODO 2：还没有实现 initial_loss")


# ===========================================================================
# TODO 3：第一层卷积的梯度范数
# ===========================================================================
def first_conv_grad_norm(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """返回损失对**第一层卷积权重**的梯度 L2 范数（Python float）。

    要求
    ----
    1. 用 `register_hook` 或 `register_full_backward_hook` 抓第一层卷积权重的梯度；
    2. 先 `model.zero_grad(set_to_none=True)`，再算 loss、`backward()`；
    3. backward 之后要 `remove()` 掉 hook（否则会一直挂着）；
    4. 返回 float，不要返回张量。

    提示：
        conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)][0]
        norms = []
        h = conv.weight.register_hook(lambda g: norms.append(float(g.norm())))
    """
    # TODO 3：写 8~12 行
    raise NotImplementedError("TODO 3：还没有实现 first_conv_grad_norm")


# ===========================================================================
# TODO 4：单批过拟合
# ===========================================================================
def overfit_single_batch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 1e-2,
) -> tuple[float, float]:
    """用 Adam 在一个固定小 batch 上反复训练，返回 (最终损失, 最终准确率)。

    要求
    ----
    1. 优化器用 `torch.optim.Adam(model.parameters(), lr=lr)`，损失用交叉熵；
    2. 每一步：zero_grad(set_to_none=True) → 前向 → 损失 → backward → step；
    3. 训练结束后在同一个 batch 上评估：返回 `loss.item()` 和准确率（Python float）；
    4. 验收：64 个样本训练 300 步，损失应降到接近 0（< 0.05）、准确率 100%。

    这个检查能证明：前向、损失、反向传播、优化器更新整条链路是通的。
    如果损失一直不降，说明链路里某处有 bug，先别急着做实验。
    """
    # TODO 4：写 15~20 行
    raise NotImplementedError("TODO 4：还没有实现 overfit_single_batch")


# ===========================================================================
# TODO 5：数值梯度校验
# ===========================================================================
def numeric_gradient_check(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
) -> float:
    """用中心差分数值梯度校验 autograd 的结果，返回最大相对误差（Python float）。

    做法
    ----
    1. `m = copy.deepcopy(model).double()`，`Xd = X.double()`（标签 y 保持 int64！）；
    2. 算一次 loss 并 backward，拿到参数 `.grad`；
    3. 取**第一个权重张量**（conv1.weight）和**最后一个权重张量**（fc3.weight），
       各随机选 4 个元素；
    4. 对每个元素：保存原值 → 分别 +eps / -eps 前向算损失 → 中心差分
       (L(+eps) - L(-eps)) / (2*eps) → 恢复原值；
    5. 相对误差 = |数值梯度 - 解析梯度| / max(|两者|, 1e-12)，返回最大值。

    要求
    ----
    - 必须转 float64：float32 下中心差分的相对误差会达到 1 量级，完全不可用；
    - 扰动/恢复参数时用 `torch.no_grad()` 包住；
    - 梯度接近 0 的元素跳过（相对误差没有意义）。
    验收：最大相对误差 < 1e-4。
    """
    # TODO 5：写 25~35 行
    raise NotImplementedError("TODO 5：还没有实现 numeric_gradient_check")


# ===========================================================================
# TODO 6：比较 ReLU 与 Sigmoid 的第一层梯度
# ===========================================================================
def compare_activation_gradients(X: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """分别构造 activation 为 relu / sigmoid 的 LeNetReg，返回它们第一层卷积的梯度范数。

    返回：{"relu": float, "sigmoid": float}

    要求
    ----
    1. 两个模型都用 `LeNetReg(activation=...)`（其余默认）；
    2. 用同一个 batch (X, y)、分别测 first_conv_grad_norm；
    3. 验收：在 MNIST 的 64 个样本上，sigmoid 的梯度范数应比 relu 小 20 倍以上。
       这个差距就是"Sigmoid + 固定学习率 SGD 学不动"的直接原因。
    """
    # TODO 6：写 8~12 行
    raise NotImplementedError("TODO 6：还没有实现 compare_activation_gradients")


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


def _data():
    if "data" not in _CACHE:
        _CACHE["data"] = load_mnist()
    return _CACHE["data"]


@check("TODO 1  LeNetReg：前向形状、参数量 44,426 / 44,470、BN/Dropout 数量")
def _test_lenet_reg() -> None:
    Xtr, _, _, _ = _data()
    model = LeNetReg()
    assert isinstance(model, nn.Module), "LeNetReg 应继承 nn.Module"
    y = model(Xtr[:4])
    assert tuple(y.shape) == (4, 10), f"输出形状应为 (4,10)，实际 {tuple(y.shape)}"
    n = sum(p.numel() for p in model.parameters())
    assert n == 44426, f"默认配置参数量应为 44,426，实际 {n}"
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    n_drop = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    assert (n_bn, n_drop) == (0, 0), f"默认配置不应有 BN/Dropout，实际 BN={n_bn}, Dropout={n_drop}"

    m_bn = LeNetReg(use_bn=True)
    n_bn_params = sum(p.numel() for p in m_bn.parameters())
    assert n_bn_params == 44470, f"加 BN 后参数量应为 44,470，实际 {n_bn_params}"
    assert sum(1 for m in m_bn.modules() if isinstance(m, nn.BatchNorm2d)) == 2, "应有 2 个 BatchNorm2d"
    assert tuple(m_bn(Xtr[:2]).shape) == (2, 10), "BN 版前向输出形状不对"

    m_drop = LeNetReg(dropout_fc=0.5)
    n_drop_params = sum(p.numel() for p in m_drop.parameters())
    assert n_drop_params == 44426, f"Dropout 不改变参数量，应为 44,426，实际 {n_drop_params}"
    assert sum(1 for m in m_drop.modules() if isinstance(m, nn.Dropout)) == 2, "应有 2 个 Dropout"

    m_both = LeNetReg(use_bn=True, dropout_fc=0.5)
    assert sum(p.numel() for p in m_both.parameters()) == 44470, "BN+Dropout 参数量应为 44,470"
    assert tuple(LeNetReg(activation="sigmoid")(Xtr[:2]).shape) == (2, 10), "sigmoid 版前向失败"


@check("TODO 2  initial_loss：未训练模型损失 ≈ ln10")
def _test_initial_loss() -> None:
    torch.manual_seed(0)
    Xtr, ytr, _, _ = _data()
    model = LeNetReg()
    loss = initial_loss(model, Xtr[:512], ytr[:512])
    assert isinstance(loss, float), f"应返回 Python float，实际 {type(loss)}"
    assert abs(loss - 2.302585) < 0.05, f"初始损失应接近 ln10=2.302585，实际 {loss:.4f}"


@check("TODO 3  first_conv_grad_norm：返回正 float、梯度已算出")
def _test_grad_norm() -> None:
    torch.manual_seed(0)
    Xtr, ytr, _, _ = _data()
    model = LeNetReg()
    norm = first_conv_grad_norm(model, Xtr[:64], ytr[:64])
    assert isinstance(norm, float), f"应返回 Python float，实际 {type(norm)}"
    assert 0.0 < norm < 100.0, f"梯度范数应是正常正数，实际 {norm}"


@check("TODO 4  overfit_single_batch：64 个样本训练 300 步后几乎零损失")
def _test_overfit() -> None:
    torch.manual_seed(3)
    Xtr, ytr, _, _ = _data()
    g = torch.Generator().manual_seed(3)
    idx = torch.randperm(len(Xtr), generator=g)[:64]
    xb, yb = Xtr[idx], ytr[idx]
    model = LeNetReg()
    final_loss, final_acc = overfit_single_batch(model, xb, yb, steps=300, lr=1e-2)
    assert isinstance(final_loss, float) and isinstance(final_acc, float), "应返回两个 float"
    assert final_loss < 0.05, f"单批过拟合后损失应 < 0.05，实际 {final_loss:.4f}"
    assert final_acc >= 0.98, f"单批过拟合后准确率应接近 100%，实际 {final_acc:.2%}"


@check("TODO 5  numeric_gradient_check：最大相对误差 < 1e-4")
def _test_numeric_grad() -> None:
    torch.manual_seed(0)
    Xtr, ytr, _, _ = _data()
    model = LeNetReg()
    err = numeric_gradient_check(model, Xtr[:8], ytr[:8], eps=1e-5)
    assert isinstance(err, float), f"应返回 Python float，实际 {type(err)}"
    assert err < 1e-4, f"数值梯度与 autograd 的最大相对误差应 < 1e-4，实际 {err:.2e}"


@check("TODO 6  compare_activation_gradients：sigmoid 比 relu 小 20 倍以上")
def _test_compare_activation() -> None:
    Xtr, ytr, _, _ = _data()
    res = compare_activation_gradients(Xtr[:64], ytr[:64])
    assert isinstance(res, dict) and set(res) == {"relu", "sigmoid"}, f"返回值不对：{res!r}"
    relu, sigmoid = float(res["relu"]), float(res["sigmoid"])
    assert relu > 0 and sigmoid > 0, "梯度范数应为正数"
    assert sigmoid < relu / 20.0, \
        f"sigmoid 的第一层梯度应比 relu 小 20 倍以上：relu={relu:.3g}, sigmoid={sigmoid:.3g}"
    print(f"     relu 梯度范数 = {relu:.4g}，sigmoid = {sigmoid:.4g}，相差 {relu / sigmoid:.1f} 倍")


def main() -> None:
    print("=" * 74)
    print("第 5 步自检：正确性验证与训练技巧")
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
    print("\n全部通过！把输出、numeric_gradient_check 的代码、思考题答案发给老师，")
    print("就可以进入第 6 步：基线模型 + 激活函数对照实验。")


if __name__ == "__main__":
    main()
