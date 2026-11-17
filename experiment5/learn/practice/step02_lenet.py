"""第 2 步：用 nn.Module 搭 LeNet-5。

目标：把第 1 步手算的卷积、池化、形状链和参数量，组装成一个真正的 PyTorch 模型；
学会 nn.Module、nn.Conv2d、nn.Linear、nn.Flatten、forward hook 和参数量统计。

规则
----
1. 只修改标了 `TODO` 的地方，其它代码不要动。
2. 写完后运行 `python step02_lenet.py`，全部 6 项自检必须 PASS。
3. 不要打开 `../code/models.py`。先自己写，卡住 20 分钟以上再来问。
4. 讲义在 `../docs/02_用nn_Module搭LeNet5.md`。
"""

from __future__ import annotations

import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

# 允许的激活函数名字（和讲义一致）
ACTIVATIONS = ("relu", "sigmoid", "tanh", "leaky_relu")


# ===========================================================================
# TODO 1：激活函数工厂
# ===========================================================================
def make_activation(name: str) -> nn.Module:
    """按名字返回一个激活函数**模块**（注意：返回的是 nn.Module，不是函数）。

    支持的名字：relu、sigmoid、tanh、leaky_relu（负半轴斜率 0.1）。
    其它名字抛 ValueError（信息里带上到底传了什么名字）。

    为什么返回模块而不是函数？因为模块可以放进 nn.Sequential，而且行为统一。
    提示：
        "relu"      -> nn.ReLU()
        "sigmoid"   -> nn.Sigmoid()
        "tanh"      -> nn.Tanh()
        "leaky_relu"-> nn.LeakyReLU(negative_slope=0.1)
    """
    # TODO 1：写 6~10 行（可以先用 if/elif，也可以自己查 nn.ModuleDict 之类的写法）
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    else:
        raise ValueError(f"未知激活函数名字 {name!r}，只能是 {ACTIVATIONS}")


# ===========================================================================
# TODO 2：一个卷积块
# ===========================================================================
class ConvBlock(nn.Module):
    """conv2d(kernel_size×kernel_size) -> activation -> pooling(2×2)。

    参数
    ----
    in_ch, out_ch : int
        输入/输出通道数。
    kernel_size : int
        卷积核大小（默认 5）。
    activation : str
        激活函数名字，交给 make_activation。
    pooling : str
        "max" -> nn.MaxPool2d(2)；"avg" -> nn.AvgPool2d(2)；其它抛 ValueError。

    要求
    ----
    1. `__init__` 里必须先调用 `super().__init__()`；
    2. 把 conv、act、pool 都赋给 `self.xxx`（这样参数才会被注册）；
    3. `forward` 里按 conv → act → pool 的顺序计算；
    4. 一个 (1, 28, 28) 输入经过 ConvBlock(1, 6, 5, "relu", "max") 后形状是 (6, 12, 12)，
       参数个数是 156（只有卷积层有参数）。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        activation: str = "relu",
        pooling: str = "max",
    ) -> None:
        # TODO 2：写 10~15 行
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.act = make_activation(activation)
        if pooling == "max":
            self.pool = nn.MaxPool2d(2)
        elif pooling == "avg":
            self.pool = nn.AvgPool2d(2)
        else:
            raise ValueError(f"未知 pooling 名字 {pooling!r}，只能是 'max' 或 'avg'")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2：写 2~4 行
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


# ===========================================================================
# TODO 3：LeNet-5
# ===========================================================================
class LeNet5(nn.Module):
    """28×28 灰度图 → 10 类 logits 的 LeNet-5。

    结构（默认参数）：
        features:   ConvBlock(1, 6, 5) -> ConvBlock(6, 16, 5)
                    输入 1×28×28 → conv1 6×24×24 → pool 6×12×12
                    → conv2 16×8×8 → pool 16×4×4
        classifier: Flatten -> Linear(16*4*4=256 → 120) -> 激活
                    -> Linear(120 → 84) -> 激活 -> Linear(84 → num_classes)
    默认参数下的可训练参数总量必须是 44,426。

    要求
    ----
    1. `__init__` 先 `super().__init__()`；
    2. `self.features = nn.Sequential(...)` 装两个 ConvBlock；
    3. `self.classifier = nn.Sequential(...)` 装 Flatten、三个 Linear、两个激活模块；
       Flatten 要用 `nn.Flatten()`（从 dim=1 开始拉平，保留 batch 维）；
    4. `forward` 返回 logits（**不要**在最后加 softmax）；
    5. 支持 activation / pooling / fc_sizes 三个可配置参数（fc_sizes 是 (fc1, fc2) 的元组）。
    """

    def __init__(
        self,
        num_classes: int = 10,
        activation: str = "relu",
        pooling: str = "max",
        fc_sizes: tuple[int, int] = (120, 84),
    ) -> None:
        # TODO 3：写 20~30 行
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 6, 5, activation=activation, pooling=pooling),
            ConvBlock(6, 16, 5, activation=activation, pooling=pooling)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, fc_sizes[0]),
            make_activation(activation),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            make_activation(activation),
            nn.Linear(fc_sizes[1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3：写 1 行
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# TODO 4：统计参数量
# ===========================================================================
def count_parameters(model: nn.Module) -> int:
    """返回模型**可训练**参数的总个数（int）。

    提示：`model.parameters()` 会遍历所有参数张量；
          每个张量用 `.numel()` 得到元素个数；只统计 `p.requires_grad` 为 True 的。
    验收：默认 LeNet5 返回 44,426；nn.Linear(10, 20) 返回 220。
    """
    # TODO 4：写 1~2 行
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===========================================================================
# TODO 5：用 forward hook 记录特征图形状
# ===========================================================================
def feature_shapes() -> list[tuple[int, int, int]]:
    """返回默认 LeNet5 各特征图（不含 batch 维）的形状列表，顺序为：

        [conv1 输出, pool1 输出, conv2 输出, pool2 输出]
        即 [(6, 24, 24), (6, 12, 12), (16, 8, 8), (16, 4, 4)]

    要求：用 `register_forward_hook` 在真实前向上记录形状，不要直接写死四个元组。

    提示：要 hook 的是卷积层和池化层，不是整个 ConvBlock（否则只能拿到池化后的形状）。
        model = LeNet5().eval()
        shapes = []
        hooks = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(m.register_forward_hook(...))
        with torch.no_grad():
            model(torch.zeros(1, 1, 28, 28))
        for h in hooks: h.remove()
    注意 hook 拿到的 `output` 形状是 (N, C, H, W)，返回时要去掉 batch 维。
    """
    # TODO 5：写 10~15 行
    model = LeNet5().eval()
    shapes = []
    hooks = []

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            hooks.append(m.register_forward_hook(lambda module, input, output: shapes.append(tuple(output.shape[1:]))))
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28))
    for h in hooks:
        h.remove()
    return shapes


# ===========================================================================
# TODO 6：logits → 概率
# ===========================================================================
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """返回模型对 x 的 softmax 概率，形状 (N, num_classes)。

    要求
    ----
    1. 返回形状 (N, C)，每一行加起来是 1（误差 < 1e-5），所有元素 >= 0；
    2. 概率最高的类别应与 `model(x).argmax(dim=1)` 一致；
    3. **推理不需要梯度**：用 `with torch.no_grad():` 包住，返回的张量不应带 grad
       （测试会检查 `p.requires_grad == False`）；
    4. 不要修改传入的 model 参数。
    提示：`F.softmax(logits, dim=1)`。
    """
    # TODO 6：写 3~5 行
    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=1)
        return proba


# ===========================================================================
# 自检代码：不要修改下面的内容
# ===========================================================================


CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


@check("TODO 1  make_activation：4 种激活函数行为正确，未知名字报错")
def _test_activation() -> None:
    # ReLU 负半轴为 0、正半轴不变
    r = make_activation("relu")
    assert isinstance(r, nn.Module), f"应返回 nn.Module，实际 {type(r)}"
    out = r(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0]])), f"ReLU 输出不对：{out}"

    s = make_activation("sigmoid")
    assert abs(float(s(torch.tensor(0.0))) - 0.5) < 1e-6, "sigmoid(0) 应为 0.5"

    t = make_activation("tanh")
    assert abs(float(t(torch.tensor(0.0)))) < 1e-6, "tanh(0) 应为 0"

    lr = make_activation("leaky_relu")
    got = float(lr(torch.tensor(-4.0)))
    assert abs(got - (-0.4)) < 1e-5, f"leaky_relu(-4) 应约等于 -0.4，实际 {got}"

    try:
        make_activation("swish")
    except ValueError:
        pass
    else:
        raise AssertionError("传入未知激活函数名时应抛 ValueError")


@check("TODO 2  ConvBlock：形状正确、参数量正确")
def _test_conv_block() -> None:
    block = ConvBlock(1, 6, 5, "relu", "max")
    assert isinstance(block, nn.Module), "ConvBlock 应继承 nn.Module"
    x = torch.randn(2, 1, 28, 28)
    y = block(x)
    assert tuple(y.shape) == (2, 6, 12, 12), f"输出形状应为 (2,6,12,12)，实际 {tuple(y.shape)}"
    assert sum(p.numel() for p in block.parameters()) == 156, \
        f"ConvBlock(1,6,5) 参数量应为 156，实际 {sum(p.numel() for p in block.parameters())}"

    block2 = ConvBlock(6, 16, 5, "sigmoid", "avg")
    y2 = block2(torch.randn(2, 6, 12, 12))
    assert tuple(y2.shape) == (2, 16, 4, 4), f"输出形状应为 (2,16,4,4)，实际 {tuple(y2.shape)}"
    assert sum(p.numel() for p in block2.parameters()) == 2416, \
        f"ConvBlock(6,16,5) 参数量应为 2416，实际 {sum(p.numel() for p in block2.parameters())}"


@check("TODO 3  LeNet5：输出 (N,10)、默认参数量 44,426")
def _test_lenet() -> None:
    model = LeNet5()
    assert isinstance(model, nn.Module), "LeNet5 应继承 nn.Module"
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert tuple(y.shape) == (4, 10), f"输出形状应为 (4,10)，实际 {tuple(y.shape)}"
    assert y.dtype == torch.float32, f"输出 dtype 应为 float32，实际 {y.dtype}"

    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    assert (n_conv, n_linear) == (2, 3), f"应有 2 个 Conv2d、3 个 Linear，实际 {n_conv} 和 {n_linear}"

    total = sum(p.numel() for p in model.parameters())
    assert total == 44426, f"默认 LeNet5 参数量应为 44,426，实际 {total}"

    # 换激活/池化后仍可前向
    model2 = LeNet5(activation="tanh", pooling="avg")
    y2 = model2(torch.randn(2, 1, 28, 28))
    assert tuple(y2.shape) == (2, 10), f"tanh+avg 版本输出应为 (2,10)，实际 {tuple(y2.shape)}"

    # 改 fc_sizes 后参数量应可手算：2,572 + 131,584 + 65,664 + 1,290 = 201,110
    model3 = LeNet5(fc_sizes=(512, 128))
    total3 = sum(p.numel() for p in model3.parameters())
    assert total3 == 201110, f"fc_sizes=(512,128) 参数量应为 201,110，实际 {total3}"


@check("TODO 4  count_parameters：44,426 / 220")
def _test_count_parameters() -> None:
    assert count_parameters(LeNet5()) == 44426, f"默认 LeNet5 应为 44,426，实际 {count_parameters(LeNet5())}"
    lin = nn.Linear(10, 20)
    assert count_parameters(lin) == 220, f"nn.Linear(10,20) 应为 220，实际 {count_parameters(lin)}"


@check("TODO 5  feature_shapes：28 → 24 → 12 → 8 → 4")
def _test_feature_shapes() -> None:
    shapes = feature_shapes()
    assert isinstance(shapes, list) and len(shapes) == 4, f"应返回长度 4 的列表，实际 {shapes!r}"
    got = [tuple(s) for s in shapes]
    expected = [(6, 24, 24), (6, 12, 12), (16, 8, 8), (16, 4, 4)]
    assert got == expected, f"特征图形状应为 {expected}，实际 {got}"


@check("TODO 6  predict_proba：形状 (N,10)、行和为 1、不带梯度")
def _test_predict_proba() -> None:
    torch.manual_seed(0)
    model = LeNet5().eval()
    x = torch.randn(8, 1, 28, 28)
    p = predict_proba(model, x)
    assert tuple(p.shape) == (8, 10), f"概率形状应为 (8,10)，实际 {tuple(p.shape)}"
    assert torch.allclose(p.sum(dim=1), torch.ones(8), atol=1e-5), \
        f"每行概率和应为 1，实际 {p.sum(dim=1)}"
    assert bool((p >= 0).all()), "概率不能为负数"
    with torch.no_grad():
        logits = model(x)
    assert torch.equal(p.argmax(dim=1), logits.argmax(dim=1)), \
        "概率最大的类别应与 logits 的 argmax 一致"
    assert not p.requires_grad, "推理应放在 torch.no_grad() 里，返回的概率不应带梯度"


def main() -> None:
    print("=" * 74)
    print("第 2 步自检：用 nn.Module 搭 LeNet-5")
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
    print("\n全部通过！把输出、LeNet5 的代码、三个思考题的答案发给老师，")
    print("就可以进入第 3 步：把 MNIST 数据装进 DataLoader。")


if __name__ == "__main__":
    main()
