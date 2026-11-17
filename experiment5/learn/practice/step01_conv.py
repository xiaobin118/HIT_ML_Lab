"""第 1 步：卷积与池化 —— 从手算到代码。

目标：理解卷积/池化的输出尺寸、参数量，并**手写**一个不调用框架的卷积和池化，
然后与 PyTorch 的结果对照；最后用输出尺寸公式推出 LeNet-5 各层特征图的形状。

规则
----
1. 只修改标了 `TODO` 的函数体，其它代码不要动。
2. 写完后运行 `python step01_conv.py`，全部 6 项自检必须 PASS。
3. 不要打开 `../code/models.py` 或 `../code/data.py`。先自己写，卡住 20 分钟以上再来问。
4. 讲义在 `../docs/01_卷积与池化.md`。

第 3、4 题要求真的用循环实现，**测试会检查函数源码里有没有偷调 F.conv2d / F.max_pool2d**。
"""

from __future__ import annotations

import traceback

import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# TODO 1：输出尺寸公式
# ===========================================================================
def out_size(h: int, w: int, k: int, stride: int = 1, padding: int = 0) -> tuple[int, int]:
    """返回卷积/池化后的 (H_out, W_out)。

    公式：
        H_out = floor((h + 2*padding - k) / stride) + 1
        W_out = floor((w + 2*padding - k) / stride) + 1

    要求：返回 (int, int) 元组；用整除 // 实现向下取整。
    """
    # TODO 1：写 2~3 行
    H_out = (h + 2 * padding - k) // stride + 1
    W_out = (w + 2 * padding - k) // stride + 1
    return H_out, W_out


# ===========================================================================
# TODO 2：一层卷积的参数量
# ===========================================================================
def conv_params(c_in: int, c_out: int, k: int, bias: bool = True) -> int:
    """返回一层卷积的可训练参数个数。

    公式：c_out * (k * k * c_in + (1 if bias else 0))
    """
    # TODO 2：写 1 行
    return c_out * (k * k * c_in + (1 if bias else 0))


# ===========================================================================
# TODO 3：手写卷积（核心题）
# ===========================================================================
def naive_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """用循环手写二维卷积（cross-correlation，不翻转核），返回输出特征图。

    参数
    ----
    x : torch.Tensor
        形状 (C_in, H, W)，单张图，float32。
    w : torch.Tensor
        形状 (C_out, C_in, k, k)，float32。
    b : torch.Tensor | None
        形状 (C_out,)；为 None 表示没有偏置。
    stride, padding : int
        步长、填充。padding 指**每侧**补 padding 圈 0。

    返回
    ----
    torch.Tensor
        形状 (C_out, H_out, W_out)，float32。

    要求
    ----
    1. 不允许调用 F.conv2d / torch.conv2d / nn.Conv2d（测试会检查源码）；
    2. 越界的坐标（padding 区域）等价于乘 0，跳过即可，不要访问负下标；
    3. 结果必须与 F.conv2d 在 1e-5 误差内一致。

    提示：先调 TODO 1 的 out_size 算输出大小；用 torch.zeros 建输出；
          六层循环的顺序见讲义第 4.2 节；初始化 acc 时考虑偏置。
    """
    # TODO 3：写 20~30 行。步骤：
    #   1) 取 C_out, C_in, k = w.shape，用 out_size 算 H_out, W_out；
    #   2) 输出 y = torch.zeros((C_out, H_out, W_out))；
    #   3) 五/六重循环：o, i, j, c, u, v；
    #   4) 坐标 (i*stride + u - padding, j*stride + v - padding) 越界就 continue；
    #   5) 累加 x[c, row, col] * w[o, c, u, v]；
    #   6) 别忘了偏置（如果有）。
    c_out, c_in, k, _ = w.shape
    H_out, W_out = out_size(x.shape[1], x.shape[2], k, stride, padding)
    y = torch.zeros((c_out, H_out, W_out), dtype=x.dtype)
    for o in range(c_out):
        for i in range(H_out):
            for j in range(W_out):
                acc = 0.0
                acc += b[o] if b is not None else 0.0
                for c in range(c_in):
                    for u in range(k):
                        for v in range(k):
                            row = i * stride + u - padding
                            col = j * stride + v - padding
                            if 0 <= row < x.shape[1] and 0 <= col < x.shape[2]:
                                acc += x[c, row, col] * w[o, c, u, v]
                y[o, i, j] = acc
    return y


# ===========================================================================
# TODO 4：手写最大池化
# ===========================================================================
def naive_maxpool2d(x: torch.Tensor, k: int = 2, stride: int = 2) -> torch.Tensor:
    """用循环手写最大池化（无 padding），返回下采样后的特征图。

    参数
    ----
    x : torch.Tensor
        形状 (C, H, W)，float32。
    k, stride : int
        窗口大小与步长。

    返回
    ----
    torch.Tensor
        形状 (C, H_out, W_out)，float32；每个位置是窗口内最大值。

    要求
    ----
    1. 不允许调用 F.max_pool2d / nn.MaxPool2d（测试会检查源码）；
    2. 结果必须与 F.max_pool2d(x[None], k, stride)[0] 一致；
    3. 用 out_size(H, W, k, stride, 0) 算输出形状。

    提示：边界处理最简单的方式是 `length = out_size(...)`，然后
          窗口范围为行 [i*stride, i*stride+k)，不要越界。
    """
    # TODO 4：写 10~15 行
    C, H, W = x.shape
    H_out, W_out = out_size(H, W, k, stride, 0)
    y = torch.zeros((C, H_out, W_out), dtype=x.dtype)
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                row_start = i * stride
                col_start = j * stride
                window = x[c, row_start:row_start + k, col_start:col_start + k]
                y[c, i, j] = window.max()
    return y


# ===========================================================================
# TODO 5：设计一个竖直边缘检测核
# ===========================================================================
def vertical_edge_kernel() -> torch.Tensor:
    """返回一个形状 (1, 1, 3, 3) 的 float32 卷积核，用来检测**竖直边缘**。

    要求
    ----
    1. 形状必须是 (1, 1, 3, 3)（一个输入通道、一个输出通道、3×3）；
    2. 在"左黑右白"的竖直边缘图上，边缘附近的响应要明显（绝对值 > 2）；
    3. 在纯黑或纯白区域内部，响应应接近 0（说明核的正负权重相互抵消）；
    4. 不要求是某一个特定的核：Sobel、Prewitt 或你自己设计的都可以，正负号反过来也行。

    提示：讲义第 1 节的核就是一个例子，但请自己写一版并想清楚为什么它能检测竖直边缘。
    """
    # TODO 5：写 1~3 行，返回一个 3x3 张量，注意 unsqueeze 成 (1,1,3,3)
    kernel = torch.tensor([[1.0, 0.0, -1.0],
                            [1.0, 0.0, -1.0],
                            [1.0, 0.0, -1.0]], dtype=torch.float32)
    return kernel.unsqueeze(0).unsqueeze(0)


# ===========================================================================
# TODO 6：用 out_size 推出 LeNet-5 各层特征图形状
# ===========================================================================
def lenet_shapes() -> list[tuple[int, int, int]]:
    """返回 LeNet-5 在 1×28×28 输入下的特征图形状列表，顺序为：

        [conv1 输出, pool1 输出, conv2 输出, pool2 输出]

    每个元素是 (C, H, W) 元组。LeNet-5 的参数：
        conv1: 1 -> 6, k=5, stride=1, padding=0
        pool1: k=2, stride=2
        conv2: 6 -> 16, k=5, stride=1, padding=0
        pool2: k=2, stride=2
    要求：每个形状都用 TODO 1 的 out_size 算出来，不要直接写死数字。
    """
    # TODO 6：写 8~12 行。提示：从 (1, 28, 28) 开始，逐层调用 out_size 更新 H、W。
    shapes = []
    C, H, W = 1, 28, 28
    #conv1
    H, W = out_size(H, W, k=5, stride=1, padding=0)
    shapes.append((6, H, W))
    #pool1
    H, W = out_size(H, W, k=2, stride=2, padding=0)
    shapes.append((6, H, W))
    #conv2
    H, W = out_size(H, W, k=5, stride=1, padding=0)
    shapes.append((16, H, W))
    #pool2
    H, W = out_size(H, W, k=2, stride=2, padding=0)
    shapes.append((16, H, W))
    return shapes


# ===========================================================================
# 自检代码：不要修改下面的内容
# ===========================================================================


CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


def _no_cheat(fn, banned: tuple[str, ...] = ("conv2d", "Conv2d", "max_pool2d", "MaxPool2d")) -> None:
    """检查函数体里没有引用被禁止的框架函数（按函数引用名检查，不看注释/文档）。"""
    code = fn.__code__
    used = set(code.co_names) | set(code.co_varnames)
    hit = [name for name in banned if name in used]
    assert not hit, f"不允许调用 {hit[0]}（本题要求手写循环）"


@check("TODO 1  out_size：5 组形状全对")
def _test_out_size() -> None:
    cases = [
        ((28, 28, 5, 1, 0), (24, 24)),
        ((28, 28, 5, 2, 0), (12, 12)),
        ((28, 28, 5, 1, 2), (28, 28)),
        ((28, 28, 5, 2, 1), (13, 13)),
        ((32, 32, 5, 1, 0), (28, 28)),
    ]
    for args, expected in cases:
        got = out_size(*args)
        assert isinstance(got, tuple) and len(got) == 2, f"out_size{args} 应返回二元元组，实际 {got!r}"
        assert tuple(got) == expected, f"out_size{args} 应为 {expected}，实际 {tuple(got)}"


@check("TODO 2  conv_params：156 / 2416 / 9248 全对")
def _test_conv_params() -> None:
    cases = [
        ((1, 6, 5), 156),
        ((6, 16, 5), 2416),
        ((1, 32, 3), 320),
        ((32, 32, 3), 9248),
        ((1, 32, 3, False), 288),
    ]
    for args, expected in cases:
        got = conv_params(*args)
        assert got == expected, f"conv_params{args} 应为 {expected}，实际 {got}"


@check("TODO 3  naive_conv2d：与 F.conv2d 一致（stride/padding 两组）")
def _test_naive_conv() -> None:
    _no_cheat(naive_conv2d)
    torch.manual_seed(0)
    x = torch.randn(3, 6, 6)
    w = torch.randn(2, 3, 3, 3)
    b = torch.randn(2)
    for stride, padding in [(1, 1), (2, 0)]:
        got = naive_conv2d(x, w, b, stride=stride, padding=padding)
        expected = F.conv2d(x[None], w, b, stride=stride, padding=padding)[0]
        assert tuple(got.shape) == tuple(expected.shape), \
            f"stride={stride},padding={padding} 输出形状应为 {tuple(expected.shape)}，实际 {tuple(got.shape)}"
        assert torch.allclose(got, expected, atol=1e-5), \
            f"stride={stride},padding={padding} 数值与 F.conv2d 不一致，最大误差 {(got - expected).abs().max():.2e}"
    # 无偏置也要对
    got = naive_conv2d(x, w, None)
    expected = F.conv2d(x[None], w, None)[0]
    assert torch.allclose(got, expected, atol=1e-5), "bias=None 时数值与 F.conv2d 不一致"


@check("TODO 4  naive_maxpool2d：与 F.max_pool2d 一致")
def _test_naive_pool() -> None:
    _no_cheat(naive_maxpool2d)
    torch.manual_seed(1)
    x = torch.randn(2, 6, 6)
    for k, s in [(2, 2), (3, 1)]:
        got = naive_maxpool2d(x, k=k, stride=s)
        expected = F.max_pool2d(x[None], kernel_size=k, stride=s)[0]
        assert tuple(got.shape) == tuple(expected.shape), \
            f"k={k},stride={s} 输出形状应为 {tuple(expected.shape)}，实际 {tuple(got.shape)}"
        assert torch.allclose(got, expected), f"k={k},stride={s} 数值与 F.max_pool2d 不一致"


@check("TODO 5  vertical_edge_kernel：边缘响应大、平坦区接近 0")
def _test_edge_kernel() -> None:
    kern = vertical_edge_kernel()
    assert isinstance(kern, torch.Tensor), f"应返回 torch.Tensor，实际 {type(kern)}"
    assert tuple(kern.shape) == (1, 1, 3, 3), f"形状应为 (1,1,3,3)，实际 {tuple(kern.shape)}"
    assert kern.dtype == torch.float32, f"dtype 应为 float32，实际 {kern.dtype}"
    assert abs(float(kern.sum())) < 1e-6, "核的正负权重应相互抵消（总和≈0）"

    # 左半黑、右半白 的 8x8 合成图
    x = torch.zeros(1, 1, 8, 8)
    x[..., 4:] = 1.0
    resp = F.conv2d(x, kern)[0, 0]          # 输出 6x6
    # 窗口完全落在纯黑区域（输出列 0~1）或纯白区域（输出列 5）时，响应应接近 0
    assert float(resp[:, :2].abs().max()) < 1e-6, \
        f"纯黑区域的响应应接近 0，实际 {float(resp[:, :2].abs().max()):.3f}"
    assert float(resp[:, 5].abs().max()) < 1e-6, \
        f"纯白区域的响应应接近 0，实际 {float(resp[:, 5].abs().max()):.3f}"
    # 边缘落在窗口内（输出列 2~3）时，响应应明显
    assert float(resp[:, 2:4].abs().max()) > 2.0, \
        f"边缘处响应应 > 2，实际 {float(resp[:, 2:4].abs().max()):.3f}"
    print(f"     核 = \n{kern[0, 0]}\n     边缘处最大响应 = {float(resp[:, 2:4].abs().max()):.2f}")


@check("TODO 6  lenet_shapes：28 → 24 → 12 → 8 → 4")
def _test_lenet_shapes() -> None:
    shapes = lenet_shapes()
    assert isinstance(shapes, list) and len(shapes) == 4, \
        f"应返回长度为 4 的列表，实际 {shapes!r}"
    got = [tuple(s) for s in shapes]
    expected = [(6, 24, 24), (6, 12, 12), (16, 8, 8), (16, 4, 4)]
    assert got == expected, f"特征图形状应为 {expected}，实际 {got}"


def main() -> None:
    print("=" * 74)
    print("第 1 步自检：卷积与池化")
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
    print("\n全部通过！把输出、naive_conv2d 的代码、三个思考题的答案发给老师，")
    print("就可以进入第 2 步：用 nn.Module 搭建 LeNet-5。")


if __name__ == "__main__":
    main()
