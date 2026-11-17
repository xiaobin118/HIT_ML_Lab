"""第 0 步：PyTorch 热手 —— 张量、设备与自动求导。

目标：把 NumPy 的思维迁移到 PyTorch，并亲手用 autograd 训练一个线性模型。

规则
----
1. 只修改标了 `TODO` 的函数体，其它代码不要动。
2. 写完后运行 `python step00_pytorch_warmup.py`，全部 7 项自检必须 PASS。
3. 不要打开 `../code/` 下的成品实现。卡住 20 分钟以上再来问。
4. 讲义在 `../docs/00_PyTorch热手.md`。

每题都写了"要求"和"提示"，照着要求写即可。提示里给了思路，但没有给完整答案。
"""

from __future__ import annotations

import traceback

import numpy as np
import torch


# ===========================================================================
# TODO 1：创建张量
# ===========================================================================
def make_tensor() -> torch.Tensor:
    """返回一个形状为 (2, 3)、dtype 为 torch.float32 的张量：
        [[1, 2, 3],
         [4, 5, 6]]

    要求：不能直接写 `torch.tensor([[1,2,3],[4,5,6]])` 就交差——它默认是 int64。
          请显式指定 dtype，或用浮点字面量。
    """
    # TODO 1：写 1 行
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # 修改为正确的张量


# ===========================================================================
# TODO 2：把张量搬到 GPU（没有 GPU 就留在 CPU）
# ===========================================================================
def to_device(x: torch.Tensor) -> torch.Tensor:
    """有 CUDA 时返回 x 在 GPU 上的版本；没有 CUDA 时原样返回 x。

    要求：返回的张量要和 x 数值相同，但 device 正确。
    """
    # TODO 2：写 2~3 行。提示：`torch.cuda.is_available()`，`.to("cuda")`
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device)


# ===========================================================================
# TODO 3：把一批 NumPy 图像转成 PyTorch 张量
# ===========================================================================
def image_batch(X: np.ndarray) -> torch.Tensor:
    """把 (N, 28, 28) 的 uint8 NumPy 数组转成 (N, 1, 28, 28) 的 float32 张量，数值缩放到 [0,1]。

    要求：
    - 形状 (N, 1, 28, 28)：补一个通道维；
    - dtype = torch.float32，数值在 [0,1]；
    - 元素和 X 的对应关系是 `X[n] / 255`。

    提示：torch.from_numpy(...)、.float()、/255.0、.unsqueeze(1)。注意顺序。
    """
    # TODO 3：写 2~3 行
    tmp = torch.from_numpy(X).float() / 255.0
    return tmp.unsqueeze(1)  # 补一个通道维


# ===========================================================================
# TODO 4：归约操作
# ===========================================================================
def reduce_ops(x: torch.Tensor) -> dict:
    """给定形状为 (4, 3) 的 float32 张量，返回一个 dict：

        "total":        所有元素之和，Python float（用 .item() 转）
        "col_mean":     每一列的均值，形状 (3,)
        "row_argmax":   每一行最大值所在列的下标，dtype 必须是 int64，形状 (4,)
        "row_max_keep": 每一行的最大值，但保持二维形状 (4, 1)

    提示：`.sum()`、`.mean(dim=?)`、`.argmax(dim=?)`、`.max(dim=?, keepdim=True).values`。
    """
    # TODO 4：写 4 行
    total = x.sum().item()
    col_mean = x.mean(dim=0)
    row_argmax = x.argmax(dim=1)
    row_max_keep = x.max(dim=1, keepdim=True).values
    return {
        "total": total,
        "col_mean": col_mean,
        "row_argmax": row_argmax,
        "row_max_keep": row_max_keep,
    }


# ===========================================================================
# TODO 5：用 autograd 训练线性回归
# ===========================================================================
def autograd_linear_fit(
    x: torch.Tensor, y: torch.Tensor, steps: int = 300, lr: float = 0.1
) -> tuple[float, float]:
    """用梯度下降拟合 y ≈ w*x + b，返回 (w, b) 的 Python float。

    参数：
    - x: 形状 (N,) 的 float32 张量，输入；
    - y: 形状 (N,) 的 float32 张量，目标是 y = 2*x + 1（无噪声）；
    - steps: 迭代次数；
    - lr: 学习率。

    要求：
    1. w、b 用 `torch.zeros(1, requires_grad=True)` 初始化；
    2. 损失用均方误差 `((w*x + b - y) ** 2).mean()`；
    3. 每一轮：loss.backward() → 手动更新参数（记得用 torch.no_grad 包住）→ 梯度清零；
    4. 训练结束后返回 float(w.item()), float(b.item())。
    验收：|w - 2| < 0.1 且 |b - 1| < 0.1。
    """
    # TODO 5：写 10~15 行。提示：
    #   w = torch.zeros(1, requires_grad=True)
    #   for _ in range(steps):
    #       loss = ...
    #       loss.backward()
    #       with torch.no_grad():
    #           w -= lr * w.grad
    #           b -= lr * b.grad
    #       w.grad.zero_()
    #       b.grad.zero_()
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(steps):
        loss = ((w * x + b - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return float(w.item()), float(b.item())

# ===========================================================================
# TODO 6：autograd 梯度 vs 手写公式
# ===========================================================================


def grad_compare(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对损失 L = mean((X @ w - y)^2)（w 初始为全 0 向量），分别用两种方式求 dL/dw。

    参数：
    - X: 形状 (N, 3) 的 float32 张量；
    - y: 形状 (N,) 的 float32 张量。

    返回：
    - auto_grad:   用 autograd 算出的梯度，形状 (3,)；
    - manual_grad: 用手写公式算出的梯度，形状 (3,)。

    手写公式（w = 0 时）：manual_grad = 2/N * Xᵀ(Xw - y)
    要求：先让 w = torch.zeros(3, requires_grad=True)，算 loss、backward，
          取出 auto_grad；再用上面公式算 manual_grad；两者误差应 < 1e-6。
    """
    # TODO 6：写 6~10 行
    N = X.shape[0]
    w = torch.zeros(3, requires_grad=True)
    loss = ((X @ w - y) ** 2).mean()
    loss.backward()
    auto_grad = w.grad.clone()
    manual_grad = (2 / N) * X.T @ (X @ w - y)
    return auto_grad, manual_grad


# ===========================================================================
# TODO 7：在 GPU 上做矩阵乘法
# ===========================================================================
def matmul_on_device(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, str]:
    """把 a、b 搬到 GPU（没有 GPU 就留在 CPU）做矩阵乘法，返回 (结果, device 字符串)。

    要求：
    - 有 CUDA 时：结果在 cuda 上，device 字符串为 "cuda"；
    - 没有 CUDA 时：结果在 cpu 上，device 字符串为 "cpu"；
    - 结果数值与 `a @ b` 相同（GPU 用 float32 计算，允许 1e-5 误差）。

    提示：device = "cuda" if torch.cuda.is_available() else "cpu"；
          a = a.to(device); b = b.to(device); return a @ b, device
    """
    # TODO 7：写 3~5 行
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = a.to(device)
    b = b.to(device)
    return a @ b, device


# ===========================================================================
# 自检代码：不要修改下面的内容
# ===========================================================================


CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


@check("TODO 1  make_tensor：形状 (2,3)、float32、数值正确")
def _test_make_tensor() -> None:
    t = make_tensor()
    assert isinstance(t, torch.Tensor), f"应返回 torch.Tensor，实际是 {type(t)}"
    assert tuple(t.shape) == (2, 3), f"形状应为 (2,3)，实际 {tuple(t.shape)}"
    assert t.dtype == torch.float32, f"dtype 应为 float32，实际 {t.dtype}"
    expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.equal(t, expected), f"数值不对：\n{t}"


@check("TODO 2  to_device：设备正确、数值不变")
def _test_to_device() -> None:
    x = torch.randn(2, 3)
    out = to_device(x)
    assert isinstance(out, torch.Tensor), f"应返回 torch.Tensor，实际是 {type(out)}"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert out.device.type == expected, f"应有 GPU 时 device='cuda'，实际 {out.device}"
    assert torch.allclose(out.cpu(), x, atol=1e-6), "搬到设备后数值变了"


@check("TODO 3  image_batch：形状 (N,1,28,28)、float32、值域 [0,1]")
def _test_image_batch() -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, size=(5, 28, 28), dtype=np.uint8)
    out = image_batch(X)
    assert isinstance(out, torch.Tensor), f"应返回 torch.Tensor，实际是 {type(out)}"
    assert tuple(out.shape) == (5, 1, 28, 28), f"形状应为 (5,1,28,28)，实际 {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"dtype 应为 float32，实际 {out.dtype}"
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "数值应缩放到 [0,1]"
    expected = torch.from_numpy(X.astype(np.float32) / 255.0).unsqueeze(1)
    assert torch.allclose(out, expected, atol=1e-6), "数值与 X/255 不一致"


@check("TODO 4  reduce_ops：sum / 列均值 / 行 argmax / keepdim")
def _test_reduce_ops() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = reduce_ops(x)
    assert isinstance(out, dict), f"应返回 dict，实际 {type(out)}"
    for key in ("total", "col_mean", "row_argmax", "row_max_keep"):
        assert key in out, f"返回的 dict 缺少 key: {key}"
    assert abs(out["total"] - 66.0) < 1e-6, f"total 应为 66.0，实际 {out['total']}"
    assert tuple(out["col_mean"].shape) == (3,), f"col_mean 形状应为 (3,)，实际 {tuple(out['col_mean'].shape)}"
    assert torch.allclose(out["col_mean"], torch.tensor([4.5, 5.5, 6.5])), \
        f"col_mean 应为 [4.5, 5.5, 6.5]，实际 {out['col_mean']}"
    assert tuple(out["row_argmax"].shape) == (4,), f"row_argmax 形状应为 (4,)，实际 {tuple(out['row_argmax'].shape)}"
    assert out["row_argmax"].dtype == torch.int64, f"row_argmax 应为 int64，实际 {out['row_argmax'].dtype}"
    assert torch.equal(out["row_argmax"], torch.tensor([2, 2, 2, 2])), \
        f"row_argmax 应为 [2,2,2,2]，实际 {out['row_argmax']}"
    assert tuple(out["row_max_keep"].shape) == (4, 1), \
        f"row_max_keep 形状应为 (4,1)，实际 {tuple(out['row_max_keep'].shape)}"
    assert torch.allclose(out["row_max_keep"].flatten(), torch.tensor([2.0, 5.0, 8.0, 11.0])), \
        f"row_max_keep 数值不对：{out['row_max_keep']}"


@check("TODO 5  autograd_linear_fit：w≈2、b≈1")
def _test_autograd_fit() -> None:
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1.0, 1.0, size=100).astype(np.float32)
    x = torch.from_numpy(x_np)
    y = 2.0 * x + 1.0
    w, b = autograd_linear_fit(x, y)
    assert abs(w - 2.0) < 0.1, f"w 应接近 2，实际 {w:.4f}"
    assert abs(b - 1.0) < 0.1, f"b 应接近 1，实际 {b:.4f}"


@check("TODO 6  grad_compare：autograd 梯度与手写公式一致")
def _test_grad_compare() -> None:
    rng = np.random.default_rng(1)
    X = torch.from_numpy(rng.normal(size=(8, 3)).astype(np.float32))
    y = torch.from_numpy(rng.normal(size=8).astype(np.float32))
    auto, manual = grad_compare(X, y)
    auto = torch.as_tensor(auto).float().flatten()
    manual = torch.as_tensor(manual).float().flatten()
    assert tuple(auto.shape) == (3,), f"auto_grad 形状应为 (3,)，实际 {tuple(auto.shape)}"
    assert tuple(manual.shape) == (3,), f"manual_grad 形状应为 (3,)，实际 {tuple(manual.shape)}"
    assert torch.allclose(auto, manual, atol=1e-6), \
        f"两种梯度不一致：\nautograd = {auto}\n手写    = {manual}"


@check("TODO 7  matmul_on_device：设备正确、结果正确")
def _test_matmul_device() -> None:
    torch.manual_seed(0)
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    out, dev = matmul_on_device(a, b)
    expected_dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == expected_dev, f"device 字符串应为 {expected_dev!r}，实际 {dev!r}"
    assert isinstance(out, torch.Tensor), f"应返回 torch.Tensor，实际 {type(out)}"
    assert out.device.type == expected_dev, f"结果应在 {expected_dev} 上，实际 {out.device}"
    assert torch.allclose(out.cpu(), a @ b, atol=1e-4), "矩阵乘结果不对"


def main() -> None:
    print("=" * 74)
    print("第 0 步自检：PyTorch 热手")
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
    print("\n全部通过！把输出、autograd_linear_fit / grad_compare 的代码、")
    print("以及你的一两句疑问发给老师，就可以进入第 1 步：卷积与池化。")


if __name__ == "__main__":
    main()
