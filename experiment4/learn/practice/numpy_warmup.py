"""第 0 步：NumPy 热手（约 10 分钟）。

每个练习只做一件事：选对 axis、配对下标、写对形状。
运行：python numpy_warmup.py
全部 PASS 后再开始 step01_linear.py。
"""

from __future__ import annotations

import numpy as np

# 类型别名：Array = 任意形状的 NumPy 数组
Array = np.ndarray


def check(name: str, got: object, expected: object, why: str = "") -> bool:
    """比较两个数组/标量是否相等，并打印说明。

    Parameters
    ----------
    name : str
        练习名。
    got : object
        你填的答案；可以是 None、标量、列表或 np.ndarray。
    expected : object
        正确答案，形状/数值应与 got 一致。
    why : str
        通过后打印的一句解释。
    """
    try:
        ok = np.allclose(np.asarray(got, dtype=float), np.asarray(expected, dtype=float))
    except Exception:
        ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if ok:
        if why:
            print(f"       {why}")
    else:
        print(f"       期望：{expected}")
        print(f"       实际：{got}")
    return ok


# ---------------------------------------------------------------------------
# 练习 1：axis
# ---------------------------------------------------------------------------
def ex1_axis() -> bool:
    X = np.arange(1, 7).reshape(2, 3)   # [[1,2,3],[4,5,6]]
    print(f"\n练习 1：X.shape = {X.shape}")

    # TODO：每一行求和，形状应为 (2,)
    row_sum = np.sum(X, axis=1)
    # TODO：每一列求平均，形状应为 (3,)
    col_mean = np.mean(X, axis=0)

    ok = check("axis=1 求行和", row_sum, [6, 15], "消掉第 1 维（列）")
    ok &= check("axis=0 求列均值", col_mean, [2.5, 3.5, 4.5], "消掉第 0 维（行）")
    return ok


# ---------------------------------------------------------------------------
# 练习 2：argmax
# ---------------------------------------------------------------------------
def ex2_argmax() -> bool:
    Z = np.array([[0.2, 0.5, 0.3],
                  [0.9, 0.05, 0.05],
                  [0.1, 0.2, 0.7]])       # 3 个样本，3 类打分
    print(f"\n练习 2：Z.shape = {Z.shape}")

    # TODO：每个样本预测分数最大的类别，形状 (3,)
    pred = np.argmax(Z, axis=1)

    ok = check("argmax(axis=1)", pred, [1, 0, 2], "返回的是下标，不是分数")
    return ok


# ---------------------------------------------------------------------------
# 练习 3：花式索引（交叉熵要用的那一步）
# ---------------------------------------------------------------------------
def ex3_fancy_index() -> bool:
    P = np.array([[0.1, 0.2, 0.7],
                  [0.6, 0.3, 0.1],
                  [0.2, 0.5, 0.3]])       # 每个样本属于 3 类的概率
    y = np.array([2, 0, 1])               # 真实类别
    print(f"\n练习 3：P.shape = {P.shape}, y = {y}")

    # TODO：取出每个样本真实类别的概率，形状 (3,)，结果应为 [0.7, 0.6, 0.5]
    p_true = P[np.arange(3), y]

    ok = check("P[np.arange(3), y]", p_true, [0.7, 0.6, 0.5],
               "行下标 0,1,2 与列下标 2,0,1 一一配对")
    return ok


# ---------------------------------------------------------------------------
# 练习 4：one-hot
# ---------------------------------------------------------------------------
def ex4_onehot() -> bool:
    y = np.array([2, 0, 1])
    print(f"\n练习 4：y = {y}")

    # TODO：构造 (3,3) 的 one-hot，第 i 行只有 y[i] 位置是 1
    Y = np.eye(3)[y]

    expected = np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])
    ok = check("np.eye(3)[y]", Y, expected, "eye 先生成单位矩阵，再用 y 取行")
    return ok


# ---------------------------------------------------------------------------
# 练习 5：广播（softmax 的减最大值）
# ---------------------------------------------------------------------------
def ex5_broadcast() -> bool:
    logits = np.array([[1.0, 2.0, 3.0],
                       [10.0, 0.0, -10.0]])
    print(f"\n练习 5：logits.shape = {logits.shape}")

    # TODO：每行减去该行最大值，形状仍为 (2,3)。
    #       提示：max(axis=1, keepdims=True) 的形状是 (2,1)。
    shifted = logits - logits.max(axis=1, keepdims=True)

    expected = np.array([[-2.0, -1.0, 0.0],
                         [0.0, -10.0, -20.0]])
    ok = check("logits - max(axis=1, keepdims=True)", shifted, expected,
               "每行最大值变成 0，为 exp 做准备")
    return ok


# ---------------------------------------------------------------------------
# 练习 6：矩阵乘法与形状
# ---------------------------------------------------------------------------
def ex6_matmul() -> bool:
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])       # (2,2)
    W = np.array([[1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0]])  # (2,3)
    print(f"\n练习 6：X{X.shape} @ W{W.shape}")

    # TODO：矩阵乘法，结果形状 (2,3)
    Z = X @ W

    expected = np.array([[1.0, 2.0, 3.0],
                         [3.0, 4.0, 7.0]])
    ok = check("X @ W", Z, expected, "维度：(2,2) @ (2,3) -> (2,3)")
    return ok


# ---------------------------------------------------------------------------
# 练习 7：数值稳定的 softmax
# ---------------------------------------------------------------------------
def ex7_softmax() -> bool:
    logits = np.array([[1000.0, 0.0, -1000.0]])
    print(f"\n练习 7：logits = {logits}")

    # TODO：先减最大值，再 exp，再除以行和。结果应为 [[1,0,0]]。
    p = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(p)
    p /= p.sum(axis=1, keepdims=True)

    expected = np.array([[1.0, 0.0, 0.0]])
    ok = check("stable softmax", p, expected, "最大值变成 exp(0)=1，避免溢出")
    return ok


# ---------------------------------------------------------------------------
# 选看：第 5 步画决策边界会用到的两个函数
# ---------------------------------------------------------------------------
def demo_grid() -> None:
    print("\n选看：meshgrid + column_stack 生成网格点")
    xs = np.linspace(-1, 1, 3)
    ys = np.linspace(-1, 1, 3)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    print(f"  xs = {xs}")
    print(f"  xx.shape = {xx.shape}, yy.shape = {yy.shape}")
    print(f"  grid.shape = {grid.shape}（9 个点，每点 2 个坐标）")
    print(f"  grid 前 3 行：\n{grid[:3]}")


def main() -> None:
    print("=" * 62)
    print("NumPy 热手：把 None 替换成正确答案")
    print("=" * 62)
    results = [
        ex1_axis(),
        ex2_argmax(),
        ex3_fancy_index(),
        ex4_onehot(),
        ex5_broadcast(),
        ex6_matmul(),
        ex7_softmax(),
    ]
    demo_grid()
    print("\n" + "=" * 62)
    if all(results):
        print(f"全部 {len(results)} 项通过，可以做第 1 步了。")
    else:
        print(f"通过 {sum(results)}/{len(results)} 项。对照 docs/00 查对应的函数。")
    print("=" * 62)


if __name__ == "__main__":
    main()
