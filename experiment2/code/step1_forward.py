import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """对数组或标量逐元素计算 sigmoid，并裁剪输入防止 exp 溢出。"""
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def linear_score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    X: shape (N, d)
    w: shape (d,)
    b: 标量
    return z: shape (N,)
    """
    return X @ w + b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    X: shape (N, d)
    w: shape (d,)
    b: 标量
    return p: shape (N,)
    """
    z = linear_score(X, w, b)
    return sigmoid(z)


if __name__ == "__main__":
    X = np.array([
        [0.0, 1.0],
        [1.0, -1.0],
        [2.0, 0.5],
        [-1.0, -2.0],
    ])
    y = np.array([1, 0, 1, 0])

    # 测试 1：w=0, b=0，全部样本得分都是 0，概率都是 0.5
    w = np.zeros(2)
    b = 0.0

    z = linear_score(X, w, b)
    p = predict_proba(X, w, b)

    print("X.shape =", X.shape)
    print("w.shape =", w.shape)
    print("z.shape =", z.shape)
    print("p.shape =", p.shape)
    print("z =", z)
    print("p =", p)

    assert z.shape == (4,)
    assert p.shape == (4,)
    assert np.allclose(p, 0.5)

    # 测试 2：手动设置非零权重
    w = np.array([1.0, -1.0])
    p = predict_proba(X, w, b)
    print("w = [1, -1], b = 0 时 p =", p)

    expected = np.array([0.26894142, 0.88079708, 0.81757448, 0.73105858])
    assert np.allclose(p, expected)
    print("step1_forward.py 测试通过")
