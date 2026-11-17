import numpy as np

from step1_forward import predict_proba


def nll_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam=0.0) -> float:
    """
    X: shape (N, d)
    y: shape (N,)
    w: shape (d,)
    b: 标量
    return loss: 标量，负对数似然损失
    """
    p = predict_proba(X, w, b)
    eps = 1e-12
    loss = -np.mean(
        y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
    )
    reg = 0.5 * lam * np.sum(w ** 2)
    return loss + reg


def gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam=0.0) -> tuple:
    """
    X: shape (N, d)
    y: shape (N,)
    w: shape (d,)
    b: 标量
    return:
        grad_w: shape (d,)
        grad_b: 标量
    """
    N = X.shape[0]
    p = predict_proba(X, w, b)
    error = p - y
    grad_w = X.T @ error / N + lam * w
    grad_b = np.mean(error)
    return grad_w, grad_b


if __name__ == "__main__":
    X = np.array([
        [0.0, 1.0],
        [1.0, -1.0],
        [2.0, 0.5],
        [-1.0, -2.0],
    ])
    y = np.array([1, 0, 1, 0])

    # 测试 1：w=0, b=0，p 全为 0.5，loss 应为 log(2)
    w = np.zeros(2)
    b = 0.0

    loss = nll_loss(X, y, w, b)
    grad_w, grad_b = gradients(X, y, w, b)

    print("loss =", loss)
    print("grad_w =", grad_w, "shape =", grad_w.shape)
    print("grad_b =", grad_b)

    assert np.isclose(loss, np.log(2))
    assert np.allclose(grad_w, np.array([-0.25, -0.5625]))
    assert np.isclose(grad_b, 0.0)

    # 测试 2：非零权重，只检查形状和标量类型
    w = np.array([1.0, -1.0])
    loss = nll_loss(X, y, w, b)
    grad_w, grad_b = gradients(X, y, w, b)

    print("非零权重的 loss =", loss)
    print("非零权重的 grad_w =", grad_w)

    assert np.ndim(loss) == 0
    assert grad_w.shape == (2,)
    assert np.ndim(grad_b) == 0

    print("step2_loss_grad.py 测试通过")
