import numpy as np

from step1_forward import predict_proba
from step2_loss_grad import nll_loss, gradients


def train_logreg_gd(X: np.ndarray, y: np.ndarray,
                    lr: float = 0.5,
                    n_iter: int = 5000,
                    tol: float = 1e-6,
                    lam: float = 0.0) -> tuple:
    """
    批量梯度下降训练无惩罚逻辑回归。

    返回:
        w: shape (d,)
        b: 标量
        history: list[float]，每一轮更新前的损失
    """
    d = X.shape[1]
    w = np.zeros(d)
    b = 0.0
    history = []

    for it in range(n_iter):
        # 1. 前向：计算当前损失
        loss = nll_loss(X, y, w, b, lam)

        # 2. 反向：计算梯度
        grad_w, grad_b = gradients(X, y, w, b, lam)

        # 3. 更新参数
        w -= lr * grad_w
        b -= lr * grad_b

        # 4. 记录更新前的损失
        history.append(float(loss))

        # 5. 收敛判断
        grad_norm = np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2)
        if grad_norm < tol:
            break

    return w, b, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 500

    X = rng.normal(size=(N, 2))
    y = (X @ np.array([1.0, 1.0]) + 0.8 * rng.normal(size=N) > 0).astype(int)

    w, b, history = train_logreg_gd(
        X, y, lr=0.5, n_iter=5000, tol=1e-6
    )

    y_pred = (predict_proba(X, w, b) >= 0.5).astype(int)
    train_acc = np.mean(y_pred == y)

    print("w =", w)
    print("b =", b)
    print("iterations =", len(history))
    print("first loss =", history[0])
    print("last loss =", history[-1])
    print("train accuracy =", train_acc)

    assert len(history) > 1
    assert history[-1] < history[0]
    assert train_acc > 0.7

    print("step3.py 测试通过")
