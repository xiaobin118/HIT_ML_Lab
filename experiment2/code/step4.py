import numpy as np

from step1_forward import predict_proba
from step3 import train_logreg_gd


def make_gaussian_data(n_per_class: int = 500,
                       rho: float = 0.8,
                       seed: int = 0) -> tuple:
    """
    生成两类相关高斯数据。

    类 0: mean = [-1, 0]
    类 1: mean = [ 1, 0]
    cov  = [[1, rho],
            [rho, 1]]

    返回:
        X: shape (2*n_per_class, 2)
        y: shape (2*n_per_class,)
    """
    rng = np.random.default_rng(seed)

    mean0 = np.array([-1.0, 0.0])
    mean1 = np.array([1.0, 0.0])
    cov = np.array([
        [1.0, rho],
        [rho, 1.0],
    ])

    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)

    X = np.vstack([X0, X1])
    y = np.r_[
        np.zeros(n_per_class, dtype=int),
        np.ones(n_per_class, dtype=int),
    ]

    idx = rng.permutation(2 * n_per_class)
    return X[idx], y[idx]


def split_dataset(X: np.ndarray, y: np.ndarray,
                  test_ratio: float = 0.3,
                  seed: int = 42) -> tuple:
    """
    返回:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))

    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
    )


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    用训练集统计量做标准化：
        X_std = (X - mu) / sigma
    返回:
        X_train_std, X_test_std, mu, sigma
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    return X_train_std, X_test_std, mu, sigma


def predict_class(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """返回 0/1 预测标签。"""
    return (predict_proba(X, w, b) >= 0.5).astype(int)


def accuracy(X: np.ndarray, y: np.ndarray,
             w: np.ndarray, b: float) -> float:
    return float(np.mean(predict_class(X, w, b) == y))


if __name__ == "__main__":
    # 生成人工相关高斯数据
    X, y = make_gaussian_data(n_per_class=500, rho=0.8, seed=0)

    print("X.shape =", X.shape)
    print("y.shape =", y.shape)
    print("y.mean() =", y.mean())
    print("class 0 mean =", X[y == 0].mean(axis=0))
    print("class 1 mean =", X[y == 1].mean(axis=0))
    print("class 0 corrcoef =\n", np.corrcoef(X[y == 0].T))
    print("class 1 corrcoef =\n", np.corrcoef(X[y == 1].T))

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=0.3, seed=42
    )

    print("X_train.shape =", X_train.shape)
    print("X_test.shape  =", X_test.shape)
    print("y_train.shape =", y_train.shape)
    print("y_test.shape  =", y_test.shape)

    # 标准化
    X_train_std, X_test_std, mu, sigma = standardize(X_train, X_test)

    print("mu.shape =", mu.shape)
    print("sigma.shape =", sigma.shape)
    print("train mean after std =", X_train_std.mean(axis=0))
    print("train std  after std =", X_train_std.std(axis=0))

    # 在标准化数据上训练逻辑回归
    w, b, history = train_logreg_gd(
        X_train_std, y_train,
        lr=0.5, n_iter=20000, tol=1e-6
    )

    print("w =", w)
    print("b =", b)
    print("iterations =", len(history))
    print("train accuracy =", accuracy(X_train_std, y_train, w, b))
    print("test  accuracy =", accuracy(X_test_std, y_test, w, b))

    assert X.shape == (1000, 2)
    assert y.shape == (1000,)
    assert X_train.shape == (700, 2)
    assert X_test.shape == (300, 2)
    assert mu.shape == (2,)
    assert sigma.shape == (2,)
    assert np.allclose(X_train_std.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X_train_std.std(axis=0), 1.0, atol=1e-8)

    print("step4.py 测试通过")
