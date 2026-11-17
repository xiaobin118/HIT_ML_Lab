# -*- coding: utf-8 -*-
"""
第 2 课：生成正弦函数 + 噪声数据（骨架）

只填写 TODO，不要改函数名。
运行：
    python step02_generate_data.py
"""

import numpy as np

SEED = 42


def generate_data(n, noise_std=0, seed=SEED):
    """
    生成训练数据。

    任务：
    1. 用 np.random.default_rng(seed) 创建随机数生成器 rng。
    2. 生成 n 个 x，x 在 [-1, 1] 上均匀分布。
    3. 生成 n 个标准正态噪声 eps。
    4. 生成 y = sin(2*pi*x) + noise_std * eps。
    5. 返回 x, y。

    要求：
    - x、y 都是一维 np.ndarray，shape 为 (n,)；
    - 不允许写 Python for 循环。
    """
    # TODO
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray = rng.uniform(-1, 1, n)
    eps: np.ndarray = rng.normal(size=n)
    y: np.ndarray = np.sin(2 * np.pi * x) + noise_std * eps
    return x, y


def make_test_data(num=600):
    """
    生成无噪声测试数据。

    任务：
    1. 用 np.linspace 在 [-1, 1] 上生成 num 个点，记为 x_test。
    2. 用真实函数计算 y_test = sin(2*pi*x_test)。
    3. 返回 x_test, y_test。

    注意：
    - 测试集不加噪声；
    - 后面用测试误差衡量模型对真实正弦函数的泛化能力。
    """
    # TODO
    x_test = np.linspace(-1, 1, num)
    y_test = np.sin(2 * np.pi * x_test)
    return x_test, y_test


def main():
    x_train, y_train = generate_data(15)
    x_test, y_test = make_test_data()

    # 写完后，增加打印，观察：
    # 1. x_train.shape, y_train.shape
    # 2. x_test.shape, y_test.shape
    # 3. x_train[:5], y_train[:5]
    # 4. np.sin(2*np.pi*x_train[:3]) 和 y_train[:3] 的差别
    # 5. 把 noise_std 改成 0，再运行一次，观察 y_train 和真实正弦是否一致
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)
    print("x_train[:5]:", x_train[:5])
    print("y_train[:5]:", y_train[:5])
    print("np.sin(2*np.pi*x_train[:3]):", np.sin(2 * np.pi * x_train[:3]))
    print("y_train[:3]:", y_train[:3])
    print("Difference:", np.sin(2 * np.pi * x_train[:3]) - y_train[:3])


if __name__ == "__main__":
    main()
