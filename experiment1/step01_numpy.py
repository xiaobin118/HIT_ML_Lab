# -*- coding: utf-8 -*-
"""
第 1 课：NumPy 数组与向量化（骨架）

规则：
1. 只填写每个函数中的 TODO，不要改函数名。
2. 每完成一个任务，就在 main() 中取消对应 print 的注释，运行:
       python step01_numpy.py
3. 报错是正常的，先根据报错自己查，再把代码/报错发给我 review。
"""

import numpy as np


def task1_create_array():
    """
    任务 1：创建一个一维数组 [0, 1, 2, 3, 4]，并返回它。
    提示：使用 np.array(...)
    numpy 的数组是 np.ndarray 类型
    ndarray 的底层是一块连续的内存, 加上dtype, shape, strides, ndim等属性, 使得它可以高效地进行向量化运算。
    """
    # TODO
    return np.array([0, 1, 2, 3, 4])


def task2_shape_and_dtype():
    """
    任务 2：创建一个 2 行 3 列的全零数组。
    打印它的 shape 和 dtype，最后返回这个数组。
    提示：np.zeros(...)，.shape，.dtype
    """
    # TODO
    return np.zeros((2, 3))


def task3_linspace():
    """
    任务 3：在区间 [-1, 1] 上生成 5 个等距点，返回这个数组。
    提示：np.linspace(start, stop, num)
    """
    # TODO
    return np.linspace(-1, 1, 5)


def task4_vectorization():
    """
    任务 4：给定 a = np.array([-1, 0, 1])，
    不使用 for 循环，计算 a**2 + 2*a + 1，返回结果。
    提示：NumPy 的四则运算会自动作用于每个元素。
    """
    # TODO
    a = np.array([-1, 0, 1])
    return a**2 + 2 * a + 1


def task5_sin():
    """
    任务 5：给定 x = np.array([0.0, 0.5, 1.0])，
    计算 y = np.sin(2 * np.pi * x)，返回 y。
    提示：np.sin，np.pi
    """
    # TODO
    x = np.array([0.0, 0.5, 1.0])
    y = np.sin(2 * np.pi * x)
    return y


def task6_random_data():
    """
    任务 6：使用 rng = np.random.default_rng(42)。
    生成 5 个在 [-1, 1] 上均匀分布的 x；
    再生成 5 个服从标准正态分布的 eps。
    返回 x, eps。
    提示：rng.uniform(low, high, size)，rng.normal(size=...)
    """
    # TODO
    rng: np.random.Generator = np.random.default_rng(42)
    x: np.ndarray = rng.uniform(-1, 1, 5)
    eps: np.ndarray = rng.normal(size=5)
    return x, eps


def task7_matrix_mul():
    """
    任务 7：给定
        A = np.array([[1, 2], [3, 4]])
        v = np.array([5, 6])
    使用 @ 运算符计算矩阵乘向量，返回结果。
    提示：结果形状应该是 (2,)。
    """
    # TODO
    A: np.ndarray = np.array([[1, 2], [3, 4]])
    v: np.ndarray = np.array([5, 6])
    return A @ v


def main():
    # 完成一个任务后，取消对应行注释。
    print("task1:", task1_create_array())
    print("task2:", task2_shape_and_dtype())
    print("task3:", task3_linspace())
    print("task4:", task4_vectorization())
    print("task5:", task5_sin())
    print("task6:", task6_random_data())
    print("task7:", task7_matrix_mul())


if __name__ == "__main__":
    main()
