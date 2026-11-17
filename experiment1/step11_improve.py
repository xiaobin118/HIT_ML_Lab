# -*- coding: utf-8 -*-
"""
第 11 课：诊断与改进模型（骨架）

运行：
    python step11_improve.py
"""

import numpy as np

from step02_generate_data import generate_data, make_test_data
from step09_polynomial_l2 import train_polynomial_l2


def run_grid():
    """
    任务：
    1. 固定测试集；
    2. 对不同的 N、degree、l2 组合训练模型；
    3. 记录 train_mse、test_mse、||w||；
    4. 返回一个 list[dict]。

    建议搜索范围：
        N       = [15, 30, 100]
        degree  = [5, 9, 15]
        l2      = [0.0, 0.01, 0.1]

    注意：
    - 每次 generate_data 都使用相同 seed，方便复现；
    - 训练轮数可以用 60000。
    """
    # TODO
    # use 网格搜索方法，遍历 N, degree, l2 的组合
    results = []

    N_list = [15, 30, 100]
    degree_list = [5, 9, 15]
    l2_list = [0.0, 0.01, 0.1]

    x_test, y_test = make_test_data(600)

    for N in N_list:
        x_train, y_train = generate_data(N, noise_std=0.2, seed=42)

        for degree in degree_list:
            for l2 in l2_list:
                result = train_polynomial_l2(
                    x_train, y_train,
                    x_test, y_test,
                    degree=degree,
                    l2=l2,
                    epochs=60000,
                )

                results.append({
                    "N": N,
                    "degree": degree,
                    "l2": l2,
                    "train_mse": result["train_mse"],
                    "test_mse": result["test_mse"],
                    "w_norm": np.linalg.norm(result["w"]),
                })

    return results


def main():
    results = run_grid()

    # 打印表格
    # 找到测试误差最小的配置
    results = run_grid()

    print(f"{'N':>4s} {'degree':>8s} {'l2':>8s} "
          f"{'train_mse':>12s} {'test_mse':>12s} {'||w||':>10s}")

    for r in results:
        print(f"{r['N']:4d} {r['degree']:8d} {r['l2']:8.4g} "
              f"{r['train_mse']:12.6f} {r['test_mse']:12.6f} "
              f"{r['w_norm']:10.4f}")

    best = min(results, key=lambda r: r["test_mse"])
    print("\nbest config:", best)


if __name__ == "__main__":
    main()
