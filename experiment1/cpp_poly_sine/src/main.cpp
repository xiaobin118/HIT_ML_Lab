#include "data.hpp"
#include "linear_algebra.hpp"
#include "matrix.hpp"
#include "polynomial_features.hpp"

#include <algorithm>
#include <iostream>

int main() {
    // 第 2 课已经把 polynomial_features 验证过了。
    // 第 3 课任务：
    // 1. 生成 N=15, noise_std=0.2 的训练数据；
    // 2. degree=5，构造 Phi；
    // 3. 构造一个权重向量 w，长度为 degree + 1；
    // 4. 用 mat_vec_mul(Phi, w) 计算 y_pred；
    // 5. 打印 y_pred.size()；
    // 6. 用一个 3x2 的小矩阵手算验证 mat_vec_mul 和 mat_t_vec_mul。
    //
    // 注意：先不要把 main 写死，先用小矩阵验证函数正确性。

    std::cout << "TODO: linear algebra operations\n";
    return 0;
}
