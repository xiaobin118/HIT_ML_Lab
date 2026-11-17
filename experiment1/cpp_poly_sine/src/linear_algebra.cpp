#include "linear_algebra.hpp"

#include <stdexcept>

std::vector<double> mat_vec_mul(const Matrix& A, const std::vector<double>& v) {
    // TODO:
    // 1. 检查 A.cols() == v.size()，不满足抛 std::invalid_argument
    // 2. result 长度 A.rows()，初始化为 0
    // 3. 三层思路：result[i] = sum_j A(i, j) * v[j]
    // 4. 返回 result
    (void)A;
    (void)v;
    throw std::logic_error("mat_vec_mul not implemented");
}

std::vector<double> mat_t_vec_mul(const Matrix& A, const std::vector<double>& v) {
    // TODO:
    // 1. 检查 A.rows() == v.size()，不满足抛 std::invalid_argument
    // 2. result 长度 A.cols()，初始化为 0
    // 3. result[j] = sum_i A(i, j) * v[i]
    // 4. 返回 result
    (void)A;
    (void)v;
    throw std::logic_error("mat_t_vec_mul not implemented");
}
