#pragma once

#include "matrix.hpp"

#include <vector>

// 计算 A @ v
// A: (N, D)
// v: (D,)
// 返回: (N,)
std::vector<double> mat_vec_mul(const Matrix& A, const std::vector<double>& v);

// 计算 A^T @ v
// A: (N, D)
// v: (N,)
// 返回: (D,)
std::vector<double> mat_t_vec_mul(const Matrix& A, const std::vector<double>& v);
