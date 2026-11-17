#pragma once

#include "matrix.hpp"

#include <vector>

// 输入 x 长度 N，输出 Phi 形状 (N, degree + 1)
// 第 i 行为 [1, x_i, x_i^2, ..., x_i^degree]
Matrix polynomial_features(const std::vector<double>& x, int degree);
