#include "polynomial_features.hpp"

#include <cstddef>
#include <stdexcept>

Matrix polynomial_features(const std::vector<double> &x, int degree) {
    // TODO:
    // 1. degree < 0 抛异常
    // 2. N = x.size()
    // 3. 创建 Matrix Phi(N, degree + 1)
    // 4. 对每个样本 i:
    //      Phi(i, 0) = 1.0
    //      Phi(i, j) = Phi(i, j - 1) * x[i]
    // 5. 返回 Phi
    (void)x;
    (void)degree;
    if (degree < 0) {
        throw std::invalid_argument("Degree must be non-negative");
    }
    std::size_t N = x.size();
    Matrix Phi(N, degree + 1);
    for (std::size_t i = 0; i < N; ++i) {
        Phi(i, 0) = 1.0;
        for (int j = 1; j <= degree; ++j) {
            Phi(i, j) = Phi(i, j - 1) * x[i];
        }
    }
    return Phi;
}
