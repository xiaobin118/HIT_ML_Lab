#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);

    double& operator()(std::size_t r, std::size_t c);
    double operator()(std::size_t r, std::size_t c) const;

    std::size_t rows() const;
    std::size_t cols() const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};
