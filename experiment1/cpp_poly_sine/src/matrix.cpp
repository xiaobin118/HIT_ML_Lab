#include "matrix.hpp"

#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    // TODO:
    // 1. 检查 r < rows_ 且 c < cols_
    // 2. 越界抛 std::out_of_range
    // 3. 返回 data_[r * cols_ + c]
    (void)r;
    (void)c;
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[r * cols_ + c];
}

double Matrix::operator()(std::size_t r, std::size_t c) const {
    // TODO:
    // 1. 检查越界
    // 2. 返回 data_[r * cols_ + c]
    (void)r;
    (void)c;
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[r * cols_ + c];
}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}
