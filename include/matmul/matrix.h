#pragma once
#include <array>
#include <iostream>
#include <vector>

class Matrix
{
private:
    std::vector<float> _data;
    std::array<std::size_t, 2> _shape;
    std::array<std::size_t, 2> _stride;

public:
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, std::vector<float> data);
    Matrix(std::vector<std::vector<float>> matrix);
    float &operator()(std::size_t i, std::size_t j);
    const float &operator()(std::size_t i, std::size_t j) const;
    friend std::ostream &operator<<(std::ostream &os, const Matrix &obj);
    const std::size_t size() const;
    const std::array<std::size_t, 2> shape() const;
    Matrix matmul_ijk(const Matrix &other) const;
    Matrix matmul_ikj(const Matrix &other) const;
    Matrix matmul_blocked_ikj(const Matrix &other, const std::size_t block_size) const;
    Matrix matmul_parallel_blocked_ikj(const Matrix &other, const std::size_t block_size) const;
};
