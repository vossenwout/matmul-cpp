#include "matmul/matrix.h"
#include <iostream>
#include <random>
#include <vector>

Matrix::Matrix(std::size_t rows, std::size_t cols) : _shape({rows, cols}), _stride({cols, 1})
{
    const int seed = 8;
    std::mt19937 gen(seed);
    float bound = 1.0f;
    for (int i = 0; i < rows * cols; i++)
    {
        _data.push_back(std::uniform_real_distribution<float>(bound, -bound)(gen));
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<float> data)
    : _data(data), _shape({rows, cols}), _stride({cols, 1})
{
    if (rows * cols != data.size())
    {
        throw std::invalid_argument("Size " + std::to_string(rows) + "," + std::to_string(cols) +
                                    "doesn't fit data with length " + std::to_string(data.size()));
    }
}
Matrix::Matrix(std::vector<std::vector<float>> matrix)
{
    for (std::size_t i = 0; i < matrix.size(); i++)
    {
        for (std::size_t j = 0; j < matrix[0].size(); j++)
        {
            _data.push_back(matrix[i][j]);
        }
    }
}

float &Matrix::operator()(std::size_t i, std::size_t j)
{
    // bound checking makes it MUCH slower so disabled
    // if (i < 0 or i >= _shape[0] or j < 0 or j >= _shape[1])
    //{
    //     throw std::invalid_argument("index " + std::to_string(i) + "," + std::to_string(j) +
    //                                "out of bounds for shape " + std::to_string(_shape[0]) + "," +
    //                               std::to_string(_shape[1]));
    // }
    return _data[i * _stride[0] + j];
}

const float &Matrix::operator()(std::size_t i, std::size_t j) const
{
    // bound checking makes it MUCH slower so disabled
    // if (i < 0 or i >= _shape[0] or j < 0 or j >= _shape[1])
    // {
    //    throw std::invalid_argument("index " + std::to_string(i) + "," + std::to_string(j) +
    //                               "out of bounds for shape " + std::to_string(_shape[0]) + "," +
    //                             std::to_string(_shape[1]));
    // }
    return _data[i * _stride[0] + j];
}

std::ostream &operator<<(std::ostream &os, const Matrix &obj)
{
    std::string string_repr = "[\n";
    for (std::size_t i = 0; i < obj._shape[0]; i++)
    {
        string_repr += " [";
        for (std::size_t j = 0; j < obj._shape[1]; j++)
        {
            string_repr += std::to_string(obj(i, j));
            if (j != obj._shape[1] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
        if (i != obj._shape[0] - 1)
        {
            string_repr += ", ";
        }
        string_repr += "\n";
    }
    string_repr += "]\n";
    os << string_repr;
    return os;
}

const std::size_t Matrix::size() const { return _data.size(); }
const std::array<std::size_t, 2> Matrix::shape() const { return _shape; }

Matrix Matrix::matmul_ijk(const Matrix &other) const
{
    std::size_t res_rows = shape()[0];
    std::size_t res_cols = other.shape()[1];
    std::vector<float> res_data(res_rows * res_cols);

    for (std::size_t i = 0; i < shape()[0]; i++)
    {
        for (std::size_t j = 0; j < other.shape()[1]; j++)
        {
            float acc = 0;
            for (std::size_t k = 0; k < shape()[1]; k++)
            {
                acc += (operator()(i, k) * other(k, j));
            }
            res_data[(i * res_cols) + j] = acc;
        }
    }
    return Matrix(res_rows, res_cols, res_data);
}

Matrix Matrix::matmul_ikj(const Matrix &other) const
{
    std::size_t res_rows = shape()[0];
    std::size_t res_cols = other.shape()[1];
    std::vector<float> res_data(res_rows * res_cols);

    for (std::size_t i = 0; i < shape()[0]; i++)
    {
        for (std::size_t k = 0; k < other.shape()[0]; k++)
        {
            float A_i_k = operator()(i, k);
            for (std::size_t j = 0; j < other.shape()[1]; j++)
            {
                res_data[i * res_cols + j] += A_i_k * other(k, j);
            }
        }
    }
    return Matrix(res_rows, res_cols, res_data);
}

Matrix Matrix::matmul_blocked_ikj(const Matrix &other, const std::size_t block_size) const
{

    // don't want to deal with edge cases
    if (shape()[0] % block_size != 0 or shape()[1] % block_size != 0 or
        other.shape()[1] % block_size != 0)
    {
        throw std::invalid_argument("block_size " + std::to_string(block_size) +
                                    "incompatible with matrix shapes");
    }

    std::size_t res_rows = shape()[0];
    std::size_t res_cols = other.shape()[1];
    std::vector<float> res_data(res_rows * res_cols);

    // indices of block top_left
    for (std::size_t ii = 0; ii < shape()[0]; ii += block_size)
    {
        // we only go through all data of B, shape()[0] / block_size times
        for (std::size_t kk = 0; kk < shape()[1]; kk += block_size)
        {
            for (std::size_t jj = 0; jj < other.shape()[1]; jj += block_size)
            {
                // indices within block
                for (std::size_t i = ii; i < ii + block_size; i++)
                {
                    for (std::size_t k = kk; k < kk + block_size; k++)
                    {
                        float i_k = operator()(i, k);
                        for (std::size_t j = jj; j < jj + block_size; j++)
                        {
                            res_data[i * res_cols + j] += i_k * other(k, j);
                        }
                    }
                }
            }
        }
    }
    return Matrix(res_rows, res_cols, res_data);
}

Matrix Matrix::matmul_parallel_blocked_ikj(const Matrix &other, const std::size_t block_size) const
{

    // don't want to deal with edge cases
    if (shape()[0] % block_size != 0 or shape()[1] % block_size != 0 or
        other.shape()[1] % block_size != 0)
    {
        throw std::invalid_argument("block_size " + std::to_string(block_size) +
                                    "incompatible with matrix shapes");
    }

    std::size_t res_rows = shape()[0];
    std::size_t res_cols = other.shape()[1];
    std::vector<float> res_data(res_rows * res_cols);

#pragma omp parallel for
    // indices of block top_left
    for (std::size_t ii = 0; ii < shape()[0]; ii += block_size)
    {
        // we only go through all data of B, shape()[0] / block_size times
        for (std::size_t kk = 0; kk < shape()[1]; kk += block_size)
        {
            for (std::size_t jj = 0; jj < other.shape()[1]; jj += block_size)
            {
                // indices within block
                for (std::size_t i = ii; i < ii + block_size; i++)
                {
                    for (std::size_t k = kk; k < kk + block_size; k++)
                    {
                        float i_k = operator()(i, k);
                        for (std::size_t j = jj; j < jj + block_size; j++)
                        {
                            res_data[i * res_cols + j] += i_k * other(k, j);
                        }
                    }
                }
            }
        }
    }
    return Matrix(res_rows, res_cols, res_data);
}
