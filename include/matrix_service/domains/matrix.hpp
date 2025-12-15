#pragma once

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

template <class T>
class Matrix {
    std::vector<T> mat;
    size_t n{0}, m{0};

 public:
    Matrix() = default;
    explicit Matrix(const std::vector<std::vector<T>>& mat);

    Matrix(const Matrix& other);

    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(const Matrix& other);

    Matrix& operator=(Matrix&& other) noexcept;

    std::pair<size_t, size_t> size() const;

    Matrix& operator+=(const Matrix& other);

    Matrix operator+(const Matrix& other) const;

    template <class U>
    Matrix& operator*=(U num);

    template <class U>
    Matrix operator*(U num) const;

    Matrix& operator*=(const Matrix& other);

    Matrix operator*(const Matrix& other) const;

    Matrix& transpose();

    Matrix transposed() const;

    typename std::vector<T>::const_iterator begin() const;

    typename std::vector<T>::iterator begin();

    typename std::vector<T>::const_iterator end() const;

    typename std::vector<T>::iterator end();

    template <class U>
    friend std::ostream& operator<<(std::ostream& out, const Matrix<U>& m);
};

#include "matrix.hpp.inl"
