#include "matrix.hpp"

template <class T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& source) {
    if (source.empty() || source.front().empty()) {
        throw std::invalid_argument("Matrix cannot be empty");
    }

    const auto expected_cols = source.front().size();
    for (const auto& row : source) {
        if (row.size() != expected_cols) {
            throw std::invalid_argument("Matrix rows must have the same length");
        }
    }

    n = source.size();
    m = expected_cols;
    mat.resize(n * m);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            mat[i * m + j] = source[i][j];
        }
    }
}

template <class T>
Matrix<T>::Matrix(const Matrix& other) = default;

template <class T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : mat(std::move(other.mat)), n(other.n), m(other.m) {
    other.n = 0;
    other.m = 0;
}

template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) = default;

template <class T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        mat = std::move(other.mat);
        n = other.n;
        m = other.m;
        other.n = 0;
        other.m = 0;
    }

    return *this;
}

template <class T>
std::pair<size_t, size_t> Matrix<T>::size() const {
    return {n, m};
}

template <class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& other) {
    if (n != other.n || m != other.m) {
        throw std::invalid_argument(
            "Matrix dimensions mismatch for addition");
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            mat[i * m + j] += other.mat[i * other.m + j];
        }
    }

    return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    Matrix res(*this);

    return res += other;
}

template <class T>
template <class U>
Matrix<T>& Matrix<T>::operator*=(U num) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            mat[i * m + j] *= num;
        }
    }

    return *this;
}

template <class T>
template <class U>
Matrix<T> Matrix<T>::operator*(U num) const {
    Matrix res(*this);

    return res *= num;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other) {
    if (m != other.n) {
        throw std::invalid_argument(
            "Matrix dimensions mismatch for multiplication");
    }

    std::vector<T> tmp(n * other.m, T{});

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < other.m; ++j) {
            T sum = T{};
            for (size_t k = 0; k < m; ++k) {
                sum += mat[i * m + k] * other.mat[k * other.m + j];
            }
            tmp[i * other.m + j] = sum;
        }
    }

    mat.swap(tmp);
    m = other.m;

    return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    Matrix res(*this);

    return res *= other;
}

template <class T>
Matrix<T>& Matrix<T>::transpose() {
    std::vector<T> tmp(mat.size(), T{});

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            tmp[j * n + i] = mat[i * m + j];
        }
    }

    mat.swap(tmp);
    std::swap(n, m);

    return *this;
}

template <class T>
Matrix<T> Matrix<T>::transposed() const {
    Matrix res(*this);

    return res.transpose();
}

template <class T>
typename std::vector<T>::const_iterator Matrix<T>::begin() const {
    return mat.cbegin();
}

template <class T>
typename std::vector<T>::iterator Matrix<T>::begin() {
    return mat.begin();
}

template <class T>
typename std::vector<T>::const_iterator Matrix<T>::end() const {
    return mat.cend();
}

template <class T>
typename std::vector<T>::iterator Matrix<T>::end() {
    return mat.end();
}

template <class T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& m) {
    for (size_t i = 0; i < m.n; ++i) {
        if (i != 0) {
            out << '\n';
        }
        for (size_t j = 0; j < m.m; ++j) {
            if (j != 0) {
                out << '\t';
            }
            out << m.mat[i * m.m + j];
        }
    }

    return out;
}
