#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <class T>
concept IsArithmetic = std::is_arithmetic_v<T>;

template <IsArithmetic T = double>
class Tensor {
 public:
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    Tensor();
    explicit Tensor(Shape shape, const T& value = T{});
    Tensor(std::initializer_list<size_t> shape, const T& value = T{});
    Tensor(Shape shape, std::vector<T> data);

    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);

    template <IsArithmetic U = T>
    static Tensor arange(U start, U end, U step = U{1});

    size_t dim() const;
    const Shape& shape() const;
    const Strides& strides() const;
    size_t numel() const;
    bool is_contiguous() const;

    T* data();
    const T* data() const;

    template <class... Index>
    T& operator()(Index... indices);

    template <class... Index>
    const T& operator()(Index... indices) const;

    T& operator()(std::initializer_list<size_t> indices);
    const T& operator()(std::initializer_list<size_t> indices) const;

    T& at(const Shape& indices);
    const T& at(const Shape& indices) const;

    Tensor clone() const;

    Tensor reshape(const Shape& new_shape) const;
    Tensor permute(const Shape& dims) const;
    Tensor transpose(size_t dim0, size_t dim1) const;
    Tensor contiguous() const;
    Tensor flatten() const;
    Tensor unsqueeze(size_t dim) const;
    Tensor squeeze(size_t dim) const;

    Tensor& fill(const T& value);

    Tensor& operator+=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(T scalar);
    Tensor operator+(T scalar) const;

    Tensor& operator-=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(T scalar);
    Tensor operator-(T scalar) const;

    Tensor& operator*=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(T scalar);
    Tensor operator*(T scalar) const;

    Tensor matmul(const Tensor& other) const;

    Tensor sum() const;
    Tensor sum(size_t dim) const;

 private:
    Shape shape_;
    Strides strides_;
    std::shared_ptr<std::vector<T>> data_;
    size_t offset_{0};

    Tensor(Shape shape, std::shared_ptr<std::vector<T>> data, Strides strides,
           size_t offset);

    static Strides make_contiguous_strides(const Shape& shape);
    static size_t compute_numel(const Shape& shape);
    size_t compute_offset(const Shape& indices) const;

    template <class Fn>
    void for_each_index(Fn&& fn) const;

    template <class Fn>
    void apply_scalar(Fn&& fn, const T& scalar);

    template <class Fn>
    void apply_tensor(const Tensor& other, Fn&& fn);
};

#include "tensor.hpp.inl"
