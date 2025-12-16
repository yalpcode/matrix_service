#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <matrix_service/domains/tensor.hpp>

template <class T = double>
class MaxPooling {
 public:
    using TensorType = Tensor<T>;

    MaxPooling(size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
               size_t stride_w = 1, size_t padding_h = 0,
               size_t padding_w = 0);

    TensorType forward(const TensorType& input) const;

 private:
    size_t kernel_h_;
    size_t kernel_w_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;

    size_t output_height(size_t input_h) const;
    size_t output_width(size_t input_w) const;
};

#include "max_pooling.hpp.inl"
