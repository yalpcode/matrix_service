#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <matrix_service/domains/tensor.hpp>

template <class T = double>
class Conv2D {
 public:
    using TensorType = Tensor<T>;

    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_h,
           size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
           size_t padding_h = 0, size_t padding_w = 0, size_t dilation_h = 1,
           size_t dilation_w = 1, size_t groups = 1, bool bias = true);

    TensorType forward(const TensorType& input) const;

    const TensorType& weight() const {
        return weight_;
    }
    const TensorType& bias() const {
        return bias_;
    }

    void load_weights(TensorType weight, TensorType bias);

 private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_h_;
    size_t kernel_w_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    size_t dilation_h_;
    size_t dilation_w_;
    size_t groups_;
    bool use_bias_;

    TensorType weight_;
    TensorType bias_;

    size_t output_height(size_t input_h) const;
    size_t output_width(size_t input_w) const;
};

#include "conv2d.hpp.inl"
