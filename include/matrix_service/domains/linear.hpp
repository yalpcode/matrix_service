#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <matrix_service/domains/tensor.hpp>

template <class T = double>
class Linear {
 public:
    using TensorType = Tensor<T>;

    Linear(size_t in_features, size_t out_features, bool bias = true);

    TensorType forward(const TensorType& input) const;

    const TensorType& weight() const {
        return weight_;
    }

    const TensorType& bias() const {
        return bias_;
    }

 private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;

    TensorType weight_;
    TensorType bias_;
};

#include "linear.hpp.inl"
