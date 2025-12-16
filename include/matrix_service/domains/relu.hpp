#pragma once

#include <type_traits>

#include <matrix_service/domains/tensor.hpp>

template <class T = double>
class ReLU {
 public:
    using TensorType = Tensor<T>;

    ReLU() = default;

    TensorType forward(const TensorType& input) const;
};

#include "relu.hpp.inl"
