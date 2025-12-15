#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <matrix_service/domains/tensor.hpp>

template <class T = double>
class Flatten {
 public:
    using TensorType = Tensor<T>;

    Flatten() = default;

    TensorType forward(const TensorType& input) const;
};

#include "flatten.hpp.inl"
