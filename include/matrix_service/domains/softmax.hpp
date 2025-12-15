#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <matrix_service/domains/tensor.hpp>

template <class T = double>
class Softmax {
 public:
    using TensorType = Tensor<T>;
    static constexpr size_t kUseLastDim = std::numeric_limits<size_t>::max();

    explicit Softmax(size_t dim = kUseLastDim);

    TensorType forward(const TensorType& input) const;

 private:
    size_t dim_;
};

#include "softmax.hpp.inl"
