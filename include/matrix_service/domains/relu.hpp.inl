#include "relu.hpp"

template <class T>
typename ReLU<T>::TensorType ReLU<T>::forward(
    const ReLU<T>::TensorType& input) const {
    static_assert(std::is_arithmetic<T>::value,
                  "ReLU expects arithmetic tensor type");

    TensorType output = input.clone();
    T* data = output.data();
    const auto total = output.numel();
    for (size_t i = 0; i < total; ++i) {
        data[i] = data[i] < T{} ? T{} : data[i];
    }
    return output;
}
