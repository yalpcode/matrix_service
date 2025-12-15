#include "flatten.hpp"

template <class T>
typename Flatten<T>::TensorType Flatten<T>::forward(
    const Flatten<T>::TensorType& input) const {
    if (input.dim() == 0) {
        throw std::invalid_argument("Flatten expects tensor rank >= 1");
    }

    const auto& shape = input.shape();
    const size_t batch = shape[0];

    size_t features = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        features *= shape[i];
    }

    TensorType contiguous = input.is_contiguous() ? input : input.contiguous();
    return contiguous.reshape({batch, features});
}
