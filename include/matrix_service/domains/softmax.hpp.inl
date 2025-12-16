#include "softmax.hpp"

#include <cmath>

template <class T>
Softmax<T>::Softmax(size_t dim) : dim_(dim) {
    static_assert(std::is_floating_point<T>::value,
                  "Softmax expects floating point tensor type");
}

template <class T>
typename Softmax<T>::TensorType Softmax<T>::forward(
    const Softmax<T>::TensorType& input) const {
    if (input.dim() == 0) {
        throw std::invalid_argument("Softmax expects tensor rank >= 1");
    }

    size_t axis = dim_ == kUseLastDim ? input.dim() - 1 : dim_;
    if (axis >= input.dim()) {
        throw std::invalid_argument("Softmax axis out of range");
    }

    const auto& shape = input.shape();
    const size_t axis_size = shape[axis];
    if (axis_size == 0) {
        throw std::invalid_argument("Softmax axis must be non-empty");
    }

    TensorType output(shape, T{});
    std::vector<size_t> idx(shape.size(), 0);
    std::vector<size_t> non_axis_dims;
    non_axis_dims.reserve(shape.size());
    for (size_t d = 0; d < shape.size(); ++d) {
        if (d != axis) {
            non_axis_dims.push_back(d);
        }
    }

    while (true) {
        T max_val = std::numeric_limits<T>::lowest();
        for (size_t k = 0; k < axis_size; ++k) {
            idx[axis] = k;
            const auto value = input.at(idx);
            if (value > max_val) {
                max_val = value;
            }
        }

        double sum = 0.0;
        for (size_t k = 0; k < axis_size; ++k) {
            idx[axis] = k;
            const double exp_val =
                std::exp(static_cast<double>(input.at(idx) - max_val));
            output.at(idx) = static_cast<T>(exp_val);
            sum += exp_val;
        }

        if (!std::isfinite(sum) || sum == 0.0) {
            throw std::runtime_error("Softmax encountered invalid normalization");
        }

        for (size_t k = 0; k < axis_size; ++k) {
            idx[axis] = k;
            output.at(idx) =
                static_cast<T>(static_cast<double>(output.at(idx)) / sum);
        }

        idx[axis] = 0;

        if (non_axis_dims.empty()) {
            break;
        }

        bool advanced = false;
        for (auto it = non_axis_dims.rbegin(); it != non_axis_dims.rend();
             ++it) {
            const auto d = *it;
            idx[d]++;
            if (idx[d] < shape[d]) {
                advanced = true;
                break;
            }
            idx[d] = 0;
        }

        if (!advanced) {
            break;
        }
    }

    return output;
}
