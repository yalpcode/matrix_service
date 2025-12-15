#include "linear.hpp"

template <class T>
Linear<T>::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(bias) {
    static_assert(std::is_arithmetic<T>::value,
                  "Linear expects arithmetic tensor type");

    if (in_features_ == 0 || out_features_ == 0) {
        throw std::invalid_argument("Linear features must be positive");
    }

    weight_ = TensorType({out_features_, in_features_});
    bias_ = TensorType({out_features_}, T{});

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    for (size_t i = 0; i < weight_.numel(); ++i) {
        weight_.data()[i] = static_cast<T>(dist(gen));
    }

    if (use_bias_) {
        for (size_t i = 0; i < bias_.numel(); ++i) {
            bias_.data()[i] = static_cast<T>(dist(gen));
        }
    } else {
        bias_.fill(T{});
    }
}

template <class T>
typename Linear<T>::TensorType Linear<T>::forward(
    const Linear<T>::TensorType& input) const {
    if (input.dim() != 2) {
        throw std::invalid_argument("Linear expects 2D input [N, in_features]");
    }

    const auto& shape = input.shape();
    const auto batch = shape[0];
    const auto in_features = shape[1];

    if (in_features != in_features_) {
        throw std::invalid_argument("Input features do not match Linear layer");
    }

    TensorType output({batch, out_features_}, T{});

    for (size_t n = 0; n < batch; ++n) {
        for (size_t o = 0; o < out_features_; ++o) {
            T acc{};
            for (size_t i = 0; i < in_features_; ++i) {
                acc += input({n, i}) * weight_({o, i});
            }
            if (use_bias_) {
                acc += bias_({o});
            }
            output({n, o}) = acc;
        }
    }

    return output;
}
