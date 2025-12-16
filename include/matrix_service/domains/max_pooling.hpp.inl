#include "max_pooling.hpp"

#include <limits>

template <class T>
MaxPooling<T>::MaxPooling(size_t kernel_h, size_t kernel_w, size_t stride_h,
                          size_t stride_w, size_t padding_h, size_t padding_w)
    : kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      padding_h_(padding_h),
      padding_w_(padding_w) {
    static_assert(std::is_arithmetic<T>::value,
                  "MaxPooling expects arithmetic tensor type");

    if (kernel_h_ == 0 || kernel_w_ == 0) {
        throw std::invalid_argument("Kernel dimensions must be positive");
    }
    if (stride_h_ == 0 || stride_w_ == 0) {
        throw std::invalid_argument("Stride dimensions must be positive");
    }
}

template <class T>
typename MaxPooling<T>::TensorType MaxPooling<T>::forward(
    const MaxPooling<T>::TensorType& input) const {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }

    const auto& in_shape = input.shape();
    const auto batch_size = in_shape[0];
    const auto channels = in_shape[1];
    const auto in_h = in_shape[2];
    const auto in_w = in_shape[3];

    if (in_h == 0 || in_w == 0) {
        throw std::invalid_argument("Input spatial dimensions must be positive");
    }

    const size_t out_h = output_height(in_h);
    const size_t out_w = output_width(in_w);

    TensorType output({batch_size, channels, out_h, out_w});

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const auto h_start =
                        static_cast<long long>(oh * stride_h_) -
                        static_cast<long long>(padding_h_);
                    const auto w_start =
                        static_cast<long long>(ow * stride_w_) -
                        static_cast<long long>(padding_w_);

                    bool has_value = false;
                    T max_val = std::numeric_limits<T>::lowest();

                    for (size_t kh = 0; kh < kernel_h_; ++kh) {
                        const auto ih =
                            h_start + static_cast<long long>(kh);
                        if (ih < 0 || ih >= static_cast<long long>(in_h)) {
                            continue;
                        }
                        for (size_t kw = 0; kw < kernel_w_; ++kw) {
                            const auto iw =
                                w_start + static_cast<long long>(kw);
                            if (iw < 0 ||
                                iw >= static_cast<long long>(in_w)) {
                                continue;
                            }

                            const auto value =
                                input({n, c, static_cast<size_t>(ih),
                                       static_cast<size_t>(iw)});
                            if (!has_value || value > max_val) {
                                max_val = value;
                                has_value = true;
                            }
                        }
                    }

                    if (!has_value) {
                        throw std::invalid_argument(
                            "Pooling window has no valid elements");
                    }

                    output({n, c, oh, ow}) = max_val;
                }
            }
        }
    }

    return output;
}

template <class T>
size_t MaxPooling<T>::output_height(size_t input_h) const {
    const auto numerator = static_cast<long long>(input_h) +
                           static_cast<long long>(2 * padding_h_) -
                           static_cast<long long>(kernel_h_);
    if (numerator < 0) {
        throw std::invalid_argument(
            "Kernel height larger than padded input height");
    }

    return static_cast<size_t>(numerator / static_cast<long long>(stride_h_) +
                               1);
}

template <class T>
size_t MaxPooling<T>::output_width(size_t input_w) const {
    const auto numerator = static_cast<long long>(input_w) +
                           static_cast<long long>(2 * padding_w_) -
                           static_cast<long long>(kernel_w_);
    if (numerator < 0) {
        throw std::invalid_argument(
            "Kernel width larger than padded input width");
    }

    return static_cast<size_t>(numerator / static_cast<long long>(stride_w_) +
                               1);
}
