#include "conv2d.hpp"

template <class T>
Conv2D<T>::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_h,
                  size_t kernel_w, size_t stride_h, size_t stride_w,
                  size_t padding_h, size_t padding_w, size_t dilation_h,
                  size_t dilation_w, size_t groups, bool bias)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      padding_h_(padding_h),
      padding_w_(padding_w),
      dilation_h_(dilation_h),
      dilation_w_(dilation_w),
      groups_(groups),
      use_bias_(bias) {
    static_assert(std::is_arithmetic<T>::value,
                  "Conv2D expects arithmetic tensor type");

    if (groups_ == 0) {
        throw std::invalid_argument("Groups must be positive");
    }
    if (in_channels_ == 0 || out_channels_ == 0) {
        throw std::invalid_argument("Channels must be positive");
    }
    if (kernel_h_ == 0 || kernel_w_ == 0) {
        throw std::invalid_argument("Kernel dimensions must be positive");
    }
    if (in_channels_ % groups_ != 0 || out_channels_ % groups_ != 0) {
        throw std::invalid_argument(
            "In and out channels must be divisible by groups");
    }

    weight_ = TensorType(
        {out_channels_, in_channels_ / groups_, kernel_h_, kernel_w_});
    bias_ = TensorType({out_channels_}, T{});

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
void Conv2D<T>::load_weights(typename Conv2D<T>::TensorType weight,
                             typename Conv2D<T>::TensorType bias) {
    const TensorType expected_w_shape(
        {out_channels_, in_channels_ / groups_, kernel_h_, kernel_w_});
    if (weight.shape() != expected_w_shape.shape()) {
        throw std::invalid_argument("Conv2D weight shape mismatch");
    }
    if (bias.shape() != TensorType({out_channels_}).shape()) {
        throw std::invalid_argument("Conv2D bias shape mismatch");
    }
    weight_ = std::move(weight);
    bias_ = std::move(bias);
}

template <class T>
typename Conv2D<T>::TensorType Conv2D<T>::forward(
    const Conv2D<T>::TensorType& input) const {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }

    const auto& in_shape = input.shape();
    const auto batch_size = in_shape[0];
    const auto channels = in_shape[1];
    const auto in_h = in_shape[2];
    const auto in_w = in_shape[3];

    if (channels != in_channels_) {
        throw std::invalid_argument("Input channels do not match Conv2D");
    }

    const size_t out_h = output_height(in_h);
    const size_t out_w = output_width(in_w);

    TensorType output({batch_size, out_channels_, out_h, out_w}, T{});

    const size_t channels_per_group = in_channels_ / groups_;
    const size_t out_channels_per_group = out_channels_ / groups_;

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            const size_t group_idx = oc / out_channels_per_group;
            const size_t in_group_offset = group_idx * channels_per_group;

            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    T acc{};
                    const auto h_start =
                        static_cast<long long>(oh * stride_h_) -
                        static_cast<long long>(padding_h_);
                    const auto w_start =
                        static_cast<long long>(ow * stride_w_) -
                        static_cast<long long>(padding_w_);

                    for (size_t ic = 0; ic < channels_per_group; ++ic) {
                        for (size_t kh = 0; kh < kernel_h_; ++kh) {
                            const auto ih =
                                h_start + static_cast<long long>(kh) *
                                             static_cast<long long>(dilation_h_);
                            if (ih < 0 || ih >= static_cast<long long>(in_h)) {
                                continue;
                            }
                            for (size_t kw = 0; kw < kernel_w_; ++kw) {
                                const auto iw =
                                    w_start + static_cast<long long>(kw) *
                                                   static_cast<long long>(
                                                       dilation_w_);
                                if (iw < 0 ||
                                    iw >= static_cast<long long>(in_w)) {
                                    continue;
                                }

                                acc += input({n, in_group_offset + ic,
                                              static_cast<size_t>(ih),
                                              static_cast<size_t>(iw)}) *
                                       weight_({oc, ic, kh, kw});
                            }
                        }
                    }

                    if (use_bias_) {
                        acc += bias_({oc});
                    }

                    output({n, oc, oh, ow}) = acc;
                }
            }
        }
    }

    return output;
}

template <class T>
size_t Conv2D<T>::output_height(size_t input_h) const {
    return (input_h + 2 * padding_h_ - dilation_h_ * (kernel_h_ - 1) - 1) /
               stride_h_ +
           1;
}

template <class T>
size_t Conv2D<T>::output_width(size_t input_w) const {
    return (input_w + 2 * padding_w_ - dilation_w_ * (kernel_w_ - 1) - 1) /
               stride_w_ +
           1;
}
