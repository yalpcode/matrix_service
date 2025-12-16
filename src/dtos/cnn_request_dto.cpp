#include <matrix_service/dtos/cnn_request_dto.hpp>
#include <stdexcept>
#include <userver/formats/parse/common_containers.hpp>
#include <vector>

CnnRequestDTO::CnnRequestDTO(const userver::formats::json::Value& json) {
    const auto images =
        json["inputs"].As<std::vector<std::vector<std::vector<double>>>>();

    if (images.empty()) {
        throw std::invalid_argument("inputs must contain at least one image");
    }

    const auto height = images.front().size();
    if (height == 0) {
        throw std::invalid_argument("image height must be positive");
    }
    const auto width = images.front().front().size();
    if (width == 0) {
        throw std::invalid_argument("image width must be positive");
    }

    for (const auto& img : images) {
        if (img.size() != height) {
            throw std::invalid_argument("all images must have same height");
        }
        for (const auto& row : img) {
            if (row.size() != width) {
                throw std::invalid_argument("all images must have same width");
            }
        }
    }

    const size_t batch = images.size();
    input_ = Tensor<double>({batch, 1, height, width});

    for (size_t n = 0; n < batch; ++n) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                input_({n, 0, h, w}) = images[n][h][w];
            }
        }
    }
    if (json.HasMember("weights")) {
        const auto weights_json = json["weights"];

        const auto conv_weight =
            weights_json["conv1"]["weight"]
                .As<std::vector<
                    std::vector<std::vector<std::vector<double>>>>>();
        const auto conv_bias =
            weights_json["conv1"]["bias"].As<std::vector<double>>();

        const auto fc1_weight = weights_json["fc1"]["weight"]
                                    .As<std::vector<std::vector<double>>>();
        const auto fc1_bias =
            weights_json["fc1"]["bias"].As<std::vector<double>>();

        const auto fc2_weight = weights_json["fc2"]["weight"]
                                    .As<std::vector<std::vector<double>>>();
        const auto fc2_bias =
            weights_json["fc2"]["bias"].As<std::vector<double>>();

        if (conv_weight.size() != 32) {
            throw std::invalid_argument(
                "conv1 weight outer dimension must be 32");
        }
        if (conv_bias.size() != 32) {
            throw std::invalid_argument("conv1 bias size must be 32");
        }
        for (const auto& oc : conv_weight) {
            if (oc.size() != 1) {
                throw std::invalid_argument(
                    "conv1 weight must have in_channels dimension 1");
            }
            for (const auto& ic : oc) {
                if (ic.size() != 3) {
                    throw std::invalid_argument(
                        "conv1 kernel height must be 3");
                }
                for (const auto& row : ic) {
                    if (row.size() != 3) {
                        throw std::invalid_argument(
                            "conv1 kernel width must be 3");
                    }
                }
            }
        }

        if (fc1_weight.size() != 128) {
            throw std::invalid_argument("fc1 weight must have 128 rows");
        }
        for (const auto& row : fc1_weight) {
            if (row.size() != 32 * 14 * 14) {
                throw std::invalid_argument(
                    "fc1 weight must have 32*14*14 columns");
            }
        }
        if (fc1_bias.size() != 128) {
            throw std::invalid_argument("fc1 bias size must be 128");
        }

        if (fc2_weight.size() != 10) {
            throw std::invalid_argument("fc2 weight must have 10 rows");
        }
        for (const auto& row : fc2_weight) {
            if (row.size() != 128) {
                throw std::invalid_argument("fc2 weight must have 128 columns");
            }
        }
        if (fc2_bias.size() != 10) {
            throw std::invalid_argument("fc2 bias size must be 10");
        }

        auto conv_w_tensor = Tensor<double>({32, 1, 3, 3});
        for (size_t oc = 0; oc < 32; ++oc) {
            for (size_t ic = 0; ic < 1; ++ic) {
                for (size_t h = 0; h < 3; ++h) {
                    for (size_t w = 0; w < 3; ++w) {
                        conv_w_tensor({oc, ic, h, w}) =
                            conv_weight[oc][ic][h][w];
                    }
                }
            }
        }
        auto conv_b_tensor = Tensor<double>({32});
        for (size_t i = 0; i < 32; ++i) {
            conv_b_tensor({i}) = conv_bias[i];
        }

        auto fc1_w_tensor = Tensor<double>({128, 32 * 14 * 14});
        for (size_t r = 0; r < 128; ++r) {
            for (size_t c = 0; c < 32 * 14 * 14; ++c) {
                fc1_w_tensor({r, c}) = fc1_weight[r][c];
            }
        }
        auto fc1_b_tensor = Tensor<double>({128});
        for (size_t i = 0; i < 128; ++i) {
            fc1_b_tensor({i}) = fc1_bias[i];
        }

        auto fc2_w_tensor = Tensor<double>({10, 128});
        for (size_t r = 0; r < 10; ++r) {
            for (size_t c = 0; c < 128; ++c) {
                fc2_w_tensor({r, c}) = fc2_weight[r][c];
            }
        }
        auto fc2_b_tensor = Tensor<double>({10});
        for (size_t i = 0; i < 10; ++i) {
            fc2_b_tensor({i}) = fc2_bias[i];
        }

        weights_.emplace();
        weights_->conv1 = {std::move(conv_w_tensor), std::move(conv_b_tensor)};
        weights_->fc1 = {std::move(fc1_w_tensor), std::move(fc1_b_tensor)};
        weights_->fc2 = {std::move(fc2_w_tensor), std::move(fc2_b_tensor)};
    }
}

const Tensor<double>& CnnRequestDTO::GetInput() const {
    return input_;
}

bool CnnRequestDTO::HasWeights() const {
    return weights_.has_value();
}

const SimpleCnnWeights& CnnRequestDTO::GetWeights() const {
    if (!weights_) {
        throw std::runtime_error("weights are not provided");
    }
    return *weights_;
}
