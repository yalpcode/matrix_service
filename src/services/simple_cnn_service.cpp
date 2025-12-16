#include <matrix_service/services/simple_cnn_service.hpp>

#include <stdexcept>

SimpleCNNService::SimpleCNNService()
    : conv1_(/*in_channels=*/1, /*out_channels=*/32, /*kernel_h=*/3,
             /*kernel_w=*/3, /*stride_h=*/1, /*stride_w=*/1, /*padding_h=*/1,
             /*padding_w=*/1),
      relu_(),
      pool1_(/*kernel_h=*/2, /*kernel_w=*/2, /*stride_h=*/2, /*stride_w=*/2),
      flatten_(),
      fc1_(/*in_features=*/32 * 14 * 14, /*out_features=*/128),
      relu2_(),
      fc2_(/*in_features=*/128, /*out_features=*/10),
      softmax_() {
}

Tensor<double> SimpleCNNService::Infer(const Tensor<double>& input) const {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }
    const auto& shape = input.shape();
    const auto batch = shape[0];
    if (shape[1] != 1) {
        throw std::invalid_argument("SimpleCNN expects single-channel input");
    }
    if (shape[2] != 28 || shape[3] != 28) {
        throw std::invalid_argument("SimpleCNN expects 28x28 spatial size");
    }
    if (batch != 1) {
        throw std::invalid_argument("SimpleCNN currently expects batch size 1");
    }

    auto x = conv1_.forward(input);
    x = relu_.forward(x);
    x = pool1_.forward(x);
    x = flatten_.forward(x);
    x = fc1_.forward(x);
    x = relu2_.forward(x);
    x = fc2_.forward(x);
    x = softmax_.forward(x);
    return x;
}

void SimpleCNNService::ApplyWeights(const SimpleCnnWeights& weights) {
    conv1_.load_weights(weights.conv1.weight, weights.conv1.bias);
    fc1_.load_weights(weights.fc1.weight, weights.fc1.bias);
    fc2_.load_weights(weights.fc2.weight, weights.fc2.bias);
}
