#include <matrix_service/domains/tensor.hpp>
#include <matrix_service/services/simple_cnn_service.hpp>

#include <cmath>
#include <stdexcept>

#include <userver/utest/utest.hpp>

UTEST(SimpleCNNService, ForwardProducesLogitsShape) {
    SimpleCNNService service;
    Tensor<double> input({1, 1, 28, 28}, 0.0);

    const auto output = service.Infer(input);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 10);

    for (size_t i = 0; i < output.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(output.data()[i]));
    }
}

UTEST(SimpleCNNService, RejectsWrongChannelCount) {
    SimpleCNNService service;
    Tensor<double> input({1, 2, 28, 28}, 0.0);
    EXPECT_THROW(service.Infer(input), std::invalid_argument);
}

UTEST(SimpleCNNService, AllowsExternalWeights) {
    SimpleCNNService service;
    SimpleCnnWeights weights;

    weights.conv1.weight = Tensor<double>({32, 1, 3, 3}, 0.0);
    weights.conv1.bias = Tensor<double>({32}, 0.0);
    weights.fc1.weight = Tensor<double>({128, 32 * 14 * 14}, 0.0);
    weights.fc1.bias = Tensor<double>({128}, 0.0);
    weights.fc2.weight = Tensor<double>({10, 128}, 0.0);
    weights.fc2.bias = Tensor<double>({10}, 1.0);

    service.ApplyWeights(weights);

    Tensor<double> input({1, 1, 28, 28}, 0.0);
    const auto output = service.Infer(input);
    double sum = 0.0;
    for (size_t i = 0; i < output.shape()[1]; ++i) {
        sum += output({0, i});
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
    const double expected = 1.0 / static_cast<double>(output.shape()[1]);
    for (size_t i = 0; i < output.shape()[1]; ++i) {
        EXPECT_NEAR(output({0, i}), expected, 1e-9);
    }
}

UTEST(SimpleCNNService, RejectsBatchMoreThanOne) {
    SimpleCNNService service;
    Tensor<double> input({2, 1, 28, 28}, 0.0);
    EXPECT_THROW(service.Infer(input), std::invalid_argument);
}
