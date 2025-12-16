#include <matrix_service/domains/flatten.hpp>
#include <matrix_service/domains/linear.hpp>
#include <matrix_service/domains/relu.hpp>
#include <matrix_service/domains/softmax.hpp>
#include <matrix_service/domains/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <userver/utest/utest.hpp>

UTEST(Flatten, FlattensBatch) {
    Flatten<double> flatten;

    Tensor<double> input({2, 2, 3});
    double val = 1.0;
    for (size_t n = 0; n < 2; ++n) {
        for (size_t c = 0; c < 2; ++c) {
            for (size_t h = 0; h < 3; ++h) {
                input({n, c, h}) = val++;
            }
        }
    }

    const auto output = flatten.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 6);

    const std::vector<double> expected_first{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (size_t i = 0; i < expected_first.size(); ++i) {
        EXPECT_DOUBLE_EQ(output({0, i}), expected_first[i]);
    }
}

UTEST(Flatten, HandlesNonContiguousInput) {
    Flatten<double> flatten;

    Tensor<double> input({2, 2, 2});
    double val = 1.0;
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data()[i] = val++;
    }

    const auto permuted = input.permute({1, 0, 2});
    const auto output = flatten.forward(permuted);
    const auto expected = permuted.contiguous().reshape({2, 4});

    for (size_t i = 0; i < output.shape()[0]; ++i) {
        for (size_t j = 0; j < output.shape()[1]; ++j) {
            EXPECT_DOUBLE_EQ(output({i, j}), expected({i, j}));
        }
    }
}

UTEST(ReLU, NonNegativeOutput) {
    ReLU<double> relu;

    Tensor<double> input({1, 2, 2});
    input({0, 0, 0}) = -1.0;
    input({0, 0, 1}) = 0.5;
    input({0, 1, 0}) = -0.2;
    input({0, 1, 1}) = 3.0;

    const auto output = relu.forward(input);
    EXPECT_DOUBLE_EQ(output({0, 0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(output({0, 0, 1}), 0.5);
    EXPECT_DOUBLE_EQ(output({0, 1, 0}), 0.0);
    EXPECT_DOUBLE_EQ(output({0, 1, 1}), 3.0);
}

UTEST(Softmax, NormalizesAlongAxis) {
    Softmax<double> softmax(/*dim=*/1);

    Tensor<double> input({2, 3});
    input({0, 0}) = 1.0;
    input({0, 1}) = 2.0;
    input({0, 2}) = 3.0;
    input({1, 0}) = -1.0;
    input({1, 1}) = 0.0;
    input({1, 2}) = 1.0;

    const auto output = softmax.forward(input);

    for (size_t row = 0; row < 2; ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < 3; ++col) {
            sum += output({row, col});
        }
        EXPECT_NEAR(sum, 1.0, 1e-9);
    }

    const auto softmax_row = [](std::initializer_list<double> values) {
        const auto max_it = std::max_element(values.begin(), values.end());
        const double max_val = *max_it;
        double denom = 0.0;
        for (const auto v : values) {
            denom += std::exp(v - max_val);
        }
        std::vector<double> result;
        result.reserve(values.size());
        for (const auto v : values) {
            result.push_back(std::exp(v - max_val) / denom);
        }
        return result;
    };

    const auto expected0 = softmax_row({1.0, 2.0, 3.0});
    const auto expected1 = softmax_row({-1.0, 0.0, 1.0});

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(output({0, i}), expected0[i], 1e-9);
        EXPECT_NEAR(output({1, i}), expected1[i], 1e-9);
    }
}

UTEST(Softmax, DefaultAxisUsesLast) {
    Softmax<double> softmax;

    Tensor<double> input({3});
    input({0}) = 0.0;
    input({1}) = 1.0;
    input({2}) = 2.0;

    const auto output = softmax.forward(input);

    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        sum += output({i});
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);

    const auto expected = []() {
        const std::vector<double> vals{0.0, 1.0, 2.0};
        const auto max_it = std::max_element(vals.begin(), vals.end());
        const double max_val = *max_it;
        double denom = 0.0;
        for (const auto v : vals) {
            denom += std::exp(v - max_val);
        }
        std::vector<double> result;
        result.reserve(vals.size());
        for (const auto v : vals) {
            result.push_back(std::exp(v - max_val) / denom);
        }
        return result;
    }();

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(output({i}), expected[i], 1e-9);
    }
}

UTEST(Linear, ForwardMatchesWeights) {
    Linear<double> linear(/*in_features=*/3, /*out_features=*/2, /*bias=*/true);

    Tensor<double> input({1, 3});
    input({0, 0}) = 1.0;
    input({0, 1}) = 2.0;
    input({0, 2}) = 3.0;

    const auto output = linear.forward(input);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);

    for (size_t o = 0; o < 2; ++o) {
        double expected = linear.bias()({o});
        for (size_t i = 0; i < 3; ++i) {
            expected += input({0, i}) * linear.weight()({o, i});
        }
        EXPECT_NEAR(output({0, o}), expected, 1e-9);
    }
}
