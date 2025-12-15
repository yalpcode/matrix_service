#include <matrix_service/domains/max_pooling.hpp>
#include <matrix_service/domains/tensor.hpp>

#include <vector>

#include <userver/utest/utest.hpp>

UTEST(MaxPooling, ForwardBasic) {
    MaxPooling<double> pool(/*kernel_h=*/2, /*kernel_w=*/2, /*stride_h=*/1,
                            /*stride_w=*/1, /*padding_h=*/0, /*padding_w=*/0);

    Tensor<double> input({1, 1, 2, 3});
    input({0, 0, 0, 0}) = 1.0;
    input({0, 0, 0, 1}) = 3.0;
    input({0, 0, 0, 2}) = 2.0;
    input({0, 0, 1, 0}) = 4.0;
    input({0, 0, 1, 1}) = 6.0;
    input({0, 0, 1, 2}) = 5.0;

    const auto output = pool.forward(input);

    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 2);

    EXPECT_DOUBLE_EQ(output({0, 0, 0, 0}), 6.0);
    EXPECT_DOUBLE_EQ(output({0, 0, 0, 1}), 6.0);
}

UTEST(MaxPooling, HandlesPaddingAndNegativeValues) {
    MaxPooling<double> pool(/*kernel_h=*/2, /*kernel_w=*/2, /*stride_h=*/1,
                            /*stride_w=*/1, /*padding_h=*/1, /*padding_w=*/1);

    Tensor<double> input({1, 1, 2, 2});
    input({0, 0, 0, 0}) = -1.0;
    input({0, 0, 0, 1}) = -2.0;
    input({0, 0, 1, 0}) = -3.0;
    input({0, 0, 1, 1}) = -4.0;

    const auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 3);
    EXPECT_EQ(output.shape()[3], 3);

    const std::vector<double> expected = {-1.0, -1.0, -2.0, -1.0, -1.0,
                                          -2.0, -3.0, -3.0, -4.0};
    size_t idx = 0;
    for (size_t h = 0; h < output.shape()[2]; ++h) {
        for (size_t w = 0; w < output.shape()[3]; ++w) {
            EXPECT_DOUBLE_EQ(output({0, 0, h, w}), expected[idx++]);
        }
    }
}
