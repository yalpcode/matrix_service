#include <matrix_service/domains/conv2d.hpp>
#include <matrix_service/domains/tensor.hpp>

#include <userver/utest/utest.hpp>

UTEST(Conv2D, ForwardUsesTensor) {
    Conv2D<double> conv(/*in_channels=*/1, /*out_channels=*/2, /*kernel_h=*/1,
                        /*kernel_w=*/1, /*stride_h=*/1, /*stride_w=*/1,
                        /*padding_h=*/0, /*padding_w=*/0, /*dilation_h=*/1,
                        /*dilation_w=*/1, /*groups=*/1, /*bias=*/true);

    Tensor<double> input({1, 1, 2, 2}, 1.0);
    const auto output = conv.forward(input);

    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);

    for (size_t oc = 0; oc < 2; ++oc) {
        const auto expected =
            conv.weight()({oc, 0, 0, 0}) + conv.bias()({oc});
        for (size_t oh = 0; oh < 2; ++oh) {
            for (size_t ow = 0; ow < 2; ++ow) {
                EXPECT_NEAR(output({0, oc, oh, ow}), expected, 1e-9);
            }
        }
    }
}

UTEST(Conv2D, GroupsAreIsolated) {
    Conv2D<double> conv(/*in_channels=*/2, /*out_channels=*/2, /*kernel_h=*/1,
                        /*kernel_w=*/1, /*stride_h=*/1, /*stride_w=*/1,
                        /*padding_h=*/0, /*padding_w=*/0, /*dilation_h=*/1,
                        /*dilation_w=*/1, /*groups=*/2, /*bias=*/false);

    Tensor<double> input({1, 2, 1, 1});
    input({0, 0, 0, 0}) = 2.0;
    input({0, 1, 0, 0}) = 3.0;

    const auto output = conv.forward(input);

    EXPECT_DOUBLE_EQ(output({0, 0, 0, 0}),
                     input({0, 0, 0, 0}) * conv.weight()({0, 0, 0, 0}));
    EXPECT_DOUBLE_EQ(output({0, 1, 0, 0}),
                     input({0, 1, 0, 0}) * conv.weight()({1, 0, 0, 0}));
}
