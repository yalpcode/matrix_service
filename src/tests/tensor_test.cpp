#include <matrix_service/domains/tensor.hpp>

#include <userver/utest/utest.hpp>

UTEST(Tensor, CreationReshapePermute) {
    Tensor<double> tensor({2, 2}, 0.0);
    tensor(0, 0) = 1.0;
    tensor(0, 1) = 2.0;
    tensor(1, 0) = 3.0;
    tensor(1, 1) = 4.0;

    const auto reshaped = tensor.reshape({4});
    EXPECT_EQ(reshaped.dim(), 1);
    EXPECT_EQ(reshaped.shape()[0], 4);
    EXPECT_DOUBLE_EQ(reshaped(3), 4.0);

    const auto permuted = tensor.permute({1, 0});
    EXPECT_EQ(permuted.shape()[0], 2);
    EXPECT_EQ(permuted.shape()[1], 2);
    EXPECT_FALSE(permuted.is_contiguous());

    const auto contiguous = permuted.contiguous();
    EXPECT_TRUE(contiguous.is_contiguous());
    EXPECT_DOUBLE_EQ(contiguous(1, 0), 2.0);
}

UTEST(Tensor, MatmulVariants) {
    Tensor<double> mat({2, 3});
    double value = 1.0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            mat(i, j) = value++;
        }
    }

    Tensor<double> vec({3});
    vec(0) = 1.0;
    vec(1) = 0.0;
    vec(2) = -1.0;

    const auto mv = mat.matmul(vec);
    EXPECT_EQ(mv.shape()[0], 2);
    EXPECT_DOUBLE_EQ(mv(0), -2.0);
    EXPECT_DOUBLE_EQ(mv(1), -2.0);

    Tensor<double> mat2({3, 2});
    mat2(0, 0) = 1.0;
    mat2(0, 1) = 2.0;
    mat2(1, 0) = 3.0;
    mat2(1, 1) = 4.0;
    mat2(2, 0) = 5.0;
    mat2(2, 1) = 6.0;

    const auto mm = mat.matmul(mat2);
    EXPECT_EQ(mm.shape()[0], 2);
    EXPECT_EQ(mm.shape()[1], 2);
    EXPECT_DOUBLE_EQ(mm(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(mm(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(mm(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(mm(1, 1), 64.0);
}

UTEST(Tensor, SumAndSqueeze) {
    Tensor<int> tensor({2, 1, 2});
    tensor(0, 0, 0) = 1;
    tensor(0, 0, 1) = 2;
    tensor(1, 0, 0) = 3;
    tensor(1, 0, 1) = 4;

    const auto squeezed = tensor.squeeze(1);
    EXPECT_EQ(squeezed.dim(), 2);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 2);

    const auto sum0 = squeezed.sum(0);
    EXPECT_EQ(sum0.shape()[0], 2);
    EXPECT_EQ(sum0(0), 4);
    EXPECT_EQ(sum0(1), 6);

    const auto sum1 = squeezed.sum(1);
    EXPECT_EQ(sum1.shape()[0], 2);
    EXPECT_EQ(sum1(0), 3);
    EXPECT_EQ(sum1(1), 7);

    const auto total = squeezed.sum();
    EXPECT_EQ(total.dim(), 0);
    EXPECT_EQ(total({}), 10);
}
