#pragma once

#include <matrix_service/domains/tensor.hpp>

struct ConvLayerWeights {
    Tensor<double> weight;
    Tensor<double> bias;
};

struct LinearLayerWeights {
    Tensor<double> weight;
    Tensor<double> bias;
};

struct SimpleCnnWeights {
    ConvLayerWeights conv1;
    LinearLayerWeights fc1;
    LinearLayerWeights fc2;
};
