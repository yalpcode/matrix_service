#pragma once

#include <matrix_service/domains/conv2d.hpp>
#include <matrix_service/domains/flatten.hpp>
#include <matrix_service/domains/linear.hpp>
#include <matrix_service/domains/max_pooling.hpp>
#include <matrix_service/domains/relu.hpp>
#include <matrix_service/domains/softmax.hpp>
#include <matrix_service/domains/tensor.hpp>
#include <matrix_service/domains/simple_cnn_weights.hpp>

class SimpleCNNService {
 public:
    SimpleCNNService();

    Tensor<double> Infer(const Tensor<double>& input) const;
    void ApplyWeights(const SimpleCnnWeights& weights);

 private:
    Conv2D<double> conv1_;
    ReLU<double> relu_;
    MaxPooling<double> pool1_;
    Flatten<double> flatten_;
    Linear<double> fc1_;
    ReLU<double> relu2_;
    Linear<double> fc2_;
    Softmax<double> softmax_;
};
