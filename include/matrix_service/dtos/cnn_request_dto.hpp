#pragma once

#include <matrix_service/domains/simple_cnn_weights.hpp>
#include <matrix_service/domains/tensor.hpp>
#include <userver/formats/json.hpp>
#include <optional>
#include <vector>

class CnnRequestDTO {
 public:
    explicit CnnRequestDTO(const userver::formats::json::Value& json);

    const Tensor<double>& GetInput() const;
    bool HasWeights() const;
    const SimpleCnnWeights& GetWeights() const;

 private:
    Tensor<double> input_;
    std::optional<SimpleCnnWeights> weights_;
};
