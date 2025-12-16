#pragma once

#include <matrix_service/domains/tensor.hpp>
#include <userver/formats/json.hpp>

class CnnResponseSchema {
 public:
    explicit CnnResponseSchema(const Tensor<double>& logits);

    userver::formats::json::Value GetJson() const;

 private:
    userver::formats::json::Value json_;
};
