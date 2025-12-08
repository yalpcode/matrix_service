#pragma once

#include <matrix_service/domains/matrix.hpp>
#include <userver/formats/json.hpp>
#include <userver/formats/json/value_builder.hpp>
#include <vector>

class MatrixResponseSchema {
    userver::formats::json::Value json;

 public:
    MatrixResponseSchema(const Matrix<double>& matrix);

    userver::formats::json::Value getJson() const;
};
