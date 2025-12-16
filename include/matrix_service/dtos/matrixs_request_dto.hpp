#pragma once

#include <matrix_service/domains/matrix.hpp>
#include <stdexcept>
#include <userver/formats/json.hpp>
#include <vector>

class MatrixsRequestDTO {
    Matrix<double> matrix_left;
    Matrix<double> matrix_right;

 public:
    explicit MatrixsRequestDTO(const userver::formats::json::Value& json);

    const Matrix<double>& getMatrixLeft() const;

    const Matrix<double>& getMatrixRight() const;
};
