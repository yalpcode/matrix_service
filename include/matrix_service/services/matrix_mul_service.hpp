#pragma once

#include <matrix_service/gtos/matrixs_request_dto.hpp>
#include <stdexcept>

class MatrixMulService {
 public:
    Matrix<double> matrixMul(
        const MatrixsRequestDTO& matrixs_request_dto) const;
};
