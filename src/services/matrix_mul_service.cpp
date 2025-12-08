#include <matrix_service/services/matrix_mul_service.hpp>

Matrix<double> MatrixMulService::matrixMul(
    const MatrixsRequestDTO& matrixs_request_dto) const {
    const auto& matrix_left = matrixs_request_dto.getMatrixLeft();
    const auto& matrix_right = matrixs_request_dto.getMatrixRight();

    const auto [left_rows, left_cols] = matrix_left.size();
    const auto [right_rows, right_cols] = matrix_right.size();

    if (left_cols != right_rows) {
        throw std::invalid_argument(
            "Matrices have incompatible dimensions for multiplication");
    }

    return matrix_left * matrix_right;
}
