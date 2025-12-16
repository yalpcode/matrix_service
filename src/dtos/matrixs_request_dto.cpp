#include <matrix_service/dtos/matrixs_request_dto.hpp>
#include <stdexcept>
#include <userver/formats/parse/common_containers.hpp>
#include <vector>

MatrixsRequestDTO::MatrixsRequestDTO(const userver::formats::json::Value& json)
    : matrix_left(json["left"].As<std::vector<std::vector<double>>>()),
      matrix_right(json["right"].As<std::vector<std::vector<double>>>()) {
    const auto [left_rows, left_cols] = matrix_left.size();
    const auto [right_rows, right_cols] = matrix_right.size();

    if (left_rows == 0 || left_cols == 0 || right_rows == 0 ||
        right_cols == 0) {
        throw std::invalid_argument("Matrices must be non-empty");
    }
}

const Matrix<double>& MatrixsRequestDTO::getMatrixLeft() const {
    return matrix_left;
}

const Matrix<double>& MatrixsRequestDTO::getMatrixRight() const {
    return matrix_right;
}
