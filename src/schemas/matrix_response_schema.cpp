#include <matrix_service/schemas/matrix_response_schema.hpp>

#include <vector>

#include <userver/formats/json/value_builder.hpp>
#include <userver/formats/serialize/common_containers.hpp>

MatrixResponseSchema::MatrixResponseSchema(const Matrix<double>& matrix) {
    const auto [rows, cols] = matrix.size();
    std::vector<std::vector<double>> serialized(rows,
                                                std::vector<double>(cols));

    auto it = matrix.begin();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j, ++it) {
            serialized[i][j] = *it;
        }
    }

    userver::formats::json::ValueBuilder builder;
    builder["result"] = std::move(serialized);
    json = builder.ExtractValue();
}

userver::formats::json::Value MatrixResponseSchema::getJson() const {
    return json;
}
