#include <matrix_service/dtos/matrixs_request_dto.hpp>
#include <matrix_service/schemas/matrix_response_schema.hpp>
#include <matrix_service/services/matrix_mul_service.hpp>
#include <stdexcept>
#include <userver/formats/json/serialize.hpp>
#include <userver/utest/utest.hpp>

UTEST(MatrixMulService, MultipliesAndSerializes) {
    const auto request_json = userver::formats::json::FromString(
        R"({"left": [[1, 2], [3, 4]], "right": [[5], [6]]})");

    MatrixsRequestDTO dto(request_json);
    MatrixMulService service;

    const auto result = service.matrixMul(dto);
    const auto [rows, cols] = result.size();
    EXPECT_EQ(rows, 2);
    EXPECT_EQ(cols, 1);

    auto it = result.begin();
    EXPECT_DOUBLE_EQ(*it++, 17);
    EXPECT_DOUBLE_EQ(*it, 39);

    MatrixResponseSchema schema(result);
    const auto response_json = schema.getJson();
    EXPECT_DOUBLE_EQ(response_json["result"][0][0].As<double>(), 17);
    EXPECT_DOUBLE_EQ(response_json["result"][1][0].As<double>(), 39);
}

UTEST(MatrixMulService, ThrowsOnIncompatibleMatrices) {
    const auto request_json = userver::formats::json::FromString(
        R"({"left": [[1, 2]], "right": [[3, 4]]})");

    MatrixsRequestDTO dto(request_json);
    MatrixMulService service;

    EXPECT_THROW(service.matrixMul(dto), std::invalid_argument);
}
