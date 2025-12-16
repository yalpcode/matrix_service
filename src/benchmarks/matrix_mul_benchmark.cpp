#include <benchmark/benchmark.h>
#include <matrix_service/dtos/matrixs_request_dto.hpp>
#include <matrix_service/services/matrix_mul_service.hpp>
#include <userver/formats/json.hpp>

static void BM_MatrixMulService(benchmark::State& state) {
    const auto request_json = userver::formats::json::FromString(
        R"({"left": [[1, 2], [3, 4]], "right": [[5, 6], [7, 8]]})");
    MatrixsRequestDTO dto(request_json);
    MatrixMulService service;

    for (auto _ : state) {
        benchmark::DoNotOptimize(service.matrixMul(dto));
    }
}

BENCHMARK(BM_MatrixMulService);
