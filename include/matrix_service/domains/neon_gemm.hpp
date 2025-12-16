#pragma once

#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include <vector>

#include "quantized_matrix.hpp"

class NeonGEMM {
 public:
    struct GemmParams {
        // Размеры блоков для кэшей под Apple M1
        size_t mc = 512;  // Блок строк для L2 кэша
        size_t kc = 256;  // Блок глубины для L2
        size_t nc = 512;  // Блок столбцов для L2

        // Размеры микроядра под Apple M1
        static constexpr size_t MR = 8;  // 8 строк за раз
        static constexpr size_t NR =
            12;  // 12 столбцов за раз (использует 12 регистров)
        static constexpr size_t KR = 4;  // 4 глубины за раз

        // bool use_prefetch = true;
        // int num_threads = 1; <- не реализовано
    };

    // Результат умножения с int32 аккумуляторами
    struct QuantizedResult {
        std::vector<int32_t> data;
        size_t rows;
        size_t cols;
        QuantizedMatrix::QuantParams params;

        // Конвертация в QuantizedMatrix
        QuantizedMatrix to_qmatrix(float output_scale,
                                   int32_t output_zp = 0) const;
    };

    // Основное умножение A (m×k) × B (k×n) → C (m×n)
    static QuantizedResult multiply(const QuantizedMatrix& A,
                                    const QuantizedMatrix& B,
                                    const GemmParams& params);

    static QuantizedResult multiply(const QuantizedMatrix& A,
                                    const QuantizedMatrix& B);

 private:
    static void micro_kernel_8x12x4(const int8_t* A_ptr, const int8_t* B_ptr,
                                    int32_t* C_ptr, size_t ldc, int32_t zp_a,
                                    int32_t zp_b);

    static void micro_kernel_simple(const int8_t* A_ptr, const int8_t* B_ptr,
                                    int32_t* C_ptr, size_t mc, size_t nc,
                                    size_t kc, size_t ldc, int32_t zp_a,
                                    int32_t zp_b);

    static void compute_block_neon(const QuantizedMatrix& A,
                                   const QuantizedMatrix& B, int32_t* C,
                                   size_t i_start, size_t j_start,
                                   size_t p_start, size_t mc, size_t nc,
                                   size_t kc, size_t m, size_t n, size_t k,
                                   int32_t zp_a, int32_t zp_b,
                                   const GemmParams& params);
};

#include "neon_gemm.hpp.inl"