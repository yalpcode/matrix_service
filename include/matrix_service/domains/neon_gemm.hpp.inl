#include "neon_gemm.hpp"

// Конвертация результата в QuantizedMatrix
QuantizedMatrix NeonGEMM::QuantizedResult::to_qmatrix(float output_scale,
                                                      int32_t output_zp) const {
    std::vector<int8_t> q_data(data.size());
    QuantizedMatrix::QuantParams out_params(output_scale, output_zp);

    for (size_t i = 0; i < data.size(); ++i) {
        float dequantized = params.dequantize(data[i]);
        float requantized =
            dequantized / output_scale + static_cast<float>(output_zp);
        q_data[i] = static_cast<int8_t>(
            std::clamp(std::round(requantized), -128.0f, 127.0f));
    }

    return QuantizedMatrix(rows, cols, q_data, out_params);
}

// Основное умножение
NeonGEMM::QuantizedResult NeonGEMM::multiply(
    const QuantizedMatrix& A, const QuantizedMatrix& B,
    const NeonGEMM::GemmParams& params) {
    size_t m = A.rows();
    size_t k = A.cols();
    size_t n = B.cols();

    if (k != B.rows()) {
        throw std::invalid_argument(
            "Matrix dimensions don't match for multiplication");
    }

    int32_t zp_a = A.quant_params().zero_point;
    int32_t zp_b = B.quant_params().zero_point;

    QuantizedMatrix::QuantParams result_params;
    result_params.scale = A.quant_params().scale * B.quant_params().scale;
    result_params.zero_point = 0;

    std::vector<int32_t> C(m * n, 0);

    for (size_t jc = 0; jc < n; jc += params.nc) {
        size_t nc = std::min(params.nc, n - jc);

        for (size_t pc = 0; pc < k; pc += params.kc) {
            size_t kc = std::min(params.kc, k - pc);

            for (size_t ic = 0; ic < m; ic += params.mc) {
                size_t mc = std::min(params.mc, m - ic);

                compute_block_neon(A, B, C.data(), ic, jc, pc, mc, nc, kc, m, n,
                                   k, zp_a, zp_b, params);
            }
        }
    }

    return {C, m, n, result_params};
}

NeonGEMM::QuantizedResult NeonGEMM::multiply(const QuantizedMatrix& A,
                                             const QuantizedMatrix& B) {
    return NeonGEMM::multiply(A, B, NeonGEMM::GemmParams{});
}

#if __ARM_NEON
void NeonGEMM::micro_kernel_8x12x4(const int8_t* A_ptr, const int8_t* B_ptr,
                                   int32_t* C_ptr, size_t ldc, int32_t zp_a,
                                   int32_t zp_b) {
    int32x4_t v_zp_a = vdupq_n_s32(zp_a);
    int32x4_t v_zp_b = vdupq_n_s32(zp_b);

    int32x4_t acc[3][2];

    // Инициализация аккумуляторов
    for (int g = 0; g < 3; ++g) {
        acc[g][0] = vdupq_n_s32(0);  // строки 0-3
        acc[g][1] = vdupq_n_s32(0);  // строки 4-7
    }

    // Векторы для строк A (предзагрузка)
    int32x4_t a_vecs[2][4];  // [row_half][lane]

    // Обрабатываем 4 глубины (KR = 4)
    for (int depth = 0; depth < 4; ++depth) {
        // Загружаем столбец из A (8 элементов = 2×int8x8_t)
        int8x8_t a_col0 = vld1_s8(A_ptr + depth * 8);
        int8x8_t a_col1 =
            vld1_s8(A_ptr + depth * 8 + 8);  // если больше 8, но у нас 8

        // Преобразуем в int16, затем в int32 и вычитаем zero_point
        int16x8_t a_col0_s16 = vmovl_s8(a_col0);
        int32x4_t a_low =
            vsubq_s32(vmovl_s16(vget_low_s16(a_col0_s16)), v_zp_a);
        int32x4_t a_high =
            vsubq_s32(vmovl_s16(vget_high_s16(a_col0_s16)), v_zp_a);

        // Загружаем строку из B (12 элементов)
        int8x16_t b_row0 = vld1q_s8(B_ptr + depth * 12);
        int8x8_t b_row1 = vld1_s8(B_ptr + depth * 12 + 16);

        // Разбираем B на 3 группы по 4 столбца
        // Группа 0: столбцы 0-3
        int8x8_t b_group0 = vget_low_s8(b_row0);
        int16x8_t b_group0_s16 = vmovl_s8(b_group0);
        int32x4_t b0 = vsubq_s32(vmovl_s16(vget_low_s16(b_group0_s16)), v_zp_b);

        // Группа 1: столбцы 4-7
        int8x8_t b_group1 = vget_high_s8(b_row0);
        int16x8_t b_group1_s16 = vmovl_s8(b_group1);
        int32x4_t b1 = vsubq_s32(vmovl_s16(vget_low_s16(b_group1_s16)), v_zp_b);

        // Группа 2: столбцы 8-11
        int8x8_t b_group2 = b_row1;
        int16x8_t b_group2_s16 = vmovl_s8(b_group2);
        int32x4_t b2 = vsubq_s32(vmovl_s16(vget_low_s16(b_group2_s16)), v_zp_b);

        // Создаем векторы для каждой строки A

        // Для строк 0-3
        a_vecs[0][0] = vdupq_laneq_s32(a_low, 0);  // строка 0: 4 копии a_low[0]
        a_vecs[0][1] = vdupq_laneq_s32(a_low, 1);  // строка 1: 4 копии a_low[1]
        a_vecs[0][2] = vdupq_laneq_s32(a_low, 2);  // строка 2: 4 копии a_low[2]
        a_vecs[0][3] = vdupq_laneq_s32(a_low, 3);  // строка 3: 4 копии a_low[3]

        // Для строк 4-7
        a_vecs[1][0] =
            vdupq_laneq_s32(a_high, 0);  // строка 4: 4 копии a_high[0]
        a_vecs[1][1] =
            vdupq_laneq_s32(a_high, 1);  // строка 5: 4 копии a_high[1]
        a_vecs[1][2] =
            vdupq_laneq_s32(a_high, 2);  // строка 6: 4 копии a_high[2]
        a_vecs[1][3] =
            vdupq_laneq_s32(a_high, 3);  // строка 7: 4 копии a_high[3]

        acc[0][0] = vmlaq_s32(acc[0][0], a_vecs[0][0], b0);  // строка 0
        acc[0][0] = vmlaq_s32(acc[0][0], a_vecs[0][1], b0);  // строка 1
        acc[0][0] = vmlaq_s32(acc[0][0], a_vecs[0][2], b0);  // строка 2
        acc[0][0] = vmlaq_s32(acc[0][0], a_vecs[0][3], b0);  // строка 3

        acc[1][0] = vmlaq_s32(acc[1][0], a_vecs[0][0], b1);  // строка 0
        acc[1][0] = vmlaq_s32(acc[1][0], a_vecs[0][1], b1);  // строка 1
        acc[1][0] = vmlaq_s32(acc[1][0], a_vecs[0][2], b1);  // строка 2
        acc[1][0] = vmlaq_s32(acc[1][0], a_vecs[0][3], b1);  // строка 3

        acc[2][0] = vmlaq_s32(acc[2][0], a_vecs[0][0], b2);  // строка 0
        acc[2][0] = vmlaq_s32(acc[2][0], a_vecs[0][1], b2);  // строка 1
        acc[2][0] = vmlaq_s32(acc[2][0], a_vecs[0][2], b2);  // строка 2
        acc[2][0] = vmlaq_s32(acc[2][0], a_vecs[0][3], b2);  // строка 3

        acc[0][1] = vmlaq_s32(acc[0][1], a_vecs[1][0], b0);  // строка 4
        acc[0][1] = vmlaq_s32(acc[0][1], a_vecs[1][1], b0);  // строка 5
        acc[0][1] = vmlaq_s32(acc[0][1], a_vecs[1][2], b0);  // строка 6
        acc[0][1] = vmlaq_s32(acc[0][1], a_vecs[1][3], b0);  // строка 7

        acc[1][1] = vmlaq_s32(acc[1][1], a_vecs[1][0], b1);  // строка 4
        acc[1][1] = vmlaq_s32(acc[1][1], a_vecs[1][1], b1);  // строка 5
        acc[1][1] = vmlaq_s32(acc[1][1], a_vecs[1][2], b1);  // строка 6
        acc[1][1] = vmlaq_s32(acc[1][1], a_vecs[1][3], b1);  // строка 7

        acc[2][1] = vmlaq_s32(acc[2][1], a_vecs[1][0], b2);  // строка 4
        acc[2][1] = vmlaq_s32(acc[2][1], a_vecs[1][1], b2);  // строка 5
        acc[2][1] = vmlaq_s32(acc[2][1], a_vecs[1][2], b2);  // строка 6
        acc[2][1] = vmlaq_s32(acc[2][1], a_vecs[1][3], b2);  // строка 7
    }

    // Сохраняем аккумуляторы в матрицу C
    // C имеет размер 8×12, сохраняем по 4 элемента за раз

    // Для каждой группы столбцов (0, 1, 2)
    for (int g = 0; g < 3; ++g) {
        // Вычисляем указатель на начало группы столбцов в C
        int32_t* C_group_ptr = C_ptr + g * 4;  // 4 столбца в группе

        // Сохраняем строки 0-3 (первая половина)
        vst1q_s32(C_group_ptr + 0 * ldc, acc[g][0]);

        // Сохраняем строки 4-7 (вторая половина)
        vst1q_s32(C_group_ptr + 4 * ldc, acc[g][1]);
    }
}
#endif
// Упрощенная версия микроядра (для неполных блоков)
void NeonGEMM::micro_kernel_simple(const int8_t* A_ptr, const int8_t* B_ptr,
                                   int32_t* C_ptr, size_t mc, size_t nc,
                                   size_t kc, size_t ldc, int32_t zp_a,
                                   int32_t zp_b) {
    std::vector<std::vector<int32_t>> acc(mc, std::vector<int32_t>(nc, 0));

    for (size_t i = 0; i < mc; ++i) {
        for (size_t k = 0; k < kc; ++k) {
            int32_t a_val = static_cast<int32_t>(A_ptr[i * kc + k]) - zp_a;

            for (size_t j = 0; j < nc; ++j) {
                int32_t b_val = static_cast<int32_t>(B_ptr[k * nc + j]) - zp_b;
                acc[i][j] += a_val * b_val;
            }
        }
    }

    // Копируем в C
    for (size_t i = 0; i < mc; ++i) {
        for (size_t j = 0; j < nc; ++j) {
            C_ptr[i * ldc + j] += acc[i][j];
        }
    }
}

void NeonGEMM::compute_block_neon(const QuantizedMatrix& A,
                                  const QuantizedMatrix& B, int32_t* C,
                                  size_t i_start, size_t j_start,
                                  size_t p_start, size_t mc, size_t nc,
                                  size_t kc, size_t m, size_t n, size_t k,
                                  int32_t zp_a, int32_t zp_b,
                                  const GemmParams& params) {
    const int8_t* A_data = A.data();
    const int8_t* B_data = B.data();

    size_t lda = A.cols();
    size_t ldb = B.cols();
    size_t ldc = n;

    for (size_t i = i_start; i < i_start + mc; i += params.MR) {
        size_t mr = std::min(params.MR, i_start + mc - i);

        for (size_t j = j_start; j < j_start + nc; j += params.NR) {
            size_t nr = std::min(params.NR, j_start + nc - j);

            for (size_t p = p_start; p < p_start + kc; p += params.KR) {
                size_t kr = std::min(params.KR, p_start + kc - p);

                // Указатели на текущие блоки
                const int8_t* A_block = A_data + i * lda + p;
                const int8_t* B_block = B_data + p * ldb + j;
                int32_t* C_block = C + i * ldc + j;

                // Вызываем микроядро
                if (mr == params.MR && nr == params.NR && kr == params.KR) {
                    #if __ARM_NEON
                    // Полный блок - используем оптимизированное ядро
                    micro_kernel_8x12x4(A_block, B_block, C_block, ldc, zp_a,
                                        zp_b);
                    #elif
                    micro_kernel_simple(A_block, B_block, C_block, mr, nr, kr,
                                        ldc, zp_a, zp_b);
                    #endif
                } else {
                    // Неполный блок - используем простую версию
                    micro_kernel_simple(A_block, B_block, C_block, mr, nr, kr,
                                        ldc, zp_a, zp_b);
                }
            }
        }
    }
}
