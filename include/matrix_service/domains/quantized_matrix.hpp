#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

template<class T>
class Tensor;

class QuantizedMatrix {
 public:
    struct QuantParams {
        float scale;
        int32_t zero_point;

        QuantParams(float s = 1.0f, int32_t zp = 0);

        int8_t quantize(float value) const;

        float dequantize(int8_t value) const;

        float dequantize(int32_t value) const;
    };

    QuantizedMatrix();
    QuantizedMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    QuantizedMatrix(size_t rows, size_t cols, const std::vector<int8_t>& data,
                    QuantParams params = QuantParams());
    QuantizedMatrix(const Tensor<float> tensor);

    static QuantizedMatrix zeros(size_t rows, size_t cols,
                                 QuantParams params = QuantParams());

    int8_t at(size_t row, size_t col) const;

    int8_t& at(size_t row, size_t col);

    float get_float(size_t row, size_t col) const;

    size_t rows() const;
    size_t cols() const;
    size_t size() const;

    const int8_t* data() const;
    int8_t* data();

    const QuantParams& quant_params() const;

    std::vector<float> dequantize() const;

    Tensor<float> dequantize_to_tensor() const;

    void print(size_t max_rows = 10, size_t max_cols = 10) const;

 private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<int8_t> data_;
    QuantParams params_;
};

#include "quantized_matrix.hpp.inl"
