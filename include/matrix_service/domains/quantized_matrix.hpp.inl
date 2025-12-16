#include "quantized_matrix.hpp"
#include "tensor.hpp"

// Параметры квантования (scale и zero-point)
inline QuantizedMatrix::QuantParams::QuantParams(float s, int32_t zp)
    : scale(s), zero_point(zp) {
}

// Квантование float -> int8
inline int8_t QuantizedMatrix::QuantParams::quantize(float value) const {
    float q = value / scale + static_cast<float>(zero_point);
    return static_cast<int8_t>(std::clamp(std::round(q), -128.0f, 127.0f));
}

// Деквантование int8 -> float
inline float QuantizedMatrix::QuantParams::dequantize(int8_t value) const {
    return scale * (static_cast<float>(value) - static_cast<float>(zero_point));
}

// Деквантование int32 -> float (для аккумуляторов)
inline float QuantizedMatrix::QuantParams::dequantize(int32_t value) const {
    return scale * (static_cast<float>(value) - static_cast<float>(zero_point));
}

// Конструктор по умолчанию
inline QuantizedMatrix::QuantizedMatrix() = default;

// Из обычной матрицы float32
inline QuantizedMatrix::QuantizedMatrix(size_t rows, size_t cols,
                                        const std::vector<float>& data)
    : rows_(rows), cols_(cols) {
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());

    params_.scale = (max_val - min_val) / 255.0f;
    params_.zero_point =
        static_cast<int32_t>(std::round(-min_val / params_.scale));

    data_.resize(rows * cols);
    for (size_t i = 0; i < data.size(); ++i) {
        data_[i] = params_.quantize(data[i]);
    }
}

// Создание с заданными параметрами
inline QuantizedMatrix::QuantizedMatrix(size_t rows, size_t cols,
                                        const std::vector<int8_t>& data,
                                        QuantParams params)
    : rows_(rows), cols_(cols), data_(data), params_(params) {
    data_.resize(rows * cols);
}

// Конструктор из тензора (предполагаем что это уже im2col результат) с
// симметричным квантованием
inline QuantizedMatrix::QuantizedMatrix(const Tensor<float> tensor) {
    if (tensor.dim() != 2) {
        throw std::invalid_argument("Tensor must be 2D for QuantizedMatrix");
    }

    rows_ = tensor.shape()[0];
    cols_ = tensor.shape()[1];

    float max_abs = 0.0f;

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            float val = static_cast<float>(tensor.at({i, j}));
            max_abs = std::max(max_abs, std::abs(val));
        }
    }

    float scale = max_abs / 127.0f;
    params_ = QuantParams(scale, 0);

    data_.resize(rows_ * cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            float val = static_cast<float>(tensor.at({i, j}));
            data_[i * cols_ + j] = params_.quantize(val);
        }
    }
}

// Создание нулевой матрицы
inline QuantizedMatrix QuantizedMatrix::zeros(size_t rows, size_t cols,
                                              QuantParams params) {
    return QuantizedMatrix(rows, cols, std::vector<int8_t>(rows * cols, 0),
                           params);
}

// Доступ к элементам
inline int8_t QuantizedMatrix::at(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}
inline int8_t& QuantizedMatrix::at(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

// Получение float значения
inline float QuantizedMatrix::get_float(size_t row, size_t col) const {
    return params_.dequantize(at(row, col));
}

// Размеры
inline size_t QuantizedMatrix::rows() const {
    return rows_;
}
inline size_t QuantizedMatrix::cols() const {
    return cols_;
}
inline size_t QuantizedMatrix::size() const {
    return rows_ * cols_;
}

// Прямой доступ к данным
inline const int8_t* QuantizedMatrix::data() const {
    return data_.data();
}
inline int8_t* QuantizedMatrix::data() {
    return data_.data();
}

// Параметры квантования
inline const QuantizedMatrix::QuantParams& QuantizedMatrix::quant_params() const {
    return params_;
}

// Деквантование всей матрицы
inline std::vector<float> QuantizedMatrix::dequantize() const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = params_.dequantize(data_[i]);
    }
    return result;
}

inline Tensor<float> QuantizedMatrix::dequantize_to_tensor() const {
    std::vector<float> float_data = dequantize();
    return Tensor<float>({rows_, cols_}, std::move(float_data));
}

// Вывод в консоль (для отладки)
inline void QuantizedMatrix::print(size_t max_rows, size_t max_cols) const {
    std::cout << "QuantizedMatrix " << rows_ << "x" << cols_
              << " (scale=" << params_.scale << ", zp=" << params_.zero_point
              << ")\n";

    for (size_t i = 0; i < std::min(rows_, max_rows); ++i) {
        for (size_t j = 0; j < std::min(cols_, max_cols); ++j) {
            std::cout << static_cast<int>(at(i, j)) << "\t";
        }
        if (cols_ > max_cols) std::cout << "...";
        std::cout << "\n";
    }
    if (rows_ > max_rows) std::cout << "...\n";
}
