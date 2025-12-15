#include "tensor.hpp"

template <IsArithmetic T>
Tensor<T>::Tensor()
    : shape_(),
      strides_(make_contiguous_strides(shape_)),
      data_(std::make_shared<std::vector<T>>(compute_numel(shape_))),
      offset_(0) {
}

template <IsArithmetic T>
Tensor<T>::Tensor(Shape shape, const T& value)
    : shape_(std::move(shape)),
      strides_(make_contiguous_strides(shape_)),
      data_(std::make_shared<std::vector<T>>(compute_numel(shape_), value)),
      offset_(0) {
}

template <IsArithmetic T>
Tensor<T>::Tensor(std::initializer_list<size_t> shape, const T& value)
    : Tensor(Shape(shape), value) {
}

template <IsArithmetic T>
Tensor<T>::Tensor(Shape shape, std::vector<T> data)
    : shape_(std::move(shape)),
      strides_(make_contiguous_strides(shape_)),
      data_(std::make_shared<std::vector<T>>(std::move(data))),
      offset_(0) {
    if (data_->size() != compute_numel(shape_)) {
        throw std::invalid_argument("Data size does not match tensor shape");
    }
}

template <IsArithmetic T>
Tensor<T>::Tensor(Shape shape, std::shared_ptr<std::vector<T>> data,
                  Strides strides, size_t offset)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      data_(std::move(data)),
      offset_(offset) {
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::zeros(const Shape& shape) {
    return Tensor(shape, T{});
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::ones(const Shape& shape) {
    return Tensor(shape, static_cast<T>(1));
}

template <IsArithmetic T>
template <class U, class>
Tensor<T> Tensor<T>::arange(U start, U end, U step) {
    if (step == U{}) {
        throw std::invalid_argument("Step for arange cannot be zero");
    }

    std::vector<T> values;
    if ((step > U{} && start >= end) || (step < U{} && start <= end)) {
        return Tensor({static_cast<size_t>(0)}, std::move(values));
    }

    for (U current = start; (step > U{} ? current < end : current > end);
         current += step) {
        values.emplace_back(static_cast<T>(current));
    }

    return Tensor({values.size()}, std::move(values));
}

template <IsArithmetic T>
size_t Tensor<T>::dim() const {
    return shape_.size();
}

template <IsArithmetic T>
const typename Tensor<T>::Shape& Tensor<T>::shape() const {
    return shape_;
}

template <IsArithmetic T>
const typename Tensor<T>::Strides& Tensor<T>::strides() const {
    return strides_;
}

template <IsArithmetic T>
size_t Tensor<T>::numel() const {
    return compute_numel(shape_);
}

template <IsArithmetic T>
bool Tensor<T>::is_contiguous() const {
    return offset_ == 0 && strides_ == make_contiguous_strides(shape_);
}

template <IsArithmetic T>
T* Tensor<T>::data() {
    return data_->data() + offset_;
}

template <IsArithmetic T>
const T* Tensor<T>::data() const {
    return data_->data() + offset_;
}

template <IsArithmetic T>
template <class... Index>
T& Tensor<T>::operator()(Index... indices) {
    static_assert(sizeof...(Index) > 0, "At least one index must be provided");
    std::array<size_t, sizeof...(Index)> idx{{static_cast<size_t>(indices)...}};
    return at(Shape(idx.begin(), idx.end()));
}

template <IsArithmetic T>
template <class... Index>
const T& Tensor<T>::operator()(Index... indices) const {
    static_assert(sizeof...(Index) > 0, "At least one index must be provided");
    std::array<size_t, sizeof...(Index)> idx{{static_cast<size_t>(indices)...}};
    return at(Shape(idx.begin(), idx.end()));
}

template <IsArithmetic T>
T& Tensor<T>::operator()(std::initializer_list<size_t> indices) {
    return at(Shape(indices));
}

template <IsArithmetic T>
const T& Tensor<T>::operator()(std::initializer_list<size_t> indices) const {
    return at(Shape(indices));
}

template <IsArithmetic T>
T& Tensor<T>::at(const Shape& indices) {
    return (*data_)[compute_offset(indices)];
}

template <IsArithmetic T>
const T& Tensor<T>::at(const Shape& indices) const {
    return (*data_)[compute_offset(indices)];
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::clone() const {
    Tensor copy(shape_, T{});
    for_each_index([&](const Shape& idx) { copy.at(idx) = at(idx); });
    return copy;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::reshape(const Shape& new_shape) const {
    if (compute_numel(new_shape) != numel()) {
        throw std::invalid_argument("Reshape would change tensor size");
    }

    if (!is_contiguous()) {
        return contiguous().reshape(new_shape);
    }

    return Tensor(new_shape, data_, make_contiguous_strides(new_shape),
                  offset_);
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::permute(const Shape& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::invalid_argument("Permutation size must match tensor rank");
    }

    Shape new_shape(dims.size());
    Strides new_strides(dims.size());
    std::vector<bool> seen(dims.size(), false);

    for (size_t i = 0; i < dims.size(); ++i) {
        const auto dim = dims[i];
        if (dim >= shape_.size() || seen[dim]) {
            throw std::invalid_argument("Invalid permutation for tensor");
        }
        seen[dim] = true;
        new_shape[i] = shape_[dim];
        new_strides[i] = strides_[dim];
    }

    return Tensor(new_shape, data_, new_strides, offset_);
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
        throw std::invalid_argument("Transpose dimensions out of range");
    }

    if (dim0 == dim1) {
        return *this;
    }

    Shape order(shape_.size());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[dim0], order[dim1]);

    return permute(order);
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor out(shape_, T{});
    for_each_index([&](const Shape& idx) { out.at(idx) = at(idx); });
    return out;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::flatten() const {
    return reshape({numel()});
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::unsqueeze(size_t dim) const {
    if (dim > shape_.size()) {
        throw std::invalid_argument("Unsqueeze dimension out of range");
    }

    Shape new_shape = shape_;
    new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(dim), 1);

    Strides new_strides = strides_;
    if (is_contiguous()) {
        new_strides = make_contiguous_strides(new_shape);
    } else {
        const auto stride_val =
            dim < strides_.size() ? strides_[dim] * shape_[dim] : size_t{1};
        new_strides.insert(
            new_strides.begin() + static_cast<std::ptrdiff_t>(dim), stride_val);
    }

    return Tensor(new_shape, data_, new_strides, offset_);
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::squeeze(size_t dim) const {
    if (dim >= shape_.size()) {
        throw std::invalid_argument("Squeeze dimension out of range");
    }

    if (shape_[dim] != 1) {
        throw std::invalid_argument(
            "Only dimensions of size 1 can be squeezed");
    }

    Shape new_shape = shape_;
    Strides new_strides = strides_;

    new_shape.erase(new_shape.begin() + static_cast<std::ptrdiff_t>(dim));
    new_strides.erase(new_strides.begin() + static_cast<std::ptrdiff_t>(dim));

    return Tensor(new_shape, data_, new_strides, offset_);
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::fill(const T& value) {
    apply_scalar(
        [](T& current, const T& replacement) { current = replacement; }, value);
    return *this;
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& other) {
    apply_tensor(other, [](T& lhs, const T& rhs) { lhs += rhs; });
    return *this;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    Tensor result = clone();
    result += other;
    return result;
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::operator+=(T scalar) {
    apply_scalar([](T& lhs, const T& value) { lhs += value; }, scalar);
    return *this;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::operator+(T scalar) const {
    Tensor result = clone();
    result += scalar;
    return result;
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::operator-=(const Tensor& other) {
    apply_tensor(other, [](T& lhs, const T& rhs) { lhs -= rhs; });
    return *this;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::operator-(const Tensor& other) const {
    Tensor result = clone();
    result -= other;
    return result;
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::operator-=(T scalar) {
    apply_scalar([](T& lhs, const T& value) { lhs -= value; }, scalar);
    return *this;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::operator-(T scalar) const {
    Tensor result = clone();
    result -= scalar;
    return result;
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::operator*=(const Tensor& other) {
    apply_tensor(other, [](T& lhs, const T& rhs) { lhs *= rhs; });
    return *this;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
    Tensor result = clone();
    result *= other;
    return result;
}

template <IsArithmetic T>
Tensor<T>& Tensor<T>::operator*=(T scalar) {
    apply_scalar([](T& lhs, const T& value) { lhs *= value; }, scalar);
    return *this;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::operator*(T scalar) const {
    Tensor result = clone();
    result *= scalar;
    return result;
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::matmul(const Tensor& other) const {
    if (dim() == 1 && other.dim() == 1) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Vector sizes must match for dot");
        }
        T acc{};
        for (size_t i = 0; i < shape_[0]; ++i) {
            acc += at({i}) * other.at({i});
        }
        return Tensor({}, std::vector<T>{acc});
    }

    if (dim() == 2 && other.dim() == 1) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument(
                "Matrix columns must match vector size for matmul");
        }

        Tensor result({shape_[0]}, T{});
        for (size_t i = 0; i < shape_[0]; ++i) {
            T acc{};
            for (size_t k = 0; k < shape_[1]; ++k) {
                acc += at({i, k}) * other.at({k});
            }
            result.at({i}) = acc;
        }
        return result;
    }

    if (dim() == 1 && other.dim() == 2) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument(
                "Vector size must match matrix rows for matmul");
        }
        Tensor result({other.shape_[1]}, T{});
        for (size_t j = 0; j < other.shape_[1]; ++j) {
            T acc{};
            for (size_t i = 0; i < shape_[0]; ++i) {
                acc += at({i}) * other.at({i, j});
            }
            result.at({j}) = acc;
        }
        return result;
    }

    if (dim() == 2 && other.dim() == 2) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument(
                "Inner matrix dimensions must agree for matmul");
        }

        Tensor result({shape_[0], other.shape_[1]}, T{});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                T acc{};
                for (size_t k = 0; k < shape_[1]; ++k) {
                    acc += at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = acc;
            }
        }
        return result;
    }

    throw std::invalid_argument(
        "Matmul is implemented for 1D and 2D tensors only");
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::sum() const {
    T acc{};
    for_each_index([&](const Shape& idx) { acc += at(idx); });
    return Tensor({}, std::vector<T>{acc});
}

template <IsArithmetic T>
Tensor<T> Tensor<T>::sum(size_t dim) const {
    if (dim >= shape_.size()) {
        throw std::invalid_argument("Sum dimension out of range");
    }

    Shape reduced_shape = shape_;
    reduced_shape.erase(reduced_shape.begin() +
                        static_cast<std::ptrdiff_t>(dim));

    Tensor result(reduced_shape, T{});

    if (result.numel() == 0 && numel() == 0) {
        return result;
    }

    result.for_each_index([&](const Shape& idx) {
        Shape expanded_idx(shape_.size(), 0);
        for (size_t i = 0, j = 0; i < shape_.size(); ++i) {
            if (i == dim) {
                continue;
            }
            expanded_idx[i] = idx[j++];
        }

        T acc{};
        for (size_t k = 0; k < shape_[dim]; ++k) {
            expanded_idx[dim] = k;
            acc += at(expanded_idx);
        }

        result.at(idx) = acc;
    });

    return result;
}

template <IsArithmetic T>
typename Tensor<T>::Strides Tensor<T>::make_contiguous_strides(
    const Shape& shape) {
    Strides strides(shape.size(), 0);
    size_t stride = 1;

    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

template <IsArithmetic T>
size_t Tensor<T>::compute_numel(const Shape& shape) {
    if (shape.empty()) {
        return 1;
    }

    size_t total = 1;
    for (const auto dim : shape) {
        total *= dim;
    }
    return total;
}

template <IsArithmetic T>
size_t Tensor<T>::compute_offset(const Shape& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimensionality mismatch");
    }

    size_t offset = offset_;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        offset += indices[i] * strides_[i];
    }

    return offset;
}

template <IsArithmetic T>
template <class Fn>
void Tensor<T>::for_each_index(Fn&& fn) const {
    const auto total = numel();
    if (total == 0) {
        return;
    }

    if (shape_.empty()) {
        fn({});
        return;
    }

    Shape idx(shape_.size(), 0);
    for (size_t linear = 0; linear < total; ++linear) {
        fn(idx);
        for (std::ptrdiff_t dim =
                 static_cast<std::ptrdiff_t>(shape_.size()) - 1;
             dim >= 0; --dim) {
            const auto d = static_cast<size_t>(dim);
            if (++idx[d] < shape_[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}

template <IsArithmetic T>
template <class Fn>
void Tensor<T>::apply_scalar(Fn&& fn, const T& scalar) {
    for_each_index(
        [&](const Shape& idx) { fn((*data_)[compute_offset(idx)], scalar); });
}

template <IsArithmetic T>
template <class Fn>
void Tensor<T>::apply_tensor(const Tensor& other, Fn&& fn) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match");
    }

    for_each_index([&](const Shape& idx) {
        fn((*data_)[compute_offset(idx)],
           other.data_->at(other.compute_offset(idx)));
    });
}
