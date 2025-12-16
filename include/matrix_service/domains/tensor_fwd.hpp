#pragma once

#include <type_traits>

template <class T>
concept IsArithmetic = std::is_arithmetic_v<T>;

template <class T>
concept IsQuantized = std::is_same_v<T, float>;

template <IsArithmetic T>
class Tensor;
