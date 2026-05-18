#pragma once

// special_value.h — enumeration and lookup tables for dtype-aware special values.
//
// Provides SpecialValue, an enum of semantically meaningful tensor fill values
// (Zero, One, NaN, Infinity, dtype min/max, etc.), together with two free
// functions:
//   is_valid_for_dtype()        — returns false for combinations that make no
//                                 sense (e.g. NaN on an integer dtype).
//   get_special_value_as_double() — returns the exact floating-point
//                                   representation of a SpecialValue for the
//                                   given dtype, suitable for masked_fill_ ops.

#include <torch/torch.h>

namespace tgen {

enum class SpecialValue {
    Zero,              // 加法零元
    One,               // 乘法单位元
    NegativeOne,       // -1 (unsigned 为 0xFF...FF)
    DTypeMin,          // dtype 表示范围下界
    DTypeMax,          // dtype 表示范围上界
    NaN,               // quiet NaN (仅浮点)
    Infinity,          // +∞ (仅浮点, 不含 e4m3fn)
    NegativeInfinity,  // -∞ (仅浮点, 不含 e4m3fn)
    Epsilon,           // 机器精度 (仅浮点)
    Tiny,              // 最小正常正数 (仅浮点)
};

bool is_valid_for_dtype(SpecialValue sv, torch::ScalarType dtype);

// WARNING: For int64 DTypeMax and uint64 DTypeMax/NegativeOne, the returned
// double is rounded (IEEE 754 cannot exactly represent 2^63-1 or 2^64-1).
// Do NOT cast the result back to int64_t/uint64_t — use get_special_value_as_int64()
// (internal to tensor_generator.cpp) for lossless integer fills.
double get_special_value_as_double(SpecialValue sv, torch::ScalarType dtype);

} // namespace tgen
