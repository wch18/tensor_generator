#pragma once

#include <torch/torch.h>

namespace detail {

inline bool is_float_dtype(torch::ScalarType dtype) {
    return c10::isFloatingType(dtype);
}

inline bool is_int_dtype(torch::ScalarType dtype) {
    return c10::isIntegralType(dtype, false);
}

inline bool is_unsigned_int_dtype(torch::ScalarType dtype) {
    return c10::isIntegralType(dtype, false) && !c10::isSignedType(dtype);
}

inline bool has_infinity(torch::ScalarType dtype) {
    return c10::isFloatingType(dtype) &&
           dtype != torch::kFloat8_e4m3fn &&
           dtype != c10::ScalarType::Float8_e4m3fnuz;
}

// dtypes where masked_fill_ / torch::where are not supported natively
inline bool needs_cast_workaround(torch::ScalarType dtype) {
    return c10::isFloat8Type(dtype) ||
           dtype == c10::ScalarType::UInt16 ||
           dtype == c10::ScalarType::UInt32 ||
           dtype == c10::ScalarType::UInt64;
}

} // namespace detail
