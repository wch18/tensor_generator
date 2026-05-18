#pragma once

// dtype_traits.h — shared dtype classification helpers (internal).
//
// These predicates are used by both special_value.cpp and tensor_generator.cpp.
// Keeping them in a single location prevents silent divergence when new dtypes
// are added to libtorch.

#include <torch/torch.h>

namespace tgen::detail {

inline bool is_float_dtype(torch::ScalarType dtype) {
    return dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
           dtype == torch::kFloat32 || dtype == torch::kFloat64 ||
           dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

inline bool is_unsigned_int_dtype(torch::ScalarType dtype) {
    return dtype == torch::kUInt8 ||
           dtype == c10::ScalarType::UInt16 ||
           dtype == c10::ScalarType::UInt32 ||
           dtype == c10::ScalarType::UInt64;
}

inline bool has_infinity(torch::ScalarType dtype) {
    return is_float_dtype(dtype) && dtype != torch::kFloat8_e4m3fn;
}

// dtypes where masked_fill_ is not supported natively
inline bool needs_cast_workaround(torch::ScalarType dtype) {
    return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2 ||
           dtype == c10::ScalarType::UInt16 ||
           dtype == c10::ScalarType::UInt32 ||
           dtype == c10::ScalarType::UInt64;
}

} // namespace tgen::detail
