#pragma once

#include <torch/torch.h>

enum class SpecialValue {
    Zero,
    One,
    NOne,       // -1 (unsigned wraps to 0xFF...FF)
    DTypeMin,
    DTypeMax,
    NaN,
    Inf,        // +inf (float only, not e4m3fn/e8m0fnu)
    NInf,       // -inf (float only, not e4m3fn/e8m0fnu)
    Epsilon,
    Tiny,
};

bool is_valid_for_dtype(SpecialValue sv, torch::ScalarType dtype);

double get_special_value_as_double(SpecialValue sv, torch::ScalarType dtype);

int64_t get_special_value_as_int64(SpecialValue sv, torch::ScalarType dtype);

// Returns a torch::Scalar holding the value in the appropriate C++ type
// (bool for kBool, int64_t for integer dtypes, double for float dtypes).
torch::Scalar get_scalar(SpecialValue sv, torch::ScalarType dtype);

template<torch::ScalarType DType>
torch::Scalar get_scalar(SpecialValue sv) {
    return get_scalar(sv, DType);
}
