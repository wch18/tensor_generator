#pragma once

#include <torch/torch.h>

enum class SpecialValue {
    Zero,
    One,
    NOne,       // -1 (unsigned wraps to 0xFF...FF)
    DTypeMin,
    DTypeMax,
    NaN,
    Inf,        // +inf (float only, not e4m3fn)
    NInf,       // -inf (float only, not e4m3fn)
    Epsilon,
    Tiny,
};

bool is_valid_for_dtype(SpecialValue sv, torch::ScalarType dtype);

double get_special_value_as_double(SpecialValue sv, torch::ScalarType dtype);
