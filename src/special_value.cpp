#include "special_value.h"
#include "dtype_traits.h"
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

using namespace detail;

bool is_valid_for_dtype(SpecialValue sv, torch::ScalarType dtype) {
    switch (sv) {
    case SpecialValue::Zero:
        // E8M0 cannot represent zero
        if (dtype == c10::ScalarType::Float8_e8m0fnu) return false;
        return true;
    case SpecialValue::One:
    case SpecialValue::DTypeMin:
    case SpecialValue::DTypeMax:
        return true;
    case SpecialValue::NOne:
        // E8M0 is unsigned, cannot represent -1
        if (dtype == c10::ScalarType::Float8_e8m0fnu) return false;
        return true;
    case SpecialValue::NaN:
        return is_float_dtype(dtype);
    case SpecialValue::Inf:
    case SpecialValue::NInf:
        return has_infinity(dtype);
    case SpecialValue::Epsilon:
    case SpecialValue::Tiny:
        return is_float_dtype(dtype);
    }
    return false;
}

double get_special_value_as_double(SpecialValue sv, torch::ScalarType dtype) {
    switch (sv) {
    case SpecialValue::Zero:
        return 0.0;

    case SpecialValue::One:
        return 1.0;

    case SpecialValue::NOne:
        switch (dtype) {
        case torch::kUInt8:  return 255.0;
        case torch::kBool:   return 1.0;
        default:
            if (dtype == c10::ScalarType::UInt16) return 65535.0;
            if (dtype == c10::ScalarType::UInt32) return 4294967295.0;
            if (dtype == c10::ScalarType::UInt64) return 18446744073709551615.0;
            return -1.0;
        }

    case SpecialValue::DTypeMin:
        switch (dtype) {
        case torch::kBool:       return 0.0;
        case torch::kUInt8:      return 0.0;
        case torch::kInt8:       return -128.0;
        case torch::kInt16:      return -32768.0;
        case torch::kInt32:      return (double)std::numeric_limits<int32_t>::lowest();
        case torch::kInt64:      return (double)std::numeric_limits<int64_t>::lowest();
        case torch::kFloat16:    return -65504.0;
        case torch::kBFloat16:   return (double)-3.3895313892515355e+38;
        case torch::kFloat32:    return (double)(-FLT_MAX);
        case torch::kFloat64:    return -DBL_MAX;
        case torch::kFloat8_e4m3fn: return -448.0;
        case torch::kFloat8_e5m2:   return -57344.0;
        default:
            // E8M0: min = 2^(-127)
            if (dtype == c10::ScalarType::Float8_e8m0fnu)
                return 5.877471754111438e-39;
            return 0.0;
        }

    case SpecialValue::DTypeMax:
        switch (dtype) {
        case torch::kBool:       return 1.0;
        case torch::kUInt8:      return 255.0;
        case torch::kInt8:       return 127.0;
        case torch::kInt16:      return 32767.0;
        case torch::kInt32:      return (double)std::numeric_limits<int32_t>::max();
        case torch::kInt64:      return (double)std::numeric_limits<int64_t>::max();
        case torch::kFloat16:    return 65504.0;
        case torch::kBFloat16:   return (double)3.3895313892515355e+38;
        case torch::kFloat32:    return (double)FLT_MAX;
        case torch::kFloat64:    return DBL_MAX;
        case torch::kFloat8_e4m3fn: return 448.0;
        case torch::kFloat8_e5m2:   return 57344.0;
        default:
            if (dtype == c10::ScalarType::Float8_e8m0fnu)
                return 1.7014118346046923e+38;  // 2^127
            if (dtype == c10::ScalarType::UInt16) return 65535.0;
            if (dtype == c10::ScalarType::UInt32) return 4294967295.0;
            if (dtype == c10::ScalarType::UInt64) return 18446744073709551615.0;
            return 0.0;
        }

    case SpecialValue::NaN:
        return std::numeric_limits<double>::quiet_NaN();

    case SpecialValue::Inf:
        return std::numeric_limits<double>::infinity();

    case SpecialValue::NInf:
        return -std::numeric_limits<double>::infinity();

    case SpecialValue::Epsilon:
        switch (dtype) {
        case torch::kFloat16:       return 9.765625e-4;
        case torch::kBFloat16:      return 7.8125e-3;
        case torch::kFloat32:       return (double)FLT_EPSILON;
        case torch::kFloat64:       return DBL_EPSILON;
        case torch::kFloat8_e4m3fn: return 0.125;
        case torch::kFloat8_e5m2:   return 0.25;
        default:
            // E8M0: next after 1.0 is 2.0, so epsilon = 1.0
            if (dtype == c10::ScalarType::Float8_e8m0fnu) return 1.0;
            return 0.0;
        }

    case SpecialValue::Tiny:
        switch (dtype) {
        case torch::kFloat16:       return 6.103515625e-5;
        case torch::kBFloat16:      return (double)FLT_MIN;
        case torch::kFloat32:       return (double)FLT_MIN;
        case torch::kFloat64:       return DBL_MIN;
        case torch::kFloat8_e4m3fn: return 0.015625;
        case torch::kFloat8_e5m2:   return 6.103515625e-5;
        default:
            // E8M0: smallest positive value = 2^(-127)
            if (dtype == c10::ScalarType::Float8_e8m0fnu)
                return 5.877471754111438e-39;
            return 0.0;
        }
    }
    return 0.0;
}

int64_t get_special_value_as_int64(SpecialValue sv, torch::ScalarType dtype) {
    switch (sv) {
    case SpecialValue::Zero: return 0;
    case SpecialValue::One:  return 1;
    case SpecialValue::NOne:
        if (dtype == torch::kUInt8) return 255;
        if (dtype == c10::ScalarType::UInt16) return 65535;
        if (dtype == c10::ScalarType::UInt32) return 4294967295LL;
        if (dtype == c10::ScalarType::UInt64) return -1LL;
        return -1;
    case SpecialValue::DTypeMin:
        if (is_unsigned_int_dtype(dtype)) return 0;
        if (dtype == torch::kInt8)  return -128;
        if (dtype == torch::kInt16) return -32768;
        if (dtype == torch::kInt32) return (int64_t)std::numeric_limits<int32_t>::lowest();
        if (dtype == torch::kInt64) return std::numeric_limits<int64_t>::lowest();
        return 0;
    case SpecialValue::DTypeMax:
        if (dtype == torch::kUInt8) return 255;
        if (dtype == c10::ScalarType::UInt16) return 65535;
        if (dtype == c10::ScalarType::UInt32) return 4294967295LL;
        if (dtype == c10::ScalarType::UInt64) return -1LL;
        if (dtype == torch::kInt8)  return 127;
        if (dtype == torch::kInt16) return 32767;
        if (dtype == torch::kInt32) return (int64_t)std::numeric_limits<int32_t>::max();
        if (dtype == torch::kInt64) return std::numeric_limits<int64_t>::max();
        return 0;
    default: return 0;
    }
}

torch::Scalar get_scalar(SpecialValue sv, torch::ScalarType dtype) {
    if (dtype == torch::kBool) {
        return get_special_value_as_double(sv, dtype) != 0.0;
    } else if (is_float_dtype(dtype)) {
        return get_special_value_as_double(sv, dtype);
    } else {
        return get_special_value_as_int64(sv, dtype);
    }
}

SpecialValue special_value_from_int(int v) {
    switch (v) {
    case 0:  return SpecialValue::Zero;
    case 1:  return SpecialValue::One;
    case -1: return SpecialValue::NOne;
    default:
        throw std::invalid_argument(
            "no SpecialValue mapping for integer " + std::to_string(v) +
            "; supported: 0 (Zero), 1 (One), -1 (NOne)");
    }
}

SpecialValue special_value_from_double(double v) {
    if (v == 0.0)  return SpecialValue::Zero;
    if (v == 1.0)  return SpecialValue::One;
    if (v == -1.0) return SpecialValue::NOne;
    if (std::isnan(v))  return SpecialValue::NaN;
    if (std::isinf(v) && v > 0) return SpecialValue::Inf;
    if (std::isinf(v) && v < 0) return SpecialValue::NInf;
    throw std::invalid_argument(
        "no SpecialValue mapping for double " + std::to_string(v) +
        "; supported: 0, 1, -1, NaN, +Inf, -Inf");
}
