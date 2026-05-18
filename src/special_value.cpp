#include "special_value.h"
#include "dtype_traits.h"
#include <cmath>
#include <cfloat>
#include <limits>

using namespace detail;

bool is_valid_for_dtype(SpecialValue sv, torch::ScalarType dtype) {
    switch (sv) {
    case SpecialValue::Zero:
    case SpecialValue::One:
    case SpecialValue::NOne:
    case SpecialValue::DTypeMin:
    case SpecialValue::DTypeMax:
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
            if (is_unsigned_int_dtype(dtype)) return 0.0;
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
        default: return 0.0;
        }

    case SpecialValue::Tiny:
        switch (dtype) {
        case torch::kFloat16:       return 6.103515625e-5;
        case torch::kBFloat16:      return (double)FLT_MIN;
        case torch::kFloat32:       return (double)FLT_MIN;
        case torch::kFloat64:       return DBL_MIN;
        case torch::kFloat8_e4m3fn: return 0.015625;
        case torch::kFloat8_e5m2:   return 6.103515625e-5;
        default: return 0.0;
        }
    }
    return 0.0;
}
