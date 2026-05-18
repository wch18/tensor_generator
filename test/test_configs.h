#pragma once

// test_configs.h — shared test configuration for tensor_generator tests.
//
// Provides TestConfig (pairing a TensorSpec + BaseDist with a dtype
// compatibility predicate), and the three factory functions used to drive
// the cartesian product parameterised suite:
//   get_all_configs()  — all TensorSpec scenarios under test
//   get_all_shapes()   — shape list covering 1-D through 5-D and large sizes
//   get_all_dtypes()   — every dtype supported by tensor_generator

#include "../src/tensor_spec.h"
#include <string>
#include <vector>

using namespace tgen;

// ============================================================
// dtype classification helpers
// ============================================================
static bool is_float_type(torch::ScalarType dt) {
    return dt == torch::kFloat16 || dt == torch::kBFloat16 ||
           dt == torch::kFloat32 || dt == torch::kFloat64 ||
           dt == torch::kFloat8_e4m3fn || dt == torch::kFloat8_e5m2;
}

static bool is_int_type(torch::ScalarType dt) {
    return dt == torch::kUInt8 || dt == torch::kInt8 ||
           dt == torch::kInt16 || dt == torch::kInt32 || dt == torch::kInt64 ||
           dt == c10::ScalarType::UInt16 ||
           dt == c10::ScalarType::UInt32 ||
           dt == c10::ScalarType::UInt64;
}

static bool is_unsigned_type(torch::ScalarType dt) {
    return dt == torch::kUInt8 ||
           dt == c10::ScalarType::UInt16 ||
           dt == c10::ScalarType::UInt32 ||
           dt == c10::ScalarType::UInt64;
}

// ============================================================
// Config definitions for cartesian product
// ============================================================

struct TestConfig {
    std::string name;
    TensorSpec spec;
    BaseDist base_dist;
    // Predicate: is this config compatible with a given dtype?
    bool (*is_compatible)(torch::ScalarType);
};

// Compatibility predicates
static bool compat_all(torch::ScalarType) { return true; }

static bool compat_float_only(torch::ScalarType dt) { return is_float_type(dt); }

static bool compat_float_with_inf(torch::ScalarType dt) {
    return is_float_type(dt) && dt != torch::kFloat8_e4m3fn;
}

static bool compat_non_bool(torch::ScalarType dt) { return dt != torch::kBool; }

static bool compat_int_only(torch::ScalarType dt) { return is_int_type(dt); }

// All test configurations
static std::vector<TestConfig> get_all_configs() {
    return {
        {"all_zeros",
         TensorSpec{{SpecialValue::Zero, 1.0}},
         {},
         compat_all},

        {"all_ones",
         TensorSpec{{SpecialValue::One, 1.0}},
         {},
         compat_all},

        {"half_zero_half_one",
         TensorSpec{{SpecialValue::Zero, 0.5}, {SpecialValue::One, 0.5}},
         {},
         compat_all},

        {"min_max_zero",
         TensorSpec{{SpecialValue::Zero, 0.3}, {SpecialValue::DTypeMin, 0.3}, {SpecialValue::DTypeMax, 0.4}},
         {},
         compat_all},

        {"neg1_min_max",
         TensorSpec{{SpecialValue::NegativeOne, 0.4}, {SpecialValue::DTypeMin, 0.3}, {SpecialValue::DTypeMax, 0.3}},
         {},
         compat_all},

        {"discrete_plus_uniform",
         TensorSpec{{SpecialValue::Zero, 0.3}, {SpecialValue::DTypeMax, 0.1}},
         BaseDist::uniform(-10.0, 10.0),
         compat_non_bool},

        {"discrete_plus_normal",
         TensorSpec{{SpecialValue::Zero, 0.2}, {SpecialValue::One, 0.1}},
         BaseDist::normal(0.0, 5.0),
         compat_non_bool},

        {"float_specials",
         TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Infinity, 0.15},
                    {SpecialValue::NegativeInfinity, 0.15}, {SpecialValue::Epsilon, 0.1},
                    {SpecialValue::Tiny, 0.1}},
         BaseDist::uniform(-1.0, 1.0),
         compat_float_with_inf},

        {"float_no_inf",
         TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Epsilon, 0.2},
                    {SpecialValue::Tiny, 0.2}, {SpecialValue::DTypeMax, 0.2}},
         BaseDist::uniform(-1.0, 1.0),
         compat_float_only},

        {"rare_event",
         TensorSpec{{SpecialValue::DTypeMax, 0.01}, {SpecialValue::Zero, 0.01}},
         BaseDist::uniform(-1.0, 1.0),
         compat_non_bool},

        {"heavy_discrete",
         TensorSpec{{SpecialValue::Zero, 0.3}, {SpecialValue::One, 0.3},
                    {SpecialValue::NegativeOne, 0.3}},
         BaseDist::uniform(-100.0, 100.0),
         compat_non_bool},

        {"int_boundary",
         TensorSpec{{SpecialValue::DTypeMin, 0.5}, {SpecialValue::DTypeMax, 0.5}},
         {},
         compat_int_only},

        {"pure_continuous",
         TensorSpec{},
         BaseDist::uniform(-5.0, 5.0),
         compat_non_bool},

        {"all_nan",
         TensorSpec{{SpecialValue::NaN, 1.0}},
         {},
         compat_float_only},

        {"all_dtypemin",
         TensorSpec{{SpecialValue::DTypeMin, 1.0}},
         {},
         compat_all},

        {"all_dtypemax",
         TensorSpec{{SpecialValue::DTypeMax, 1.0}},
         {},
         compat_all},
    };
}

// ============================================================
// Shape definitions
// ============================================================
static std::vector<std::vector<int64_t>> get_all_shapes() {
    return {
        {1},                    // single element
        {7},                    // small 1D
        {100},                  // medium 1D
        {3, 4},                 // 2D
        {2, 3, 4},             // 3D
        {2, 3, 4, 5},          // 4D
        {2, 3, 4, 5, 6},       // 5D
        {1000000},             // large 1D (for statistical verification)
    };
}

// ============================================================
// DType definitions
// ============================================================
static std::vector<torch::ScalarType> get_all_dtypes() {
    return {
        torch::kUInt8,              // unsigned 8-bit
        c10::ScalarType::UInt16,    // unsigned 16-bit
        c10::ScalarType::UInt32,    // unsigned 32-bit
        c10::ScalarType::UInt64,    // unsigned 64-bit
        torch::kInt8,               // signed 8-bit
        torch::kInt16,              // signed 16-bit
        torch::kInt32,              // signed 32-bit
        torch::kInt64,              // signed 64-bit
        torch::kFloat16,            // IEEE 754 half
        torch::kBFloat16,           // Brain float 16
        torch::kFloat32,            // single precision
        torch::kFloat64,            // double precision
        torch::kBool,               // boolean
        torch::kFloat8_e4m3fn,      // FP8 e4m3 (no inf)
        torch::kFloat8_e5m2,        // FP8 e5m2
    };
}
