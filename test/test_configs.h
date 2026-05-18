#pragma once

// test_configs.h — shared test configuration for tensor_generator tests.

#include "../src/tensor_spec.h"
#include <string>
#include <vector>

// ============================================================
// dtype classification helpers (delegating to c10)
// ============================================================
static bool is_float_type(torch::ScalarType dt) {
    return c10::isFloatingType(dt);
}

static bool is_int_type(torch::ScalarType dt) {
    return c10::isIntegralType(dt, false);
}

// ============================================================
// Config definitions for cartesian product
// ============================================================

struct TestConfig {
    std::string name;
    TensorSpec spec;
    BaseDist base_dist;
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
         TensorSpec{{SpecialValue::NOne, 0.4}, {SpecialValue::DTypeMin, 0.3}, {SpecialValue::DTypeMax, 0.3}},
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
         TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Inf, 0.15},
                    {SpecialValue::NInf, 0.15}, {SpecialValue::Epsilon, 0.1},
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
                    {SpecialValue::NOne, 0.3}},
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

        {"pure_randint",
         TensorSpec{},
         BaseDist::randint(-100, 100),
         compat_int_only},

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
        {1},
        {7},
        {100},
        {3, 4},
        {2, 3, 4},
        {2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {1000000},
    };
}

// ============================================================
// DType definitions
// ============================================================
static std::vector<torch::ScalarType> get_all_dtypes() {
    return {
        torch::kUInt8,
        c10::ScalarType::UInt16,
        c10::ScalarType::UInt32,
        c10::ScalarType::UInt64,
        torch::kInt8,
        torch::kInt16,
        torch::kInt32,
        torch::kInt64,
        torch::kFloat16,
        torch::kBFloat16,
        torch::kFloat32,
        torch::kFloat64,
        torch::kBool,
        torch::kFloat8_e4m3fn,
        torch::kFloat8_e5m2,
    };
}
