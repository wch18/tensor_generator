#pragma once

// tensor_spec.h — declarative specification for tensor content generation.
//
// TensorSpec describes the probability distribution used to fill a tensor:
//   - A list of DiscreteEntry items, each pairing a SpecialValue with its
//     probability mass (must sum to ≤ 1.0).
//   - An optional BaseDist (uniform or normal) that receives the remaining
//     probability mass (1 - sum_of_discrete_probabilities).
//
// Usage examples:
//
//   // 100 % NaN (float-only)
//   TensorSpec{{SpecialValue::NaN, 1.0}}
//
//   // 30 % zeros, 70 % uniform in [-1, 1]
//   TensorSpec{{SpecialValue::Zero, 0.3}}.with_uniform(-1.0, 1.0)
//
//   // 20 % NaN, 10 % +inf, rest uniform
//   TensorSpec{{SpecialValue::NaN, 0.2},
//              {SpecialValue::Infinity, 0.1}}.with_uniform(-5.0, 5.0)
//
//   // Pure normal distribution, no discrete mass
//   TensorSpec{}.with_normal(0.0, 1.0)

#include "special_value.h"
#include <initializer_list>
#include <vector>

namespace tgen {

enum class BaseDistType { Uniform, Normal };

struct BaseDist {
    BaseDistType type = BaseDistType::Uniform;
    double param1 = 0.0;  // uniform: low,  normal: mean
    double param2 = 1.0;  // uniform: high, normal: std
};

struct DiscreteEntry {
    SpecialValue value;
    double probability;
};

struct TensorSpec {
    std::vector<DiscreteEntry> discrete;
    BaseDist base_dist = {};

    TensorSpec() = default;

    TensorSpec(std::initializer_list<DiscreteEntry> entries)
        : discrete(entries) {}

    explicit TensorSpec(std::vector<DiscreteEntry> entries)
        : discrete(std::move(entries)) {}

    TensorSpec(std::initializer_list<DiscreteEntry> entries, BaseDist dist)
        : discrete(entries), base_dist(dist) {}

    TensorSpec& with_base_dist(BaseDistType type, double p1, double p2) {
        base_dist = {type, p1, p2};
        return *this;
    }

    TensorSpec& with_uniform(double lo, double hi) {
        base_dist = {BaseDistType::Uniform, lo, hi};
        return *this;
    }

    TensorSpec& with_normal(double mean, double std) {
        base_dist = {BaseDistType::Normal, mean, std};
        return *this;
    }
};

} // namespace tgen
