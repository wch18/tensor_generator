#pragma once

// tensor_spec.h — declarative specification for tensor content generation.
//
// TensorSpec describes the discrete probability distribution used to fill a
// tensor: a list of DiscreteEntry items, each pairing a SpecialValue with its
// probability mass (must sum to <= 1.0).
//
// BaseDist describes the continuous base distribution that receives the
// remaining probability mass (1 - sum_of_discrete_probabilities). It is
// supplied separately to generate_tensor() rather than embedded in TensorSpec.
//
// Usage examples:
//
//   // 100% NaN (float-only)
//   TensorSpec{{SpecialValue::NaN, 1.0}}
//
//   // 30% zeros, 70% uniform in [-1, 1] (default base dist)
//   TensorSpec{{SpecialValue::Zero, 0.3}}
//
//   // 20% NaN, 10% +inf, rest uniform in [-5, 5]
//   generate_tensor(shape, dtype,
//       TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Infinity, 0.1}},
//       BaseDist::uniform(-5.0, 5.0))
//
//   // Pure normal distribution, no discrete mass
//   generate_tensor(shape, dtype, TensorSpec{}, BaseDist::normal(0.0, 1.0))

#include "special_value.h"
#include <initializer_list>
#include <vector>

namespace tgen {

enum class BaseDistType { Uniform, Normal };

struct BaseDist {
    BaseDistType type = BaseDistType::Uniform;
    double param1 = -1.0;  // uniform: low,  normal: mean
    double param2 = 1.0;   // uniform: high, normal: std

    static BaseDist uniform(double lo, double hi) {
        return {BaseDistType::Uniform, lo, hi};
    }

    static BaseDist normal(double mean, double std) {
        return {BaseDistType::Normal, mean, std};
    }
};

struct DiscreteEntry {
    SpecialValue value;
    double probability;
};

struct TensorSpec {
    std::vector<DiscreteEntry> discrete;

    TensorSpec() = default;

    TensorSpec(std::initializer_list<DiscreteEntry> entries)
        : discrete(entries) {}

    explicit TensorSpec(std::vector<DiscreteEntry> entries)
        : discrete(std::move(entries)) {}
};

} // namespace tgen
