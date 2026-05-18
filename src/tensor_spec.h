#pragma once

// tensor_spec.h — declarative specification for tensor content generation.
//
// TensorSpec: discrete distribution (list of SpecialValue + probability pairs).
// BaseDist: continuous/integer base distribution, supports torch::Generator.
//
// BaseDist can be implicitly constructed from a torch::Generator, yielding
// a default Uniform(-1, 1) with that generator attached.

#include "special_value.h"
#include <initializer_list>
#include <vector>

enum class BaseDistType { Uniform, Normal, RandInt };

struct BaseDist {
    BaseDistType type = BaseDistType::Uniform;
    double param1 = -1.0;  // uniform: low,  normal: mean, randint: low (inclusive)
    double param2 = 1.0;   // uniform: high, normal: std,  randint: high (exclusive)
    torch::Generator generator;

    BaseDist() = default;

    // Implicit conversion from Generator: default Uniform(-1, 1) with generator
    BaseDist(torch::Generator gen)
        : type(BaseDistType::Uniform), param1(-1.0), param2(1.0),
          generator(std::move(gen)) {}

    static BaseDist uniform(double lo, double hi) {
        BaseDist d;
        d.type = BaseDistType::Uniform;
        d.param1 = lo;
        d.param2 = hi;
        return d;
    }

    static BaseDist normal(double mean, double std) {
        BaseDist d;
        d.type = BaseDistType::Normal;
        d.param1 = mean;
        d.param2 = std;
        return d;
    }

    static BaseDist randint(int64_t lo, int64_t hi) {
        BaseDist d;
        d.type = BaseDistType::RandInt;
        d.param1 = (double)lo;
        d.param2 = (double)hi;
        return d;
    }

    BaseDist with_generator(torch::Generator gen) const {
        BaseDist copy = *this;
        copy.generator = std::move(gen);
        return copy;
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
