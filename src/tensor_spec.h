#pragma once

// tensor_spec.h — declarative specification for tensor content generation.
//
// TensorSpec: discrete distribution (list of value + probability pairs).
// DiscreteValue: holds either a SpecialValue or a raw numeric literal.
// BaseDist: continuous/integer base distribution, supports torch::Generator.
//
// DiscreteEntry accepts SpecialValue, int, or double as the value:
//   {SpecialValue::Zero, 0.3}   — explicit SpecialValue
//   {0, 0.3}                    — int 0 maps to SpecialValue::Zero
//   {2, 0.3}                    — raw int value 2
//   {3.14, 0.3}                 — raw double value 3.14
//
// Well-known numerics (0, 1, -1, NaN, ±Inf) are promoted to their
// SpecialValue equivalents for dtype-aware behavior (e.g. -1 → 0xFF
// on uint8). All other numerics are stored as raw values.

#include "special_value.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

enum class BaseDistType {
    Uniform, Normal, RandInt,
    Exponential, Poisson, LogNormal, Cauchy, Bernoulli
};

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

    static BaseDist uniform(double lo, double hi,
                            std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::Uniform;
        d.param1 = lo;
        d.param2 = hi;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist normal(double mean, double std,
                           std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::Normal;
        d.param1 = mean;
        d.param2 = std;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist randint(int64_t lo, int64_t hi,
                            std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::RandInt;
        d.param1 = (double)lo;
        d.param2 = (double)hi;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist exponential(double lambda = 1.0,
                                std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::Exponential;
        d.param1 = lambda;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist poisson(double rate = 1.0,
                            std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::Poisson;
        d.param1 = rate;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist lognormal(double mean = 0.0, double std = 1.0,
                              std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::LogNormal;
        d.param1 = mean;
        d.param2 = std;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist cauchy(double median = 0.0, double sigma = 1.0,
                           std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::Cauchy;
        d.param1 = median;
        d.param2 = sigma;
        return seed ? d.with_seed(*seed) : d;
    }

    static BaseDist bernoulli(double p = 0.5,
                              std::optional<int64_t> seed = std::nullopt) {
        BaseDist d;
        d.type = BaseDistType::Bernoulli;
        d.param1 = p;
        return seed ? d.with_seed(*seed) : d;
    }

    BaseDist with_generator(torch::Generator gen) const {
        BaseDist copy = *this;
        copy.generator = std::move(gen);
        return copy;
    }

    BaseDist with_seed(int64_t seed) const {
        auto gen = at::detail::createCPUGenerator();
        gen.set_current_seed(seed);
        return with_generator(std::move(gen));
    }
};

class DiscreteValue {
public:
    DiscreteValue(SpecialValue sv) : special_(true), sv_(sv), raw_(0.0) {}

    DiscreteValue(int v) : special_(true), sv_(SpecialValue::Zero), raw_(v) {
        switch (v) {
        case 0:  sv_ = SpecialValue::Zero; break;
        case 1:  sv_ = SpecialValue::One;  break;
        case -1: sv_ = SpecialValue::NOne; break;
        default: special_ = false;
        }
    }

    DiscreteValue(double v) : special_(false), sv_(SpecialValue::Zero), raw_(v) {
        if (v == 0.0)                      { special_ = true; sv_ = SpecialValue::Zero; }
        else if (v == 1.0)                 { special_ = true; sv_ = SpecialValue::One;  }
        else if (v == -1.0)                { special_ = true; sv_ = SpecialValue::NOne; }
        else if (std::isnan(v))            { special_ = true; sv_ = SpecialValue::NaN;  }
        else if (std::isinf(v) && v > 0)   { special_ = true; sv_ = SpecialValue::Inf;  }
        else if (std::isinf(v) && v < 0)   { special_ = true; sv_ = SpecialValue::NInf; }
    }

    bool is_special() const { return special_; }
    SpecialValue as_special_value() const {
        assert(special_ && "as_special_value() called on a raw DiscreteValue");
        return sv_;
    }
    double raw() const { return raw_; }

    bool operator==(const DiscreteValue& o) const {
        if (special_ != o.special_) return false;
        if (special_) return sv_ == o.sv_;
        if (std::isnan(raw_) && std::isnan(o.raw_)) return true;
        return raw_ == o.raw_;
    }
    bool operator==(SpecialValue sv) const { return special_ && sv_ == sv; }
    bool operator!=(const DiscreteValue& o) const { return !(*this == o); }
    bool operator!=(SpecialValue sv) const { return !(*this == sv); }

    bool is_valid_for_dtype(torch::ScalarType dtype) const {
        if (special_) return ::is_valid_for_dtype(sv_, dtype);
        return true;
    }

    double as_double(torch::ScalarType dtype) const {
        if (special_) return get_special_value_as_double(sv_, dtype);
        return raw_;
    }

    int64_t as_int64(torch::ScalarType dtype) const {
        if (special_) return get_special_value_as_int64(sv_, dtype);
        if (raw_ < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
            raw_ >= static_cast<double>(std::numeric_limits<int64_t>::max()) + 1.0)
            throw std::invalid_argument("raw value out of int64 range");
        return static_cast<int64_t>(raw_);
    }

private:
    bool special_;
    SpecialValue sv_;
    double raw_;
};

struct DiscreteEntry {
    DiscreteValue value;
    double probability;

    DiscreteEntry(SpecialValue v, double p) : value(v), probability(p) {}
    DiscreteEntry(int v, double p) : value(v), probability(p) {}
    DiscreteEntry(double v, double p) : value(v), probability(p) {}
};

struct TensorSpec {
    std::vector<DiscreteEntry> discrete;

    TensorSpec() = default;

    TensorSpec(std::initializer_list<DiscreteEntry> entries)
        : discrete(entries) {}

    explicit TensorSpec(std::vector<DiscreteEntry> entries)
        : discrete(std::move(entries)) {}
};
