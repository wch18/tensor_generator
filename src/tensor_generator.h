#pragma once

#include "tensor_spec.h"
#include "dtype_traits.h"
#include <cmath>
#include <optional>
#include <stdexcept>

namespace detail {

inline void safe_masked_fill(torch::Tensor& result, const torch::Tensor& mask,
                             const DiscreteValue& dv, torch::ScalarType dtype) {
    if (dtype == torch::kBool) {
        result.masked_fill_(mask, dv.as_double(dtype) != 0.0);
    } else if (is_float_dtype(dtype) && !needs_cast_workaround(dtype)) {
        result.masked_fill_(mask, dv.as_double(dtype));
    } else if (is_float_dtype(dtype)) {
        auto work = result.to(torch::kFloat64);
        work.masked_fill_(mask, dv.as_double(dtype));
        result = work.to(dtype);
    } else if (dtype == c10::ScalarType::UInt64) {
        auto work = result.to(torch::kInt64);
        work.masked_fill_(mask, dv.as_int64(dtype));
        result = work.to(dtype);
    } else if (needs_cast_workaround(dtype)) {
        auto work = result.to(torch::kFloat64);
        work.masked_fill_(mask, (double)dv.as_int64(dtype));
        result = work.to(dtype);
    } else {
        result.masked_fill_(mask, dv.as_int64(dtype));
    }
}

} // namespace detail

inline torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},
    std::optional<int64_t> seed = std::nullopt)
{
    using namespace detail;

    if (seed.has_value()) {
        torch::manual_seed(seed.value());
    }

    // --- Resolve discrete entries (with dtype fallbacks) ---
    // Track original sum for count mode validation (before NaN skip)
    std::vector<DiscreteEntry> discrete;
    double p_sum = 0.0;
    double original_sum = 0.0;
    for (auto& e : spec.discrete) {
        if (e.probability < 0)
            throw std::invalid_argument("probability must be non-negative");
        original_sum += e.probability;
        if (e.value.is_valid_for_dtype(dtype)) {
            discrete.push_back(e);
        } else if (e.value.is_special()) {
            auto sv = e.value.as_special_value();
            if (sv == SpecialValue::Inf || sv == SpecialValue::NInf) {
                SpecialValue fallback = (sv == SpecialValue::Inf)
                    ? SpecialValue::DTypeMax : SpecialValue::DTypeMin;
                discrete.push_back({fallback, e.probability});
            } else if (sv == SpecialValue::NaN) {
                continue;
            } else {
                throw std::invalid_argument(
                    "discrete value not valid for given dtype");
            }
        } else {
            throw std::invalid_argument(
                "discrete value not valid for given dtype");
        }
        p_sum += discrete.back().probability;
    }

    // --- Auto-detect count mode ---
    // Use original_sum (pre-skip) so NaN removal doesn't break detection.
    bool count_mode = spec.count_mode;
    if (!count_mode && !spec.discrete.empty()) {
        bool all_positive_int = true;
        for (auto& e : spec.discrete) {
            double p = e.probability;
            if (p < 1.0 || p != std::floor(p)) {
                all_positive_int = false;
                break;
            }
        }
        if (all_positive_int && original_sum > 1.0 + 1e-9)
            count_mode = true;
    }

    // --- Mode-specific validation ---
    int64_t numel = 1;
    for (auto d : shape) numel *= d;

    bool all_discrete;
    if (count_mode) {
        // Validate against original user-declared sum so NaN-skipped
        // entries still count toward the overflow check.
        int64_t original_total = 0;
        for (auto& e : spec.discrete) {
            double p = e.probability;
            if (p != std::floor(p) || p < 0)
                throw std::invalid_argument(
                    "count mode requires non-negative integer counts");
            original_total += (int64_t)p;
        }
        if (original_total > numel)
            throw std::invalid_argument(
                "discrete counts sum (" + std::to_string(original_total) +
                ") exceeds tensor size (" + std::to_string(numel) + ")");
        int64_t resolved_count = 0;
        for (auto& e : discrete)
            resolved_count += (int64_t)e.probability;
        all_discrete = (resolved_count == numel);
    } else {
        if (p_sum > 1.0 + 1e-9)
            throw std::invalid_argument("discrete probabilities exceed 1.0");
        all_discrete = (p_sum >= 1.0 - 1e-9);
    }

    // --- Validate BaseDist / dtype compatibility ---
    if (!all_discrete) {
        if (base_dist.type == BaseDistType::RandInt) {
            if (is_float_dtype(dtype))
                throw std::invalid_argument(
                    "RandInt base distribution is incompatible with float dtype");
            if (dtype == torch::kBool)
                throw std::invalid_argument(
                    "RandInt base distribution is incompatible with bool dtype");
        }
        if (base_dist.type == BaseDistType::Poisson) {
            if (is_float_dtype(dtype))
                throw std::invalid_argument(
                    "Poisson base distribution is incompatible with float dtype");
            if (dtype == torch::kBool)
                throw std::invalid_argument(
                    "Poisson base distribution is incompatible with bool dtype");
        }
    }

    // --- Generate base tensor ---
    auto f64_opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto i64_opts = torch::TensorOptions().dtype(torch::kInt64);
    auto gen = base_dist.generator;

    torch::Tensor result;
    if (all_discrete) {
        if (dtype == c10::ScalarType::UInt64) {
            result = torch::zeros(shape, i64_opts).to(dtype);
        } else if (needs_cast_workaround(dtype)) {
            result = torch::zeros(shape, f64_opts).to(dtype);
        } else {
            result = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
        }
    } else {
        torch::Tensor base;
        switch (base_dist.type) {
        case BaseDistType::Uniform:
            if (gen.defined())
                base = torch::rand(shape, gen, f64_opts);
            else
                base = torch::rand(shape, f64_opts);
            base = base * (base_dist.param2 - base_dist.param1) + base_dist.param1;
            break;
        case BaseDistType::Normal:
            if (gen.defined())
                base = torch::randn(shape, gen, f64_opts);
            else
                base = torch::randn(shape, f64_opts);
            base = base * base_dist.param2 + base_dist.param1;
            break;
        case BaseDistType::RandInt: {
            int64_t lo = (int64_t)base_dist.param1;
            int64_t hi = (int64_t)base_dist.param2;
            if (gen.defined())
                base = torch::randint(lo, hi, shape, gen, i64_opts);
            else
                base = torch::randint(lo, hi, shape, i64_opts);
            break;
        }
        case BaseDistType::Exponential: {
            if (gen.defined())
                base = torch::rand(shape, gen, f64_opts);
            else
                base = torch::rand(shape, f64_opts);
            base = -torch::log1p(-base) / base_dist.param1;
            break;
        }
        case BaseDistType::Poisson: {
            auto lambda_tensor = torch::full(shape, base_dist.param1, f64_opts);
            if (gen.defined())
                base = torch::poisson(lambda_tensor, gen);
            else
                base = torch::poisson(lambda_tensor);
            base = base.to(torch::kInt64);
            break;
        }
        case BaseDistType::LogNormal: {
            if (gen.defined())
                base = torch::randn(shape, gen, f64_opts);
            else
                base = torch::randn(shape, f64_opts);
            base = torch::exp(base * base_dist.param2 + base_dist.param1);
            break;
        }
        case BaseDistType::Cauchy: {
            if (gen.defined())
                base = torch::rand(shape, gen, f64_opts);
            else
                base = torch::rand(shape, f64_opts);
            base = base_dist.param1 +
                   base_dist.param2 * torch::tan(M_PI * (base - 0.5));
            break;
        }
        case BaseDistType::Bernoulli: {
            if (gen.defined())
                base = torch::rand(shape, gen, f64_opts);
            else
                base = torch::rand(shape, f64_opts);
            base = (base < base_dist.param1).to(torch::kFloat64);
            break;
        }
        }
        result = base.to(dtype);
    }

    if (discrete.empty() || numel == 0)
        return result;

    // --- Apply discrete values ---
    if (count_mode) {
        // Count mode: permutation-based exact placement
        torch::Tensor perm;
        if (gen.defined())
            perm = torch::randperm(numel, gen, i64_opts);
        else
            perm = torch::randperm(numel, i64_opts);

        int64_t offset = 0;
        for (auto& e : discrete) {
            int64_t count = (int64_t)e.probability;
            if (count <= 0) continue;

            auto indices = perm.slice(0, offset, offset + count);
            auto mask = torch::zeros({numel}, torch::kBool);
            mask.index_fill_(0, indices, true);
            mask = mask.view(shape);
            safe_masked_fill(result, mask, e.value, dtype);
            offset += count;
        }
    } else {
        // Probability mode: selector-based stochastic placement
        torch::Tensor selector;
        if (gen.defined())
            selector = torch::rand(shape, gen, f64_opts);
        else
            selector = torch::rand(shape, f64_opts);

        int last_active = -1;
        for (int i = (int)discrete.size() - 1; i >= 0; --i) {
            if (discrete[i].probability > 0) {
                last_active = i;
                break;
            }
        }

        double cum = 0.0;
        for (int i = 0; i < (int)discrete.size(); ++i) {
            if (discrete[i].probability <= 0) continue;

            torch::Tensor mask;
            if (i == last_active && all_discrete) {
                mask = (selector >= cum);
            } else {
                mask = (selector >= cum) & (selector < cum + discrete[i].probability);
            }

            safe_masked_fill(result, mask, discrete[i].value, dtype);
            cum += discrete[i].probability;
        }
    }

    return result;
}
