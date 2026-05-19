#include "tensor_generator.h"
#include "dtype_traits.h"
#include <cmath>
#include <stdexcept>

using namespace detail;

static void safe_masked_fill(torch::Tensor& result, const torch::Tensor& mask,
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

torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist,
    std::optional<int64_t> seed)
{
    if (seed.has_value()) {
        torch::manual_seed(seed.value());
    }

    // Resolve discrete entries with fallbacks for incompatible dtypes:
    //   Inf  on non-float / no-infinity float → DTypeMax
    //   NInf on non-float / no-infinity float → DTypeMin
    //   NaN  on non-float                     → silently skipped
    std::vector<DiscreteEntry> discrete;
    double p_sum = 0.0;
    for (auto& e : spec.discrete) {
        if (e.probability < 0)
            throw std::invalid_argument("probability must be non-negative");
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
    if (p_sum > 1.0 + 1e-9)
        throw std::invalid_argument("discrete probabilities exceed 1.0");

    bool all_discrete = (p_sum >= 1.0 - 1e-9);

    // Validate BaseDist / dtype compatibility
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
            // Inverse CDF: -ln(1 - U) / lambda
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
            // exp(mean + std * Z)
            base = torch::exp(base * base_dist.param2 + base_dist.param1);
            break;
        }
        case BaseDistType::Cauchy: {
            if (gen.defined())
                base = torch::rand(shape, gen, f64_opts);
            else
                base = torch::rand(shape, f64_opts);
            // Inverse CDF: median + sigma * tan(pi * (U - 0.5))
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

    if (discrete.empty() || result.numel() == 0)
        return result;

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

    return result;
}
