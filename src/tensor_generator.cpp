#include "tensor_generator.h"
#include "dtype_traits.h"
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

using namespace detail;

static int64_t get_special_value_as_int64(SpecialValue sv, torch::ScalarType dtype) {
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

static void safe_masked_fill(torch::Tensor& result, const torch::Tensor& mask,
                             SpecialValue sv, torch::ScalarType dtype) {
    if (dtype == torch::kBool) {
        result.masked_fill_(mask, get_special_value_as_double(sv, dtype) != 0.0);
    } else if (is_float_dtype(dtype) && !needs_cast_workaround(dtype)) {
        result.masked_fill_(mask, get_special_value_as_double(sv, dtype));
    } else if (is_float_dtype(dtype)) {
        auto work = result.to(torch::kFloat64);
        work.masked_fill_(mask, get_special_value_as_double(sv, dtype));
        result = work.to(dtype);
    } else if (dtype == c10::ScalarType::UInt64) {
        auto work = result.to(torch::kInt64);
        work.masked_fill_(mask, get_special_value_as_int64(sv, dtype));
        result = work.to(dtype);
    } else if (needs_cast_workaround(dtype)) {
        auto work = result.to(torch::kFloat64);
        work.masked_fill_(mask, (double)get_special_value_as_int64(sv, dtype));
        result = work.to(dtype);
    } else {
        result.masked_fill_(mask, get_special_value_as_int64(sv, dtype));
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

    auto& discrete = spec.discrete;

    double p_sum = 0.0;
    for (auto& e : discrete) {
        if (e.probability < 0)
            throw std::invalid_argument("probability must be non-negative");
        if (!is_valid_for_dtype(e.value, dtype))
            throw std::invalid_argument("SpecialValue not valid for given dtype");
        p_sum += e.probability;
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
