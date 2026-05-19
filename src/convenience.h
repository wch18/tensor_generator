#pragma once

#include "tensor_generator.h"
#include <c10/util/Optional.h>

// ============================================================================
// Convenience APIs for common tensor generation patterns.
//
// All functions use Uniform(0, 1) as default base distribution (except
// make_*_tensor which directly specify the distribution).
// ============================================================================

// --- 1a: Base distribution tensor generators --------------------------------

inline torch::Tensor make_uniform_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double lo = 0.0, double hi = 1.0,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::uniform(lo, hi);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_normal_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double mean = 0.0, double std = 1.0,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::normal(mean, std);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_randint_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kInt32,
    int64_t lo = 0, int64_t hi = 100,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::randint(lo, hi);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_exponential_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double lambda = 1.0,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::exponential(lambda);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_poisson_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kInt32,
    double rate = 1.0,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::poisson(rate);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_lognormal_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double mean = 0.0, double std = 1.0,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::lognormal(mean, std);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_cauchy_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double median = 0.0, double sigma = 1.0,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::cauchy(median, sigma);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

inline torch::Tensor make_bernoulli_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double p = 0.5,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::bernoulli(p);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{}, dist);
}

// --- 1b: Single special value -----------------------------------------------

inline torch::Tensor tensor_with_one_sv(
    SpecialValue sv,
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    double ratio = 0.01,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::uniform(0.0, 1.0);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype, TensorSpec{{sv, ratio}}, dist);
}

#define DEFINE_TENSOR_WITH_ONE_SV(EnumVal, short_name)                         \
inline torch::Tensor tensor_with_##short_name(                                 \
    torch::IntArrayRef shape,                                                  \
    torch::ScalarType dtype = torch::kFloat32,                                 \
    double short_name##_ratio = 0.01,                                          \
    c10::optional<torch::Generator> generator = c10::nullopt) {                \
    return tensor_with_one_sv(                                                 \
        SpecialValue::EnumVal, shape, dtype, short_name##_ratio, generator);   \
}

DEFINE_TENSOR_WITH_ONE_SV(Zero, zero)
DEFINE_TENSOR_WITH_ONE_SV(One, one)
DEFINE_TENSOR_WITH_ONE_SV(NOne, none)
DEFINE_TENSOR_WITH_ONE_SV(DTypeMin, min)
DEFINE_TENSOR_WITH_ONE_SV(DTypeMax, max)
DEFINE_TENSOR_WITH_ONE_SV(NaN, nan)
DEFINE_TENSOR_WITH_ONE_SV(Inf, inf)
DEFINE_TENSOR_WITH_ONE_SV(NInf, ninf)
DEFINE_TENSOR_WITH_ONE_SV(Epsilon, epsilon)
DEFINE_TENSOR_WITH_ONE_SV(Tiny, tiny)

#undef DEFINE_TENSOR_WITH_ONE_SV

// --- 1c: Two special values -------------------------------------------------

inline torch::Tensor tensor_with_two_sv(
    SpecialValue sv1, double ratio1,
    SpecialValue sv2, double ratio2,
    torch::IntArrayRef shape,
    torch::ScalarType dtype = torch::kFloat32,
    c10::optional<torch::Generator> generator = c10::nullopt) {
    BaseDist dist = BaseDist::uniform(0.0, 1.0);
    if (generator.has_value()) dist = dist.with_generator(*generator);
    return generate_tensor(shape, dtype,
        TensorSpec{{sv1, ratio1}, {sv2, ratio2}}, dist);
}

#define DEFINE_TENSOR_WITH_TWO_SV(Enum1, name1, Enum2, name2)                  \
inline torch::Tensor tensor_with_##name1##_##name2(                            \
    torch::IntArrayRef shape,                                                  \
    torch::ScalarType dtype = torch::kFloat32,                                 \
    double name1##_ratio = 0.01,                                               \
    double name2##_ratio = 0.01,                                               \
    c10::optional<torch::Generator> generator = c10::nullopt) {                \
    return tensor_with_two_sv(                                                 \
        SpecialValue::Enum1, name1##_ratio,                                    \
        SpecialValue::Enum2, name2##_ratio,                                    \
        shape, dtype, generator);                                              \
}

// All C(10,2) = 45 pairs in alphabetical order by short name
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, Inf, inf)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, DTypeMax, max)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, DTypeMin, min)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, NaN, nan)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, NInf, ninf)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, NOne, none)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, One, one)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(Epsilon, epsilon, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, DTypeMax, max)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, DTypeMin, min)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, NaN, nan)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, NInf, ninf)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, NOne, none)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, One, one)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(Inf, inf, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, DTypeMin, min)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, NaN, nan)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, NInf, ninf)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, NOne, none)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, One, one)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMax, max, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMin, min, NaN, nan)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMin, min, NInf, ninf)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMin, min, NOne, none)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMin, min, One, one)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMin, min, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(DTypeMin, min, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(NaN, nan, NInf, ninf)
DEFINE_TENSOR_WITH_TWO_SV(NaN, nan, NOne, none)
DEFINE_TENSOR_WITH_TWO_SV(NaN, nan, One, one)
DEFINE_TENSOR_WITH_TWO_SV(NaN, nan, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(NaN, nan, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(NInf, ninf, NOne, none)
DEFINE_TENSOR_WITH_TWO_SV(NInf, ninf, One, one)
DEFINE_TENSOR_WITH_TWO_SV(NInf, ninf, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(NInf, ninf, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(NOne, none, One, one)
DEFINE_TENSOR_WITH_TWO_SV(NOne, none, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(NOne, none, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(One, one, Tiny, tiny)
DEFINE_TENSOR_WITH_TWO_SV(One, one, Zero, zero)
DEFINE_TENSOR_WITH_TWO_SV(Tiny, tiny, Zero, zero)

#undef DEFINE_TENSOR_WITH_TWO_SV
