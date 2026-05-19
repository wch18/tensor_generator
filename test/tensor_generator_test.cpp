#include <gtest/gtest.h>
#include "../src/tensor_generator.h"
#include "test_configs.h"
#include <cmath>
#include <tuple>
#include <string>
#include <vector>
#include <sstream>

// ============================================================
// Helpers
// ============================================================

static void assert_probability(int64_t count, int64_t total, double expected_prob,
                               double sigma_tolerance = 5.0) {
    if (total == 0) return;
    double expected = total * expected_prob;
    double stddev = std::sqrt(total * expected_prob * (1.0 - expected_prob));
    double lo = expected - std::max(sigma_tolerance * stddev, 3.0);
    double hi = expected + std::max(sigma_tolerance * stddev, 3.0);
    EXPECT_GE(count, (int64_t)std::floor(lo))
        << "count=" << count << " expected~" << expected << " (p=" << expected_prob << ")";
    EXPECT_LE(count, (int64_t)std::ceil(hi))
        << "count=" << count << " expected~" << expected << " (p=" << expected_prob << ")";
}

static std::string dtype_name(torch::ScalarType dt) {
    if (dt == c10::ScalarType::UInt16) return "uint16";
    if (dt == c10::ScalarType::UInt32) return "uint32";
    if (dt == c10::ScalarType::UInt64) return "uint64";
    if (dt == c10::ScalarType::Float8_e8m0fnu) return "fp8_e8m0fnu";
    switch (dt) {
    case torch::kUInt8:          return "uint8";
    case torch::kInt8:           return "int8";
    case torch::kInt16:          return "int16";
    case torch::kInt32:          return "int32";
    case torch::kInt64:          return "int64";
    case torch::kFloat16:        return "float16";
    case torch::kBFloat16:       return "bfloat16";
    case torch::kFloat32:        return "float32";
    case torch::kFloat64:        return "float64";
    case torch::kBool:           return "bool";
    case torch::kFloat8_e4m3fn:  return "fp8_e4m3fn";
    case torch::kFloat8_e5m2:    return "fp8_e5m2";
    default:                     return "unknown";
    }
}

static std::string shape_str(torch::IntArrayRef shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ",";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

// ============================================================
// Parameterized test: Shape x DType x Config cartesian product
// ============================================================

struct CartesianTestParam {
    std::vector<int64_t> shape;
    torch::ScalarType dtype;
    int config_idx;
};

class CartesianTest : public ::testing::TestWithParam<CartesianTestParam> {};

TEST_P(CartesianTest, GenerateAndVerify) {
    auto param = GetParam();
    auto configs = get_all_configs();
    auto& cfg = configs[param.config_idx];

    if (!cfg.is_compatible(param.dtype)) {
        GTEST_SKIP() << "Config '" << cfg.name << "' not compatible with "
                     << dtype_name(param.dtype);
    }

    int64_t seed = 42 + param.config_idx * 1000 +
                   (int)param.dtype * 100 + param.shape.back();

    torch::IntArrayRef shape(param.shape);
    torch::Tensor t;

    ASSERT_NO_THROW(t = generate_tensor(shape, param.dtype, cfg.spec, cfg.base_dist, seed))
        << "Failed for shape=" << shape_str(shape)
        << " dtype=" << dtype_name(param.dtype)
        << " config=" << cfg.name;

    ASSERT_EQ(t.dim(), (int64_t)param.shape.size());
    for (size_t i = 0; i < param.shape.size(); ++i) {
        ASSERT_EQ(t.size(i), param.shape[i]);
    }
    ASSERT_EQ(t.scalar_type(), param.dtype);

    int64_t N = t.numel();
    if (N == 0) return;

    auto tf = t.to(torch::kFloat64);

    double p_sum = 0.0;
    for (auto& entry : cfg.spec.discrete) {
        p_sum += entry.probability;
    }
    bool has_continuous = (p_sum < 1.0 - 1e-9);

    bool low_resolution_dtype = is_int_type(param.dtype) || param.dtype == torch::kBool ||
        param.dtype == torch::kFloat8_e4m3fn || param.dtype == torch::kFloat8_e5m2 ||
        param.dtype == c10::ScalarType::Float8_e8m0fnu;
    bool relaxed_verification = low_resolution_dtype && has_continuous;

    struct ValueGroup {
        double value;
        double total_prob;
        bool is_nan;
        bool is_pos_inf;
        bool is_neg_inf;
    };
    std::vector<ValueGroup> groups;
    for (auto& entry : cfg.spec.discrete) {
        double v = entry.value.as_double(param.dtype);
        bool found = false;
        for (auto& g : groups) {
            bool same = false;
            if (std::isnan(v) && g.is_nan) same = true;
            else if (std::isinf(v) && v > 0 && g.is_pos_inf) same = true;
            else if (std::isinf(v) && v < 0 && g.is_neg_inf) same = true;
            else if (!std::isnan(v) && !std::isinf(v) && !g.is_nan && !g.is_pos_inf
                     && !g.is_neg_inf && g.value == v) same = true;
            if (same) {
                g.total_prob += entry.probability;
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({v, entry.probability,
                              std::isnan(v),
                              std::isinf(v) && v > 0,
                              std::isinf(v) && v < 0});
        }
    }

    for (auto& g : groups) {
        int64_t count;
        if (g.is_nan) {
            count = torch::isnan(tf).sum().item<int64_t>();
        } else if (g.is_pos_inf) {
            count = (tf == std::numeric_limits<double>::infinity()).sum().item<int64_t>();
        } else if (g.is_neg_inf) {
            count = (tf == -std::numeric_limits<double>::infinity()).sum().item<int64_t>();
        } else {
            count = (tf == g.value).sum().item<int64_t>();
        }

        if (N >= 10000) {
            if (relaxed_verification) {
                double expected = N * g.total_prob;
                double stddev = std::sqrt(N * g.total_prob * (1.0 - g.total_prob));
                EXPECT_GE(count, (int64_t)(expected - 5.0 * stddev))
                    << "count=" << count << " too low for p=" << g.total_prob
                    << " dtype=" << dtype_name(param.dtype) << " config=" << cfg.name;
            } else {
                assert_probability(count, N, g.total_prob);
            }
        } else if (g.total_prob >= 1.0 - 1e-9) {
            EXPECT_EQ(count, N)
                << "With p=1.0, all elements should match. dtype="
                << dtype_name(param.dtype) << " config=" << cfg.name;
        }
    }

    if (p_sum >= 1.0 - 1e-9 && N >= 1000) {
        int64_t total_discrete = 0;
        for (auto& g : groups) {
            if (g.is_nan) {
                total_discrete += torch::isnan(tf).sum().item<int64_t>();
            } else if (g.is_pos_inf) {
                total_discrete += (tf == std::numeric_limits<double>::infinity()).sum().item<int64_t>();
            } else if (g.is_neg_inf) {
                total_discrete += (tf == -std::numeric_limits<double>::infinity()).sum().item<int64_t>();
            } else {
                total_discrete += (tf == g.value).sum().item<int64_t>();
            }
        }
        EXPECT_EQ(total_discrete, N)
            << "All-discrete config should cover all elements. dtype="
            << dtype_name(param.dtype) << " config=" << cfg.name;
    }
}

static std::vector<CartesianTestParam> generate_cartesian_params() {
    auto shapes = get_all_shapes();
    auto dtypes = get_all_dtypes();
    auto configs = get_all_configs();

    std::vector<CartesianTestParam> params;
    for (auto& shape : shapes) {
        for (auto dtype : dtypes) {
            for (int ci = 0; ci < (int)configs.size(); ++ci) {
                if (configs[ci].is_compatible(dtype)) {
                    params.push_back(CartesianTestParam{shape, dtype, ci});
                }
            }
        }
    }
    return params;
}

struct CartesianTestNameGenerator {
    std::string operator()(const ::testing::TestParamInfo<CartesianTestParam>& info) const {
        auto& p = info.param;
        auto configs = get_all_configs();
        std::ostringstream name;
        name << "shape";
        for (auto d : p.shape) name << "_" << d;
        name << "__" << dtype_name(p.dtype);
        name << "__" << configs[p.config_idx].name;
        return name.str();
    }
};

INSTANTIATE_TEST_SUITE_P(
    ShapeXDTypeXConfig,
    CartesianTest,
    ::testing::ValuesIn(generate_cartesian_params()),
    CartesianTestNameGenerator());

// ============================================================
// Focused unit tests
// ============================================================

// --- Validation tests ---
TEST(Validation, FloatOnlyValues) {
    std::vector<torch::ScalarType> float_types = {
        torch::kFloat16, torch::kBFloat16, torch::kFloat32, torch::kFloat64,
        torch::kFloat8_e4m3fn, torch::kFloat8_e5m2, c10::ScalarType::Float8_e8m0fnu};
    std::vector<torch::ScalarType> int_types = {
        torch::kUInt8, c10::ScalarType::UInt16, c10::ScalarType::UInt32,
        c10::ScalarType::UInt64, torch::kInt8, torch::kInt16, torch::kInt32,
        torch::kInt64, torch::kBool};

    for (auto dt : float_types) {
        EXPECT_TRUE(is_valid_for_dtype(SpecialValue::NaN, dt));
        EXPECT_TRUE(is_valid_for_dtype(SpecialValue::Epsilon, dt));
        EXPECT_TRUE(is_valid_for_dtype(SpecialValue::Tiny, dt));
    }
    for (auto dt : int_types) {
        EXPECT_FALSE(is_valid_for_dtype(SpecialValue::NaN, dt));
        EXPECT_FALSE(is_valid_for_dtype(SpecialValue::Epsilon, dt));
        EXPECT_FALSE(is_valid_for_dtype(SpecialValue::Tiny, dt));
    }
}

TEST(Validation, InfinityE4M3) {
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::Inf, torch::kFloat8_e4m3fn));
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::NInf, torch::kFloat8_e4m3fn));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::Inf, torch::kFloat8_e5m2));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::NaN, torch::kFloat8_e4m3fn));
}

TEST(Validation, E8M0Constraints) {
    auto dt = c10::ScalarType::Float8_e8m0fnu;
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::Zero, dt));
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::NOne, dt));
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::Inf, dt));
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::NInf, dt));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::One, dt));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::DTypeMin, dt));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::DTypeMax, dt));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::NaN, dt));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::Epsilon, dt));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::Tiny, dt));
}

// --- Error handling tests ---
TEST(ErrorHandling, NegativeProbability) {
    EXPECT_THROW(generate_tensor({10}, torch::kFloat32, {{SpecialValue::Zero, -0.1}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, ProbExceedsOne) {
    EXPECT_THROW(generate_tensor({10}, torch::kFloat32,
                     {{SpecialValue::Zero, 0.6}, {SpecialValue::One, 0.6}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, EpsilonOnInt) {
    EXPECT_THROW(generate_tensor({10}, torch::kInt64, {{SpecialValue::Epsilon, 0.1}}),
                 std::invalid_argument);
}

// --- Fallback tests: Inf/NInf → DTypeMax/DTypeMin, NaN → skip ---
TEST(Fallback, InfOnIntegerFallsToDTypeMax) {
    auto t = generate_tensor({1000}, torch::kInt32,
        {{SpecialValue::Inf, 1.0}});
    EXPECT_TRUE((t == std::numeric_limits<int32_t>::max()).all().item<bool>());
}

TEST(Fallback, NInfOnIntegerFallsToDTypeMin) {
    auto t = generate_tensor({1000}, torch::kInt64,
        {{SpecialValue::NInf, 1.0}});
    EXPECT_TRUE((t == std::numeric_limits<int64_t>::lowest()).all().item<bool>());
}

TEST(Fallback, InfOnUInt8FallsTo255) {
    auto t = generate_tensor({100}, torch::kUInt8,
        {{SpecialValue::Inf, 1.0}});
    EXPECT_TRUE((t.to(torch::kInt64) == 255).all().item<bool>());
}

TEST(Fallback, NInfOnUInt8FallsTo0) {
    auto t = generate_tensor({100}, torch::kUInt8,
        {{SpecialValue::NInf, 1.0}});
    EXPECT_TRUE((t.to(torch::kInt64) == 0).all().item<bool>());
}

TEST(Fallback, InfOnBoolFallsToTrue) {
    auto t = generate_tensor({100}, torch::kBool,
        {{SpecialValue::Inf, 1.0}});
    EXPECT_TRUE(t.all().item<bool>());
}

TEST(Fallback, NInfOnBoolFallsToFalse) {
    auto t = generate_tensor({100}, torch::kBool,
        {{SpecialValue::NInf, 1.0}});
    EXPECT_FALSE(t.any().item<bool>());
}

TEST(Fallback, InfOnE4M3FallsToDTypeMax) {
    auto t = generate_tensor({100}, torch::kFloat8_e4m3fn,
        {{SpecialValue::Inf, 1.0}});
    float val = t.to(torch::kFloat32).max().item<float>();
    EXPECT_FLOAT_EQ(val, 448.0f);
}

TEST(Fallback, NInfOnE4M3FallsToDTypeMin) {
    auto t = generate_tensor({100}, torch::kFloat8_e4m3fn,
        {{SpecialValue::NInf, 1.0}});
    float val = t.to(torch::kFloat32).min().item<float>();
    EXPECT_FLOAT_EQ(val, -448.0f);
}

TEST(Fallback, InfOnE8M0FallsToDTypeMax) {
    auto t = generate_tensor({100}, c10::ScalarType::Float8_e8m0fnu,
        {{SpecialValue::Inf, 1.0}});
    double val = t.to(torch::kFloat64).max().item<double>();
    EXPECT_DOUBLE_EQ(val, 1.7014118346046923e+38);  // 2^127
}

TEST(Fallback, NInfOnE8M0FallsToDTypeMin) {
    auto t = generate_tensor({100}, c10::ScalarType::Float8_e8m0fnu,
        {{SpecialValue::NInf, 1.0}});
    double val = t.to(torch::kFloat64).min().item<double>();
    EXPECT_DOUBLE_EQ(val, 5.877471754111438e-39);   // 2^-127
}

TEST(Fallback, NaNOnIntegerSilentlySkipped) {
    auto t = generate_tensor({1000}, torch::kInt32,
        {{SpecialValue::NaN, 0.5}},
        BaseDist::randint(10, 20), 42);
    auto t64 = t.to(torch::kFloat64);
    EXPECT_GE(t64.min().item<double>(), 10.0);
    EXPECT_LE(t64.max().item<double>(), 19.0);
}

TEST(Fallback, NaNOnBoolSilentlySkipped) {
    auto t = generate_tensor({1000}, torch::kBool,
        {{SpecialValue::NaN, 0.3}, {SpecialValue::One, 0.7}});
    EXPECT_TRUE(t.all().item<bool>());
}

TEST(Fallback, NaNSkippedLeavesBaseDist) {
    auto t = generate_tensor({10000}, torch::kInt64,
        {{SpecialValue::NaN, 1.0}},
        BaseDist::randint(0, 100), 55);
    auto t64 = t.to(torch::kFloat64);
    EXPECT_GE(t64.min().item<double>(), 0.0);
    EXPECT_LE(t64.max().item<double>(), 99.0);
}

TEST(Fallback, MixedFallbackAndNative) {
    auto t = generate_tensor({100000}, torch::kInt32,
        {{SpecialValue::Inf, 0.2},
         {SpecialValue::NInf, 0.2},
         {SpecialValue::NaN, 0.1},
         {SpecialValue::Zero, 0.5}},
        BaseDist::randint(50, 100), 42);
    auto max_count = (t == std::numeric_limits<int32_t>::max()).sum().item<int64_t>();
    auto min_count = (t == std::numeric_limits<int32_t>::lowest()).sum().item<int64_t>();
    auto zero_count = (t == 0).sum().item<int64_t>();
    assert_probability(max_count, 100000, 0.2);
    assert_probability(min_count, 100000, 0.2);
    assert_probability(zero_count, 100000, 0.5);
    // NaN was skipped, remaining 10% filled by base dist (randint 50..99)
    auto base_count = 100000 - max_count - min_count - zero_count;
    assert_probability(base_count, 100000, 0.1);
}

TEST(Fallback, InfNInfOnFloatWithInfinityUnchanged) {
    auto t = generate_tensor({10000}, torch::kFloat32,
        {{SpecialValue::Inf, 0.5}, {SpecialValue::NInf, 0.5}});
    auto pinf_count = (t == std::numeric_limits<float>::infinity()).sum().item<int64_t>();
    auto ninf_count = (t == -std::numeric_limits<float>::infinity()).sum().item<int64_t>();
    EXPECT_EQ(pinf_count + ninf_count, 10000);
    assert_probability(pinf_count, 10000, 0.5);
}

TEST(ErrorHandling, RandIntOnFloat) {
    EXPECT_THROW(generate_tensor({10}, torch::kFloat32, TensorSpec{}, BaseDist::randint(0, 10)),
                 std::invalid_argument);
}

TEST(ErrorHandling, RandIntOnBool) {
    EXPECT_THROW(generate_tensor({10}, torch::kBool, TensorSpec{}, BaseDist::randint(0, 2)),
                 std::invalid_argument);
}

// --- Edge case tests ---
TEST(EdgeCase, EmptyTensor) {
    auto t = generate_tensor({2, 0, 3}, torch::kFloat32, {{SpecialValue::NaN, 0.5}});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.scalar_type(), torch::kFloat32);
}

TEST(EdgeCase, SingleElementAllDiscrete) {
    for (int trial = 0; trial < 100; ++trial) {
        auto t = generate_tensor({1}, torch::kFloat32, {{SpecialValue::NaN, 1.0}});
        EXPECT_TRUE(std::isnan(t.item<float>()));
    }
}

TEST(EdgeCase, SingleElementMixed) {
    int zero_count = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        auto t = generate_tensor({1}, torch::kFloat32,
            {{SpecialValue::Zero, 0.5}},
            BaseDist::uniform(1.0, 2.0),
            42 + trial);
        float val = t.item<float>();
        if (val == 0.0f) {
            zero_count++;
        } else {
            EXPECT_GE(val, 1.0f);
            EXPECT_LE(val, 2.0f);
        }
    }
    EXPECT_GT(zero_count, 400);
    EXPECT_LT(zero_count, 600);
}

TEST(EdgeCase, ZeroProbabilityEntries) {
    auto t = generate_tensor({1000}, torch::kFloat32,
        {{SpecialValue::NaN, 0.0}, {SpecialValue::One, 1.0}});
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 0);
    EXPECT_EQ((t == 1.0f).sum().item<int64_t>(), 1000);
}

TEST(EdgeCase, EmptyDiscreteList) {
    auto t = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{}, BaseDist::uniform(0.0, 1.0), 123);
    EXPECT_GE(t.min().item<float>(), 0.0f);
    EXPECT_LE(t.max().item<float>(), 1.0f);
}

TEST(EdgeCase, VerySmallProbability) {
    const int64_t N = 1000000;
    auto t = generate_tensor({N}, torch::kFloat32,
        {{SpecialValue::Inf, 0.001}},
        BaseDist::uniform(-1.0, 1.0), 77);
    auto inf_count = (t == std::numeric_limits<float>::infinity()).sum().item<int64_t>();
    assert_probability(inf_count, N, 0.001);
}

// --- RandInt tests ---
TEST(RandInt, Int32Range) {
    auto t = generate_tensor({100000}, torch::kInt32,
        TensorSpec{}, BaseDist::randint(-50, 50), 42);
    auto t64 = t.to(torch::kFloat64);
    EXPECT_GE(t64.min().item<double>(), -50.0);
    EXPECT_LE(t64.max().item<double>(), 49.0);
}

TEST(RandInt, Int64Range) {
    auto t = generate_tensor({100000}, torch::kInt64,
        TensorSpec{}, BaseDist::randint(0, 1000), 43);
    EXPECT_GE(t.min().item<int64_t>(), 0);
    EXPECT_LE(t.max().item<int64_t>(), 999);
}

TEST(RandInt, UInt8Range) {
    auto t = generate_tensor({100000}, torch::kUInt8,
        TensorSpec{}, BaseDist::randint(0, 256), 44);
    auto t64 = t.to(torch::kFloat64);
    EXPECT_GE(t64.min().item<double>(), 0.0);
    EXPECT_LE(t64.max().item<double>(), 255.0);
}

TEST(RandInt, WithDiscreteOverlay) {
    auto t = generate_tensor({100000}, torch::kInt32,
        {{SpecialValue::Zero, 0.3}},
        BaseDist::randint(100, 200), 45);
    auto zero_count = (t == 0).sum().item<int64_t>();
    assert_probability(zero_count, 100000, 0.3);
    auto non_zero = t.masked_select(t != 0);
    auto nz64 = non_zero.to(torch::kFloat64);
    EXPECT_GE(nz64.min().item<double>(), 100.0);
    EXPECT_LE(nz64.max().item<double>(), 199.0);
}

// --- Generator tests ---
static torch::Generator make_cpu_gen(uint64_t seed) {
    auto gen = at::detail::createCPUGenerator();
    gen.set_current_seed(seed);
    return gen;
}

TEST(Generator, ImplicitConversion) {
    auto gen = make_cpu_gen(99);
    // Implicit conversion: Generator -> BaseDist (default Uniform(-1, 1))
    auto t = generate_tensor({1000}, torch::kFloat32, TensorSpec{}, gen);
    EXPECT_GE(t.min().item<float>(), -1.0f);
    EXPECT_LE(t.max().item<float>(), 1.0f);
}

TEST(Generator, Reproducibility) {
    auto t1 = generate_tensor({1000}, torch::kFloat32, TensorSpec{},
        BaseDist::uniform(0.0, 1.0).with_generator(make_cpu_gen(123)));

    auto t2 = generate_tensor({1000}, torch::kFloat32, TensorSpec{},
        BaseDist::uniform(0.0, 1.0).with_generator(make_cpu_gen(123)));

    EXPECT_TRUE(torch::equal(t1, t2));
}

TEST(Generator, Isolation) {
    torch::manual_seed(42);
    auto ref1 = torch::rand({10});

    torch::manual_seed(42);
    generate_tensor({1000}, torch::kFloat32, TensorSpec{}, make_cpu_gen(999));
    auto ref2 = torch::rand({10});

    EXPECT_TRUE(torch::equal(ref1, ref2));
}

TEST(Generator, WithRandInt) {
    auto t1 = generate_tensor({1000}, torch::kInt32, TensorSpec{},
        BaseDist::randint(0, 100).with_generator(make_cpu_gen(77)));

    auto t2 = generate_tensor({1000}, torch::kInt32, TensorSpec{},
        BaseDist::randint(0, 100).with_generator(make_cpu_gen(77)));

    EXPECT_TRUE(torch::equal(t1, t2));
}

// --- UInt8 specific tests ---
TEST(UInt8Specific, NOneIs255) {
    auto t = generate_tensor({10000}, torch::kUInt8,
        {{SpecialValue::NOne, 1.0}});
    auto t64 = t.to(torch::kFloat64);
    EXPECT_EQ((t64 == 255.0).sum().item<int64_t>(), 10000);
}

TEST(UInt8Specific, DTypeMinIs0_DTypeMaxIs255) {
    auto t = generate_tensor({10000}, torch::kUInt8,
        {{SpecialValue::DTypeMin, 0.5}, {SpecialValue::DTypeMax, 0.5}});
    auto t64 = t.to(torch::kFloat64);
    auto zero_count = (t64 == 0).sum().item<int64_t>();
    auto ff_count = (t64 == 255).sum().item<int64_t>();
    EXPECT_EQ(zero_count + ff_count, 10000);
    assert_probability(zero_count, 10000, 0.5);
}

TEST(UInt8Specific, ContinuousClampedTo0_255) {
    auto t = generate_tensor({10000}, torch::kUInt8,
        TensorSpec{}, BaseDist::uniform(0.0, 256.0), 333);
    auto t64 = t.to(torch::kFloat64);
    EXPECT_GE(t64.min().item<double>(), 0.0);
    EXPECT_LE(t64.max().item<double>(), 255.0);
}

// --- UInt16 specific tests ---
TEST(UInt16Specific, NOneIs65535) {
    auto t = generate_tensor({10000}, c10::ScalarType::UInt16,
        {{SpecialValue::NOne, 1.0}});
    auto t64 = t.to(torch::kFloat64);
    EXPECT_EQ((t64 == 65535.0).sum().item<int64_t>(), 10000);
}

TEST(UInt16Specific, DTypeMinIs0_DTypeMaxIs65535) {
    auto t = generate_tensor({10000}, c10::ScalarType::UInt16,
        {{SpecialValue::DTypeMin, 0.5}, {SpecialValue::DTypeMax, 0.5}});
    auto t64 = t.to(torch::kFloat64);
    auto zero_count = (t64 == 0).sum().item<int64_t>();
    auto max_count = (t64 == 65535.0).sum().item<int64_t>();
    EXPECT_EQ(zero_count + max_count, 10000);
    assert_probability(zero_count, 10000, 0.5);
}

TEST(UInt16Specific, AllDiscreteNoleak) {
    auto t = generate_tensor({50000}, c10::ScalarType::UInt16,
        {{SpecialValue::Zero, 0.3},
         {SpecialValue::One, 0.3},
         {SpecialValue::DTypeMax, 0.4}},
        {}, 555);
    auto t64 = t.to(torch::kFloat64);
    auto total = (t64 == 0).sum().item<int64_t>() +
                 (t64 == 1).sum().item<int64_t>() +
                 (t64 == 65535.0).sum().item<int64_t>();
    EXPECT_EQ(total, 50000);
}

// --- UInt32 specific tests ---
TEST(UInt32Specific, NOneIs4294967295) {
    auto t = generate_tensor({10000}, c10::ScalarType::UInt32,
        {{SpecialValue::NOne, 1.0}});
    auto t64 = t.to(torch::kFloat64);
    EXPECT_EQ((t64 == 4294967295.0).sum().item<int64_t>(), 10000);
}

TEST(UInt32Specific, DTypeMinIs0_DTypeMaxIs4294967295) {
    auto t = generate_tensor({10000}, c10::ScalarType::UInt32,
        {{SpecialValue::DTypeMin, 0.5}, {SpecialValue::DTypeMax, 0.5}});
    auto t64 = t.to(torch::kFloat64);
    auto zero_count = (t64 == 0).sum().item<int64_t>();
    auto max_count = (t64 == 4294967295.0).sum().item<int64_t>();
    EXPECT_EQ(zero_count + max_count, 10000);
    assert_probability(zero_count, 10000, 0.5);
}

TEST(UInt32Specific, AllDiscreteNoleak) {
    auto t = generate_tensor({50000}, c10::ScalarType::UInt32,
        {{SpecialValue::Zero, 0.25},
         {SpecialValue::One, 0.25},
         {SpecialValue::NOne, 0.25},
         {SpecialValue::DTypeMax, 0.25}},
        {}, 666);
    auto t64 = t.to(torch::kFloat64);
    auto total = (t64 == 0).sum().item<int64_t>() +
                 (t64 == 1).sum().item<int64_t>() +
                 (t64 == 4294967295.0).sum().item<int64_t>();
    EXPECT_EQ(total, 50000);
}

// --- UInt64 specific tests ---
TEST(UInt64Specific, DTypeMinIs0) {
    auto t = generate_tensor({1000}, c10::ScalarType::UInt64,
        {{SpecialValue::DTypeMin, 1.0}});
    auto t64 = t.to(torch::kFloat64);
    EXPECT_EQ((t64 == 0).sum().item<int64_t>(), 1000);
}

TEST(UInt64Specific, ZeroAndOne) {
    auto t = generate_tensor({50000}, c10::ScalarType::UInt64,
        {{SpecialValue::Zero, 0.5}, {SpecialValue::One, 0.5}},
        {}, 777);
    auto t64 = t.to(torch::kFloat64);
    auto zero_count = (t64 == 0).sum().item<int64_t>();
    auto one_count = (t64 == 1).sum().item<int64_t>();
    EXPECT_EQ(zero_count + one_count, 50000);
    assert_probability(zero_count, 50000, 0.5);
}

// --- Float special value correctness ---
TEST(FloatSpecials, Float16ValueCorrectness) {
    auto t = generate_tensor({1}, torch::kFloat16,
        {{SpecialValue::Epsilon, 1.0}});
    float val = t.to(torch::kFloat32).item<float>();
    EXPECT_FLOAT_EQ(val, 9.765625e-4f);
}

TEST(FloatSpecials, BFloat16ValueCorrectness) {
    auto t = generate_tensor({1}, torch::kBFloat16,
        {{SpecialValue::Epsilon, 1.0}});
    float val = t.to(torch::kFloat32).item<float>();
    EXPECT_FLOAT_EQ(val, 7.8125e-3f);
}

TEST(FloatSpecials, Float8E4M3FN_MaxValue) {
    auto t = generate_tensor({1}, torch::kFloat8_e4m3fn,
        {{SpecialValue::DTypeMax, 1.0}});
    float val = t.to(torch::kFloat32).item<float>();
    EXPECT_FLOAT_EQ(val, 448.0f);
}

TEST(FloatSpecials, Float8E5M2_MaxValue) {
    auto t = generate_tensor({1}, torch::kFloat8_e5m2,
        {{SpecialValue::DTypeMax, 1.0}});
    float val = t.to(torch::kFloat32).item<float>();
    EXPECT_FLOAT_EQ(val, 57344.0f);
}

TEST(FloatSpecials, Float64Epsilon) {
    auto t = generate_tensor({1}, torch::kFloat64,
        {{SpecialValue::Epsilon, 1.0}});
    double val = t.item<double>();
    EXPECT_DOUBLE_EQ(val, DBL_EPSILON);
}

TEST(FloatSpecials, Float64Tiny) {
    auto t = generate_tensor({1}, torch::kFloat64,
        {{SpecialValue::Tiny, 1.0}});
    double val = t.item<double>();
    EXPECT_DOUBLE_EQ(val, DBL_MIN);
}

// --- Integer boundary tests ---
TEST(IntBoundary, Int8MinMax) {
    auto t = generate_tensor({1}, torch::kInt8, {{SpecialValue::DTypeMin, 1.0}});
    EXPECT_EQ(t.item<int8_t>(), -128);
    t = generate_tensor({1}, torch::kInt8, {{SpecialValue::DTypeMax, 1.0}});
    EXPECT_EQ(t.item<int8_t>(), 127);
}

TEST(IntBoundary, Int16MinMax) {
    auto t = generate_tensor({1}, torch::kInt16, {{SpecialValue::DTypeMin, 1.0}});
    EXPECT_EQ(t.item<int16_t>(), -32768);
    t = generate_tensor({1}, torch::kInt16, {{SpecialValue::DTypeMax, 1.0}});
    EXPECT_EQ(t.item<int16_t>(), 32767);
}

TEST(IntBoundary, Int32MinMax) {
    auto t = generate_tensor({1}, torch::kInt32, {{SpecialValue::DTypeMin, 1.0}});
    EXPECT_EQ(t.item<int32_t>(), std::numeric_limits<int32_t>::lowest());
    t = generate_tensor({1}, torch::kInt32, {{SpecialValue::DTypeMax, 1.0}});
    EXPECT_EQ(t.item<int32_t>(), std::numeric_limits<int32_t>::max());
}

TEST(IntBoundary, Int64MinMax) {
    auto t = generate_tensor({1}, torch::kInt64, {{SpecialValue::DTypeMin, 1.0}});
    EXPECT_EQ(t.item<int64_t>(), std::numeric_limits<int64_t>::lowest());
    t = generate_tensor({1}, torch::kInt64, {{SpecialValue::DTypeMax, 1.0}});
    EXPECT_EQ(t.item<int64_t>(), std::numeric_limits<int64_t>::max());
}

// --- Bool dtype tests ---
TEST(BoolDType, ZeroIsFalseOneIsTrue) {
    auto t = generate_tensor({1000}, torch::kBool,
        {{SpecialValue::Zero, 0.5}, {SpecialValue::One, 0.5}},
        {}, 42);
    auto true_count = t.sum().item<int64_t>();
    assert_probability(true_count, 1000, 0.5);
}

TEST(BoolDType, DTypeMinFalse_DTypeMaxTrue) {
    auto t = generate_tensor({1}, torch::kBool, {{SpecialValue::DTypeMin, 1.0}});
    EXPECT_FALSE(t.item<bool>());
    t = generate_tensor({1}, torch::kBool, {{SpecialValue::DTypeMax, 1.0}});
    EXPECT_TRUE(t.item<bool>());
}

// --- Performance tests ---
TEST(Performance, FourMillionFloat32) {
    auto t = generate_tensor({4000000}, torch::kFloat32,
        TensorSpec{{SpecialValue::Zero, 0.1},
                   {SpecialValue::NaN, 0.05},
                   {SpecialValue::Inf, 0.02},
                   {SpecialValue::DTypeMin, 0.03}},
        BaseDist::uniform(-100.0, 100.0), 999);

    EXPECT_EQ(t.numel(), 4000000);
    auto nan_count = torch::isnan(t).sum().item<int64_t>();
    auto zero_count = (t == 0.0f).sum().item<int64_t>();
    assert_probability(nan_count, 4000000, 0.05);
    assert_probability(zero_count, 4000000, 0.1);
}

TEST(Performance, FourMillionInt64) {
    auto t = generate_tensor({4000000}, torch::kInt64,
        TensorSpec{{SpecialValue::Zero, 0.2},
                   {SpecialValue::DTypeMax, 0.1},
                   {SpecialValue::NOne, 0.1}},
        BaseDist::uniform(-1000.0, 1000.0), 1000);

    EXPECT_EQ(t.numel(), 4000000);
    auto zero_count = (t == 0).sum().item<int64_t>();
    assert_probability(zero_count, 4000000, 0.2);
}

// --- Continuous distribution tests ---
TEST(ContinuousDist, UniformRange) {
    auto t = generate_tensor({500000}, torch::kFloat32,
        TensorSpec{}, BaseDist::uniform(-7.5, 3.5), 1111);
    EXPECT_GE(t.min().item<float>(), -7.5f);
    EXPECT_LE(t.max().item<float>(), 3.5f);
    float mean = t.mean().item<float>();
    EXPECT_NEAR(mean, -2.0f, 0.05f);
}

TEST(ContinuousDist, NormalMeanStd) {
    auto t = generate_tensor({500000}, torch::kFloat64,
        TensorSpec{}, BaseDist::normal(5.0, 3.0), 2222);
    double mean = t.mean().item<double>();
    double std_val = t.std().item<double>();
    EXPECT_NEAR(mean, 5.0, 0.03);
    EXPECT_NEAR(std_val, 3.0, 0.03);
}

// --- All-discrete leakage tests ---
TEST(AllDiscrete, NoLeakageMultipleTrials) {
    for (int trial = 0; trial < 50; ++trial) {
        auto t = generate_tensor({100000}, torch::kFloat32,
            {{SpecialValue::Zero, 0.1},
             {SpecialValue::One, 0.2},
             {SpecialValue::NOne, 0.3},
             {SpecialValue::DTypeMax, 0.4}},
            {}, trial * 7);

        auto total = (t == 0).sum().item<int64_t>() +
                     (t == 1).sum().item<int64_t>() +
                     (t == -1).sum().item<int64_t>() +
                     (t == FLT_MAX).sum().item<int64_t>();
        EXPECT_EQ(total, 100000)
            << "Leakage detected at trial " << trial;
    }
}

TEST(AllDiscrete, NoLeakageWithFloatingProbs) {
    for (int trial = 0; trial < 50; ++trial) {
        auto t = generate_tensor({50000}, torch::kInt32,
            {{SpecialValue::Zero, 0.1},
             {SpecialValue::One, 0.2},
             {SpecialValue::NOne, 0.3},
             {SpecialValue::DTypeMin, 0.4}},
            {}, 1000 + trial);

        auto total = (t == 0).sum().item<int64_t>() +
                     (t == 1).sum().item<int64_t>() +
                     (t == -1).sum().item<int64_t>() +
                     (t == (int32_t)std::numeric_limits<int32_t>::lowest()).sum().item<int64_t>();
        EXPECT_EQ(total, 50000)
            << "Leakage detected at trial " << trial;
    }
}

// --- Seed tests ---
TEST(Seed, SameSeedProducesSameResult) {
    auto t1 = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{{SpecialValue::Zero, 0.3}},
        BaseDist::uniform(-5.0, 5.0), 12345);
    auto t2 = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{{SpecialValue::Zero, 0.3}},
        BaseDist::uniform(-5.0, 5.0), 12345);
    EXPECT_TRUE(torch::equal(t1, t2));
}

TEST(Seed, DifferentSeedProducesDifferentResult) {
    auto t1 = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{}, BaseDist::uniform(0.0, 1.0), 111);
    auto t2 = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{}, BaseDist::uniform(0.0, 1.0), 222);
    EXPECT_FALSE(torch::equal(t1, t2));
}

// --- get_scalar tests ---
TEST(GetScalar, FloatValues) {
    auto s = get_scalar(SpecialValue::NaN, torch::kFloat32);
    EXPECT_TRUE(std::isnan(s.toDouble()));

    s = get_scalar(SpecialValue::Inf, torch::kFloat64);
    EXPECT_EQ(s.toDouble(), std::numeric_limits<double>::infinity());

    s = get_scalar(SpecialValue::Epsilon, torch::kFloat32);
    EXPECT_FLOAT_EQ((float)s.toDouble(), FLT_EPSILON);
}

TEST(GetScalar, IntValues) {
    auto s = get_scalar(SpecialValue::DTypeMax, torch::kInt64);
    EXPECT_EQ(s.toLong(), std::numeric_limits<int64_t>::max());

    s = get_scalar(SpecialValue::DTypeMin, torch::kInt32);
    EXPECT_EQ(s.toLong(), std::numeric_limits<int32_t>::lowest());

    s = get_scalar(SpecialValue::NOne, torch::kUInt8);
    EXPECT_EQ(s.toLong(), 255);
}

TEST(GetScalar, BoolValues) {
    auto s = get_scalar(SpecialValue::Zero, torch::kBool);
    EXPECT_FALSE(s.toBool());

    s = get_scalar(SpecialValue::One, torch::kBool);
    EXPECT_TRUE(s.toBool());
}

TEST(GetScalar, TemplateVersion) {
    auto s = get_scalar<torch::kFloat32>(SpecialValue::DTypeMax);
    EXPECT_FLOAT_EQ((float)s.toDouble(), FLT_MAX);

    s = get_scalar<torch::kInt64>(SpecialValue::DTypeMin);
    EXPECT_EQ(s.toLong(), std::numeric_limits<int64_t>::lowest());
}

// ============================================================
// DiscreteEntry implicit numeric conversion
// ============================================================

TEST(DiscreteEntryConversion, IntToSpecialValue) {
    DiscreteEntry e0(0, 0.1);
    EXPECT_EQ(e0.value, SpecialValue::Zero);
    EXPECT_DOUBLE_EQ(e0.probability, 0.1);

    DiscreteEntry e1(1, 0.2);
    EXPECT_EQ(e1.value, SpecialValue::One);

    DiscreteEntry en1(-1, 0.3);
    EXPECT_EQ(en1.value, SpecialValue::NOne);
}

TEST(DiscreteEntryConversion, DoubleToSpecialValue) {
    DiscreteEntry e0(0.0, 0.1);
    EXPECT_EQ(e0.value, SpecialValue::Zero);

    DiscreteEntry e1(1.0, 0.2);
    EXPECT_EQ(e1.value, SpecialValue::One);

    DiscreteEntry en1(-1.0, 0.3);
    EXPECT_EQ(en1.value, SpecialValue::NOne);

    DiscreteEntry enan(std::numeric_limits<double>::quiet_NaN(), 0.1);
    EXPECT_EQ(enan.value, SpecialValue::NaN);

    DiscreteEntry einf(std::numeric_limits<double>::infinity(), 0.1);
    EXPECT_EQ(einf.value, SpecialValue::Inf);

    DiscreteEntry eninf(-std::numeric_limits<double>::infinity(), 0.1);
    EXPECT_EQ(eninf.value, SpecialValue::NInf);
}

TEST(DiscreteEntryConversion, RawIntValues) {
    DiscreteEntry e2(2, 0.1);
    EXPECT_FALSE(e2.value.is_special());
    EXPECT_DOUBLE_EQ(e2.value.raw(), 2.0);

    DiscreteEntry e42(42, 0.5);
    EXPECT_FALSE(e42.value.is_special());
    EXPECT_DOUBLE_EQ(e42.value.raw(), 42.0);

    DiscreteEntry en128(-128, 0.1);
    EXPECT_FALSE(en128.value.is_special());
    EXPECT_DOUBLE_EQ(en128.value.raw(), -128.0);
}

TEST(DiscreteEntryConversion, RawDoubleValues) {
    DiscreteEntry e_pi(3.14, 0.1);
    EXPECT_FALSE(e_pi.value.is_special());
    EXPECT_DOUBLE_EQ(e_pi.value.raw(), 3.14);

    DiscreteEntry e_half(0.5, 0.2);
    EXPECT_FALSE(e_half.value.is_special());
    EXPECT_DOUBLE_EQ(e_half.value.raw(), 0.5);

    DiscreteEntry e_neg(-99.9, 0.1);
    EXPECT_FALSE(e_neg.value.is_special());
    EXPECT_DOUBLE_EQ(e_neg.value.raw(), -99.9);
}

TEST(DiscreteEntryConversion, RawIntGeneratesCorrectly) {
    auto t = generate_tensor({10000}, torch::kInt32, TensorSpec{{2, 1.0}});
    EXPECT_TRUE((t == 2).all().item<bool>());

    auto t2 = generate_tensor({10000}, torch::kFloat32, TensorSpec{{42, 0.5}},
                              BaseDist::uniform(0.0, 1.0), 777);
    auto count_42 = (t2 == 42.0f).sum().item<int64_t>();
    EXPECT_GT(count_42, 4000);
    EXPECT_LT(count_42, 6000);
}

TEST(DiscreteEntryConversion, RawDoubleGeneratesCorrectly) {
    auto t = generate_tensor({10000}, torch::kFloat32, TensorSpec{{3.14, 1.0}});
    EXPECT_TRUE(torch::allclose(t, torch::full({10000}, 3.14f)));

    auto t2 = generate_tensor({10000}, torch::kFloat64,
                              TensorSpec{{-99.5, 0.3}},
                              BaseDist::uniform(0.0, 1.0), 888);
    auto count = (t2 == -99.5).sum().item<int64_t>();
    EXPECT_GT(count, 2500);
    EXPECT_LT(count, 3500);
}

TEST(DiscreteEntryConversion, RawValueMixedWithSpecial) {
    auto t = generate_tensor({10000}, torch::kFloat32,
                             TensorSpec{{SpecialValue::NaN, 0.2}, {42, 0.3}},
                             BaseDist::uniform(0.0, 1.0), 999);
    auto nan_count = torch::isnan(t).sum().item<int64_t>();
    auto val42_count = (t == 42.0f).sum().item<int64_t>();
    EXPECT_GT(nan_count, 1500);
    EXPECT_LT(nan_count, 2500);
    EXPECT_GT(val42_count, 2500);
    EXPECT_LT(val42_count, 3500);
}

TEST(DiscreteEntryConversion, RawValueOnIntegerDtype) {
    auto t = generate_tensor({100}, torch::kUInt8, TensorSpec{{200, 1.0}});
    auto t64 = t.to(torch::kInt64);
    EXPECT_TRUE((t64 == 200).all().item<bool>());
}

TEST(DiscreteEntryConversion, RawValueAlwaysValidForDtype) {
    DiscreteEntry e(42, 0.1);
    EXPECT_TRUE(e.value.is_valid_for_dtype(torch::kFloat32));
    EXPECT_TRUE(e.value.is_valid_for_dtype(torch::kInt32));
    EXPECT_TRUE(e.value.is_valid_for_dtype(torch::kBool));
    EXPECT_TRUE(e.value.is_valid_for_dtype(torch::kUInt8));
}

TEST(DiscreteEntryConversion, RawDoubleTruncatesOnIntDtype) {
    auto t = generate_tensor({100}, torch::kInt32, TensorSpec{{3.14, 1.0}});
    EXPECT_TRUE((t == 3).all().item<bool>());

    auto t2 = generate_tensor({100}, torch::kInt64, TensorSpec{{-7.9, 1.0}});
    EXPECT_TRUE((t2 == -7).all().item<bool>());

    auto t3 = generate_tensor({100}, torch::kUInt8, TensorSpec{{3.14, 1.0}});
    EXPECT_TRUE((t3.to(torch::kInt64) == 3).all().item<bool>());
}

TEST(DiscreteEntryConversion, RawValueOnUnsignedDtypes) {
    auto t16 = generate_tensor({100}, c10::ScalarType::UInt16, TensorSpec{{1000, 1.0}});
    EXPECT_TRUE((t16.to(torch::kInt64) == 1000).all().item<bool>());

    auto t32 = generate_tensor({100}, c10::ScalarType::UInt32, TensorSpec{{70000, 1.0}});
    EXPECT_TRUE((t32.to(torch::kInt64) == 70000).all().item<bool>());

    auto t64 = generate_tensor({100}, c10::ScalarType::UInt64, TensorSpec{{12345, 1.0}});
    EXPECT_TRUE((t64.to(torch::kInt64) == 12345).all().item<bool>());
}

TEST(DiscreteEntryConversion, RawValueOverflowInt64Throws) {
    EXPECT_THROW(
        generate_tensor({10}, torch::kInt64, TensorSpec{{1e30, 1.0}}),
        std::invalid_argument);
    EXPECT_THROW(
        generate_tensor({10}, torch::kInt32, TensorSpec{{-1e30, 1.0}}),
        std::invalid_argument);
}

TEST(DiscreteEntryConversion, RawValueOnBoolDtype) {
    auto t = generate_tensor({100}, torch::kBool, TensorSpec{{42, 1.0}});
    EXPECT_TRUE(t.all().item<bool>());

    auto t0 = generate_tensor({100}, torch::kBool, TensorSpec{{0, 1.0}});
    EXPECT_FALSE(t0.any().item<bool>());
}

TEST(DiscreteEntryConversion, NegativeZeroMapsToZero) {
    DiscreteEntry e(-0.0, 0.1);
    EXPECT_EQ(e.value, SpecialValue::Zero);
}

TEST(DiscreteEntryConversion, ImplicitZeroOnE8M0Throws) {
    EXPECT_THROW(
        generate_tensor({10}, c10::ScalarType::Float8_e8m0fnu, TensorSpec{{0, 0.5}}),
        std::invalid_argument);
}

TEST(DiscreteEntryConversion, ImplicitNOneOnE8M0Throws) {
    EXPECT_THROW(
        generate_tensor({10}, c10::ScalarType::Float8_e8m0fnu, TensorSpec{{-1, 0.5}}),
        std::invalid_argument);
}

TEST(DiscreteEntryConversion, ImplicitNaNOnIntegerSkipped) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    // NaN on integer is silently skipped, not rejected
    auto t1 = generate_tensor({100}, torch::kInt32, TensorSpec{{nan, 0.5}},
                              BaseDist::randint(0, 10), 42);
    EXPECT_EQ(t1.numel(), 100);
    auto t2 = generate_tensor({100}, torch::kInt64, TensorSpec{{nan, 0.1}},
                              BaseDist::randint(0, 10), 43);
    EXPECT_EQ(t2.numel(), 100);
}

TEST(DiscreteEntryConversion, ImplicitInfOnNoInfDtypeFallback) {
    double inf = std::numeric_limits<double>::infinity();
    // Inf on e4m3fn → DTypeMax (448)
    auto t1 = generate_tensor({100}, torch::kFloat8_e4m3fn, TensorSpec{{inf, 1.0}});
    EXPECT_FLOAT_EQ(t1.to(torch::kFloat32).max().item<float>(), 448.0f);
    // Inf on int32 → DTypeMax
    auto t2 = generate_tensor({100}, torch::kInt32, TensorSpec{{inf, 1.0}});
    EXPECT_TRUE((t2 == std::numeric_limits<int32_t>::max()).all().item<bool>());
    // Inf on bool → true
    auto t3 = generate_tensor({100}, torch::kBool, TensorSpec{{inf, 1.0}});
    EXPECT_TRUE(t3.all().item<bool>());
}

TEST(DiscreteEntryConversion, ImplicitNInfOnNoInfDtypeFallback) {
    double ninf = -std::numeric_limits<double>::infinity();
    // NInf on e4m3fn → DTypeMin (-448)
    auto t1 = generate_tensor({100}, torch::kFloat8_e4m3fn, TensorSpec{{ninf, 1.0}});
    EXPECT_FLOAT_EQ(t1.to(torch::kFloat32).min().item<float>(), -448.0f);
    // NInf on uint8 → 0
    auto t2 = generate_tensor({100}, torch::kUInt8, TensorSpec{{ninf, 1.0}});
    EXPECT_TRUE((t2.to(torch::kInt64) == 0).all().item<bool>());
}

TEST(DiscreteEntryConversion, ImplicitValueNegativeProbThrows) {
    EXPECT_THROW(
        generate_tensor({10}, torch::kFloat32, TensorSpec{{0, -0.1}}),
        std::invalid_argument);
}

TEST(DiscreteEntryConversion, ImplicitValueProbExceedsOneThrows) {
    EXPECT_THROW(
        generate_tensor({10}, torch::kFloat32, TensorSpec{{0, 0.6}, {1, 0.6}}),
        std::invalid_argument);
}

TEST(DiscreteEntryConversion, ExplicitSpecialValueStillWorks) {
    DiscreteEntry e(SpecialValue::DTypeMax, 0.5);
    EXPECT_EQ(e.value, SpecialValue::DTypeMax);
    EXPECT_DOUBLE_EQ(e.probability, 0.5);
}

TEST(DiscreteEntryConversion, TensorSpecWithImplicitValues) {
    // {0, 0.1} equivalent to {SpecialValue::Zero, 0.1}
    TensorSpec spec1{{0, 0.1}};
    TensorSpec spec2{{SpecialValue::Zero, 0.1}};
    ASSERT_EQ(spec1.discrete.size(), 1u);
    EXPECT_EQ(spec1.discrete[0].value, spec2.discrete[0].value);
    EXPECT_DOUBLE_EQ(spec1.discrete[0].probability, spec2.discrete[0].probability);
}

TEST(DiscreteEntryConversion, TensorSpecMixedEntries) {
    TensorSpec spec{{0, 0.3}, {SpecialValue::NaN, 0.2}, {1, 0.1}};
    ASSERT_EQ(spec.discrete.size(), 3u);
    EXPECT_EQ(spec.discrete[0].value, SpecialValue::Zero);
    EXPECT_EQ(spec.discrete[1].value, SpecialValue::NaN);
    EXPECT_EQ(spec.discrete[2].value, SpecialValue::One);
}

TEST(DiscreteEntryConversion, GenerateTensorEquivalence) {
    auto shape = std::vector<int64_t>{10000};
    auto dtype = torch::kFloat32;
    int64_t seed = 12345;

    auto t_explicit = generate_tensor(shape, dtype,
        TensorSpec{{SpecialValue::Zero, 0.3}, {SpecialValue::NaN, 0.2}},
        BaseDist::uniform(0.0, 1.0), seed);

    auto t_implicit = generate_tensor(shape, dtype,
        TensorSpec{{0, 0.3}, {std::numeric_limits<double>::quiet_NaN(), 0.2}},
        BaseDist::uniform(0.0, 1.0), seed);

    EXPECT_TRUE(t_explicit.equal(t_implicit) ||
                torch::allclose(
                    t_explicit.nan_to_num(0), t_implicit.nan_to_num(0)));

    auto nan_mask_e = torch::isnan(t_explicit);
    auto nan_mask_i = torch::isnan(t_implicit);
    EXPECT_EQ(nan_mask_e.sum().item<int64_t>(), nan_mask_i.sum().item<int64_t>());
}

// ============================================================
// Count mode tests
// ============================================================

// Helper: make a CPU generator with a given seed
static torch::Generator make_gen(uint64_t seed) {
    auto gen = at::detail::createCPUGenerator();
    gen.set_current_seed(seed);
    return gen;
}

// --- Auto-detection ---

TEST(CountMode, AutoDetectSingleEntry) {
    // {NaN, 5} → sum=5 > 1, all int ≥ 1 → count mode
    auto t = generate_tensor({100}, torch::kFloat32, {{SpecialValue::NaN, 5.0}},
                             BaseDist::uniform(0.0, 1.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 5);
}

TEST(CountMode, AutoDetectMultiEntry) {
    // {NaN, 3}, {Inf, 2} → sum=5 > 1 → count mode
    auto t = generate_tensor({100}, torch::kFloat32,
        {{SpecialValue::NaN, 3.0}, {SpecialValue::Inf, 2.0}},
        BaseDist::uniform(0.0, 1.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 3);
    auto inf_count = (t == std::numeric_limits<float>::infinity()).sum().item<int64_t>();
    EXPECT_EQ(inf_count, 2);
}

TEST(CountMode, AutoDetectTwoOnes) {
    // {Zero, 1}, {One, 1} → sum=2 > 1 → count mode
    auto t = generate_tensor({100}, torch::kFloat32,
        {{SpecialValue::Zero, 1.0}, {SpecialValue::One, 1.0}},
        BaseDist::uniform(5.0, 10.0), 42);
    EXPECT_EQ((t == 0.0f).sum().item<int64_t>(), 1);
    EXPECT_EQ((t == 1.0f).sum().item<int64_t>(), 1);
}

TEST(CountMode, NoAutoDetectSingleOne) {
    // {NaN, 1.0} → sum=1 ≤ 1 → probability mode (100% NaN)
    auto t = generate_tensor({100}, torch::kFloat32, {{SpecialValue::NaN, 1.0}});
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 100);
}

TEST(CountMode, NoAutoDetectFractional) {
    // {NaN, 0.5} → not integer ≥ 1 → probability mode
    auto t = generate_tensor({10000}, torch::kFloat32,
        {{SpecialValue::NaN, 0.5}}, BaseDist::uniform(0.0, 1.0), 42);
    auto nan_count = torch::isnan(t).sum().item<int64_t>();
    EXPECT_GT(nan_count, 4500);
    EXPECT_LT(nan_count, 5500);
}

TEST(CountMode, NoAutoDetectMixed) {
    // {NaN, 1}, {Inf, 0.5} → not all integers → probability mode
    // sum = 1.5 > 1, but 0.5 is not integer → stays prob mode → throws
    EXPECT_THROW(
        generate_tensor({100}, torch::kFloat32,
            {{SpecialValue::NaN, 1.0}, {SpecialValue::Inf, 0.5}}),
        std::invalid_argument);
}

// --- Explicit with_counts() ---

TEST(CountMode, ExplicitSingleCount) {
    // Edge case: with_counts({{NaN, 1}}) → exactly 1 NaN
    auto spec = TensorSpec::with_counts({{SpecialValue::NaN, 1.0}});
    auto t = generate_tensor({100}, torch::kFloat32, spec,
                             BaseDist::uniform(0.0, 1.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 1);
}

TEST(CountMode, ExplicitZeroCount) {
    // with_counts({{NaN, 0}}) → 0 NaN, no-op
    auto spec = TensorSpec::with_counts({{SpecialValue::NaN, 0.0}});
    auto t = generate_tensor({100}, torch::kFloat32, spec,
                             BaseDist::uniform(1.0, 2.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 0);
}

// --- All-count (total_count == numel) ---

TEST(CountMode, AllCountSingleValue) {
    auto t = generate_tensor({10}, torch::kFloat32, {{SpecialValue::NaN, 10.0}});
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 10);
}

TEST(CountMode, AllCountMultiValue) {
    auto t = generate_tensor({10}, torch::kFloat32,
        {{SpecialValue::Zero, 4.0}, {SpecialValue::One, 3.0},
         {SpecialValue::NaN, 3.0}});
    EXPECT_EQ((t == 0.0f).sum().item<int64_t>(), 4);
    EXPECT_EQ((t == 1.0f).sum().item<int64_t>(), 3);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 3);
}

// --- Shapes: 1D through 5D ---

TEST(CountMode, Shape1D) {
    auto t = generate_tensor({50}, torch::kFloat32,
        {{SpecialValue::NaN, 5.0}, {SpecialValue::Inf, 3.0}},
        BaseDist::uniform(0.0, 1.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 5);
    EXPECT_EQ((t == std::numeric_limits<float>::infinity()).sum().item<int64_t>(), 3);
}

TEST(CountMode, Shape2D) {
    auto t = generate_tensor({10, 20}, torch::kFloat32,
        {{SpecialValue::NaN, 15.0}, {SpecialValue::Zero, 10.0}},
        BaseDist::uniform(1.0, 2.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 15);
    EXPECT_EQ((t == 0.0f).sum().item<int64_t>(), 10);
    EXPECT_EQ(t.sizes(), (std::vector<int64_t>{10, 20}));
}

TEST(CountMode, Shape3D) {
    auto t = generate_tensor({4, 5, 6}, torch::kFloat64,
        {{SpecialValue::NaN, 10.0}, {SpecialValue::Inf, 5.0}},
        BaseDist::normal(0.0, 1.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 10);
    auto inf_count = (t == std::numeric_limits<double>::infinity()).sum().item<int64_t>();
    EXPECT_EQ(inf_count, 5);
    EXPECT_EQ(t.sizes(), (std::vector<int64_t>{4, 5, 6}));
}

TEST(CountMode, Shape4D) {
    auto t = generate_tensor({2, 3, 4, 5}, torch::kFloat32,
        {{SpecialValue::NaN, 20.0}, {SpecialValue::NInf, 10.0}},
        BaseDist::uniform(-1.0, 1.0), 42);
    int64_t numel = 2 * 3 * 4 * 5;
    EXPECT_EQ(t.numel(), numel);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 20);
    EXPECT_EQ((t == -std::numeric_limits<float>::infinity()).sum().item<int64_t>(), 10);
}

TEST(CountMode, Shape5D) {
    auto t = generate_tensor({2, 2, 3, 2, 5}, torch::kFloat32,
        {{SpecialValue::Zero, 10.0}, {SpecialValue::DTypeMax, 5.0}},
        BaseDist::uniform(1.0, 2.0), 42);
    int64_t numel = 2 * 2 * 3 * 2 * 5;
    EXPECT_EQ(t.numel(), numel);
    EXPECT_EQ((t == 0.0f).sum().item<int64_t>(), 10);
    EXPECT_EQ((t == FLT_MAX).sum().item<int64_t>(), 5);
}

// --- Integer and other dtypes ---

TEST(CountMode, Int32Dtype) {
    auto t = generate_tensor({100}, torch::kInt32,
        {{SpecialValue::Zero, 10.0}, {SpecialValue::DTypeMax, 5.0}},
        BaseDist::randint(1, 100), 42);
    EXPECT_EQ((t == 0).sum().item<int64_t>(), 10);
    EXPECT_EQ((t == std::numeric_limits<int32_t>::max()).sum().item<int64_t>(), 5);
}

TEST(CountMode, Int64Dtype) {
    auto t = generate_tensor({200}, torch::kInt64,
        {{SpecialValue::DTypeMin, 8.0}, {SpecialValue::DTypeMax, 7.0}},
        BaseDist::randint(0, 1000), 42);
    EXPECT_EQ((t == std::numeric_limits<int64_t>::lowest()).sum().item<int64_t>(), 8);
    EXPECT_EQ((t == std::numeric_limits<int64_t>::max()).sum().item<int64_t>(), 7);
}

TEST(CountMode, UInt8Dtype) {
    auto t = generate_tensor({50}, torch::kUInt8,
        {{SpecialValue::DTypeMax, 3.0}, {SpecialValue::Zero, 2.0}},
        BaseDist::randint(1, 100), 42);
    auto t64 = t.to(torch::kInt64);
    EXPECT_EQ((t64 == 255).sum().item<int64_t>(), 3);
    EXPECT_EQ((t64 == 0).sum().item<int64_t>(), 2);
}

TEST(CountMode, BoolDtype) {
    auto t = generate_tensor({100}, torch::kBool,
        {{SpecialValue::One, 10.0}, {SpecialValue::Zero, 10.0}});
    // total_count=20, numel=100, all_discrete=false
    // But base_dist is Uniform(-1,1) and bool → each element casts to bool
    // Just check the counts are exact for discrete entries
    // Actually bool + count mode: base doesn't throw, let's test with all_discrete
    auto t_all = generate_tensor({20}, torch::kBool,
        {{SpecialValue::One, 10.0}, {SpecialValue::Zero, 10.0}});
    EXPECT_EQ(t_all.sum().item<int64_t>(), 10);
}

// --- Literal values in count mode ---

TEST(CountMode, LiteralIntValue) {
    auto t = generate_tensor({100}, torch::kInt32,
        {{42, 7.0}, {SpecialValue::Zero, 3.0}},
        BaseDist::randint(1, 10), 42);
    EXPECT_EQ((t == 42).sum().item<int64_t>(), 7);
    EXPECT_EQ((t == 0).sum().item<int64_t>(), 3);
}

TEST(CountMode, LiteralDoubleValue) {
    auto t = generate_tensor({100}, torch::kFloat64,
        {{3.14, 5.0}, {SpecialValue::NaN, 3.0}},
        BaseDist::uniform(0.0, 1.0), 42);
    EXPECT_EQ((t == 3.14).sum().item<int64_t>(), 5);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 3);
}

// --- Base distribution interaction ---

TEST(CountMode, RemainingFilledByBaseDist) {
    auto t = generate_tensor({1000}, torch::kFloat32,
        {{SpecialValue::NaN, 10.0}},
        BaseDist::uniform(100.0, 200.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 10);
    auto non_nan = t.masked_select(~torch::isnan(t));
    EXPECT_EQ(non_nan.numel(), 990);
    EXPECT_GE(non_nan.min().item<float>(), 100.0f);
    EXPECT_LE(non_nan.max().item<float>(), 200.0f);
}

// --- Reproducibility ---

TEST(CountMode, Reproducible) {
    auto gen1 = BaseDist::uniform(0.0, 1.0).with_seed(777);
    auto gen2 = BaseDist::uniform(0.0, 1.0).with_seed(777);
    auto t1 = generate_tensor({100}, torch::kFloat32,
        {{SpecialValue::NaN, 5.0}, {SpecialValue::Inf, 3.0}}, gen1);
    auto t2 = generate_tensor({100}, torch::kFloat32,
        {{SpecialValue::NaN, 5.0}, {SpecialValue::Inf, 3.0}}, gen2);

    auto nan1 = torch::isnan(t1);
    auto nan2 = torch::isnan(t2);
    EXPECT_TRUE(nan1.equal(nan2));
    EXPECT_TRUE(t1.nan_to_num(0).equal(t2.nan_to_num(0)));
}

// --- Fallback in count mode ---

TEST(CountMode, InfFallbackOnIntDtype) {
    // Inf → DTypeMax on int32, count mode
    auto t = generate_tensor({50}, torch::kInt32,
        {{SpecialValue::Inf, 5.0}, {SpecialValue::NInf, 3.0}},
        BaseDist::randint(0, 100), 42);
    EXPECT_EQ((t == std::numeric_limits<int32_t>::max()).sum().item<int64_t>(), 5);
    EXPECT_EQ((t == std::numeric_limits<int32_t>::lowest()).sum().item<int64_t>(), 3);
}

TEST(CountMode, NaNSkippedOnIntDtype) {
    // NaN on int is silently skipped; count not applied
    auto t = generate_tensor({50}, torch::kInt32,
        {{SpecialValue::NaN, 5.0}, {SpecialValue::Zero, 3.0}},
        BaseDist::randint(10, 20), 42);
    // NaN was skipped, so only Zero count=3 applies
    // sum after resolution = 3, which is still > 1 since with_counts
    // Actually auto-detect: original entries have 5+3=8 ints, but after
    // NaN skip, resolved discrete has only {Zero, 3.0}. p_sum=3 > 1 → count mode
    EXPECT_EQ((t == 0).sum().item<int64_t>(), 3);
}

// --- Error cases ---

TEST(CountMode, ErrorCountExceedsNumel) {
    EXPECT_THROW(
        generate_tensor({10}, torch::kFloat32,
            {{SpecialValue::NaN, 8.0}, {SpecialValue::Inf, 5.0}}),
        std::invalid_argument);
}

TEST(CountMode, ErrorExplicitCountExceedsNumel) {
    auto spec = TensorSpec::with_counts({{SpecialValue::NaN, 20.0}});
    EXPECT_THROW(
        generate_tensor({10}, torch::kFloat32, spec),
        std::invalid_argument);
}

TEST(CountMode, ErrorNonIntegerInExplicitCountMode) {
    auto spec = TensorSpec::with_counts({{SpecialValue::NaN, 3.5}});
    EXPECT_THROW(
        generate_tensor({100}, torch::kFloat32, spec),
        std::invalid_argument);
}

TEST(CountMode, ErrorNegativeCountInExplicitMode) {
    auto spec = TensorSpec::with_counts({{SpecialValue::NaN, -1.0}});
    EXPECT_THROW(
        generate_tensor({100}, torch::kFloat32, spec),
        std::invalid_argument);
}

// --- Stress: large tensor ---

TEST(CountMode, LargeTensor) {
    auto t = generate_tensor({1000000}, torch::kFloat32,
        {{SpecialValue::NaN, 1000.0}, {SpecialValue::Inf, 500.0},
         {SpecialValue::Zero, 200.0}},
        BaseDist::uniform(-1.0, 1.0), 42);
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 1000);
    EXPECT_EQ((t == std::numeric_limits<float>::infinity()).sum().item<int64_t>(), 500);
    EXPECT_EQ((t == 0.0f).sum().item<int64_t>(), 200);
}

// --- No overlap between discrete positions ---

TEST(CountMode, NoOverlapBetweenValues) {
    for (int trial = 0; trial < 20; ++trial) {
        auto t = generate_tensor({100}, torch::kFloat32,
            {{SpecialValue::NaN, 30.0}, {SpecialValue::Inf, 30.0},
             {SpecialValue::Zero, 30.0}},
            BaseDist::uniform(1.0, 2.0), 42 + trial);
        auto nan_count = torch::isnan(t).sum().item<int64_t>();
        auto inf_count = (t == std::numeric_limits<float>::infinity()).sum().item<int64_t>();
        auto zero_count = (t == 0.0f).sum().item<int64_t>();
        EXPECT_EQ(nan_count, 30) << "trial=" << trial;
        EXPECT_EQ(inf_count, 30) << "trial=" << trial;
        EXPECT_EQ(zero_count, 30) << "trial=" << trial;
        // remaining 10 should be from base dist [1, 2)
        auto remaining = t.masked_select(~torch::isnan(t) &
                                         (t != std::numeric_limits<float>::infinity()) &
                                         (t != 0.0f));
        EXPECT_EQ(remaining.numel(), 10) << "trial=" << trial;
    }
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
