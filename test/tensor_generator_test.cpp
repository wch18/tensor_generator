#include <gtest/gtest.h>
#include "../src/tensor_generator.h"
#include "test_configs.h"
#include <cmath>
#include <tuple>
#include <string>
#include <vector>
#include <sstream>

using namespace tgen;

// ============================================================
// Helpers
// ============================================================

static void assert_probability(int64_t count, int64_t total, double expected_prob,
                               double sigma_tolerance = 5.0) {
    if (total == 0) return;
    double expected = total * expected_prob;
    double stddev = std::sqrt(total * expected_prob * (1.0 - expected_prob));
    // For very small expected counts, use wider tolerance
    double lo = expected - std::max(sigma_tolerance * stddev, 3.0);
    double hi = expected + std::max(sigma_tolerance * stddev, 3.0);
    EXPECT_GE(count, (int64_t)std::floor(lo))
        << "count=" << count << " expected≈" << expected << " (p=" << expected_prob << ")";
    EXPECT_LE(count, (int64_t)std::ceil(hi))
        << "count=" << count << " expected≈" << expected << " (p=" << expected_prob << ")";
}

static std::string dtype_name(torch::ScalarType dt) {
    if (dt == c10::ScalarType::UInt16) return "uint16";
    if (dt == c10::ScalarType::UInt32) return "uint32";
    if (dt == c10::ScalarType::UInt64) return "uint64";
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
// Parameterized test: Shape × DType × Config cartesian product
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

    // Skip if config not compatible with this dtype
    if (!cfg.is_compatible(param.dtype)) {
        GTEST_SKIP() << "Config '" << cfg.name << "' not compatible with "
                     << dtype_name(param.dtype);
    }

    torch::manual_seed(42 + param.config_idx * 1000 +
                       (int)param.dtype * 100 + param.shape.back());

    torch::IntArrayRef shape(param.shape);
    torch::Tensor t;

    // Should not throw
    ASSERT_NO_THROW(t = generate_tensor(shape, param.dtype, cfg.spec))
        << "Failed for shape=" << shape_str(shape)
        << " dtype=" << dtype_name(param.dtype)
        << " config=" << cfg.name;

    // Verify shape
    ASSERT_EQ(t.dim(), (int64_t)param.shape.size());
    for (size_t i = 0; i < param.shape.size(); ++i) {
        ASSERT_EQ(t.size(i), param.shape[i]);
    }

    // Verify dtype
    ASSERT_EQ(t.scalar_type(), param.dtype);

    int64_t N = t.numel();
    if (N == 0) return;

    // Convert to float64 for analysis (handles all types)
    auto tf = t.to(torch::kFloat64);

    // Verify discrete values are correctly represented
    double p_sum = 0.0;
    for (auto& entry : cfg.spec.discrete) {
        p_sum += entry.probability;
    }
    bool has_continuous = (p_sum < 1.0 - 1e-9);

    // For integer/fp8/bool dtypes with continuous parts, continuous values quantized
    // to the low-resolution dtype may coincidentally equal a discrete value.
    // In that case the observed count is higher than expected → only check lower bound.
    bool low_resolution_dtype = is_int_type(param.dtype) || param.dtype == torch::kBool ||
        param.dtype == torch::kFloat8_e4m3fn || param.dtype == torch::kFloat8_e5m2;
    bool relaxed_verification = low_resolution_dtype && has_continuous;

    // Group entries by actual value to handle aliasing (e.g. Zero=DTypeMin for unsigned)
    struct ValueGroup {
        double value;
        double total_prob;
        bool is_nan;
        bool is_pos_inf;
        bool is_neg_inf;
    };
    std::vector<ValueGroup> groups;
    for (auto& entry : cfg.spec.discrete) {
        double v = get_special_value_as_double(entry.value, param.dtype);
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
                << "With p≈1.0, all elements should match. dtype="
                << dtype_name(param.dtype) << " config=" << cfg.name;
        }
    }

    // If all_discrete (p_sum ≈ 1.0), verify no "leak" elements
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

// Generate the full cartesian product of parameters
static std::vector<CartesianTestParam> generate_cartesian_params() {
    auto shapes = get_all_shapes();
    auto dtypes = get_all_dtypes();
    auto configs = get_all_configs();

    std::vector<CartesianTestParam> params;
    for (auto& shape : shapes) {
        for (auto dtype : dtypes) {
            for (int ci = 0; ci < (int)configs.size(); ++ci) {
                // Only include if config is compatible
                if (configs[ci].is_compatible(dtype)) {
                    params.push_back(CartesianTestParam{shape, dtype, ci});
                }
            }
        }
    }
    return params;
}

// Custom test name for readability
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
// Focused unit tests (non-parameterized, for specific edge cases)
// ============================================================

// --- Validation tests ---
TEST(Validation, FloatOnlyValues) {
    std::vector<torch::ScalarType> float_types = {
        torch::kFloat16, torch::kBFloat16, torch::kFloat32, torch::kFloat64,
        torch::kFloat8_e4m3fn, torch::kFloat8_e5m2};
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
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::Infinity, torch::kFloat8_e4m3fn));
    EXPECT_FALSE(is_valid_for_dtype(SpecialValue::NegativeInfinity, torch::kFloat8_e4m3fn));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::Infinity, torch::kFloat8_e5m2));
    EXPECT_TRUE(is_valid_for_dtype(SpecialValue::NaN, torch::kFloat8_e4m3fn));
}

// --- Error handling tests ---
TEST(ErrorHandling, NaNOnInteger) {
    EXPECT_THROW(generate_tensor({10}, torch::kInt32, {{SpecialValue::NaN, 0.5}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, InfOnE4M3) {
    EXPECT_THROW(generate_tensor({10}, torch::kFloat8_e4m3fn, {{SpecialValue::Infinity, 0.1}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, NegativeProbability) {
    EXPECT_THROW(generate_tensor({10}, torch::kFloat32, {{SpecialValue::Zero, -0.1}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, ProbExceedsOne) {
    EXPECT_THROW(generate_tensor({10}, torch::kFloat32,
                     {{SpecialValue::Zero, 0.6}, {SpecialValue::One, 0.6}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, InfOnBool) {
    EXPECT_THROW(generate_tensor({10}, torch::kBool, {{SpecialValue::Infinity, 0.1}}),
                 std::invalid_argument);
}

TEST(ErrorHandling, EpsilonOnInt) {
    EXPECT_THROW(generate_tensor({10}, torch::kInt64, {{SpecialValue::Epsilon, 0.1}}),
                 std::invalid_argument);
}

// --- Edge case tests ---
TEST(EdgeCase, EmptyTensor) {
    auto t = generate_tensor({2, 0, 3}, torch::kFloat32, {{SpecialValue::NaN, 0.5}});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.scalar_type(), torch::kFloat32);
}

TEST(EdgeCase, SingleElementAllDiscrete) {
    // p=1.0 with N=1: must always produce the expected value
    for (int trial = 0; trial < 100; ++trial) {
        auto t = generate_tensor({1}, torch::kFloat32, {{SpecialValue::NaN, 1.0}});
        EXPECT_TRUE(std::isnan(t.item<float>()));
    }
}

TEST(EdgeCase, SingleElementMixed) {
    // N=1, p=0.5: should produce either Zero or continuous value
    torch::manual_seed(42);
    int zero_count = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        auto t = generate_tensor({1}, torch::kFloat32,
            TensorSpec{{SpecialValue::Zero, 0.5}}.with_uniform(1.0, 2.0));
        float val = t.item<float>();
        if (val == 0.0f) {
            zero_count++;
        } else {
            EXPECT_GE(val, 1.0f);
            EXPECT_LE(val, 2.0f);
        }
    }
    // Expect ~500 zeros (5σ tolerance: ±56)
    EXPECT_GT(zero_count, 400);
    EXPECT_LT(zero_count, 600);
}

TEST(EdgeCase, ZeroProbabilityEntries) {
    // Entries with p=0 should be skipped
    auto t = generate_tensor({1000}, torch::kFloat32,
        {{SpecialValue::NaN, 0.0}, {SpecialValue::One, 1.0}});
    EXPECT_EQ(torch::isnan(t).sum().item<int64_t>(), 0);
    EXPECT_EQ((t == 1.0f).sum().item<int64_t>(), 1000);
}

TEST(EdgeCase, EmptyDiscreteList) {
    auto t = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{}.with_uniform(0.0, 1.0));
    EXPECT_GE(t.min().item<float>(), 0.0f);
    EXPECT_LE(t.max().item<float>(), 1.0f);
}

TEST(EdgeCase, VerySmallProbability) {
    torch::manual_seed(77);
    // p=0.001 with N=1M → expect ~1000 hits
    const int64_t N = 1000000;
    auto t = generate_tensor({N}, torch::kFloat32,
        TensorSpec{{SpecialValue::Infinity, 0.001}}.with_uniform(-1.0, 1.0));
    auto inf_count = (t == std::numeric_limits<float>::infinity()).sum().item<int64_t>();
    assert_probability(inf_count, N, 0.001);
}

// --- UInt8 specific tests ---
TEST(UInt8Specific, NegativeOneIs255) {
    auto t = generate_tensor({10000}, torch::kUInt8,
        {{SpecialValue::NegativeOne, 1.0}});
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
    torch::manual_seed(333);
    auto t = generate_tensor({10000}, torch::kUInt8,
        TensorSpec{}.with_uniform(0.0, 256.0));
    auto t64 = t.to(torch::kFloat64);
    EXPECT_GE(t64.min().item<double>(), 0.0);
    EXPECT_LE(t64.max().item<double>(), 255.0);
}

// --- UInt16 specific tests ---
TEST(UInt16Specific, NegativeOneIs65535) {
    auto t = generate_tensor({10000}, c10::ScalarType::UInt16,
        {{SpecialValue::NegativeOne, 1.0}});
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
    torch::manual_seed(555);
    auto t = generate_tensor({50000}, c10::ScalarType::UInt16,
        {{SpecialValue::Zero, 0.3},
         {SpecialValue::One, 0.3},
         {SpecialValue::DTypeMax, 0.4}});
    auto t64 = t.to(torch::kFloat64);
    auto total = (t64 == 0).sum().item<int64_t>() +
                 (t64 == 1).sum().item<int64_t>() +
                 (t64 == 65535.0).sum().item<int64_t>();
    EXPECT_EQ(total, 50000);
}

// --- UInt32 specific tests ---
TEST(UInt32Specific, NegativeOneIs4294967295) {
    auto t = generate_tensor({10000}, c10::ScalarType::UInt32,
        {{SpecialValue::NegativeOne, 1.0}});
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
    torch::manual_seed(666);
    auto t = generate_tensor({50000}, c10::ScalarType::UInt32,
        {{SpecialValue::Zero, 0.25},
         {SpecialValue::One, 0.25},
         {SpecialValue::NegativeOne, 0.25},
         {SpecialValue::DTypeMax, 0.25}});
    auto t64 = t.to(torch::kFloat64);
    // NegativeOne and DTypeMax are both 4294967295 for uint32
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
    torch::manual_seed(777);
    auto t = generate_tensor({50000}, c10::ScalarType::UInt64,
        {{SpecialValue::Zero, 0.5}, {SpecialValue::One, 0.5}});
    auto t64 = t.to(torch::kFloat64);
    auto zero_count = (t64 == 0).sum().item<int64_t>();
    auto one_count = (t64 == 1).sum().item<int64_t>();
    EXPECT_EQ(zero_count + one_count, 50000);
    assert_probability(zero_count, 50000, 0.5);
}

// --- Float special value correctness across dtypes ---
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

// --- Integer boundary value correctness ---
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
        {{SpecialValue::Zero, 0.5}, {SpecialValue::One, 0.5}});
    auto true_count = t.sum().item<int64_t>();
    assert_probability(true_count, 1000, 0.5);
}

TEST(BoolDType, DTypeMinFalse_DTypeMaxTrue) {
    auto t = generate_tensor({1}, torch::kBool, {{SpecialValue::DTypeMin, 1.0}});
    EXPECT_FALSE(t.item<bool>());
    t = generate_tensor({1}, torch::kBool, {{SpecialValue::DTypeMax, 1.0}});
    EXPECT_TRUE(t.item<bool>());
}

// --- Large-scale performance sanity ---
TEST(Performance, FourMillionFloat32) {
    torch::manual_seed(999);
    auto t = generate_tensor({4000000}, torch::kFloat32,
        TensorSpec{{SpecialValue::Zero, 0.1},
                   {SpecialValue::NaN, 0.05},
                   {SpecialValue::Infinity, 0.02},
                   {SpecialValue::DTypeMin, 0.03}}.with_uniform(-100.0, 100.0));

    EXPECT_EQ(t.numel(), 4000000);
    auto nan_count = torch::isnan(t).sum().item<int64_t>();
    auto zero_count = (t == 0.0f).sum().item<int64_t>();
    assert_probability(nan_count, 4000000, 0.05);
    assert_probability(zero_count, 4000000, 0.1);
}

TEST(Performance, FourMillionInt64) {
    torch::manual_seed(1000);
    auto t = generate_tensor({4000000}, torch::kInt64,
        TensorSpec{{SpecialValue::Zero, 0.2},
                   {SpecialValue::DTypeMax, 0.1},
                   {SpecialValue::NegativeOne, 0.1}}.with_uniform(-1000.0, 1000.0));

    EXPECT_EQ(t.numel(), 4000000);
    auto zero_count = (t == 0).sum().item<int64_t>();
    assert_probability(zero_count, 4000000, 0.2);
}

// --- Continuous distribution property tests ---
TEST(ContinuousDist, UniformRange) {
    torch::manual_seed(1111);
    auto t = generate_tensor({500000}, torch::kFloat32,
        TensorSpec{}.with_uniform(-7.5, 3.5));
    EXPECT_GE(t.min().item<float>(), -7.5f);
    EXPECT_LE(t.max().item<float>(), 3.5f);
    float mean = t.mean().item<float>();
    EXPECT_NEAR(mean, -2.0f, 0.05f);  // expected mean of U(-7.5, 3.5) = -2.0
}

TEST(ContinuousDist, NormalMeanStd) {
    torch::manual_seed(2222);
    auto t = generate_tensor({500000}, torch::kFloat64,
        TensorSpec{}.with_normal(5.0, 3.0));
    double mean = t.mean().item<double>();
    double std_val = t.std().item<double>();
    EXPECT_NEAR(mean, 5.0, 0.03);
    EXPECT_NEAR(std_val, 3.0, 0.03);
}

// --- All-discrete leakage test (the critical floating-point edge case) ---
TEST(AllDiscrete, NoLeakageMultipleTrials) {
    // Run 50 trials to stress the floating-point boundary
    for (int trial = 0; trial < 50; ++trial) {
        torch::manual_seed(trial * 7);
        auto t = generate_tensor({100000}, torch::kFloat32,
            {{SpecialValue::Zero, 0.1},
             {SpecialValue::One, 0.2},
             {SpecialValue::NegativeOne, 0.3},
             {SpecialValue::DTypeMax, 0.4}});

        auto total = (t == 0).sum().item<int64_t>() +
                     (t == 1).sum().item<int64_t>() +
                     (t == -1).sum().item<int64_t>() +
                     (t == FLT_MAX).sum().item<int64_t>();
        EXPECT_EQ(total, 100000)
            << "Leakage detected at trial " << trial;
    }
}

TEST(AllDiscrete, NoLeakageWithFloatingProbs) {
    // 0.1 + 0.2 + 0.3 + 0.4 should sum to 1.0 but floating-point
    // addition may not be exact: 0.1+0.2 = 0.30000000000000004 in float64
    for (int trial = 0; trial < 50; ++trial) {
        torch::manual_seed(1000 + trial);
        auto t = generate_tensor({50000}, torch::kInt32,
            {{SpecialValue::Zero, 0.1},
             {SpecialValue::One, 0.2},
             {SpecialValue::NegativeOne, 0.3},
             {SpecialValue::DTypeMin, 0.4}});

        auto total = (t == 0).sum().item<int64_t>() +
                     (t == 1).sum().item<int64_t>() +
                     (t == -1).sum().item<int64_t>() +
                     (t == (int32_t)std::numeric_limits<int32_t>::lowest()).sum().item<int64_t>();
        EXPECT_EQ(total, 50000)
            << "Leakage detected at trial " << trial;
    }
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
