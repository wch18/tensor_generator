#include <gtest/gtest.h>
#include "convenience.h"
#include "strategies.h"
#include <cmath>
#include <set>
#include <algorithm>

// ============================================================================
// Tests for convenience.h
// ============================================================================

// --- 1a: make_*_tensor tests ------------------------------------------------

TEST(MakeUniformTensor, BasicShape) {
    auto t = make_uniform_tensor({500}, torch::kFloat32, -2.0, 2.0);
    EXPECT_EQ(t.sizes(), std::vector<int64_t>{500});
    EXPECT_EQ(t.dtype(), torch::kFloat32);
    EXPECT_TRUE((t >= -2.0).all().item<bool>());
    EXPECT_TRUE((t <= 2.0).all().item<bool>());
}

TEST(MakeUniformTensor, MultiDim) {
    auto t = make_uniform_tensor({10, 20, 30}, torch::kFloat64);
    EXPECT_EQ(t.numel(), 6000);
    EXPECT_EQ(t.dtype(), torch::kFloat64);
}

TEST(MakeUniformTensor, WithGenerator) {
    auto gen1 = at::detail::createCPUGenerator();
    gen1.set_current_seed(42);
    auto t1 = make_uniform_tensor({100}, torch::kFloat32, 0.0, 1.0, gen1);

    auto gen2 = at::detail::createCPUGenerator();
    gen2.set_current_seed(42);
    auto t2 = make_uniform_tensor({100}, torch::kFloat32, 0.0, 1.0, gen2);

    EXPECT_TRUE(t1.equal(t2));
}

TEST(MakeNormalTensor, MeanAndStd) {
    auto t = make_normal_tensor({100000}, torch::kFloat64, 5.0, 2.0);
    double mean = t.mean().item<double>();
    double std_val = t.std().item<double>();
    EXPECT_NEAR(mean, 5.0, 0.1);
    EXPECT_NEAR(std_val, 2.0, 0.1);
}

TEST(MakeRandintTensor, Range) {
    auto t = make_randint_tensor({10000}, torch::kInt32, -10, 10);
    EXPECT_TRUE((t >= -10).all().item<bool>());
    EXPECT_TRUE((t < 10).all().item<bool>());
}

TEST(MakeRandintTensor, DtypeValidation) {
    EXPECT_THROW(make_randint_tensor({10}, torch::kFloat32, 0, 10), std::invalid_argument);
}

// --- 1b: tensor_with_one_sv tests -------------------------------------------

TEST(TensorWithOneSV, GenericInterface) {
    auto t = tensor_with_one_sv(SpecialValue::Zero, {10000}, torch::kFloat32, 0.5);
    int count = (t == 0).sum().item<int>();
    EXPECT_GT(count, 4500);
    EXPECT_LT(count, 5500);
}

TEST(TensorWithOneSV, NanRatio) {
    auto t = tensor_with_nan({10000}, torch::kFloat32, 0.2);
    int nan_count = t.isnan().sum().item<int>();
    EXPECT_GT(nan_count, 1700);
    EXPECT_LT(nan_count, 2300);
}

TEST(TensorWithOneSV, InfRatio) {
    auto t = tensor_with_inf({10000}, torch::kFloat64, 0.1);
    int inf_count = t.isinf().sum().item<int>();
    EXPECT_GT(inf_count, 700);
    EXPECT_LT(inf_count, 1300);
}

TEST(TensorWithOneSV, ZeroRatio) {
    auto t = tensor_with_zero({10000}, torch::kFloat32, 0.3);
    int count = (t == 0).sum().item<int>();
    EXPECT_GT(count, 2700);
    EXPECT_LT(count, 3300);
}

TEST(TensorWithOneSV, FallbackAndInvalidDtype) {
    // NaN on int32 → silently skipped, no throw
    EXPECT_NO_THROW(tensor_with_nan({10}, torch::kInt32, 0.1));
    // Inf on int64 → fallback to DTypeMax, no throw
    EXPECT_NO_THROW(tensor_with_inf({10}, torch::kInt64, 0.1));
    // Epsilon on uint8 → still invalid (no fallback defined)
    EXPECT_THROW(tensor_with_epsilon({10}, torch::kUInt8, 0.1), std::invalid_argument);
}

TEST(TensorWithOneSV, RatioZero) {
    auto t = tensor_with_nan({100}, torch::kFloat32, 0.0);
    EXPECT_EQ(t.isnan().sum().item<int>(), 0);
}

TEST(TensorWithOneSV, RatioOne) {
    auto t = tensor_with_nan({100}, torch::kFloat32, 1.0);
    EXPECT_EQ(t.isnan().sum().item<int>(), 100);
}

TEST(TensorWithOneSV, DefaultBaseDist) {
    auto t = tensor_with_zero({10000}, torch::kFloat32, 0.01);
    auto non_zero = t.masked_select(t != 0);
    EXPECT_TRUE((non_zero >= 0.0).all().item<bool>());
    EXPECT_TRUE((non_zero <= 1.0).all().item<bool>());
}

TEST(TensorWithOneSV, AllNamedVariants) {
    tensor_with_zero({10}, torch::kFloat32, 0.1);
    tensor_with_one({10}, torch::kFloat32, 0.1);
    tensor_with_none({10}, torch::kFloat32, 0.1);
    tensor_with_min({10}, torch::kFloat32, 0.1);
    tensor_with_max({10}, torch::kFloat32, 0.1);
    tensor_with_nan({10}, torch::kFloat32, 0.1);
    tensor_with_inf({10}, torch::kFloat32, 0.1);
    tensor_with_ninf({10}, torch::kFloat32, 0.1);
    tensor_with_epsilon({10}, torch::kFloat32, 0.1);
    tensor_with_tiny({10}, torch::kFloat32, 0.1);
}

// --- 1c: tensor_with_two_sv tests -------------------------------------------

TEST(TensorWithTwoSV, GenericInterface) {
    auto t = tensor_with_two_sv(
        SpecialValue::NaN, 0.1, SpecialValue::Inf, 0.1,
        {10000}, torch::kFloat32);
    int nan_c = t.isnan().sum().item<int>();
    int inf_c = t.isinf().sum().item<int>();
    EXPECT_GT(nan_c, 700);
    EXPECT_LT(nan_c, 1300);
    EXPECT_GT(inf_c, 700);
    EXPECT_LT(inf_c, 1300);
}

TEST(TensorWithTwoSV, InfNan) {
    auto t = tensor_with_inf_nan({10000}, torch::kFloat64, 0.15, 0.15);
    int nan_c = t.isnan().sum().item<int>();
    int inf_c = t.isinf().sum().item<int>();
    EXPECT_GT(nan_c, 1200);
    EXPECT_LT(nan_c, 1800);
    EXPECT_GT(inf_c, 1200);
    EXPECT_LT(inf_c, 1800);
}

TEST(TensorWithTwoSV, MaxMin) {
    auto t = tensor_with_max_min({10000}, torch::kInt32, 0.2, 0.2);
    auto max_val = torch::tensor({std::numeric_limits<int32_t>::max()}, torch::kInt32);
    auto min_val = torch::tensor({std::numeric_limits<int32_t>::min()}, torch::kInt32);
    int max_c = (t == max_val).sum().item<int>();
    int min_c = (t == min_val).sum().item<int>();
    EXPECT_GT(max_c, 1700);
    EXPECT_LT(max_c, 2300);
    EXPECT_GT(min_c, 1700);
    EXPECT_LT(min_c, 2300);
}

TEST(TensorWithTwoSV, SumExceedsOne) {
    EXPECT_THROW(
        tensor_with_inf_nan({10}, torch::kFloat32, 0.6, 0.6),
        std::invalid_argument);
}

TEST(TensorWithTwoSV, FallbackSVForDtype) {
    // Inf→DTypeMax, NaN→skipped on int32; no throw
    auto t = tensor_with_inf_nan({1000}, torch::kInt32, 0.3, 0.2);
    auto max_count = (t == std::numeric_limits<int32_t>::max()).sum().item<int64_t>();
    EXPECT_GT(max_count, 200);
}

TEST(TensorWithTwoSV, BothZeroRatio) {
    auto t = tensor_with_inf_nan({100}, torch::kFloat32, 0.0, 0.0);
    EXPECT_EQ(t.isnan().sum().item<int>(), 0);
    EXPECT_EQ(t.isinf().sum().item<int>(), 0);
}

TEST(TensorWithTwoSV, OneFullOneZero) {
    auto t = tensor_with_inf_nan({100}, torch::kFloat32, 1.0, 0.0);
    EXPECT_EQ(t.isinf().sum().item<int>(), 100);
    EXPECT_EQ(t.isnan().sum().item<int>(), 0);
}

// ============================================================================
// Tests for strategies.h
// ============================================================================

// --- 2a: random_strategy tests ----------------------------------------------

TEST(RandomStrategy, ProducesRequestedCount) {
    auto cases = random_strategy(10, torch::kFloat32);
    EXPECT_EQ((int)cases.size(), 10);
}

TEST(RandomStrategy, StrictMode) {
    auto cases = random_strategy(20, torch::kFloat32, 0.5, true);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        EXPECT_NEAR(sum, 0.5, 1e-6);
    }
}

TEST(RandomStrategy, NonStrictBound) {
    auto cases = random_strategy(50, torch::kFloat64, 0.3, false);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        EXPECT_LE(sum, 0.3 + 1e-6);
        EXPECT_GT(sum, 0.0);
    }
}

TEST(RandomStrategy, IntegerDtype) {
    auto cases = random_strategy(10, torch::kInt32);
    for (auto& tc : cases) {
        for (auto& e : tc.spec.discrete) {
            EXPECT_TRUE(e.value.is_valid_for_dtype( torch::kInt32));
        }
    }
}

TEST(RandomStrategy, EmptyForBoolIfNoValid) {
    auto cases = random_strategy(10, torch::kBool, 1.0, false);
    for (auto& tc : cases) {
        for (auto& e : tc.spec.discrete) {
            EXPECT_TRUE(e.value.is_valid_for_dtype( torch::kBool));
        }
    }
}

TEST(RandomStrategy, Reproducible) {
    auto gen1 = at::detail::createCPUGenerator();
    gen1.set_current_seed(123);
    auto cases1 = random_strategy(5, torch::kFloat32, 0.5, false, gen1);

    auto gen2 = at::detail::createCPUGenerator();
    gen2.set_current_seed(123);
    auto cases2 = random_strategy(5, torch::kFloat32, 0.5, false, gen2);

    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(cases1[i].spec.discrete.size(), cases2[i].spec.discrete.size());
        for (size_t j = 0; j < cases1[i].spec.discrete.size(); ++j) {
            EXPECT_EQ(cases1[i].spec.discrete[j].value, cases2[i].spec.discrete[j].value);
            EXPECT_DOUBLE_EQ(cases1[i].spec.discrete[j].probability,
                             cases2[i].spec.discrete[j].probability);
        }
    }
}

TEST(RandomStrategy, PMaxOne) {
    auto cases = random_strategy(10, torch::kFloat32, 1.0, true);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST(RandomStrategy, GeneratesValidTensors) {
    auto cases = random_strategy(10, torch::kFloat32, 0.5);
    for (auto& tc : cases) {
        EXPECT_NO_THROW(generate_tensor({100}, torch::kFloat32, tc.spec, tc.dist));
    }
}

// --- 2b: fixed_categories_strategy tests ------------------------------------

TEST(FixedCategoriesStrategy, UsesOnlySpecifiedCategories) {
    std::vector<SpecialValue> cats = {SpecialValue::NaN, SpecialValue::Inf};
    auto cases = fixed_categories_strategy(cats, 10, torch::kFloat32);
    for (auto& tc : cases) {
        ASSERT_EQ(tc.spec.discrete.size(), 2u);
        EXPECT_EQ(tc.spec.discrete[0].value, SpecialValue::NaN);
        EXPECT_EQ(tc.spec.discrete[1].value, SpecialValue::Inf);
    }
}

TEST(FixedCategoriesStrategy, StrictSumEqualsTarget) {
    std::vector<SpecialValue> cats = {SpecialValue::Zero, SpecialValue::One, SpecialValue::NOne};
    auto cases = fixed_categories_strategy(cats, 20, torch::kFloat32, 0.7, true);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        EXPECT_NEAR(sum, 0.7, 1e-6);
    }
}

TEST(FixedCategoriesStrategy, EmptyCategories) {
    auto cases = fixed_categories_strategy({}, 10, torch::kFloat32);
    EXPECT_TRUE(cases.empty());
}

TEST(FixedCategoriesStrategy, SingleCategory) {
    auto cases = fixed_categories_strategy({SpecialValue::NaN}, 5, torch::kFloat32, 0.5);
    for (auto& tc : cases) {
        ASSERT_EQ(tc.spec.discrete.size(), 1u);
        EXPECT_EQ(tc.spec.discrete[0].value, SpecialValue::NaN);
        EXPECT_LE(tc.spec.discrete[0].probability, 0.5 + 1e-9);
    }
}

// --- 2c: single_sv_strategy / dual_sv_strategy tests ------------------------

TEST(SingleSVStrategy, StrictMode) {
    auto cases = single_sv_strategy(SpecialValue::NaN, 10, torch::kFloat32, 0.3, true);
    for (auto& tc : cases) {
        ASSERT_EQ(tc.spec.discrete.size(), 1u);
        EXPECT_EQ(tc.spec.discrete[0].value, SpecialValue::NaN);
        EXPECT_DOUBLE_EQ(tc.spec.discrete[0].probability, 0.3);
    }
}

TEST(SingleSVStrategy, NonStrictBound) {
    auto cases = single_sv_strategy(SpecialValue::Inf, 20, torch::kFloat32, 0.5, false);
    for (auto& tc : cases) {
        EXPECT_LE(tc.spec.discrete[0].probability, 0.5 + 1e-9);
        EXPECT_GT(tc.spec.discrete[0].probability, 0.0);
    }
}

TEST(DualSVStrategy, StrictBoth) {
    auto cases = dual_sv_strategy(
        SpecialValue::NaN, SpecialValue::Inf, 10, torch::kFloat32,
        0.2, 0.3, true, true);
    for (auto& tc : cases) {
        ASSERT_EQ(tc.spec.discrete.size(), 2u);
        EXPECT_DOUBLE_EQ(tc.spec.discrete[0].probability, 0.2);
        EXPECT_DOUBLE_EQ(tc.spec.discrete[1].probability, 0.3);
    }
}

TEST(DualSVStrategy, NonStrictBound) {
    auto cases = dual_sv_strategy(
        SpecialValue::NaN, SpecialValue::Zero, 20, torch::kFloat32,
        0.4, 0.4, false, false);
    for (auto& tc : cases) {
        double r0 = tc.spec.discrete[0].probability;
        double r1 = tc.spec.discrete[1].probability;
        EXPECT_LE(r0 + r1, 1.0 + 1e-9);
        EXPECT_LE(r0, 0.4 + 1e-9);
        EXPECT_LE(r1, 0.4 + 1e-9);
    }
}

TEST(DualSVStrategy, ClampsCombinedToOne) {
    // When both strict and sum > 1.0, should throw
    EXPECT_THROW(
        dual_sv_strategy(SpecialValue::NaN, SpecialValue::Inf, 1,
                         torch::kFloat32, 0.9, 0.9, true, true),
        std::invalid_argument);

    // When only one is strict, non-strict side gets clamped
    auto cases = dual_sv_strategy(
        SpecialValue::NaN, SpecialValue::Inf, 20, torch::kFloat32,
        0.9, 0.9, true, false);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        EXPECT_LE(sum, 1.0 + 1e-9);
    }
}

TEST(DualSVStrategy, GeneratesValidTensors) {
    auto cases = dual_sv_strategy(
        SpecialValue::NaN, SpecialValue::Inf, 5, torch::kFloat32, 0.3, 0.3);
    for (auto& tc : cases) {
        EXPECT_NO_THROW(generate_tensor({100}, torch::kFloat32, tc.spec, tc.dist));
    }
}

// --- 2d-i: grid_single_value tests ------------------------------------------

TEST(GridSingleValue, FloatCoversAllValid) {
    auto cases = grid_single_value(torch::kFloat32);
    EXPECT_EQ((int)cases.size(), 10);
    for (auto& tc : cases) {
        ASSERT_EQ(tc.spec.discrete.size(), 1u);
        EXPECT_DOUBLE_EQ(tc.spec.discrete[0].probability, 1.0);
    }
}

TEST(GridSingleValue, IntOnlyValidSVs) {
    auto cases = grid_single_value(torch::kInt32);
    for (auto& tc : cases) {
        EXPECT_TRUE(tc.spec.discrete[0].value.is_valid_for_dtype(torch::kInt32));
    }
    EXPECT_EQ((int)cases.size(), 5);
}

TEST(GridSingleValue, BoolDtype) {
    auto cases = grid_single_value(torch::kBool);
    for (auto& tc : cases) {
        EXPECT_TRUE(tc.spec.discrete[0].value.is_valid_for_dtype(torch::kBool));
    }
}

TEST(GridSingleValue, AllDiscreteGenerates) {
    auto cases = grid_single_value(torch::kFloat32);
    for (auto& tc : cases) {
        auto t = generate_tensor({100}, torch::kFloat32, tc.spec, tc.dist);
        EXPECT_EQ(t.numel(), 100);
    }
}

// --- 2d-ii: grid_full_mix tests ---------------------------------------------

TEST(GridFullMix, MinCountSatisfied) {
    auto cases = grid_full_mix(torch::kFloat32, 50);
    EXPECT_GE((int)cases.size(), 50);
}

TEST(GridFullMix, ProportionsSumToOne) {
    auto cases = grid_full_mix(torch::kFloat32, 20);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        EXPECT_LE(sum, 1.0 + 1e-9);
        // Each proportion is a multiple of 0.1
        for (auto& e : tc.spec.discrete) {
            double rounded = std::round(e.probability * 10.0) / 10.0;
            EXPECT_NEAR(e.probability, rounded, 1e-9);
        }
    }
}

TEST(GridFullMix, UniqueConfigs) {
    auto cases = grid_full_mix(torch::kFloat32, 50);
    std::set<std::string> labels;
    for (auto& tc : cases) labels.insert(tc.label);
    EXPECT_EQ(labels.size(), cases.size());
}

TEST(GridFullMix, IntDtype) {
    auto cases = grid_full_mix(torch::kInt32, 20);
    EXPECT_GE((int)cases.size(), 20);
    for (auto& tc : cases) {
        for (auto& e : tc.spec.discrete)
            EXPECT_TRUE(e.value.is_valid_for_dtype( torch::kInt32));
    }
}

TEST(GridFullMix, Reproducible) {
    auto gen1 = at::detail::createCPUGenerator();
    gen1.set_current_seed(77);
    auto cases1 = grid_full_mix(torch::kFloat32, 30, gen1);

    auto gen2 = at::detail::createCPUGenerator();
    gen2.set_current_seed(77);
    auto cases2 = grid_full_mix(torch::kFloat32, 30, gen2);

    ASSERT_EQ(cases1.size(), cases2.size());
    for (size_t i = 0; i < cases1.size(); ++i) {
        EXPECT_EQ(cases1[i].label, cases2[i].label);
    }
}

TEST(GridFullMix, GeneratesValidTensors) {
    auto cases = grid_full_mix(torch::kFloat32, 10);
    for (auto& tc : cases) {
        EXPECT_NO_THROW(generate_tensor({50}, torch::kFloat32, tc.spec, tc.dist));
    }
}

// --- 2d-iii: grid_pairwise tests --------------------------------------------

TEST(GridPairwise, FivePerPair) {
    auto cases = grid_pairwise(torch::kFloat32);
    // C(10,2) = 45 pairs * 5 = 225
    EXPECT_EQ((int)cases.size(), 225);
}

TEST(GridPairwise, IntDtype) {
    auto cases = grid_pairwise(torch::kInt32);
    // C(5,2) = 10 pairs * 5 = 50
    EXPECT_EQ((int)cases.size(), 50);
}

TEST(GridPairwise, GridPoints25Pct) {
    auto cases = grid_pairwise(torch::kFloat32);
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) {
            sum += e.probability;
            // Each must be a multiple of 0.25
            double rounded = std::round(e.probability * 4.0) / 4.0;
            EXPECT_NEAR(e.probability, rounded, 1e-9);
        }
        // Total special value proportion is either 0.25, 0.5, 0.75, or 1.0
        double rounded_sum = std::round(sum * 4.0) / 4.0;
        EXPECT_NEAR(sum, rounded_sum, 1e-9);
    }
}

TEST(GridPairwise, AllDiscreteAtEndpoints) {
    auto cases = grid_pairwise(torch::kFloat32);
    int all_discrete_count = 0;
    for (auto& tc : cases) {
        double sum = 0;
        for (auto& e : tc.spec.discrete) sum += e.probability;
        if (sum >= 1.0 - 1e-9) all_discrete_count++;
    }
    // Each pair has (1.0, 0.0) and (0.0, 1.0) = 2 all-discrete configs
    // But (1.0, 0.0) means only 1 entry in discrete, (0.0, 1.0) same
    // Both are all-discrete since sum = 1.0
    EXPECT_GT(all_discrete_count, 0);
}

TEST(GridPairwise, GeneratesValidTensors) {
    auto cases = grid_pairwise(torch::kFloat32);
    for (auto& tc : cases) {
        EXPECT_NO_THROW(generate_tensor({50}, torch::kFloat32, tc.spec, tc.dist));
    }
}

TEST(GridPairwise, BoolDtype) {
    auto cases = grid_pairwise(torch::kBool);
    for (auto& tc : cases) {
        for (auto& e : tc.spec.discrete)
            EXPECT_TRUE(e.value.is_valid_for_dtype( torch::kBool));
    }
}

// ============================================================================
// Integration: strategy -> generate_tensor pipeline
// ============================================================================

TEST(StrategyIntegration, RandomStrategyPipeline) {
    auto cases = random_strategy(10, torch::kFloat32, 0.5);
    for (auto& tc : cases) {
        auto t = generate_tensor({1000}, torch::kFloat32, tc.spec, tc.dist);
        EXPECT_EQ(t.sizes(), std::vector<int64_t>{1000});
        EXPECT_EQ(t.dtype(), torch::kFloat32);
    }
}

TEST(StrategyIntegration, GridSingleValuePipeline) {
    auto cases = grid_single_value(torch::kFloat64);
    for (auto& tc : cases) {
        auto t = generate_tensor({500}, torch::kFloat64, tc.spec, tc.dist);
        EXPECT_EQ(t.numel(), 500);
    }
}

TEST(StrategyIntegration, GridFullMixPipeline) {
    auto cases = grid_full_mix(torch::kFloat32, 10);
    for (auto& tc : cases) {
        auto t = generate_tensor({200}, torch::kFloat32, tc.spec, tc.dist);
        EXPECT_FALSE(t.requires_grad());
    }
}

TEST(StrategyIntegration, LoopPattern) {
    std::vector<int64_t> shapes[] = {{100}, {10, 10}, {5, 5, 4}};
    auto cases = grid_pairwise(torch::kFloat32);

    int tested = 0;
    for (auto& shape : shapes) {
        for (auto& tc : cases) {
            auto t = generate_tensor(shape, torch::kFloat32, tc.spec, tc.dist);
            EXPECT_EQ(t.numel(), 100);
            tested++;
            if (tested > 30) break;
        }
        if (tested > 30) break;
    }
    EXPECT_GT(tested, 30);
}
