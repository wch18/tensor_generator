#pragma once

#include "tensor_spec.h"
#include <string>
#include <vector>
#include <c10/util/Optional.h>

// ============================================================================
// Strategy generators — produce vectors of TestCase that can be iterated to
// generate tensors with generate_tensor(shape, dtype, tc.spec, tc.dist).
// ============================================================================

struct TestCase {
    TensorSpec spec;
    BaseDist dist;
    std::string label;
};

// --- 2a: Fully random strategy ----------------------------------------------
// Randomly selects SpecialValue types AND ratios.
// exclude: SpecialValues to exclude from the candidate pool.
// p_max: total probability ceiling (0 < p_max <= 1).
// strict: if true, probabilities sum to exactly p_max; otherwise <= p_max.
std::vector<TestCase> random_strategy(
    int count,
    torch::ScalarType dtype,
    const std::vector<SpecialValue>& exclude = {},
    double p_max = 1.0,
    bool strict = false,
    c10::optional<torch::Generator> generator = c10::nullopt);

// --- 2b: Fixed categories strategy ------------------------------------------
// User specifies which SpecialValues to use; ratios are randomized.
std::vector<TestCase> fixed_categories_strategy(
    const std::vector<SpecialValue>& categories,
    int count,
    torch::ScalarType dtype,
    double p_max = 1.0,
    bool strict = false,
    c10::optional<torch::Generator> generator = c10::nullopt);

// --- 2c: Single/dual category strategy --------------------------------------
// Named wrappers with explicit probability targets per category.
// p: target probability for the category.
// strict: if true, probability is exactly p; otherwise random in (0, p].
std::vector<TestCase> single_sv_strategy(
    SpecialValue sv,
    int count,
    torch::ScalarType dtype,
    double p = 0.5,
    bool strict = false,
    c10::optional<torch::Generator> generator = c10::nullopt);

std::vector<TestCase> dual_sv_strategy(
    SpecialValue sv1, SpecialValue sv2,
    int count,
    torch::ScalarType dtype,
    double p0 = 0.3, double p1 = 0.3,
    bool strict0 = false, bool strict1 = false,
    c10::optional<torch::Generator> generator = c10::nullopt);

// --- 2d: Grid strategies ----------------------------------------------------

// 2d-i: One TestCase per valid SpecialValue at 100% probability.
std::vector<TestCase> grid_single_value(torch::ScalarType dtype);

// 2d-ii: All types (valid SVs + normal) mixed at 10% grid granularity.
// Randomly samples >= min_count unique grid-aligned combinations.
std::vector<TestCase> grid_full_mix(
    torch::ScalarType dtype,
    int min_count = 50,
    c10::optional<torch::Generator> generator = c10::nullopt);

// 2d-iii: Pairwise combinations at 25% grid granularity.
// For each valid pair, generates 5 configs: (100,0), (75,25), (50,50), (25,75), (0,100).
std::vector<TestCase> grid_pairwise(torch::ScalarType dtype);
