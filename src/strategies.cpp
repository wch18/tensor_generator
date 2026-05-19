#include "strategies.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

// All SpecialValue members in enum order.
static const SpecialValue kAllSpecialValues[] = {
    SpecialValue::Zero, SpecialValue::One, SpecialValue::NOne,
    SpecialValue::DTypeMin, SpecialValue::DTypeMax,
    SpecialValue::NaN, SpecialValue::Inf, SpecialValue::NInf,
    SpecialValue::Epsilon, SpecialValue::Tiny,
};

static const char* sv_short_name(SpecialValue sv) {
    switch (sv) {
    case SpecialValue::Zero: return "zero";
    case SpecialValue::One: return "one";
    case SpecialValue::NOne: return "none";
    case SpecialValue::DTypeMin: return "min";
    case SpecialValue::DTypeMax: return "max";
    case SpecialValue::NaN: return "nan";
    case SpecialValue::Inf: return "inf";
    case SpecialValue::NInf: return "ninf";
    case SpecialValue::Epsilon: return "epsilon";
    case SpecialValue::Tiny: return "tiny";
    }
    return "unknown";
}

static std::vector<SpecialValue> valid_svs_for_dtype(torch::ScalarType dtype) {
    std::vector<SpecialValue> result;
    for (auto sv : kAllSpecialValues) {
        if (is_valid_for_dtype(sv, dtype))
            result.push_back(sv);
    }
    return result;
}

static torch::Generator ensure_gen(c10::optional<torch::Generator> gen) {
    if (gen.has_value()) return *gen;
    return torch::Generator();
}

static double rand_double(torch::Generator& gen) {
    auto t = gen.defined()
        ? torch::rand({1}, gen, torch::TensorOptions().dtype(torch::kFloat64))
        : torch::rand({1}, torch::TensorOptions().dtype(torch::kFloat64));
    return t.item<double>();
}

static int64_t rand_int(torch::Generator& gen, int64_t lo, int64_t hi) {
    if (lo >= hi) return lo;
    auto t = gen.defined()
        ? torch::randint(lo, hi, {1}, gen, torch::TensorOptions().dtype(torch::kInt64))
        : torch::randint(lo, hi, {1}, torch::TensorOptions().dtype(torch::kInt64));
    return t.item<int64_t>();
}

// Generate k random proportions that sum to target using Dirichlet(1,...,1).
static std::vector<double> random_proportions(
    int k, double target, torch::Generator& gen) {
    if (k <= 0) return {};
    std::vector<double> vals(k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        double u = rand_double(gen);
        vals[i] = -std::log(std::max(u, 1e-10));
        sum += vals[i];
    }
    for (int i = 0; i < k; ++i)
        vals[i] = vals[i] / sum * target;
    return vals;
}

static std::string format_label(const std::vector<std::pair<SpecialValue, double>>& entries,
                                double normal_ratio) {
    std::ostringstream oss;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) oss << "+";
        oss << sv_short_name(entries[i].first) << "="
            << static_cast<int>(std::round(entries[i].second * 100)) << "%";
    }
    if (normal_ratio > 0.001) {
        oss << "+normal=" << static_cast<int>(std::round(normal_ratio * 100)) << "%";
    }
    return oss.str();
}

// --- 2a: Fully random -------------------------------------------------------

std::vector<TestCase> random_strategy(
    int count,
    torch::ScalarType dtype,
    double p_max,
    bool strict,
    c10::optional<torch::Generator> generator,
    const std::vector<SpecialValue>& exclude)
{
    auto gen = ensure_gen(generator);
    auto valid = valid_svs_for_dtype(dtype);
    if (!exclude.empty()) {
        valid.erase(
            std::remove_if(valid.begin(), valid.end(),
                [&](SpecialValue sv) {
                    return std::find(exclude.begin(), exclude.end(), sv)
                           != exclude.end();
                }),
            valid.end());
    }
    if (valid.empty()) return {};

    std::vector<TestCase> result;
    result.reserve(count);

    for (int i = 0; i < count; ++i) {
        int k = static_cast<int>(rand_int(gen, 1, (int64_t)valid.size() + 1));

        // Shuffle and pick first k
        std::vector<SpecialValue> chosen(valid.begin(), valid.end());
        for (int j = (int)chosen.size() - 1; j > 0; --j) {
            int idx = static_cast<int>(rand_int(gen, 0, j + 1));
            std::swap(chosen[j], chosen[idx]);
        }
        chosen.resize(k);

        double target = strict ? p_max : rand_double(gen) * p_max;
        double min_target = std::min(0.001, p_max);
        if (target < min_target) target = min_target;
        auto proportions = random_proportions(k, target, gen);

        TensorSpec spec;
        std::vector<std::pair<SpecialValue, double>> entries;
        for (int j = 0; j < k; ++j) {
            spec.discrete.push_back({chosen[j], proportions[j]});
            entries.push_back({chosen[j], proportions[j]});
        }

        double normal_ratio = 1.0 - target;
        TestCase tc;
        tc.spec = std::move(spec);
        tc.dist = BaseDist::uniform(0.0, 1.0);
        tc.label = "random[" + std::to_string(i) + "] " + format_label(entries, normal_ratio);
        result.push_back(std::move(tc));
    }
    return result;
}

// --- 2b: Fixed categories ----------------------------------------------------

std::vector<TestCase> fixed_categories_strategy(
    const std::vector<SpecialValue>& categories,
    int count,
    torch::ScalarType dtype,
    double p_max,
    bool strict,
    c10::optional<torch::Generator> generator)
{
    auto gen = ensure_gen(generator);
    int k = (int)categories.size();
    if (k == 0) return {};

    std::vector<TestCase> result;
    result.reserve(count);

    for (int i = 0; i < count; ++i) {
        double target = strict ? p_max : rand_double(gen) * p_max;
        double min_target = std::min(0.001, p_max);
        if (target < min_target) target = min_target;
        auto proportions = random_proportions(k, target, gen);

        TensorSpec spec;
        std::vector<std::pair<SpecialValue, double>> entries;
        for (int j = 0; j < k; ++j) {
            spec.discrete.push_back({categories[j], proportions[j]});
            entries.push_back({categories[j], proportions[j]});
        }

        double normal_ratio = 1.0 - target;
        TestCase tc;
        tc.spec = std::move(spec);
        tc.dist = BaseDist::uniform(0.0, 1.0);
        tc.label = "fixed_cat[" + std::to_string(i) + "] " + format_label(entries, normal_ratio);
        result.push_back(std::move(tc));
    }
    return result;
}

// --- 2c: Single/dual --------------------------------------------------------

std::vector<TestCase> single_sv_strategy(
    SpecialValue sv,
    int count,
    torch::ScalarType dtype,
    double p,
    bool strict,
    c10::optional<torch::Generator> generator)
{
    auto gen = ensure_gen(generator);
    std::vector<TestCase> result;
    result.reserve(count);

    for (int i = 0; i < count; ++i) {
        double ratio = strict ? p : rand_double(gen) * p;
        if (!strict && ratio < std::min(0.001, p)) ratio = std::min(0.001, p);

        TensorSpec spec;
        spec.discrete.push_back({sv, ratio});

        TestCase tc;
        tc.spec = std::move(spec);
        tc.dist = BaseDist::uniform(0.0, 1.0);
        tc.label = std::string("single[") + sv_short_name(sv) + "] " +
                   sv_short_name(sv) + "=" +
                   std::to_string(static_cast<int>(std::round(ratio * 100))) + "%";
        result.push_back(std::move(tc));
    }
    return result;
}

std::vector<TestCase> dual_sv_strategy(
    SpecialValue sv1, SpecialValue sv2,
    int count,
    torch::ScalarType dtype,
    double p0, double p1,
    bool strict0, bool strict1,
    c10::optional<torch::Generator> generator)
{
    if (strict0 && strict1 && p0 + p1 > 1.0 + 1e-9)
        throw std::invalid_argument(
            "dual_sv_strategy: strict p0 + p1 exceeds 1.0");

    auto gen = ensure_gen(generator);
    std::vector<TestCase> result;
    result.reserve(count);

    for (int i = 0; i < count; ++i) {
        double r0 = strict0 ? p0 : rand_double(gen) * p0;
        double r1 = strict1 ? p1 : rand_double(gen) * p1;
        if (!strict0 && r0 < std::min(0.001, p0)) r0 = std::min(0.001, p0);
        if (!strict1 && r1 < std::min(0.001, p1)) r1 = std::min(0.001, p1);

        if (r0 + r1 > 1.0) {
            double scale = 1.0 / (r0 + r1);
            r0 *= scale;
            r1 *= scale;
        }

        TensorSpec spec;
        spec.discrete.push_back({sv1, r0});
        spec.discrete.push_back({sv2, r1});

        TestCase tc;
        tc.spec = std::move(spec);
        tc.dist = BaseDist::uniform(0.0, 1.0);
        tc.label = std::string("dual[") + sv_short_name(sv1) + "+" +
                   sv_short_name(sv2) + "] " +
                   sv_short_name(sv1) + "=" +
                   std::to_string(static_cast<int>(std::round(r0 * 100))) + "% " +
                   sv_short_name(sv2) + "=" +
                   std::to_string(static_cast<int>(std::round(r1 * 100))) + "%";
        result.push_back(std::move(tc));
    }
    return result;
}

// --- 2d-i: Single value grid ------------------------------------------------

std::vector<TestCase> grid_single_value(torch::ScalarType dtype) {
    auto valid = valid_svs_for_dtype(dtype);
    std::vector<TestCase> result;
    result.reserve(valid.size());

    for (auto sv : valid) {
        TensorSpec spec;
        spec.discrete.push_back({sv, 1.0});

        TestCase tc;
        tc.spec = std::move(spec);
        tc.dist = BaseDist::uniform(0.0, 1.0);
        tc.label = std::string("grid_single[") + sv_short_name(sv) + "=100%]";
        result.push_back(std::move(tc));
    }
    return result;
}

// --- 2d-ii: Full mix grid (10% granularity) ---------------------------------

std::vector<TestCase> grid_full_mix(
    torch::ScalarType dtype,
    int min_count,
    c10::optional<torch::Generator> generator)
{
    auto gen = ensure_gen(generator);
    auto valid = valid_svs_for_dtype(dtype);
    int k = (int)valid.size() + 1; // +1 for normal

    // Represent each configuration as a vector of slot counts (sum = 10).
    // Use a set of string keys to deduplicate.
    std::set<std::string> seen;
    std::vector<TestCase> result;

    int max_attempts = min_count * 20;
    for (int attempt = 0; attempt < max_attempts && (int)result.size() < min_count; ++attempt) {
        // Distribute 10 units among k categories by random assignment
        std::vector<int> slots(k, 0);
        for (int u = 0; u < 10; ++u) {
            int bin = static_cast<int>(rand_int(gen, 0, (int64_t)k));
            slots[bin]++;
        }

        // Build dedup key
        std::string key;
        for (int s : slots) key += std::to_string(s) + ",";
        if (seen.count(key)) continue;
        seen.insert(key);

        // Build TestCase — slots[0..k-2] are SVs, slots[k-1] is normal
        TensorSpec spec;
        std::vector<std::pair<SpecialValue, double>> entries;
        for (int j = 0; j < (int)valid.size(); ++j) {
            if (slots[j] > 0) {
                double p = slots[j] * 0.1;
                spec.discrete.push_back({valid[j], p});
                entries.push_back({valid[j], p});
            }
        }
        double normal_ratio = slots[k - 1] * 0.1;

        TestCase tc;
        tc.spec = std::move(spec);
        tc.dist = BaseDist::uniform(0.0, 1.0);
        tc.label = "grid_mix " + format_label(entries, normal_ratio);
        result.push_back(std::move(tc));
    }
    return result;
}

// --- 2d-iii: Pairwise grid (25% granularity) --------------------------------

std::vector<TestCase> grid_pairwise(torch::ScalarType dtype) {
    auto valid = valid_svs_for_dtype(dtype);
    // 5 grid points per pair: (100,0), (75,25), (50,50), (25,75), (0,100)
    static const double kGrid[][2] = {
        {1.00, 0.00}, {0.75, 0.25}, {0.50, 0.50}, {0.25, 0.75}, {0.00, 1.00}
    };

    std::vector<TestCase> result;
    for (int i = 0; i < (int)valid.size(); ++i) {
        for (int j = i + 1; j < (int)valid.size(); ++j) {
            for (auto& g : kGrid) {
                TensorSpec spec;
                std::vector<std::pair<SpecialValue, double>> entries;
                if (g[0] > 0.001) {
                    spec.discrete.push_back({valid[i], g[0]});
                    entries.push_back({valid[i], g[0]});
                }
                if (g[1] > 0.001) {
                    spec.discrete.push_back({valid[j], g[1]});
                    entries.push_back({valid[j], g[1]});
                }

                TestCase tc;
                tc.spec = std::move(spec);
                tc.dist = BaseDist::uniform(0.0, 1.0);
                tc.label = std::string("grid_pair[") + sv_short_name(valid[i]) + "+" +
                           sv_short_name(valid[j]) + "] " + format_label(entries, 0.0);
                result.push_back(std::move(tc));
            }
        }
    }
    return result;
}
