// test_rng_trajectory — verify deterministic RNG trajectories across calls.
//
// This is NOT a gtest. Run it twice: both runs must print identical output.
// Usage:
//   ./test_rng_trajectory           # run 1: prints checksums
//   ./test_rng_trajectory           # run 2: must match run 1 exactly
//
// Verifies the three RNG modes:
//   Mode 1: seed param on generate_tensor  → same tensor every call
//   Mode 2: no seed, no generator          → different tensors, non-reproducible
//   Mode 3: BaseDist with seed             → different tensors, reproducible

#include "tensor_generator.h"
#include <iostream>
#include <iomanip>

static double checksum(const torch::Tensor& t) {
    return t.to(torch::kFloat64).sum().item<double>();
}

static void print_checksums(const char* label, const torch::Tensor& t1,
                            const torch::Tensor& t2, const torch::Tensor& t3) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << label << ":\n"
              << "  call1=" << checksum(t1)
              << "  call2=" << checksum(t2)
              << "  call3=" << checksum(t3) << "\n";
}

int main() {
    auto shape = std::vector<int64_t>{1000};
    auto dtype = torch::kFloat32;
    auto spec  = TensorSpec{{SpecialValue::Zero, 0.3}};

    // ================================================================
    // Mode 1: seed on generate_tensor → every call identical
    // ================================================================
    auto m1_t1 = generate_tensor(shape, dtype, spec, BaseDist::uniform(0, 1), /*seed=*/42);
    auto m1_t2 = generate_tensor(shape, dtype, spec, BaseDist::uniform(0, 1), /*seed=*/42);
    auto m1_t3 = generate_tensor(shape, dtype, spec, BaseDist::uniform(0, 1), /*seed=*/42);

    bool m1_all_same = m1_t1.equal(m1_t2) && m1_t2.equal(m1_t3);
    std::cout << "[Mode 1] seed param: all identical = "
              << (m1_all_same ? "PASS" : "FAIL") << "\n";
    print_checksums("[Mode 1]", m1_t1, m1_t2, m1_t3);

    // ================================================================
    // Mode 3: BaseDist::uniform(..., seed) → deterministic but advancing
    // (test Mode 3 before Mode 2 so global RNG doesn't interfere)
    // ================================================================
    {
        auto dist = BaseDist::uniform(0.0, 1.0, /*seed=*/42);
        auto m3_t1 = generate_tensor(shape, dtype, spec, dist);
        auto m3_t2 = generate_tensor(shape, dtype, spec, dist);
        auto m3_t3 = generate_tensor(shape, dtype, spec, dist);

        bool m3_all_diff = !m3_t1.equal(m3_t2) && !m3_t2.equal(m3_t3) && !m3_t1.equal(m3_t3);
        std::cout << "\n[Mode 3] BaseDist with seed: all different = "
                  << (m3_all_diff ? "PASS" : "FAIL") << "\n";
        print_checksums("[Mode 3]", m3_t1, m3_t2, m3_t3);
        std::cout << "[Mode 3] (these checksums must be identical across program runs)\n";
    }

    // ================================================================
    // Mode 3b: two independent BaseDist with same seed → same sequence
    // ================================================================
    {
        auto dist_a = BaseDist::normal(0.0, 1.0, /*seed=*/99);
        auto dist_b = BaseDist::normal(0.0, 1.0, /*seed=*/99);

        auto a1 = generate_tensor(shape, dtype, spec, dist_a);
        auto a2 = generate_tensor(shape, dtype, spec, dist_a);
        auto b1 = generate_tensor(shape, dtype, spec, dist_b);
        auto b2 = generate_tensor(shape, dtype, spec, dist_b);

        bool match = a1.equal(b1) && a2.equal(b2);
        std::cout << "\n[Mode 3b] two generators same seed: sequences match = "
                  << (match ? "PASS" : "FAIL") << "\n";
        print_checksums("[Mode 3b]", a1, a2, b1);
        std::cout << "[Mode 3b] (these checksums must be identical across program runs)\n";
    }

    // ================================================================
    // Mode 3c: all distribution types with seed
    // ================================================================
    {
        std::cout << "\n[Mode 3c] all dist types with seed:\n";
        auto spec_empty = TensorSpec{};
        auto int_spec = TensorSpec{{42, 0.2}};

        auto u = generate_tensor(shape, dtype, spec_empty, BaseDist::uniform(-1, 1, 7));
        auto n = generate_tensor(shape, dtype, spec_empty, BaseDist::normal(0, 1, 7));
        auto r = generate_tensor(shape, torch::kInt32, int_spec, BaseDist::randint(0, 100, 7));
        auto e = generate_tensor(shape, dtype, spec_empty, BaseDist::exponential(1.0, 7));
        auto p = generate_tensor(shape, torch::kInt32, int_spec, BaseDist::poisson(5.0, 7));
        auto l = generate_tensor(shape, dtype, spec_empty, BaseDist::lognormal(0, 1, 7));
        auto c = generate_tensor(shape, dtype, spec_empty, BaseDist::cauchy(0, 1, 7));
        auto b = generate_tensor(shape, dtype, spec_empty, BaseDist::bernoulli(0.5, 7));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  uniform="     << checksum(u)
                  << "  normal="      << checksum(n)
                  << "  randint="     << checksum(r)
                  << "  exponential=" << checksum(e)
                  << "  poisson="     << checksum(p)
                  << "  lognormal="   << checksum(l)
                  << "  cauchy="      << checksum(c)
                  << "  bernoulli="   << checksum(b) << "\n";
        std::cout << "  (these checksums must be identical across program runs)\n";
    }

    // ================================================================
    // Mode 2: no seed, no generator → non-reproducible
    // (intentionally last — no determinism claim, just show they differ)
    // ================================================================
    auto m2_t1 = generate_tensor(shape, dtype, spec, BaseDist::uniform(0, 1));
    auto m2_t2 = generate_tensor(shape, dtype, spec, BaseDist::uniform(0, 1));
    auto m2_t3 = generate_tensor(shape, dtype, spec, BaseDist::uniform(0, 1));

    bool m2_all_diff = !m2_t1.equal(m2_t2) && !m2_t2.equal(m2_t3);
    std::cout << "\n[Mode 2] no seed: all different = "
              << (m2_all_diff ? "PASS" : "FAIL") << "\n";
    print_checksums("[Mode 2]", m2_t1, m2_t2, m2_t3);
    std::cout << "[Mode 2] (these checksums will differ across program runs)\n";

    return 0;
}
