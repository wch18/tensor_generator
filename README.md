# tensor_generator

A C++ library for generating libtorch tensors with controlled distributions — combining discrete special values (NaN, Inf, dtype min/max, etc.) with continuous/integer base distributions (uniform/normal/randint).

Designed for testing numerical code that must handle edge cases across all PyTorch scalar types.

## Features

- **15 dtype support**: uint8/16/32/64, int8/16/32/64, float16/bfloat16/float32/float64, bool, float8_e4m3fn, float8_e5m2
- **Discrete special values**: Zero, One, NOne, DTypeMin, DTypeMax, NaN, Inf, NInf, Epsilon, Tiny
- **Base distributions**: Uniform, Normal, RandInt (with dtype compatibility validation)
- **Probability mixing**: Combine discrete values at specified probabilities with a continuous/integer background
- **Convenience APIs**: One-liner functions for common patterns (`tensor_with_nan`, `tensor_with_inf_nan`, etc.)
- **Strategy generators**: Produce test case configurations for systematic coverage (random, grid, pairwise)
- **torch::Generator support**: BaseDist accepts a Generator for isolated RNG state; implicit conversion from Generator
- **Reproducible**: Optional seed parameter or per-distribution Generator
- **Dtype-aware validation**: Rejects invalid combinations at runtime (e.g., NaN on int, RandInt on float)

## Requirements

| Dependency | Version |
|-----------|---------|
| PyTorch (libtorch) | >= 2.9 |
| C++ standard | C++17 |
| CMake | >= 3.18 |
| GoogleTest | (for tests only) |

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j$(nproc)
```

Run tests:

```bash
./tensor_generator_test              # Core tests (1700+)
./test_convenience_strategies        # Convenience & strategy tests (60)
```

## Quick Start

```cpp
#include "tensor_generator.h"

// Pure uniform in [-1, 1] (default)
auto t1 = generate_tensor({1000}, torch::kFloat32, TensorSpec{});

// 30% zeros, rest uniform in [-5, 5]
auto t2 = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::Zero, 0.3}},
    BaseDist::uniform(-5.0, 5.0));

// 20% NaN + 10% Inf, rest normal(0, 1)
auto t3 = generate_tensor({1000}, torch::kFloat64,
    TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Inf, 0.1}},
    BaseDist::normal(0.0, 1.0));
```

## Convenience APIs

```cpp
#include "convenience.h"

// --- Base distribution generators ---
auto t1 = make_uniform_tensor({1000}, torch::kFloat32, 0.0, 1.0);
auto t2 = make_normal_tensor({1000}, torch::kFloat64, 0.0, 1.0);
auto t3 = make_randint_tensor({1000}, torch::kInt32, -100, 100);

// --- Single special value (10 named functions) ---
auto t4 = tensor_with_nan({1000}, torch::kFloat32, 0.1);
auto t5 = tensor_with_inf({1000}, torch::kFloat32, 0.05);
auto t6 = tensor_with_zero({1000}, torch::kFloat32, 0.2);
// Also: tensor_with_one, tensor_with_none, tensor_with_min, tensor_with_max,
//       tensor_with_ninf, tensor_with_epsilon, tensor_with_tiny

// --- Generic single SV ---
auto t7 = tensor_with_one_sv(SpecialValue::NaN, {1000}, torch::kFloat32, 0.1);

// --- Two special values (45 named functions) ---
auto t8 = tensor_with_inf_nan({1000}, torch::kFloat32, 0.1, 0.05);
auto t9 = tensor_with_max_min({1000}, torch::kInt32, 0.2, 0.2);
// All C(10,2) = 45 pairs available

// --- Generic two SVs ---
auto t10 = tensor_with_two_sv(
    SpecialValue::NaN, 0.1, SpecialValue::Inf, 0.05,
    {1000}, torch::kFloat32);

// --- With generator for reproducibility ---
auto gen = at::detail::createCPUGenerator();
gen.set_current_seed(42);
auto t11 = tensor_with_nan({1000}, torch::kFloat32, 0.1, gen);
```

## Strategy Generators

```cpp
#include "strategies.h"

// Strategy generators produce TestCase vectors for systematic iteration:
//   struct TestCase { TensorSpec spec; BaseDist dist; std::string label; };

// --- 2a: Fully random (types & ratios randomized) ---
auto cases = random_strategy(50, torch::kFloat32, /*p_max=*/0.5, /*strict=*/false);

// --- 2b: Fixed categories (user picks SVs, ratios randomized) ---
auto cases = fixed_categories_strategy(
    {SpecialValue::NaN, SpecialValue::Inf, SpecialValue::Zero},
    20, torch::kFloat32, /*p_max=*/0.8, /*strict=*/true);

// --- 2c: Single/dual with target probability ---
auto cases = single_sv_strategy(SpecialValue::NaN, 10, torch::kFloat32,
    /*p=*/0.3, /*strict=*/true);
auto cases = dual_sv_strategy(SpecialValue::NaN, SpecialValue::Inf,
    10, torch::kFloat32,
    /*p0=*/0.2, /*p1=*/0.1, /*strict0=*/false, /*strict1=*/true);

// --- 2d: Grid strategies ---
auto single = grid_single_value(torch::kFloat32);       // 10 cases (one per SV at 100%)
auto mixed  = grid_full_mix(torch::kFloat32, 50);       // >=50 unique 10%-grid configs
auto pairs  = grid_pairwise(torch::kFloat32);           // 225 cases (45 pairs × 5 grid points)

// --- Iterate to generate tensors ---
for (auto& tc : cases) {
    auto tensor = generate_tensor({1000}, torch::kFloat32, tc.spec, tc.dist);
    // test your numerical code with tensor...
}
```

## Core API

```cpp
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},              // default: Uniform(-1, 1)
    std::optional<int64_t> seed = std::nullopt);
```

**BaseDist construction:**

```cpp
BaseDist{}                          // Uniform(-1, 1)
BaseDist::uniform(lo, hi)           // Uniform [lo, hi]
BaseDist::normal(mean, std)         // Normal N(mean, std)
BaseDist::randint(lo, hi)           // Integer uniform [lo, hi)
BaseDist(torch::Generator gen)      // Implicit: Uniform(-1,1) + generator
dist.with_generator(gen)            // Attach generator to any distribution
```

**Dtype/BaseDist compatibility:**

| BaseDist | Float dtypes | Int dtypes | Bool |
|----------|:---:|:---:|:---:|
| Uniform | OK | OK (truncated) | OK |
| Normal | OK | OK (truncated) | OK |
| RandInt | ERROR | OK | ERROR |

**SpecialValue enum:**

| Name | Description |
|------|-------------|
| `Zero` | Additive identity |
| `One` | Multiplicative identity |
| `NOne` | -1 (unsigned: 0xFF...FF) |
| `DTypeMin` | Dtype representable minimum |
| `DTypeMax` | Dtype representable maximum |
| `NaN` | Quiet NaN (float only) |
| `Inf` | +infinity (float, not e4m3fn) |
| `NInf` | -infinity (float, not e4m3fn) |
| `Epsilon` | Machine epsilon (float only) |
| `Tiny` | Smallest positive normal (float only) |

## Project Structure

```
src/
  tensor_generator.h/.cpp  — public API (generate_tensor)
  tensor_spec.h            — TensorSpec, BaseDist, DiscreteEntry
  special_value.h/.cpp     — SpecialValue enum and dtype-aware value table
  dtype_traits.h           — internal dtype classification helpers
  convenience.h            — convenience wrapper APIs (header-only)
  strategies.h/.cpp        — strategy generators (TestCase, random/grid/pairwise)
test/
  tensor_generator_test.cpp         — 1700+ core test cases
  test_configs.h                    — shared test configurations
  test_convenience_strategies.cpp   — 60 convenience & strategy tests
docs/
  design.md                — detailed design document
```

## License

Internal use.
