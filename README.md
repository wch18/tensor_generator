# tensor_generator

A C++ library for generating libtorch tensors with controlled distributions — combining discrete special values (NaN, Inf, dtype min/max, etc.) with continuous/integer base distributions (uniform/normal/randint).

Designed for testing numerical code that must handle edge cases across all PyTorch scalar types.

## Features

- **15 dtype support**: uint8/16/32/64, int8/16/32/64, float16/bfloat16/float32/float64, bool, float8_e4m3fn, float8_e5m2
- **Discrete special values**: Zero, One, NOne, DTypeMin, DTypeMax, NaN, Inf, NInf, Epsilon, Tiny
- **Base distributions**: Uniform, Normal, RandInt (with dtype compatibility validation)
- **Probability mixing**: Combine discrete values at specified probabilities with a continuous/integer background
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
./tensor_generator_test
```

## Usage

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

// RandInt for integer dtypes
auto t4 = generate_tensor({1000}, torch::kInt32,
    TensorSpec{{SpecialValue::Zero, 0.1}},
    BaseDist::randint(-100, 100));

// With torch::Generator (implicit conversion)
auto gen = at::detail::createCPUGenerator();
gen.set_current_seed(42);
auto t5 = generate_tensor({1000}, torch::kFloat32, TensorSpec{}, gen);

// Generator attached to a specific distribution
auto t6 = generate_tensor({1000}, torch::kFloat32, TensorSpec{},
    BaseDist::uniform(0.0, 1.0).with_generator(gen));

// With global seed
auto t7 = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NOne, 0.2}},
    BaseDist::uniform(-1.0, 1.0),
    /*seed=*/42);
```

## API

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
test/
  tensor_generator_test.cpp — 1700+ test cases
  test_configs.h            — shared test configurations
docs/
  design.md                — detailed design document
```

## License

Internal use.
