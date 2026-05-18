# tensor_generator

A C++ library for generating libtorch tensors with controlled distributions — combining discrete special values (NaN, Inf, dtype min/max, etc.) with continuous base distributions (uniform/normal).

Designed for testing numerical code that must handle edge cases across all PyTorch scalar types.

## Features

- **15 dtype support**: uint8/16/32/64, int8/16/32/64, float16/bfloat16/float32/float64, bool, float8_e4m3fn, float8_e5m2
- **Discrete special values**: Zero, One, NegativeOne, DTypeMin, DTypeMax, NaN, ±Infinity, Epsilon, Tiny
- **Continuous base distributions**: Uniform and Normal, with configurable parameters
- **Probability mixing**: Combine discrete values at specified probabilities with a continuous background
- **Reproducible**: Optional seed parameter for deterministic generation
- **Dtype-aware validation**: Rejects invalid combinations (e.g., NaN on integer types) at runtime

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
using namespace tgen;

// Pure uniform in [-1, 1] (default base distribution)
auto t1 = generate_tensor({1000}, torch::kFloat32, TensorSpec{});

// 30% zeros, rest uniform in [-5, 5]
auto t2 = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::Zero, 0.3}},
    BaseDist::uniform(-5.0, 5.0));

// 20% NaN + 10% Inf, rest normal(0, 1)
auto t3 = generate_tensor({1000}, torch::kFloat64,
    TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Infinity, 0.1}},
    BaseDist::normal(0.0, 1.0));

// All-discrete: 50% dtype-min, 50% dtype-max
auto t4 = generate_tensor({1000}, torch::kInt32,
    TensorSpec{{SpecialValue::DTypeMin, 0.5}, {SpecialValue::DTypeMax, 0.5}});

// With explicit seed for reproducibility
auto t5 = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::Zero, 0.1}},
    BaseDist::uniform(-1.0, 1.0),
    /*seed=*/42);
```

## API

```cpp
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},          // default: Uniform(-1, 1)
    std::optional<int64_t> seed = std::nullopt);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `shape` | Tensor dimensions |
| `dtype` | Target scalar type |
| `spec` | Discrete entries: list of `{SpecialValue, probability}` pairs (sum <= 1.0) |
| `base_dist` | Continuous distribution for remaining probability mass. Default: `Uniform(-1, 1)` |
| `seed` | Optional RNG seed for reproducibility |

**BaseDist construction:**

```cpp
BaseDist::uniform(lo, hi)   // Uniform distribution in [lo, hi]
BaseDist::normal(mean, std) // Normal distribution N(mean, std)
BaseDist{}                  // Default: Uniform(-1, 1)
```

## Project Structure

```
src/
  tensor_generator.h/.cpp  — public API (generate_tensor)
  tensor_spec.h            — TensorSpec, BaseDist, DiscreteEntry definitions
  special_value.h/.cpp     — SpecialValue enum and dtype-aware value table
  dtype_traits.h           — internal dtype classification helpers
test/
  tensor_generator_test.cpp — 1600+ test cases (cartesian product + focused)
  test_configs.h            — shared test configurations
docs/
  design.md                — detailed design document
```

## License

Internal use.
