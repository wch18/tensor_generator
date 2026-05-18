# tensor_generator Design Document

## Architecture Overview

The library is split into three modules, each with a single responsibility:

```
src/
  special_value.h / special_value.cpp   — dtype-aware special value table
  tensor_spec.h                         — declarative distribution spec (header-only)
  tensor_generator.h / tensor_generator.cpp — generation engine (public API)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `special_value` | Maps `SpecialValue` enum members to exact numeric representations per dtype; validates dtype/value compatibility |
| `tensor_spec` | Aggregates a list of discrete `(SpecialValue, probability)` pairs with an optional continuous base distribution into a single `TensorSpec` struct |
| `tensor_generator` | Implements the generation algorithm; exposes `generate_tensor()` |

The dependency graph is strictly layered:

```
tensor_generator → tensor_spec → special_value → <torch/torch.h>
```

There are no circular dependencies.

---

## Public API

```cpp
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},            // default: Uniform(-1, 1)
    std::optional<int64_t> seed = std::nullopt);
```

`TensorSpec` contains only discrete entries. `BaseDist` is a separate parameter
controlling the continuous base distribution. `seed` enables reproducibility
without requiring the caller to manage global RNG state.

---

## The "Base Layer + Mask Overlay" Algorithm

`generate_tensor(shape, dtype, spec, base_dist, seed)` works in three phases:

### Phase 1 — Base Tensor

Let `p_sum = sum of all discrete entry probabilities`.

- If `p_sum >= 1.0 - 1e-9` (all-discrete): allocate a zero-initialised tensor in `dtype` (via a safe intermediate type — see "Cast Workarounds" below).
- Otherwise: sample a continuous base tensor using the `BaseDist` (uniform `U(lo, hi)` or normal `N(mu, sigma)`) in `float64`, then cast to `dtype`.

### Phase 2 — Selector Tensor

Draw a `float64` uniform tensor `selector` in `[0, 1)` with the same `shape`. This is independent of the base tensor and is reused across all discrete entries.

### Phase 3 — Mask Overlay

Walk the discrete entries in order, tracking a running cumulative probability `cum`:

```
for each entry (value, p) in spec.discrete:
    if p == 0: skip
    if this is the last non-zero entry AND all_discrete:
        mask = (selector >= cum)          # absorb any rounding slack
    else:
        mask = (selector >= cum) AND (selector < cum + p)
    masked_fill_(result, mask, value)
    cum += p
```

The special handling of the last entry ensures that when probabilities are specified as decimal fractions (e.g. `0.1 + 0.2 + 0.3 + 0.4`), whose floating-point sum may exceed or fall short of exactly `1.0`, no element is left at its initialisation value by accident.

---

## SpecialValue Table

The table below shows the concrete value written into the tensor for each `SpecialValue` enum member, per dtype. "two's complement wrap" means the bit pattern of `-1` in the signed domain, reinterpreted as the unsigned type.

| SpecialValue | uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 | bool | fp8_e4m3fn | fp8_e5m2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | false | 0.0 | 0.0 |
| One | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | 1.0 |
| NegativeOne | 255 | 65535 | 4294967295 | 2^64-1 | -1 | -1 | -1 | -1 | -1.0 | -1.0 | -1.0 | -1.0 | true | -1.0 | -1.0 |
| DTypeMin | 0 | 0 | 0 | 0 | -128 | -32768 | INT32_MIN | INT64_MIN | -65504.0 | -3.39e+38 | -FLT_MAX | -DBL_MAX | false | -448.0 | -57344.0 |
| DTypeMax | 255 | 65535 | 4294967295 | 2^64-1 | 127 | 32767 | INT32_MAX | INT64_MAX | 65504.0 | 3.39e+38 | FLT_MAX | DBL_MAX | true | 448.0 | 57344.0 |
| NaN | — | — | — | — | — | — | — | — | NaN | NaN | NaN | NaN | — | NaN | NaN |
| Infinity | — | — | — | — | — | — | — | — | +inf | +inf | +inf | +inf | — | — | +inf |
| NegativeInfinity | — | — | — | — | — | — | — | — | -inf | -inf | -inf | -inf | — | — | -inf |
| Epsilon | — | — | — | — | — | — | — | — | 9.77e-4 | 7.81e-3 | FLT_EPSILON | DBL_EPSILON | — | 0.125 | 0.25 |
| Tiny | — | — | — | — | — | — | — | — | 6.10e-5 | FLT_MIN | FLT_MIN | DBL_MIN | — | 0.015625 | 6.10e-5 |

"—" means the value is not valid for that dtype (`is_valid_for_dtype()` returns `false`).

---

## Edge Case Handling

### Floating-Point Accumulation in All-Discrete Specs

When all discrete probabilities sum to exactly `1.0` in theory, their floating-point representations may sum to slightly more or less than `1.0`. The naive range check `selector < cum` can leave a tiny sliver of the selector range unassigned. The fix is to use `selector >= cum` (without an upper bound) for the last non-zero entry, absorbing any rounding slack unconditionally.

### uint64 Two's Complement

PyTorch does not support `masked_fill_` on `uint64` tensors directly. The workaround casts the result tensor to `int64`, fills via `masked_fill_` (the int64 bit pattern is identical due to two's complement), then casts back to `uint64`. This correctly represents `NegativeOne` as `2^64 - 1` and `DTypeMax` as the same value.

### FP8 (float8_e4m3fn / float8_e5m2) Cast Workaround

PyTorch's `masked_fill_` does not accept a scalar when the target dtype is FP8. The workaround promotes the tensor to `float64`, performs the fill, then casts back. This preserves the exact representable FP8 value since all special values chosen for FP8 are representable in `float64` without loss.

### float8_e4m3fn Has No Infinity

The `e4m3fn` encoding reserves the bit pattern that would represent infinity for an additional NaN value. Therefore `Infinity` and `NegativeInfinity` are explicitly rejected by `is_valid_for_dtype()` for this dtype.

### Boolean Dtype

`bool` is treated as a 1-bit unsigned integer. `DTypeMin` = `false` (0), `DTypeMax` = `true` (1). `NegativeOne` maps to `true` (same as `One` via the two's complement rule). All float-only values (NaN, Inf, Epsilon, Tiny) are invalid.

---

## Testing Strategy

Tests are structured in two layers:

### 1. Parameterized Cartesian Product Suite (`CartesianTest`)

The suite instantiates `shape × dtype × config` combinations. Incompatible combinations are skipped via `GTEST_SKIP()` using per-config compatibility predicates so the total registered test count remains manageable.

| Dimension | Count | Values |
|-----------|-------|--------|
| Shapes | 8 | 1-D (sizes 1, 7, 100, 1M), 2-D, 3-D, 4-D, 5-D |
| Dtypes | 15 | All supported dtypes (see table below) |
| Configs | 16 | Various probability distributions |

Verification strategy per combination:
- Shape and dtype of the output are asserted exactly.
- For `N >= 10,000`: observed counts of discrete values are checked against their expected counts using a 5-sigma Gaussian tolerance (with a minimum slack of 3 counts for very rare values).
- For all-discrete specs (`p_sum ≈ 1.0`): the total count of all discrete values is asserted to equal `N` (leak check).
- Low-resolution dtypes (integers, FP8, bool) with a continuous component use a relaxed lower-bound-only check, since quantisation can cause continuous samples to coincide with discrete values.

### 2. Focused Unit Tests (non-parameterized)

These cover specific behaviours that the cartesian sweep may exercise only statistically:

| Group | What is tested |
|-------|---------------|
| `Validation` | `is_valid_for_dtype()` for float-only and e4m3fn-specific rules |
| `ErrorHandling` | `std::invalid_argument` thrown for NaN-on-int, inf-on-e4m3fn, negative probability, excess probability |
| `EdgeCase` | Empty tensor, single-element, zero-probability entries, very small probability (p=0.001 at N=1M) |
| `UInt{8,16,32,64}Specific` | Exact two's complement values, no-leak checks |
| `FloatSpecials` | Exact epsilon and tiny values for each float dtype |
| `IntBoundary` | Exact min/max for int8/16/32/64 |
| `BoolDType` | Zero=false, One=true, DTypeMin/Max |
| `Performance` | 4M-element tensors; statistical checks |
| `ContinuousDist` | Uniform range bounds and mean; normal mean and std |
| `AllDiscrete` | 50-trial leakage stress test for floating-point accumulation fix |

---

## Supported Dtypes Table

| dtype | PyTorch constant | Bits | Signed | Float |
|-------|-----------------|------|--------|-------|
| uint8 | `torch::kUInt8` | 8 | No | No |
| uint16 | `c10::ScalarType::UInt16` | 16 | No | No |
| uint32 | `c10::ScalarType::UInt32` | 32 | No | No |
| uint64 | `c10::ScalarType::UInt64` | 64 | No | No |
| int8 | `torch::kInt8` | 8 | Yes | No |
| int16 | `torch::kInt16` | 16 | Yes | No |
| int32 | `torch::kInt32` | 32 | Yes | No |
| int64 | `torch::kInt64` | 64 | Yes | No |
| float16 | `torch::kFloat16` | 16 | Yes | Yes |
| bfloat16 | `torch::kBFloat16` | 16 | Yes | Yes |
| float32 | `torch::kFloat32` | 32 | Yes | Yes |
| float64 | `torch::kFloat64` | 64 | Yes | Yes |
| bool | `torch::kBool` | 1 | No | No |
| float8_e4m3fn | `torch::kFloat8_e4m3fn` | 8 | Yes | Yes (no inf) |
| float8_e5m2 | `torch::kFloat8_e5m2` | 8 | Yes | Yes |
