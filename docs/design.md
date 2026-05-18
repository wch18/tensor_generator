# tensor_generator Design Document

## Architecture Overview

The library is split into three modules, each with a single responsibility:

```
src/
  dtype_traits.h                        — shared dtype classification predicates
  special_value.h / special_value.cpp   — dtype-aware special value table
  tensor_spec.h                         — declarative distribution spec (header-only)
  tensor_generator.h / tensor_generator.cpp — generation engine (public API)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `dtype_traits` | Shared inline predicates: is_float_dtype, is_unsigned_int_dtype, has_infinity, needs_cast_workaround |
| `special_value` | Maps SpecialValue enum to exact numeric representations per dtype; validates dtype/value compatibility |
| `tensor_spec` | Aggregates discrete (SpecialValue, probability) pairs with an optional BaseDist into a single TensorSpec struct |
| `tensor_generator` | Implements the generation algorithm; exposes generate_tensor() |

Dependency graph (strictly layered, no cycles):
```
tensor_generator → tensor_spec → special_value → <torch/torch.h>
                 → dtype_traits
```

---

## Algorithm: Base Layer + Mask Overlay

`generate_tensor(shape, dtype, spec)` works in three phases:

### Phase 1 — Base Tensor
- If `p_sum >= 1.0 - 1e-9` (all-discrete): allocate zeros in target dtype.
- Otherwise: sample from BaseDist in float64, then cast to target dtype.

### Phase 2 — Selector Tensor
Draw a float64 uniform tensor in [0, 1) with the same shape.

### Phase 3 — Mask Overlay
Walk discrete entries in order with cumulative probability `cum`:
- Last non-zero entry in all-discrete mode: `mask = (selector >= cum)` (absorbs rounding slack)
- Otherwise: `mask = (selector >= cum) & (selector < cum + p)`
- Apply `safe_masked_fill(result, mask, special_value, dtype)`

---

## Edge Case Handling

| Edge Case | Solution |
|-----------|----------|
| FP accumulation in all-discrete | Last entry uses `>=` without upper bound |
| uint64 DTypeMax (2^64-1 ≠ representable in float64) | Use int64 intermediate with -1LL bit pattern, cast to uint64 |
| FP8 no masked_fill_ | Promote to float64, fill, cast back |
| uint16/32 no masked_fill_ | Promote to float64 (values fit), fill, cast back |
| int64 DTypeMax through double | Use get_special_value_as_int64() to avoid UB |
| e4m3fn has no infinity | is_valid_for_dtype() rejects Infinity/NegativeInfinity |
| bool NegativeOne | Maps to true (same as One) |

---

## Supported DTypes (15)

| dtype | Bits | Signed | Float | Notes |
|-------|------|--------|-------|-------|
| uint8 | 8 | No | No | |
| uint16 | 16 | No | No | needs_cast_workaround |
| uint32 | 32 | No | No | needs_cast_workaround |
| uint64 | 64 | No | No | int64 bit-trick for max |
| int8 | 8 | Yes | No | |
| int16 | 16 | Yes | No | |
| int32 | 32 | Yes | No | |
| int64 | 64 | Yes | No | int64 path for DTypeMax |
| float16 | 16 | Yes | Yes | |
| bfloat16 | 16 | Yes | Yes | |
| float32 | 32 | Yes | Yes | |
| float64 | 64 | Yes | Yes | |
| bool | 1 | No | No | |
| fp8_e4m3fn | 8 | Yes | Yes | no infinity |
| fp8_e5m2 | 8 | Yes | Yes | |

---

## Testing Strategy

1643 GTest cases structured as:
- **Cartesian product** (1600 cases): 8 shapes × 15 dtypes × 16 configs
- **Focused unit tests** (43 cases): validation, error handling, edge cases, boundary values, performance

Verification modes:
- Full statistical (5σ): for float types with N >= 10000
- Lower-bound only: for low-resolution dtypes (int, fp8, bool) where continuous → discrete aliasing inflates counts
- Exact equality: for p=1.0 single-value specs
- Leak detection: all-discrete specs must cover 100% of elements
