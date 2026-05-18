#pragma once

// tensor_generator.h — public API for dtype-aware tensor generation.
//
// Exposes a single function, generate_tensor(), that creates a libtorch
// tensor whose elements are drawn from the distribution described by a
// TensorSpec and BaseDist.

#include "tensor_spec.h"
#include <optional>

namespace tgen {

// generate_tensor — fill a new tensor according to the given spec.
//
// Algorithm ("base layer + mask overlay"):
//   1. If the discrete probabilities sum to < 1.0, generate a base tensor
//      from the BaseDist (uniform or normal), cast to `dtype`.
//      Otherwise allocate a zero-initialised tensor.
//   2. Draw a uniform selector tensor in [0, 1) with the same shape.
//   3. For each DiscreteEntry (in order), compute a boolean mask for the
//      selector range [cum, cum + p) and call masked_fill_ with the
//      appropriate value.  The last active entry uses selector >= cum so
//      that floating-point rounding in cumulative sums cannot leave any
//      element unassigned in all-discrete specs.
//
// Parameters:
//   shape     — desired tensor dimensions (empty shape -> 0-element tensor).
//   dtype     — target scalar type; determines which SpecialValues are legal.
//   spec      — discrete distribution specification (see TensorSpec).
//   base_dist — continuous base distribution for remaining probability mass.
//               Defaults to Uniform(-1, 1).
//   seed      — optional RNG seed. If provided, calls torch::manual_seed()
//               before generation for reproducibility.
//
// Throws std::invalid_argument if:
//   - any entry has a negative probability,
//   - any SpecialValue is incompatible with `dtype` (e.g. NaN on int32),
//   - the discrete probabilities sum to more than 1.0 + 1e-9.
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},
    std::optional<int64_t> seed = std::nullopt);

} // namespace tgen
