#pragma once

#include "tensor_spec.h"
#include <optional>

// generate_tensor — fill a new tensor according to the given spec.
//
// Parameters:
//   shape     — desired tensor dimensions.
//   dtype     — target scalar type.
//   spec      — discrete distribution (SpecialValue + probability pairs).
//   base_dist — continuous/integer base distribution. Default: Uniform(-1, 1).
//               Supports RandInt for integer dtypes.
//               Can carry a torch::Generator for isolated RNG state.
//   seed      — optional global RNG seed (calls torch::manual_seed).
//
// Throws std::invalid_argument if:
//   - any entry has a negative probability,
//   - any SpecialValue is incompatible with dtype,
//   - discrete probabilities sum to more than 1.0 + 1e-9,
//   - RandInt base_dist used with a float or bool dtype.
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},
    std::optional<int64_t> seed = std::nullopt);
