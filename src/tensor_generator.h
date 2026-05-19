#pragma once

#include "tensor_spec.h"
#include <optional>

// generate_tensor — fill a new tensor according to the given spec.
//
// Parameters:
//   shape     — desired tensor dimensions.
//   dtype     — target scalar type.
//   spec      — discrete distribution (value + probability pairs).
//               Values can be SpecialValue enums, int, or double literals.
//   base_dist — continuous/integer base distribution. Default: Uniform(-1, 1).
//               Supports: Uniform, Normal, RandInt, Exponential, Poisson,
//                         LogNormal, Cauchy, Bernoulli.
//               Can carry a torch::Generator for isolated RNG state.
//   seed      — optional global RNG seed (calls torch::manual_seed).
//               When base_dist carries a generator (via seed param or
//               with_generator/with_seed), all RNG draws use that
//               isolated generator and this seed has no effect.
//
// Throws std::invalid_argument if:
//   - any entry has a negative probability,
//   - any discrete value is incompatible with dtype,
//   - discrete probabilities sum to more than 1.0 + 1e-9,
//   - RandInt/Poisson base_dist used with a float or bool dtype.
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},
    std::optional<int64_t> seed = std::nullopt);
