#pragma once

// The console texture layout (xg_xs.cpp), for the D3D12 layer: a subresource of a texture in title memory, linear.
// A pointer, so that the layer does not link the shim; null without it.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rt::xg {

struct LinearSubresource {
    std::vector<std::byte> bytes;
    std::uint32_t row_pitch = 0, depth_pitch = 0;
};

// `desc` is the title's 56-byte description (its layout word names the swizzle mode). False when it cannot be laid out
// or addressed, or when the texture's bytes are not all readable.
using DetileFn = bool (*)(const void* desc, const void* memory, std::uint32_t subresource, LinearSubresource& out);
inline DetileFn detile_fn = nullptr;

}  // namespace rt::xg
