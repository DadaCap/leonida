#pragma once

// The console's size for a texture (xg_xs.cpp), for the D3D12 device proxy's GetResourceAllocationInfo. A pointer, so
// that the proxy does not link the xg_xs shim; null in a process without it.
#include <cstdint>

namespace rt::xg {

// `desc`: the title's 56-byte D3D12_RESOURCE_DESC. 0 when the shim cannot lay it out; else its bytes, and `alignment`.
using TextureSizeFn = std::uint64_t (*)(const void* desc, std::uint64_t& alignment);
inline TextureSizeFn texture_size_fn = nullptr;

}  // namespace rt::xg
