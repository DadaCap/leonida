#pragma once

#include <cstddef>
#include <cstdint>

namespace rt::xmem {

bool commit_physical_range(const void* address, std::size_t size);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);

// Writes `size` bytes into the physical pages `page_array` names (64 KiB each), from byte `offset` of the first,
// committing them. False for a page that is not live; the pages before it are written. The array must be readable.
bool write_physical_pages(const std::uint64_t* page_array, std::size_t offset, const void* bytes, std::size_t size);

// The same, for a caller that does not link the shim; null in a process without it.
using WritePhysicalFn = bool (*)(const std::uint64_t* page_array, std::size_t offset, const void* bytes, std::size_t size);
inline WritePhysicalFn write_physical_fn = nullptr;

}
