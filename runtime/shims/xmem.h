#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rt::xmem {

bool commit_physical_range(const void* address, std::size_t size);
// The same, for a caller that does not link the shim; null without it.
inline bool (*commit_range_fn)(const void* address, std::size_t size) = nullptr;
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);

// Committed, zeroed memory of the D3D12 layer that the title's placeholder reservation is taken from, so that its
// GPU-visible windows are in the layer's address space (docs/d3d12_vulkan.md 3.3). Null: no pool.
void set_window_pool(void* base, std::size_t size);

// The host region every GPU-visible allocation of the title comes from, made on first use; the D3D12 layer imports it
// whole as its address arena and keeps the first layer_bytes for itself. False when it is off or cannot be made.
// Whether [address, address + size) lies in that region, which stays committed and writable. VirtualQuery there walks
// every page of the 14 GiB, some 100 ms a call, so a check for readable or writable memory asks this first.
inline std::atomic<std::uintptr_t> region_start{0}, region_end{0};
inline bool in_gpu_region(const void* address, std::size_t size) {
    const auto at = reinterpret_cast<std::uintptr_t>(address);
    const std::uintptr_t end = region_end.load(std::memory_order_acquire);
    return end && at >= region_start.load(std::memory_order_relaxed) && at < end && size <= end - at;
}
bool gpu_region(void*& base, std::size_t& size, std::size_t& layer_bytes);

// The window of GPU-visible address space the title reserved that is too large for the region (24 GiB measured): no
// memory of its own, only the physical pages mapped into it. The D3D12 layer mirrors those mappings on the GPU.
inline std::atomic<std::uintptr_t> gpu_window_start{0}, gpu_window_end{0};
struct GpuWindowPage {
    std::uintptr_t address;   // 64 KiB of the window
    void* section = nullptr;  // the host mapping of the whole section its page is in, committed; null: unmapped
    std::size_t section_bytes = 0, offset = 0;
};
// Called with each map call's changes, under xmem's lock. Null: changes are kept for when it is set.
using GpuWindowFn = void (*)(const GpuWindowPage* pages, std::size_t count);
inline std::atomic<GpuWindowFn> gpu_window_fn{nullptr};
inline std::atomic<void (*)(void* section)> gpu_section_gone_fn{nullptr};
inline void (*gpu_window_flush)() = nullptr;  // set with the shim

// Told of every write into the title's memory that the title did not make itself (DirectStorage, a page's bytes moving
// to another address): the D3D12 layer refreshes the images placed there. Null without that layer.
using WrittenFn = void (*)(std::uintptr_t address, std::size_t size);
inline std::atomic<WrittenFn> written_fn{nullptr};
inline void title_memory_written(std::uintptr_t address, std::size_t size) {
    if (const WrittenFn fn = written_fn.load(std::memory_order_acquire)) fn(address, size);
}

// Writes `size` bytes into the physical pages `pages` names (64 KiB each), from byte `offset` of the first. Set by the
// shim when it is linked, as title_write_fn below. False for a page that is not live.
using PhysicalWriteFn = bool (*)(const std::uint64_t* pages, std::size_t offset, const void* bytes, std::size_t size);
inline PhysicalWriteFn physical_write_fn = nullptr;
// After a write by address: the pages it touched are copied to their other addresses.
inline void (*propagate_fn)(std::uintptr_t address, std::size_t size) = nullptr;

// Writes `size` bytes at `address` in the title's memory, through the host pages xmem manages. False when the
// range is not mapped, or not writable where it is.
bool write_title_bytes(const void* address, const void* bytes, std::size_t size);

// The host write into the title's memory that the console's value writes (list slots 69 and 70) need. The shim
// owns the implementation and sets this pointer when it is linked into the process. The D3D12 layer calls the
// write through the pointer, so the D3D12 library does not have to link this shim - and with it the export table
// and the vectored exception handler - into every consumer of that library.
using TitleWriteFn = bool (*)(const void* address, const void* bytes, std::size_t size);
inline TitleWriteFn title_write_fn = nullptr;

// The write through the registered pointer. False when the shim is not linked into the process: the
// console's value writes stay dropped, the way they were before the shim answered them.
inline bool call_title_write(const void* address, const void* bytes, std::size_t size) {
    if (!title_write_fn) return false;
    return title_write_fn(address, bytes, size);
}
// Console queue slot 31, CopyPageMappingsBatchX (d3d12_xs_device.cpp): maps at `destination` the physical pages that
// are mapped at `source` now, `pages` 64 KiB pages one after another, or the one page at `source` every time with
// `repeat_first`. A source page with nothing mapped leaves its destination page unmapped; a null `source` unmaps them
// all. False, with the last error set, for an unaligned address or a destination outside a physical-page window.
// Set by xmem.cpp when it is linked: the command-object proxy is also compiled into tests without it.
using CopyPageMappingsFn = bool (*)(void* destination, const void* source, std::uint64_t pages, bool repeat_first);
inline CopyPageMappingsFn copy_page_mappings_fn = nullptr;

}
