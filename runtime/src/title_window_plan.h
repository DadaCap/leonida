#pragma once
// Where a title-window mapping sits and how much of a stream pool one mapping serves, without a device
// (docs/d3d12_vulkan.md section 3.3 "Title-chosen addresses").
//
// The device's resolve_title_memory makes the mapping; these decide its range first, so a test can count mappings
// without Vulkan. Two rules matter here:
//   - a window is page-aligned, capped at one request's maximum and charged to LEONIDA_TITLE_WINDOWS_MB; one that
//     does not fit is not made, and the address is backed another way or written as a zero-size view, never raw;
//   - a stream pool (the title's CPU view paired with a GPU-only alias, gfxgeom section 3) is one reservation of
//     16-24 GiB, so it cannot be one allocation: one mapping serves a page-aligned readable run of it, and every
//     request inside that run reuses it instead of asking for a 128 KiB window each. A mapping never covers bytes
//     another one already serves, in either view of the pool: a range entered over a live one replaces it whole
//     (TitleVaAliases::add), and the copy that range answered from would be left answered by nothing.
#include "title_va_alias.h"

#include <algorithm>
#include <cstdint>

namespace leonida {

// A page-aligned mapping of the title's memory: [start, start + size).
struct TitleWindowRange {
    std::uint64_t start{};
    std::uint64_t size{};
};

// `size` bytes more fit in the title-window budget [used, cap).
inline bool title_window_fits(std::uint64_t cap, std::uint64_t used, std::uint64_t size) {
    return used <= cap && size <= cap - used;
}

// The page-aligned range a window made for [va, va + bytes) covers: inside the free span [lo, hi), aligned to
// `align`, at most `most` bytes of the request, and charged against [used, cap). False when the span holds no page
// for it or the budget has no room.
inline bool title_window_range(std::uint64_t va, std::uint64_t bytes, std::uint64_t lo, std::uint64_t hi,
                               std::uint64_t align, std::uint64_t most, std::uint64_t cap, std::uint64_t used,
                               TitleWindowRange& out) {
    if (!align || (align & (align - 1)) != 0 || !most || va < lo || hi <= va) return false;
    const std::uint64_t want = (std::min)(bytes ? bytes : std::uint64_t(1), most);
    const std::uint64_t start = (std::max)(va & ~(align - 1), lo);
    const std::uint64_t raw_end = want > UINT64_MAX - va ? UINT64_MAX : va + want;
    std::uint64_t end = raw_end > UINT64_MAX - (align - 1) ? UINT64_MAX : (raw_end + align - 1) & ~(align - 1);
    end = (std::min)(end, hi);
    if (end <= va || end <= start) return false;
    const TitleWindowRange range{start, end - start};
    if (!title_window_fits(cap, used, range.size)) return false;
    out = range;
    return true;
}

// The extent of a stream pool one mapping serves for the request at `va`: from its page to the end of the readable
// run (`readable_end`, exclusive), at most `most` bytes, inside the pool [pool_lo, pool_hi). The same extent comes
// back for every request inside it, so each extent costs one mapping, not one per request.
inline bool title_pool_extent(std::uint64_t pool_lo, std::uint64_t pool_hi, std::uint64_t va,
                              std::uint64_t readable_end, std::uint64_t align, std::uint64_t most,
                              TitleWindowRange& out) {
    if (!align || (align & (align - 1)) != 0 || pool_hi <= pool_lo || va < pool_lo || va >= pool_hi ||
        readable_end <= va)
        return false;
    const std::uint64_t start = (std::max)(va & ~(align - 1), pool_lo);
    std::uint64_t end = (std::min)(readable_end, pool_hi);
    end = end > UINT64_MAX - (align - 1) ? UINT64_MAX : (end + align - 1) & ~(align - 1);
    if (most) end = (std::min)(end, start > UINT64_MAX - most ? UINT64_MAX : start + most);
    end = (std::min)(end, pool_hi);
    if (end <= va || end <= start) return false;
    out = {start, end - start};
    return true;
}

// title_pool_extent inside the free span (TitleVaAliases::gap) around the request at `va` and around `twin`, the same
// byte in the pair's other view (`va` itself when there is none): it ends at the next mapping in either view, and
// starts after the last one before the request. False when a range already holds `va` or `twin` (that mapping serves
// the request) or no page is left for it.
inline bool title_pool_free_extent(const TitleVaAliases& aliases, std::uint64_t pool_lo, std::uint64_t pool_hi,
                                   std::uint64_t va, std::uint64_t twin, std::uint64_t readable_end,
                                   std::uint64_t align, std::uint64_t most, TitleWindowRange& out) {
    std::uint64_t lo = 0, hi = 0, twin_lo = 0, twin_hi = 0;
    aliases.gap(va, lo, hi);
    aliases.gap(twin, twin_lo, twin_hi);
    if (lo > va || twin_lo > twin) return false;
    // The twin's span, moved to `va`'s view.
    lo = (std::max)(lo, va - (std::min)(va, twin - twin_lo));
    hi = (std::min)(hi, va + (std::min)(UINT64_MAX - va, twin_hi - twin));
    return title_pool_extent((std::max)(pool_lo, lo), (std::min)(pool_hi, hi), va, readable_end, align, most, out);
}

}  // namespace leonida
