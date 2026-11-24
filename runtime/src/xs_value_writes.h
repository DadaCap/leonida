#pragma once

// The console's Write32BitValueBottomOfPipeX and Write64BitValueBottomOfPipeX (console command list slots 69
// and 70): the record the Xbox-ABI list proxy keeps, and the function that puts a batch of records into the
// title's memory once the submission that carried them has completed on the GPU.
//
// The console writes the value at the bottom of the pipe: after every command that precedes it in the queue has
// run. VIctory (crates/victory-gpu/src/list.rs, GPL-3.0) does the same thing on the PC: it defers the write
// until the list has run and then writes the value into the guest's RAM from the CPU. This follows that model,
// with VIctory's write32/write64/defer_write as its reference: the write is not a GPU command, it is a CPU write
// into the title's memory, made after the submission that recorded it has completed.
//
// The two shapes are the console's, as measured with xcl (crates/xcl/src/hle/d3d12x/list_x.rs, GPL-2.0):
//   slot 69  Write32BitValueBottomOfPipeX(address, u32 value, flush, flags)
//   slot 70  Write64BitValueBottomOfPipeX(address, u64 value, flush, flags)
// Slot 70's flags 1 and 2 ask for a GPU and a system timestamp instead of the value the call carried. The 32-bit
// form's flags are not measured, so its value is written as it was given.
//
// This header holds the record and the types the layers share, and nothing else. The recording is the list
// proxy's (shims/d3d12_xs_device.cpp), the hand-over to the queue that submits is the proxy's too, and the order
// in which the submissions' batches are applied is the Vulkan queue's (src/vulkan_backend.h). The repository keeps
// such pure logic in a small header of its own so that it can be checked on the CPU without a GPU
// (src/buffer_va_index.h, src/statistics_sum.h); the test (test/xs_value_writes_test.cpp) does that.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace leonida {

// The console list slots that write a value into memory.
inline constexpr unsigned kConsoleWrite32BitValueSlot = 71;
inline constexpr unsigned kConsoleWrite64BitValueSlot = 72;

// One recorded write: the console VA the write names, the value the call carried, the width the slot writes and
// the call's flags.
struct ValueWrite {
    std::uint64_t address = 0;  // the console VA the write names
    std::uint64_t value = 0;    // the value the call carried
    std::uint32_t bytes = 0;    // 4 (slot 69) or 8 (slot 70)
    std::uint32_t flags = 0;    // the call's flags
};

// The width a console list slot writes, or 0 for a slot that writes no value.
inline std::uint32_t value_write_width(unsigned console_slot) {
    if (console_slot == kConsoleWrite32BitValueSlot) return 4;
    if (console_slot == kConsoleWrite64BitValueSlot) return 8;
    return 0;
}

// Where a console value write lands: the CPU write into the title's memory. The device sets the sink; the
// queue's completion callback uses it.
using ValueWriteSink = std::function<bool(std::uint64_t va, const void* bytes, std::uint32_t size)>;

// Applies the writes of one submission, in the order the lists recorded them. A write whose kind has no clock, or
// whose address the sink refuses, is skipped. Returns how many writes landed.
//
// Slot 70's flags 1 and 2 ask for the GPU and the system clock instead of the value the call carried; the 32-bit
// form's flags are not measured, so its value is written as it was given.
inline std::size_t apply_value_writes(const std::vector<ValueWrite>& batch, const ValueWriteSink& write,
                                      const std::function<std::uint64_t()>& gpu_clock,
                                      const std::function<std::uint64_t()>& system_clock) {
    std::size_t applied = 0;
    for (const ValueWrite& record : batch) {
        std::uint64_t value = record.value;
        if (record.bytes == 8 && (record.flags & 1u)) {
            if (!gpu_clock) continue;
            value = gpu_clock();
        } else if (record.bytes == 8 && (record.flags & 2u)) {
            if (!system_clock) continue;
            value = system_clock();
        }
        if (!write) continue;
        // The value is written little-endian, as the console's x64 does: the low 'bytes' of the word.
        if (write(record.address, &value, record.bytes)) ++applied;
    }
    return applied;
}

}  // namespace leonida
