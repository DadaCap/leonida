// CPU-only regression: the console's value-write list slots (69 and 70).
//
// What it checks:
//   - slot 69 writes 32 bits, slot 70 a 64-bit one, and no other slot writes a value;
//   - slot 70's flags 1 and 2 ask for a GPU and a system timestamp, not for the value the call carried;
//   - the writes of a submission are applied in the order the lists recorded them;
//   - an address the host cannot write is skipped, not written through.
//
// The hand-over from the queue proxy to the queue that submits them is in the Xbox-ABI shim layer and has its
// own test (test/xs_value_writes_handover_test.cpp).
//
// No game data, no GPU, no Windows header: the model is a byte array.
#include "../src/xs_value_writes.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "xs_value_writes_test: %s\n", what);
    ++g_failures;
}

// The title's memory: a byte array the sink writes into, at an address the test chooses.
constexpr std::uint64_t kBase = 0x1000;
constexpr std::size_t kSize = 256;

struct Memory {
    std::uint8_t bytes[kSize]{};

    bool write(std::uint64_t va, const void* data, std::uint32_t size) {
        if (va < kBase || va - kBase + size > kSize) return false;  // not ours
        std::memcpy(bytes + (va - kBase), data, size);
        return true;
    }

    std::uint64_t at(std::uint64_t va, std::uint32_t size) const {
        std::uint64_t value = 0;
        std::memcpy(&value, bytes + (va - kBase), size);
        return value;
    }
};

leonida::ValueWriteSink sink_for(Memory& memory) {
    return [&memory](std::uint64_t va, const void* data, std::uint32_t size) {
        return memory.write(va, data, size);
    };
}

// A record the way the list proxy makes it: the width comes from the slot, and the 32-bit form keeps 32 bits.
leonida::ValueWrite record(unsigned slot, std::uint64_t address, std::uint64_t value, std::uint32_t flags) {
    leonida::ValueWrite write;
    write.address = address;
    write.bytes = leonida::value_write_width(slot);
    write.value = write.bytes == 4 ? (value & 0xffffffffull) : value;
    write.flags = flags;
    return write;
}

}  // namespace

int main() {
    using namespace leonida;

    // ---- the slots: 69 writes 32 bits, 70 a 64-bit one, and nothing else writes a value ----
    {
        check(value_write_width(kConsoleWrite32BitValueSlot) == 4, "slot 69 writes 32 bits");
        check(value_write_width(kConsoleWrite64BitValueSlot) == 8, "slot 70 writes 64 bits");
        check(value_write_width(68) == 0, "slot 68 writes nothing");
        check(value_write_width(71) == 0, "slot 71 writes nothing");

        const ValueWrite narrow = record(kConsoleWrite32BitValueSlot, kBase, 0x1122334455667788ull, 0);
        check(narrow.bytes == 4, "the 32-bit record says 4 bytes");
        check(narrow.value == 0x55667788ull, "the 32-bit record keeps 32 bits of the value");
        const ValueWrite wide = record(kConsoleWrite64BitValueSlot, kBase + 8, 0x0123456789abcdefull, 0);
        check(wide.bytes == 8, "the 64-bit record says 8 bytes");
        check(wide.value == 0x0123456789abcdefull, "the 64-bit record keeps the whole value");
        check(wide.flags == 0, "the record keeps the flags");
    }

    // ---- the writes of a submission land in the order the lists recorded them ----
    {
        Memory memory;
        std::vector<ValueWrite> batch;
        batch.push_back(record(kConsoleWrite32BitValueSlot, kBase, 0x1111, 0));
        batch.push_back(record(kConsoleWrite64BitValueSlot, kBase + 8, 0x2222, 0));

        const std::size_t applied = apply_value_writes(batch, sink_for(memory), {}, {});
        check(applied == 2, "both writes land");
        check(memory.at(kBase, 4) == 0x1111, "the 32-bit write is where it was asked for");
        check(memory.at(kBase + 8, 8) == 0x2222, "the 64-bit write is where it was asked for");
    }

    // ---- a timestamp write takes the clock's value, not the value the call carried ----
    {
        Memory memory;
        std::vector<ValueWrite> batch;
        batch.push_back(record(kConsoleWrite64BitValueSlot, kBase, 0x1234, 1));    // the GPU clock
        batch.push_back(record(kConsoleWrite64BitValueSlot, kBase + 8, 0x1234, 2));  // the system clock

        const std::function<std::uint64_t()> gpu = [] { return 0x5555ull; };
        const std::function<std::uint64_t()> system = [] { return 0x6666ull; };
        const std::size_t applied = apply_value_writes(batch, sink_for(memory), gpu, system);
        check(applied == 2, "both timestamp writes land");
        check(memory.at(kBase, 8) == 0x5555, "flag 1 writes the GPU clock");
        check(memory.at(kBase + 8, 8) == 0x6666, "flag 2 writes the system clock");

        // Without a clock the write is skipped: the value the call carried is not written instead.
        Memory other;
        const std::size_t none = apply_value_writes(batch, sink_for(other), {}, {});
        check(none == 0, "a timestamp write without a clock is skipped");
        check(other.at(kBase, 8) == 0, "the carried value is not written without a clock");
    }

    // ---- an address the host cannot write is skipped ----
    {
        Memory memory;
        std::vector<ValueWrite> batch;
        batch.push_back(record(kConsoleWrite32BitValueSlot, kBase + kSize, 0x3333, 0));  // past the memory
        const std::size_t applied = apply_value_writes(batch, sink_for(memory), {}, {});
        check(applied == 0, "a write the sink refuses does not land");
    }

    // ---- the 32-bit form's flags are not measured: its value is written as it was given ----
    {
        Memory memory;
        std::vector<ValueWrite> batch;
        batch.push_back(record(kConsoleWrite32BitValueSlot, kBase, 0x4444, 1));
        const std::size_t applied = apply_value_writes(batch, sink_for(memory), [] { return 0x9999ull; }, {});
        check(applied == 1, "the 32-bit write lands");
        check(memory.at(kBase, 4) == 0x4444, "the 32-bit form writes the value it carried");
    }

    if (g_failures) {
        std::fprintf(stderr, "xs_value_writes_test: %d checks failed\n", g_failures);
        return 1;
    }
    std::puts("the console's value-write slots record and apply their writes");
    return 0;
}
