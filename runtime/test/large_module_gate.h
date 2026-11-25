// How many large SPIR-V modules vk_pipeline_check lets into the driver at once, by a memory budget.
//
// Measured on RADV GFX1201 (Mesa 25.3.6) with the Mesa shader cache off: a 5.7 MB translated ray-tracing compute
// module peaks at 1.96 GB of process memory compiling alone (~330 bytes per SPIR-V byte), a 0.36 MB one at 94 MB.
// Compiling the large ones strictly one at a time left the other threads idle: 40 modules (27 over 4 MiB) on 6
// threads took 308 s exclusively and 75 s in parallel, 7.0 GB and 8.7 GB peak. kBytesPerSpirvByte keeps a margin
// over the measurement.
//
// The process grew past one module's peak even one at a time because glibc keeps freed memory in per-thread arenas
// (6 threads: 6.2 GB; MALLOC_ARENA_MAX=1: 2.1 GB, same modules). release_free_memory() hands it back after each
// large module.
//
// A worker used to take the next module and then wait in the gate when it was large and didn't fit, so with the
// budget full the other modules sat untaken behind it and the pool ran below --jobs. LargeModuleQueue now puts such
// a module back and the worker takes the next one.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace leonida {

inline constexpr uint64_t kBytesPerSpirvByte = 512;

inline uint64_t large_module_estimate(uint64_t spirv_bytes) { return spirv_bytes * kBytesPerSpirvByte; }

#if defined(__linux__)
// The first unsigned number in a file ("max" and unreadable files give no value).
inline bool read_u64_file(const std::string& path, uint64_t& value) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned long long v = 0;
    const bool ok = std::fscanf(f, "%llu", &v) == 1;
    std::fclose(f);
    value = v;
    return ok;
}

// The room left under the cgroup v2 memory.max of this process and of every cgroup above it (a container or
// systemd-run -p MemoryMax=...), or UINT64_MAX when none sets a limit. cgroup_root is a parameter for the tests.
inline uint64_t cgroup_memory_room(const std::string& self_cgroup_file = "/proc/self/cgroup",
                                   const std::string& cgroup_root = "/sys/fs/cgroup") {
    std::FILE* f = std::fopen(self_cgroup_file.c_str(), "r");
    if (!f) return UINT64_MAX;
    char line[4096];
    std::string rel;
    while (std::fgets(line, sizeof(line), f)) {
        std::string l(line);
        if (l.rfind("0::", 0) == 0) {   // the cgroup v2 line
            rel = l.substr(3);
            while (!rel.empty() && (rel.back() == '\n' || rel.back() == '\r')) rel.pop_back();
            break;
        }
    }
    std::fclose(f);
    uint64_t room = UINT64_MAX;
    for (std::string dir = rel; !dir.empty();) {
        // In a cgroup namespace, "/" can be a constrained cgroup mounted as the visible root.
        const std::string base = dir == "/" ? cgroup_root : cgroup_root + dir;
        uint64_t max = 0, current = 0;
        if (read_u64_file(base + "/memory.max", max) && read_u64_file(base + "/memory.current", current))
            room = std::min(room, max > current ? max - current : 0);
        const auto slash = dir.find_last_of('/');
        if (dir == "/" || slash == std::string::npos) break;
        dir.resize(slash == 0 ? 1 : slash);
    }
    return room;
}
#endif

// Memory the OS could give this process now, or 0 when it can't tell (then the caller keeps the old exclusive gate).
inline uint64_t available_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? uint64_t(status.ullAvailPhys) : 0;
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { found = true; break; }
    std::fclose(f);
    if (!found) return 0;
    const uint64_t room = cgroup_memory_room();
    return std::max<uint64_t>(1, std::min(uint64_t(kb) * 1024, room));   // 0 would mean "unknown"
#else
    return 0;
#endif
}

// Return memory freed by the driver's compiler to the OS (glibc keeps it in the arena of the thread that freed it).
inline void release_free_memory() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

// Lets large modules in while their estimates fit the budget and fewer than max_inflight are in. One always gets in
// when none is, so a module bigger than the whole budget still compiles (alone among the large ones).
class LargeModuleGate {
public:
    LargeModuleGate(uint64_t budget, unsigned max_inflight)
        : budget_(budget), max_inflight_(max_inflight ? max_inflight : 1) {}

    bool admits(uint64_t estimate) const {
        std::lock_guard<std::mutex> g(mu_);
        return admits_locked(estimate);
    }
    void enter(uint64_t estimate) {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return admits_locked(estimate); });
        used_ += estimate;
        ++inflight_;
    }
    // enter() only if the gate admits the module now; false, with nothing taken, otherwise.
    bool try_enter(uint64_t estimate) {
        std::lock_guard<std::mutex> g(mu_);
        if (!admits_locked(estimate)) return false;
        used_ += estimate;
        ++inflight_;
        return true;
    }
    void leave(uint64_t estimate) {
        {
            std::lock_guard<std::mutex> g(mu_);
            used_ -= estimate;
            --inflight_;
        }
        cv_.notify_all();
    }
    unsigned inflight() const {
        std::lock_guard<std::mutex> g(mu_);
        return inflight_;
    }
    uint64_t budget() const { return budget_; }
    unsigned max_inflight() const { return max_inflight_; }

    // Holds a place from construction until destruction (or release()), then hands freed memory back.
    class Ticket {
    public:
        Ticket() = default;
        Ticket(LargeModuleGate& gate, uint64_t estimate) : gate_(&gate), estimate_(estimate) { gate.enter(estimate); }
        // A place only if the gate admits the module now (as std::unique_lock with std::try_to_lock); empty if not.
        Ticket(LargeModuleGate& gate, uint64_t estimate, std::try_to_lock_t)
            : gate_(gate.try_enter(estimate) ? &gate : nullptr), estimate_(estimate) {}
        Ticket(Ticket&& o) noexcept : gate_(o.gate_), estimate_(o.estimate_) { o.gate_ = nullptr; }
        Ticket& operator=(Ticket&& o) noexcept {
            if (this != &o) { release(); gate_ = o.gate_; estimate_ = o.estimate_; o.gate_ = nullptr; }
            return *this;
        }
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        ~Ticket() { release(); }
        explicit operator bool() const { return gate_ != nullptr; }
        void release() {
            if (!gate_) return;
            gate_->leave(estimate_);
            gate_ = nullptr;
            release_free_memory();
        }
    private:
        LargeModuleGate* gate_ = nullptr;
        uint64_t estimate_ = 0;
    };

private:
    bool admits_locked(uint64_t estimate) const {
        if (inflight_ == 0) return true;
        return inflight_ < max_inflight_ && used_ + estimate <= budget_;
    }

    const uint64_t budget_;
    const unsigned max_inflight_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint64_t used_ = 0;
    unsigned inflight_ = 0;
};

// The order vk_pipeline_check's workers take modules 0..count-1 in. A large module the gate doesn't admit yet is put
// back (admit() returns false) and the worker takes the next module. The oldest module put back goes first as soon
// as the gate admits it, and no later large module passes it, so none waits forever; once every index is out, a
// worker waits for it. Nothing is put back while the gate admits each large module as it comes (one worker, or room
// under the budget), so then the order is 0, 1, 2... as before.
class LargeModuleQueue {
public:
    struct Claim {
        size_t index = 0;
        LargeModuleGate::Ticket ticket;   // a module put back comes back with its place in the gate
    };

    LargeModuleQueue(LargeModuleGate& gate, size_t count) : gate_(gate), count_(count) {}

    // The next module, or none when every one is out. Drop the previous claim first: its ticket counts in the gate.
    std::optional<Claim> take() {
        std::lock_guard<std::mutex> g(mu_);
        if (!waiting_.empty()) {
            const auto [index, estimate] = waiting_.front();
            LargeModuleGate::Ticket ticket(gate_, estimate, std::try_to_lock);
            // Nothing new left: wait for its place, holding mu_ so no other module passes it. The gate frees places
            // without mu_, and a worker that blocks on mu_ holds no ticket.
            if (!ticket && next_ == count_) ticket = LargeModuleGate::Ticket(gate_, estimate);
            if (ticket) {
                waiting_.pop_front();
                return Claim{index, std::move(ticket)};
            }
        }
        if (next_ == count_) return std::nullopt;
        return Claim{next_++, {}};
    }

    // A large module's place in the gate: true when the claim holds one (it came back with it, or the gate admits
    // the module now); false when the module was put back, and the worker drops the claim and takes the next one.
    bool admit(Claim& claim, uint64_t estimate) {
        if (claim.ticket) return true;
        std::lock_guard<std::mutex> g(mu_);
        if (waiting_.empty()) {   // otherwise the module waiting goes first
            claim.ticket = LargeModuleGate::Ticket(gate_, estimate, std::try_to_lock);
            if (claim.ticket) return true;
        }
        waiting_.emplace_back(claim.index, estimate);
        return false;
    }

private:
    LargeModuleGate& gate_;
    const size_t count_;
    std::mutex mu_;
    size_t next_ = 0;
    std::deque<std::pair<size_t, uint64_t>> waiting_;   // put back: index and estimate, oldest first
};

}  // namespace leonida
