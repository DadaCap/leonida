#pragma once

// LEONIDA_GPU_DEBUG=1 (diagnostic): when the Vulkan device is lost, what VK_EXT_device_fault says (the addresses that
// faulted) and the last commands each queue reached (VK_NV_device_diagnostic_checkpoints). Off: nothing is enabled.

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace leonida {

struct GpuDebug {
    struct Marker {
        const char* kind = "";
        std::uint64_t pipeline = 0;
        std::uint32_t a = 0, b = 0, c = 0;
        std::uint64_t target = 0;
        std::uint32_t user[12] = {};
    };
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkCmdSetCheckpointNV set_checkpoint = nullptr;
    PFN_vkGetQueueCheckpointDataNV checkpoint_data = nullptr;
    PFN_vkGetDeviceFaultInfoEXT fault_info = nullptr;
    PFN_vkDeviceWaitIdle wait_idle = nullptr;
    std::mutex mutex;
    std::vector<VkQueue> queues;
    std::unordered_map<std::uint64_t, std::string> names;  // pipeline id -> its shaders
    std::vector<Marker> ring = std::vector<Marker>(1u << 18);
    std::atomic<std::uint32_t> next{0};
    std::atomic<bool> reported{false}, dump_now{false};  // dump_now: LEONIDA_GPU_DUMP_AT=1000 waits for it

    const std::uint8_t* arena = nullptr;  // the arena's host mapping, when all of it is host memory
    std::uint64_t arena_bytes = 0;
    std::atomic<std::uint64_t> heap_base{0};
    void dump(const char* what, std::uint64_t va, unsigned dwords) const {
        if (!arena || va < 0x10000 || va + dwords * 4 > arena_bytes) return;
        std::fprintf(stderr, "[gpu]     %s %llx:", what, (unsigned long long)va);
        for (unsigned i = 0; i < dwords; ++i) std::fprintf(stderr, " %x", reinterpret_cast<const std::uint32_t*>(arena + va)[i]);
        std::fprintf(stderr, "\n");
    }
    static bool wanted() {
        static const bool on = std::getenv("LEONIDA_GPU_DEBUG") != nullptr;
        return on;
    }
    // LEONIDA_GPU_SKIP=lib_cs_1,lib_cs_2 (diagnostic, with LEONIDA_GPU_DEBUG): their dispatches are not recorded.
    bool skips(std::uint64_t pipeline) {
        static const char* const list = wanted() ? std::getenv("LEONIDA_GPU_SKIP") : nullptr;
        if (!list) return false;
        const std::lock_guard lock(mutex);
        const auto it = names.find(pipeline);
        if (it == names.end()) return false;
        const std::string name = it->second.substr(it->second.rfind('/') + 1);
        return !name.empty() && std::string(list).find(name) != std::string::npos;
    }
    void mark(VkCommandBuffer commands, const Marker& marker) {
        if (!set_checkpoint) return;
        Marker& at = ring[next.fetch_add(1) % ring.size()];
        at = marker;
        set_checkpoint(commands, &at);
        // LEONIDA_GPU_DUMP_AT=s: the first 160 commands recorded after s seconds.
        static const std::uint64_t dump_at = std::getenv("LEONIDA_GPU_DUMP_AT") ? std::strtoull(std::getenv("LEONIDA_GPU_DUMP_AT"), nullptr, 10) : 0;
        static const auto started = std::chrono::steady_clock::now();
        static std::atomic<int> left{160};
        if (dump_at && (dump_at < 1000 ? std::chrono::steady_clock::now() - started > std::chrono::seconds(dump_at) : dump_now.load()) && left-- > 0) {
            const std::lock_guard lock(mutex);
            std::fprintf(stderr, "[dump] %s %s (%u %u %u) target %llx user", marker.kind, names.count(marker.pipeline) ? names[marker.pipeline].c_str() : "", marker.a, marker.b, marker.c, (unsigned long long)marker.target);
            for (std::uint32_t v : marker.user) std::fprintf(stderr, " %x", v);
            std::fprintf(stderr, "\n");
            for (std::uint32_t v : marker.user) if (v >= 0x10000) dump("table", v, 64);
        }
    }
    void report() {
        if (!wanted() || reported.exchange(true)) return;
        std::fprintf(stderr, "[gpu] the Vulkan device is lost\n");
        // The driver answers the fault query only after this has returned VK_ERROR_DEVICE_LOST (pull request 1028).
        if (wait_idle) wait_idle(device);
        if (fault_info) {
            VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
            if (fault_info(device, &counts, nullptr) >= 0) {
                std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount + 1);
                std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount + 1);
                VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
                info.pAddressInfos = addresses.data();
                info.pVendorInfos = vendor.data();
                counts.vendorBinarySize = 0;
                fault_info(device, &counts, &info);
                std::fprintf(stderr, "[gpu] fault: %s\n", info.description);
                for (std::uint32_t i = 0; i < counts.addressInfoCount; ++i)
                    std::fprintf(stderr, "[gpu]   address type %d at %llx (precision %llx)\n", int(addresses[i].addressType),
                                 (unsigned long long)addresses[i].reportedAddress, (unsigned long long)addresses[i].addressPrecision);
                for (std::uint32_t i = 0; i < counts.vendorInfoCount; ++i)
                    std::fprintf(stderr, "[gpu]   vendor %s code %llx data %llx\n", vendor[i].description,
                                 (unsigned long long)vendor[i].vendorFaultCode, (unsigned long long)vendor[i].vendorFaultData);
            }
        }
        const std::lock_guard lock(mutex);
        for (VkQueue queue : queues) {
            std::uint32_t count = 0;
            if (!checkpoint_data) break;
            checkpoint_data(queue, &count, nullptr);
            std::vector<VkCheckpointDataNV> data(count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
            checkpoint_data(queue, &count, data.data());
            for (const auto& d : data) {
                const Marker& m = *static_cast<const Marker*>(d.pCheckpointMarker);
                std::fprintf(stderr, "[gpu]   queue %p stage %x: #%zu %s pso %llx %s (%u %u %u) target %llx user", (void*)queue, unsigned(d.stage),
                             std::size_t(&m - ring.data()), m.kind, (unsigned long long)m.pipeline, names.count(m.pipeline) ? names[m.pipeline].c_str() : "", m.a, m.b, m.c,
                             (unsigned long long)m.target);
                for (std::uint32_t v : m.user) std::fprintf(stderr, " %x", v);
                std::fprintf(stderr, "\n");
                if (d.stage != 1) continue;
                for (std::uint32_t v : m.user) {
                    if (v < 0x10000) continue;
                    dump("table", v, 48);
                    if (!arena || std::uint64_t{v} + 192 > arena_bytes) continue;
                    const auto* t = reinterpret_cast<const std::uint32_t*>(arena + v);
                    for (unsigned k = 0; k + 4 <= 48; k += 4)
                        if ((t[k + 1] & 0xffff) < 4 && t[k + 2] && (t[k + 3] & 0xfff) == 0xfac) dump(("V# at +" + std::to_string(k * 4)).c_str(), t[k] | std::uint64_t{t[k + 1] & 0xffff} << 32, 24), dump("  its end", (t[k] | std::uint64_t{t[k + 1] & 0xffff} << 32) + (t[k + 2] > 64 && !(t[k + 1] >> 16) ? t[k + 2] - 64 : 0), 16);
                }
            }
        }
        std::fflush(stderr);
    }
};
inline GpuDebug& gpu_debug() {
    static auto* d = new GpuDebug;
    return *d;
}

}  // namespace leonida
