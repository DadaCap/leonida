// leonida::gpu_times (runtime/src/gpu_times.h) over a fake query pool. No GPU, no game data.
#include "../src/gpu_times.h"

namespace {
int g_failures = 0;
void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}
std::vector<std::uint64_t> g_ticks;
std::uint32_t g_written = 0, g_last = 0;
VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*,
                                      VkQueryPool* pool) {
    static std::uintptr_t next = 0;
    *pool = VkQueryPool(++next);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL reset(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) { g_written = 0; }
VKAPI_ATTR void VKAPI_CALL write(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t query) {
    ++g_written;
    g_last = query;
}
VKAPI_ATTR VkResult VKAPI_CALL results(VkDevice, VkQueryPool, std::uint32_t, std::uint32_t count, size_t, void* out,
                                       VkDeviceSize, VkQueryResultFlags) {
    if (g_ticks.size() < count) return VK_NOT_READY;
    std::copy(g_ticks.begin(), g_ticks.begin() + count, static_cast<std::uint64_t*>(out));
    return VK_SUCCESS;
}
}  // namespace

int main() {
    namespace gt = leonida::gpu_times;
    _putenv_s("LEONIDA_GPU_TIMES", "1");
    gt::note(7, {"lib_cs_1"});
    gt::note(8, {"vs_1", nullptr, "ps_1"});
    const gt::Functions f{create, destroy, reset, write, results};
    gt::QueueTimes times(VK_NULL_HANDLE, f, "direct", 32, 1e6);  // a tick is a millisecond
    gt::Recorder* r = times.begin(VK_NULL_HANDLE);
    r->step(VK_NULL_HANDLE, gt::Kind::dispatch, 7, 64);
    r->step(VK_NULL_HANDLE, gt::Kind::dispatch, 7, 32);
    r->step(VK_NULL_HANDLE, gt::Kind::draw, 8);
    r->step(VK_NULL_HANDLE, gt::Kind::draw, 8);
    r->step(VK_NULL_HANDLE, gt::Kind::other);
    r->end(VK_NULL_HANDLE);
    check(r->parts.size() == 4 && g_written == 5 && g_last == 4, "each dispatch is a part, a run of draws is one");
    times.submitted(r, 3);
    g_ticks = {0xfffffff0, 0xfffffffa, 0x5, 0x6, 0x8};  // 10, 11 (past 32 bits), 1, 2 ms
    times.collect(2);
    check(gt::log().submissions == 0, "a submission in flight is not read");
    times.collect(3);
    const auto top = gt::log().totals.top(2);
    check(top.size() == 2 && top[0].first.second == "dispatch lib_cs_1" && top[0].second.ms == 21 &&
              top[0].second.max_ms == 11 && top[0].second.parts == 2 && top[0].second.groups == 96 &&
              top[0].second.max_groups == 64,
          "a pipeline's parts are summed, the ticks wrap at the valid bits");
    check(gt::log().queue_ms["direct"] == 24, "and the queue's");
    check(gt::Totals::describe(gt::Part{gt::Kind::draw, 8, {"vs_1", nullptr, "ps_1"}}) == "draw vs_1 + ps_1" &&
              gt::Totals::describe(gt::Part{gt::Kind::dispatch, 0x10, {}}) == "dispatch pso 0x10",
          "what it is called");

    r = times.begin(VK_NULL_HANDLE);
    for (unsigned i = 0; i < 5000; ++i) r->step(VK_NULL_HANDLE, gt::Kind::dispatch, 7, 1);
    r->end(VK_NULL_HANDLE);
    check(g_last < gt::Recorder::kQueries && r->parts.back().kind == gt::Kind::untimed &&
              r->untimed + r->parts.size() - 1 == 5000,
          "past the pool's queries the rest is one part, counted");
    times.submitted(r, 4);
    times.collect(4);
    check(gt::log().unread == 1, "results that are not there are not waited for");
    std::printf("%d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
