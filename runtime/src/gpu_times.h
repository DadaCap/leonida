#pragma once
// LEONIDA_GPU_TIMES=1 (local diagnostic): GPU time by pipeline, from timestamps in every submission.
//
// One timestamp (BOTTOM_OF_PIPE: everything recorded before it has finished) is written between the timed parts of a
// command buffer: each dispatch, each draw (and what it needs first: the indirect clamp pass, attachment barriers),
// acceleration structure builds, and the runs of everything else (barriers, copies, clears). A part's time is from the timestamp before it to the one after it, so it
// is what that part added to the queue's time: parts that ran at the same time are not counted twice, the parts of one
// queue never sum to more than its wall time, and a part that ran while a longer one before it did shows little. Parts
// of different queues do overlap. The timestamps make the parts no less parallel (they are no barriers).
//
// Draws of the pipelines that took most in the last period are sampled, at most kSamples a period: the indirect
// records and count they really read, and what they ran (mesh invocations, primitives, fragments). A sampled draw has
// a barrier and a copy after it, so the one after it pays a little.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida::gpu_times {

inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_GPU_TIMES");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}

using Shaders = std::array<const char*, 3>;  // names that live as long as the device
enum class Kind : std::uint8_t { dispatch, dispatch_indirect, draw, draw_indirect, setup, build, other, untimed, count };
inline const char* kind_name(Kind kind) {
    static constexpr const char* names[] = {"dispatch", "dispatch-indirect", "draw", "draw-indirect",
                                            "draw setup (clamp pass, attachment barriers)", "as-build",
                                            "barriers/copies/clears", "past the query limit"};
    return names[unsigned(kind)];
}

struct Names {
    std::mutex mutex;
    std::unordered_map<std::uintptr_t, Shaders> by_pipeline;
};
inline Names& names() {
    static Names n;
    return n;
}
inline void note(std::uintptr_t pipeline, const Shaders& shaders) {
    if (!enabled()) return;
    std::lock_guard lock(names().mutex);
    names().by_pipeline[pipeline] = shaders;
}

struct Part {
    Kind kind = Kind::other;
    std::uintptr_t pipeline = 0;
    Shaders shaders{};
    std::uint32_t commands = 0;
    std::uint64_t groups = 0;  // thread groups of direct dispatches; max_count of indirect draws and dispatches
    bool counted = false;      // with a count buffer
};

// What all queues measured, until it is logged.
class Totals {
public:
    struct Row {
        double ms = 0, max_ms = 0;
        std::uint64_t parts = 0, commands = 0, groups = 0, max_groups = 0, counted = 0;
    };
    using Key = std::pair<std::string, std::string>;  // queue, what

    static std::string describe(const Part& part) {
        std::string out = kind_name(part.kind);
        bool named = false;
        for (const char* shader : part.shaders)
            if (shader) out += named ? " + " : " ", out += shader, named = true;
        if (!named && part.pipeline) {
            char id[32];
            std::snprintf(id, sizeof(id), " pso 0x%llx", static_cast<unsigned long long>(part.pipeline));
            out += id;
        }
        return out;
    }
    void add(const char* queue, const Part& part, double ms) {
        Row& row = rows_[{queue, describe(part)}];
        row.ms += ms;
        row.max_ms = (std::max)(row.max_ms, ms);
        ++row.parts;
        row.commands += part.commands;
        row.groups += part.groups;
        row.max_groups = (std::max)(row.max_groups, part.groups);
        row.counted += part.counted;
        if (part.pipeline) by_pipeline_[part.pipeline] += ms;
    }
    std::vector<std::pair<Key, Row>> top(std::size_t n) const {
        std::vector<std::pair<Key, Row>> out(rows_.begin(), rows_.end());
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
        if (out.size() > n) out.resize(n);
        return out;
    }
    void clear() {
        rows_.clear();
        by_pipeline_.clear();
    }
    std::vector<std::uintptr_t> slowest(std::size_t n) const {
        std::vector<std::pair<double, std::uintptr_t>> all;
        for (const auto& [pipeline, ms] : by_pipeline_) all.emplace_back(ms, pipeline);
        std::sort(all.rbegin(), all.rend());
        std::vector<std::uintptr_t> out;
        for (std::size_t i = 0; i < all.size() && i < n; ++i) out.push_back(all[i].second);
        return out;
    }

private:
    std::map<Key, Row> rows_;
    std::unordered_map<std::uintptr_t, double> by_pipeline_;
};

constexpr unsigned kSamples = 64, kSamplesPerPipeline = 8;

struct Log {
    std::mutex mutex;
    std::vector<std::uintptr_t> slow;  // the pipelines whose draws are sampled in this period
    std::map<std::uintptr_t, unsigned> sampled;
    unsigned samples = 0;
    std::vector<std::pair<double, std::string>> lines;  // the samples read: ms, text
    Totals totals;
    std::map<std::string, double> queue_ms;
    std::uint64_t submissions = 0, untimed = 0, unread = 0, draws = 0;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
};
inline Log& log() {
    static Log l;
    return l;
}
// With the log's mutex held: every 10 s.
inline void print_locked(Log& l) {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - l.since).count();
    if (seconds < 10) return;
    std::string queues;
    for (const auto& [queue, ms] : l.queue_ms) {
        char text[96];
        std::snprintf(text, sizeof(text), " %s %.1f ms", queue.c_str(), ms);
        queues += text;
    }
    std::fprintf(stderr, "[rt] gpu-times: %llu submissions finished in %.1f s;%s; %llu commands past the query limit, "
                         "%llu submissions not read; one vkCmdBeginRendering (LOAD, STORE) per draw, %.0f a second\n",
                 static_cast<unsigned long long>(l.submissions), seconds, queues.c_str(),
                 static_cast<unsigned long long>(l.untimed), static_cast<unsigned long long>(l.unread),
                 double(l.draws) / seconds);
    for (const auto& [key, row] : l.totals.top(25))
        std::fprintf(stderr, "[rt] gpu-times: %.1f ms x %llu max %.1f ms %s %s; %llu commands, groups or indirect max_count %llu max %llu, %llu with a count buffer\n",
                     row.ms, static_cast<unsigned long long>(row.parts), row.max_ms, key.first.c_str(),
                     key.second.c_str(), static_cast<unsigned long long>(row.commands),
                     static_cast<unsigned long long>(row.groups), static_cast<unsigned long long>(row.max_groups),
                     static_cast<unsigned long long>(row.counted));
    std::sort(l.lines.rbegin(), l.lines.rend());
    for (std::size_t i = 0; i < l.lines.size(); ++i)
        std::fprintf(stderr, "[rt] gpu-times: sample %.2f ms %s\n", l.lines[i].first, l.lines[i].second.c_str());
    l.lines.clear();
    l.slow = l.totals.slowest(16);
    l.sampled.clear();
    l.samples = 0;
    l.totals.clear();
    l.queue_ms.clear();
    l.submissions = l.untimed = l.unread = l.draws = 0;
    l.since = now;
}

struct Functions {
    PFN_vkCreateQueryPool create = nullptr;
    PFN_vkDestroyQueryPool destroy = nullptr;
    PFN_vkCmdResetQueryPool reset = nullptr;
    PFN_vkCmdWriteTimestamp write = nullptr;
    PFN_vkGetQueryPoolResults results = nullptr;
    bool complete() const { return create && destroy && reset && write && results; }

    // Sampling; all of them or none.
    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements requirements = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    PFN_vkBindBufferMemory bind = nullptr;
    PFN_vkMapMemory map = nullptr;
    PFN_vkCmdCopyBuffer copy = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkCmdBeginQuery begin_query = nullptr;
    PFN_vkCmdEndQuery end_query = nullptr;
    VkPhysicalDeviceMemoryProperties memory{};
    bool statistics = false, mesh_queries = false;  // the features
    bool sampling() const {
        return create_buffer && destroy_buffer && requirements && allocate && free && bind && map && copy && barrier &&
               begin_query && end_query && memory.memoryTypeCount;
    }
};

// The parts of one command buffer. step() comes before each command.
class Recorder {
public:
    static constexpr std::uint32_t kQueries = 16384;
    static constexpr std::uint32_t kRecords = 1024;  // of a sample
    static constexpr VkDeviceSize kSampleBytes = 16 + kRecords * 12;

    struct Sample {
        std::size_t part;
        std::uint32_t max_count, stride;
        VkDeviceSize bytes;  // of records copied
        bool counted, mesh, queried;
    };
    std::vector<Sample> samples;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const std::uint8_t* mapped = nullptr;
    VkQueryPool mesh_pool = VK_NULL_HANDLE, vertex_pool = VK_NULL_HANDLE, primitives = VK_NULL_HANDLE;

    VkQueryPool pool = VK_NULL_HANDLE;
    std::vector<Part> parts;  // parts[i] lies between queries i and i + 1
    std::uint64_t untimed = 0;

    void begin(const Functions& f, VkCommandBuffer cb) {
        f_ = &f;
        parts.clear();
        untimed = 0;
        samples.clear();
        sampling_ = false;
        f.reset(cb, pool, 0, kQueries);
        if (mapped) {
            std::lock_guard lock(log().mutex);
            slow_ = log().slow;
        }
        if (!slow_.empty())
            for (const VkQueryPool p : {mesh_pool, vertex_pool, primitives})
                if (p) f.reset(cb, p, 0, kSamples);
        f.write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0);
    }
    void step(VkCommandBuffer cb, Kind kind, std::uintptr_t pipeline = 0, std::uint64_t groups = 0,
              bool counted = false) {
        if (open_ && (parts.back().kind == Kind::untimed ||
                      (kind == parts.back().kind && pipeline == parts.back().pipeline && (kind == Kind::other || kind == Kind::build)))) {
            ++parts.back().commands;
            untimed += parts.back().kind == Kind::untimed;
            return;
        }
        end(cb);
        if (parts.size() + 2 >= kQueries) kind = Kind::untimed, pipeline = 0, ++untimed;
        Part part{kind, pipeline, {}, 1, groups, counted};
        if (pipeline) {
            std::lock_guard lock(names().mutex);
            if (const auto it = names().by_pipeline.find(pipeline); it != names().by_pipeline.end())
                part.shaders = it->second;
        }
        parts.push_back(part);
        open_ = true;
    }
    // Before an indirect draw's rendering scope, after its step(); sample_end after the scope. `queries`: no statistics
    // query of the application's is active.
    void sample_begin(VkCommandBuffer cb, std::uintptr_t pipeline, bool mesh, bool queries) {
        if (slow_.empty() || parts.empty() || parts.back().kind != Kind::draw_indirect || samples.size() == kSamples ||
            std::find(slow_.begin(), slow_.end(), pipeline) == slow_.end())
            return;
        {
            Log& l = log();
            std::lock_guard lock(l.mutex);
            if (l.samples == kSamples || l.sampled[pipeline] == kSamplesPerPipeline) return;
            ++l.samples;
            ++l.sampled[pipeline];
        }
        const VkQueryPool stats = mesh ? mesh_pool : vertex_pool;
        samples.push_back({parts.size() - 1, 0, 0, 0, false, mesh, queries && stats});
        sampling_ = true;
        if (!samples.back().queried) return;
        f_->begin_query(cb, stats, std::uint32_t(samples.size() - 1), 0);
        if (mesh && primitives) f_->begin_query(cb, primitives, std::uint32_t(samples.size() - 1), 0);
    }
    void sample_end(VkCommandBuffer cb, VkBuffer records, VkDeviceSize offset, std::uint32_t stride,
                    std::uint32_t max_count, VkBuffer count, VkDeviceSize count_offset) {
        if (!std::exchange(sampling_, false)) return;
        Sample& s = samples.back();
        const auto index = std::uint32_t(samples.size() - 1);
        if (s.queried) {
            f_->end_query(cb, s.mesh ? mesh_pool : vertex_pool, index);
            if (s.mesh && primitives) f_->end_query(cb, primitives, index);
        }
        end(cb);  // the draw's time ends here, before the copies
        // A mesh or vertex draw record is 12 to 20 bytes.
        const std::uint32_t record = s.mesh ? 12 : 20;
        s.max_count = max_count;
        s.stride = stride;
        s.counted = count != VK_NULL_HANDLE;
        const std::uint64_t fit = stride >= record ? (kRecords * 12 - record) / stride + 1 : 0;
        const std::uint64_t copied = (std::min)(fit, std::uint64_t(max_count));
        s.bytes = copied ? (copied - 1) * stride + (s.mesh ? 12 : 16) : 0;
        const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                                     VK_ACCESS_TRANSFER_READ_BIT};
        f_->barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0,
                    nullptr);
        const VkDeviceSize at = index * kSampleBytes;
        if (s.counted) {
            const VkBufferCopy region{count_offset, at, 4};
            f_->copy(cb, count, buffer, 1, &region);
        }
        if (s.bytes) {
            const VkBufferCopy region{offset, at + 16, s.bytes};
            f_->copy(cb, records, buffer, 1, &region);
        }
        // The application's writes to the arguments after this come after the copy.
        const VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                    VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        f_->barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                    1, &after, 0, nullptr, 0, nullptr);
    }
    // What a sample read, once the submission is done.
    std::string describe(const Functions& f, VkDevice device, std::size_t index) const {
        const Sample& s = samples[index];
        const std::uint8_t* at = mapped + index * kSampleBytes;
        std::uint32_t count = s.max_count;
        if (s.counted) std::memcpy(&count, at, 4);
        const std::uint64_t drew = (std::min)(count, s.max_count);
        std::uint64_t read = 0, empty = 0, work = 0, most = 0;
        std::array<std::uint32_t, 4> largest{};
        for (VkDeviceSize o = 0; read < drew && s.bytes && o + (s.mesh ? 12 : 16) <= s.bytes; o += s.stride, ++read) {
            std::array<std::uint32_t, 4> w{};
            std::memcpy(w.data(), at + 16 + o, s.mesh ? 12 : 16);
            const std::uint64_t n = s.mesh ? std::uint64_t(w[0]) * w[1] * w[2] : std::uint64_t(w[0]) * w[1];
            empty += !n;
            work += n;
            if (n > most) most = n, largest = w;
        }
        std::array<std::uint64_t, 2> stats{};
        std::uint64_t made = 0;
        const char* queried = "no statistics";
        char text[512], counters[160] = "";
        if (s.queried && f.results(device, s.mesh ? mesh_pool : vertex_pool, std::uint32_t(index), 1, sizeof(stats),
                                   stats.data(), sizeof(stats), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            if (s.mesh && primitives)
                f.results(device, primitives, std::uint32_t(index), 1, sizeof(made), &made, sizeof(made),
                          VK_QUERY_RESULT_64_BIT);
            // In bit order: fragment invocations before mesh invocations, vertex invocations before fragment ones.
            if (s.mesh)
                std::snprintf(counters, sizeof(counters), "mesh invocations %llu, primitives %llu, fragments %llu",
                              static_cast<unsigned long long>(stats[1]), static_cast<unsigned long long>(made),
                              static_cast<unsigned long long>(stats[0]));
            else
                std::snprintf(counters, sizeof(counters), "vertex invocations %llu, fragments %llu",
                              static_cast<unsigned long long>(stats[0]), static_cast<unsigned long long>(stats[1]));
            queried = counters;
        }
        std::snprintf(text, sizeof(text),
                      "%s: max_count %u stride %u, count %s%u; %llu records read of %llu drawn, %llu empty, %s %llu in all, "
                      "most %llu (%u, %u, %u); %s",
                      Totals::describe(parts[s.part]).c_str(), s.max_count, s.stride, s.counted ? "buffer " : "fixed ",
                      count, static_cast<unsigned long long>(read), static_cast<unsigned long long>(drew),
                      static_cast<unsigned long long>(empty), s.mesh ? "groups" : "vertices x instances",
                      static_cast<unsigned long long>(work), static_cast<unsigned long long>(most), largest[0],
                      largest[1], largest[2], queried);
        return text;
    }
    // Closes the part a command left open; also at the end of the command buffer.
    void end(VkCommandBuffer cb) {
        if (!open_) return;
        open_ = false;
        f_->write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, std::uint32_t(parts.size()));
    }

private:
    const Functions* f_ = nullptr;
    bool open_ = false;
    bool sampling_ = false;
    std::vector<std::uintptr_t> slow_;
};

// A queue's recorders: one per submission in flight, used again once it has finished. Used by one thread at a time.
class QueueTimes {
public:
    QueueTimes(VkDevice device, const Functions& functions, std::string queue, unsigned valid_bits, double period_ns,
               bool samples = false)
        : device_(device), f_(functions), queue_(std::move(queue)),
          mask_(valid_bits >= 64 ? ~0ull : (1ull << valid_bits) - 1), period_ms_(period_ns / 1e6),
          samples_(samples && functions.sampling()) {}
    // The queue is idle by then.
    ~QueueTimes() {
        for (Recorder* r : all_) {
            f_.destroy(device_, r->pool, nullptr);
            for (const VkQueryPool p : {r->mesh_pool, r->vertex_pool, r->primitives})
                if (p) f_.destroy(device_, p, nullptr);
            if (r->buffer) f_.destroy_buffer(device_, r->buffer, nullptr);
            if (r->memory) f_.free(device_, r->memory, nullptr);
            delete r;
        }
    }
    QueueTimes(const QueueTimes&) = delete;
    QueueTimes& operator=(const QueueTimes&) = delete;

    Recorder* begin(VkCommandBuffer cb) {
        if (free_.empty()) {
            if (all_.size() >= 256) return nullptr;
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = Recorder::kQueries;
            VkQueryPool pool = VK_NULL_HANDLE;
            if (f_.create(device_, &info, nullptr, &pool) != VK_SUCCESS) return nullptr;
            all_.push_back(new Recorder);
            all_.back()->pool = pool;
            if (samples_) add_sampling(*all_.back());
            free_.push_back(all_.back());
        }
        Recorder* r = free_.back();
        free_.pop_back();
        r->begin(f_, cb);
        return r;
    }
    void submitted(Recorder* r, std::uint64_t value) { pending_.emplace_back(value, r); }
    void discard(Recorder* r) { free_.push_back(r); }

    // Reads the submissions up to `completed`. No wait: a result that is not there is counted as not read.
    void collect(std::uint64_t completed) {
        if (pending_.empty() || pending_.front().first > completed) return;
        Log& l = log();
        std::lock_guard lock(l.mutex);
        while (!pending_.empty() && pending_.front().first <= completed) {
            Recorder* const r = pending_.front().second;
            pending_.pop_front();
            free_.push_back(r);
            const std::uint32_t count = std::uint32_t(r->parts.size()) + 1;
            ticks_.resize(count);
            if (f_.results(device_, r->pool, 0, count, count * sizeof(std::uint64_t), ticks_.data(), sizeof(std::uint64_t),
                           VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
                ++l.unread;
                continue;
            }
            ++l.submissions;
            l.untimed += r->untimed;
            for (std::uint32_t i = 0; i + 1 < count; ++i) {
                const double ms = double((ticks_[i + 1] - ticks_[i]) & mask_) * period_ms_;
                l.totals.add(queue_.c_str(), r->parts[i], ms);
                l.queue_ms[queue_] += ms;
                l.draws += r->parts[i].kind == Kind::draw || r->parts[i].kind == Kind::draw_indirect;
            }
            for (std::size_t s = 0; s < r->samples.size(); ++s) {
                const std::size_t i = r->samples[s].part;
                l.lines.emplace_back(double((ticks_[i + 1] - ticks_[i]) & mask_) * period_ms_, r->describe(f_, device_, s));
            }
        }
        print_locked(l);
    }

private:
    // Without it (no memory, no feature) the recorder takes no samples, or samples without statistics.
    void add_sampling(Recorder& r) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = kSamples * Recorder::kSampleBytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (f_.create_buffer(device_, &info, nullptr, &r.buffer) != VK_SUCCESS) return;
        VkMemoryRequirements need{};
        f_.requirements(device_, r.buffer, &need);
        constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = UINT32_MAX;
        for (const VkMemoryPropertyFlags want : {kHost | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, kHost})
            for (std::uint32_t i = 0; i < f_.memory.memoryTypeCount && type == UINT32_MAX; ++i)
                if ((need.memoryTypeBits & (1u << i)) && (f_.memory.memoryTypes[i].propertyFlags & want) == want) type = i;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = need.size;
        allocate.memoryTypeIndex = type;
        void* mapped = nullptr;
        if (type == UINT32_MAX || f_.allocate(device_, &allocate, nullptr, &r.memory) != VK_SUCCESS ||
            f_.bind(device_, r.buffer, r.memory, 0) != VK_SUCCESS ||
            f_.map(device_, r.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
            return;
        r.mapped = static_cast<const std::uint8_t*>(mapped);
        if (!f_.statistics) return;
        VkQueryPoolCreateInfo pool{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        pool.queryCount = kSamples;
        pool.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        pool.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                  VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        if (f_.create(device_, &pool, nullptr, &r.vertex_pool) != VK_SUCCESS) r.vertex_pool = VK_NULL_HANDLE;
        if (!f_.mesh_queries) return;
        pool.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                  VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
        if (f_.create(device_, &pool, nullptr, &r.mesh_pool) != VK_SUCCESS) r.mesh_pool = VK_NULL_HANDLE;
        pool.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
        pool.pipelineStatistics = 0;
        if (f_.create(device_, &pool, nullptr, &r.primitives) != VK_SUCCESS) r.primitives = VK_NULL_HANDLE;
    }

    VkDevice device_;
    Functions f_;
    std::string queue_;
    std::uint64_t mask_;
    double period_ms_;
    bool samples_;
    std::vector<Recorder*> all_, free_;
    std::deque<std::pair<std::uint64_t, Recorder*>> pending_;  // by the queue's progress value
    std::vector<std::uint64_t> ticks_;
};

}  // namespace leonida::gpu_times
