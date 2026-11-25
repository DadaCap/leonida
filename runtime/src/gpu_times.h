#pragma once
// LEONIDA_GPU_TIMES=1 (local diagnostic): GPU time by pipeline, from timestamps in every submission.
//
// One timestamp (BOTTOM_OF_PIPE: everything recorded before it has finished) is written between the timed parts of a
// command buffer: each dispatch, each run of draws of one pipeline, acceleration structure builds, and the runs of
// everything else (barriers, copies, clears). A part's time is from the timestamp before it to the one after it, so it
// is what that part added to the queue's time: parts that ran at the same time are not counted twice, the parts of one
// queue never sum to more than its wall time, and a part that ran while a longer one before it did shows little. Parts
// of different queues do overlap. The timestamps make the parts no less parallel (they are no barriers).
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
enum class Kind : std::uint8_t { dispatch, dispatch_indirect, draw, build, other, untimed, count };
inline const char* kind_name(Kind kind) {
    static constexpr const char* names[] = {"dispatch", "dispatch-indirect", "draw", "as-build",
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
    std::uint64_t groups = 0;  // thread groups of direct dispatches; the most indirect ones
};

// What all queues measured, until it is logged.
class Totals {
public:
    struct Row {
        double ms = 0, max_ms = 0;
        std::uint64_t parts = 0, commands = 0, groups = 0, max_groups = 0;
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
    }
    std::vector<std::pair<Key, Row>> top(std::size_t n) const {
        std::vector<std::pair<Key, Row>> out(rows_.begin(), rows_.end());
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
        if (out.size() > n) out.resize(n);
        return out;
    }
    void clear() { rows_.clear(); }

private:
    std::map<Key, Row> rows_;
};

struct Log {
    std::mutex mutex;
    Totals totals;
    std::map<std::string, double> queue_ms;
    std::uint64_t submissions = 0, untimed = 0, unread = 0;
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
                         "%llu submissions not read\n",
                 static_cast<unsigned long long>(l.submissions), seconds, queues.c_str(),
                 static_cast<unsigned long long>(l.untimed), static_cast<unsigned long long>(l.unread));
    for (const auto& [key, row] : l.totals.top(25))
        std::fprintf(stderr, "[rt] gpu-times: %.1f ms x %llu max %.1f ms %s %s; %llu commands, groups %llu max %llu\n",
                     row.ms, static_cast<unsigned long long>(row.parts), row.max_ms, key.first.c_str(),
                     key.second.c_str(), static_cast<unsigned long long>(row.commands),
                     static_cast<unsigned long long>(row.groups), static_cast<unsigned long long>(row.max_groups));
    l.totals.clear();
    l.queue_ms.clear();
    l.submissions = l.untimed = l.unread = 0;
    l.since = now;
}

struct Functions {
    PFN_vkCreateQueryPool create = nullptr;
    PFN_vkDestroyQueryPool destroy = nullptr;
    PFN_vkCmdResetQueryPool reset = nullptr;
    PFN_vkCmdWriteTimestamp write = nullptr;
    PFN_vkGetQueryPoolResults results = nullptr;
    bool complete() const { return create && destroy && reset && write && results; }
};

// The parts of one command buffer. step() comes before each command.
class Recorder {
public:
    static constexpr std::uint32_t kQueries = 4096;

    VkQueryPool pool = VK_NULL_HANDLE;
    std::vector<Part> parts;  // parts[i] lies between queries i and i + 1
    std::uint64_t untimed = 0;

    void begin(const Functions& f, VkCommandBuffer cb) {
        f_ = &f;
        parts.clear();
        untimed = 0;
        f.reset(cb, pool, 0, kQueries);
        f.write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0);
    }
    void step(VkCommandBuffer cb, Kind kind, std::uintptr_t pipeline = 0, std::uint64_t groups = 0) {
        if (open_ && (parts.back().kind == Kind::untimed ||
                      (kind == parts.back().kind && pipeline == parts.back().pipeline && kind != Kind::dispatch &&
                       kind != Kind::dispatch_indirect))) {
            ++parts.back().commands;
            untimed += parts.back().kind == Kind::untimed;
            return;
        }
        end(cb);
        if (parts.size() + 2 >= kQueries) kind = Kind::untimed, pipeline = 0, ++untimed;
        Part part{kind, pipeline, {}, 1, groups};
        if (pipeline) {
            std::lock_guard lock(names().mutex);
            if (const auto it = names().by_pipeline.find(pipeline); it != names().by_pipeline.end())
                part.shaders = it->second;
        }
        parts.push_back(part);
        open_ = true;
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
};

// A queue's recorders: one per submission in flight, used again once it has finished. Used by one thread at a time.
class QueueTimes {
public:
    QueueTimes(VkDevice device, const Functions& functions, std::string queue, unsigned valid_bits, double period_ns)
        : device_(device), f_(functions), queue_(std::move(queue)),
          mask_(valid_bits >= 64 ? ~0ull : (1ull << valid_bits) - 1), period_ms_(period_ns / 1e6) {}
    // The queue is idle by then.
    ~QueueTimes() {
        for (Recorder* r : all_) {
            f_.destroy(device_, r->pool, nullptr);
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
            }
        }
        print_locked(l);
    }

private:
    VkDevice device_;
    Functions f_;
    std::string queue_;
    std::uint64_t mask_;
    double period_ms_;
    std::vector<Recorder*> all_, free_;
    std::deque<std::pair<std::uint64_t, Recorder*>> pending_;  // by the queue's progress value
    std::vector<std::uint64_t> ticks_;
};

}  // namespace leonida::gpu_times
