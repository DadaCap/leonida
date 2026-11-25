#pragma once

// LOCAL (not upstream): what happened between two PresentX calls of the title, shown in a small label on the game
// window's title bar and written as one CSV row a frame (XboxPresentation::PresentX calls show()).
//   LEONIDA_TITLE_STATS=0     turns both off
//   LEONIDA_FRAME_LOG=<file>  where the rows go (default: frame_stats.csv in the working directory)
//
// Why a label of our own and not the window's title: the game's window belongs to a title thread that processed one
// WM_SETTEXT in two minutes. A send with a timeout showed one stale value; DefWindowProc(WM_SETTEXT) called from the
// present thread set the text once and never returned, which hung the game.

#include <atomic>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>

#include <psapi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <thread>
#endif

namespace leonida::frame_stats {
// Counted where they happen, read and cleared at each present.
inline std::atomic<std::uint32_t> draws{0}, dispatches{0};       // Draw*/DispatchMesh and Dispatch recorded
inline std::atomic<std::uint32_t> indirect{0};                   // ExecuteIndirect recorded
inline std::atomic<std::uint32_t> submissions{0};                // ExecuteCommandLists calls
inline std::atomic<std::uint32_t> builds{0}, build_ms{0};        // pipelines built and the time that took
inline std::atomic<std::uint32_t> xma_pumps{0};                  // passes of the XMA decoder's worker
inline std::atomic<std::uint32_t> xma_voices{0};                 // XMA voices decoding at the last pass (not cleared)

#ifdef _WIN32
inline std::atomic<HWND> window{nullptr};  // the game's window (set where the presenter is made)
inline std::atomic<HWND> label{nullptr};   // ours: a click-through, never-activated popup with its own thread

inline void label_thread() {
    const HWND made = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT |
                                      WS_EX_LAYERED, L"STATIC", L"Leonida", WS_POPUP | SS_LEFT | SS_NOPREFIX, 0, 0,
                                      640, 20, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
    if (!made) return;
    SetLayeredWindowAttributes(made, 0, 230, LWA_ALPHA);
    SendMessageW(made, WM_SETFONT, reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
    label.store(made);
    MSG message;
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

// One caller at a time: the title presents from one thread.
inline void show() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_TITLE_STATS");
        return !(v && *v == '0');
    }();
    const HWND game = window.load();
    if (!on || !game) return;
    using clock = std::chrono::steady_clock;
    static const auto first = clock::now();
    static auto last = first;
    static std::uint64_t frame = 0;
    static RECT placed{};
    static wchar_t text[160];  // static: a send that times out is still delivered later
    static unsigned private_mb = 0;
    static const bool started = (std::thread(label_thread).detach(), true);
    (void)started;
    static std::FILE* const rows = [] {
        const char* path = std::getenv("LEONIDA_FRAME_LOG");
        std::FILE* f = std::fopen(path && *path ? path : "frame_stats.csv", "w");
        if (f)
            std::fputs("t_ms,frame,frame_ms,draws,dispatches,indirect,submissions,pipeline_builds,pipeline_build_ms,"
                       "xma_voices,xma_pumps,private_mb\n", f);
        return f;
    }();
    const auto now = clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last).count();
    if (ms < 1.0) return;  // a second PresentX of the same frame (the intro makes two back to back)
    last = now;
    const unsigned d = draws.exchange(0), c = dispatches.exchange(0), i = indirect.exchange(0);
    const unsigned s = submissions.exchange(0), b = builds.exchange(0), bm = build_ms.exchange(0);
    const unsigned pumps = xma_pumps.exchange(0), voices = xma_voices.load();
    if (frame++ % 30 == 0) {
        PROCESS_MEMORY_COUNTERS_EX memory{};
        memory.cb = sizeof(memory);
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
                                    sizeof(memory)))
            private_mb = unsigned(memory.PrivateUsage >> 20);
    }
    if (rows) {
        std::fprintf(rows, "%.1f,%llu,%.1f,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                     std::chrono::duration<double, std::milli>(now - first).count(),
                     static_cast<unsigned long long>(frame), ms, d, c, i, s, b, bm, voices, pumps, private_mb);
        if (frame % 30 == 0 || ms > 100.0) std::fflush(rows);  // a crash loses at most a second of rows
    }
    const HWND ours = label.load();
    if (!ours) return;
    // Follow the game's window: its rectangle is read from the system, no message to its thread.
    RECT at{};
    if (GetWindowRect(game, &at) && (at.left != placed.left || at.top != placed.top || frame == 1)) {
        placed = at;
        SetWindowPos(ours, HWND_TOPMOST, at.left + 160, at.top + 6, 640, 20,
                     SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_ASYNCWINDOWPOS);
    }
    swprintf(text, 159, L" frame %llu | %.1f ms (%.0f fps) | %u draws, %u indirect, %u dispatches | %u submissions",
             static_cast<unsigned long long>(frame), ms, 1000.0 / ms, d, i, c, s);
    DWORD_PTR ignored = 0;  // our own thread answers at once; the timeout only guards PresentX
    SendMessageTimeoutW(ours, WM_SETTEXT, 0, reinterpret_cast<LPARAM>(text), SMTO_NORMAL, 5, &ignored);
}
#endif
}  // namespace leonida::frame_stats
