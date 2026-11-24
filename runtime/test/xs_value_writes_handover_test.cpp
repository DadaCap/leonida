// The hand-over of the console's value writes (list slots 69 and 70) from the queue proxy to the queue
// that submits them: the proxy stages the records it collected from the lists, and the queue takes them during
// the call the proxy forwards.
//
// No game data, no GPU: the records are plain data.
#include "../shims/d3d12_xs_device.h"

#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "xs_value_writes_handover_test: %s\n", what);
    ++g_failures;
}

}  // namespace

int main() {
    using namespace rt::d3d12_xs;

    std::vector<leonida::ValueWrite> staged;
    staged.push_back(leonida::ValueWrite{0x1000, 7, 4, 0});

    stage_value_writes(staged);
    const std::vector<leonida::ValueWrite> taken = take_staged_value_writes();
    check(taken.size() == 1, "the queue takes the staged batch");
    check(!taken.empty() && taken[0].value == 7, "the staged batch is the one the proxy collected");
    check(take_staged_value_writes().empty(), "nothing is staged twice");

    stage_value_writes(staged);
    clear_staged_value_writes();
    check(take_staged_value_writes().empty(), "a batch the proxy dropped does not reach the next queue");

    if (g_failures) {
        std::fprintf(stderr, "xs_value_writes_handover_test: %d checks failed\n", g_failures);
        return 1;
    }
    std::puts("the console's value writes reach the queue that submits them");
    return 0;
}
