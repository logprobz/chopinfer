/* Optional profiler annotations. This helper submits no device work. */
#include <nvtx3/nvToolsExt.h>

void chopinfer_profile_range_v1(const char *name, int begin) {
    if (begin) {
        nvtxRangePushA(name);
    } else {
        nvtxRangePop();
    }
}
