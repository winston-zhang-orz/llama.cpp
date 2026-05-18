// ggml-gcu: process-level topsaten init refcount. See gcu_pool.h.

#include "gcu_pool.h"

static std::mutex g_init_mu;
static int        g_init_refcount = 0;

void gcu_global_init_inc() {
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_init_refcount++ == 0) {
        TOPSATEN_CHECK(topsatenInit());
    }
}

void gcu_global_init_dec() {
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (--g_init_refcount == 0) {
        TOPSATEN_CHECK(topsatenFinalize());
    }
}
