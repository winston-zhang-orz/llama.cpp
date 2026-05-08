// ggml-gcu: Enflame GCU (TOPS) backend
//
// Spec: docs/superpowers/specs/2026-05-08-gcu-s60-backend-design.md
// Plan: docs/superpowers/plans/2026-05-08-gcu-s60-backend-mvp1.md
//
// MVP-1: skeleton with stubs for the public API. No ops yet.

#include "ggml-gcu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <topsaten/topsaten.h>
#include <tops/tops_runtime.h>

#include <cstdio>

// === Stubs (filled in subsequent phases) =============================

extern "C" {

int32_t ggml_backend_gcu_get_device_count(void) {
    int count = 0;
    if (topsGetDeviceCount(&count) != topsSuccess) {
        return 0;
    }
    return count;
}

void ggml_backend_gcu_get_device_description(int32_t /*device*/, char * description, size_t description_size) {
    snprintf(description, description_size, "Enflame GCU");
}

void ggml_backend_gcu_get_device_memory(int32_t /*device*/, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
}

bool ggml_backend_is_gcu(ggml_backend_t /*backend*/) {
    return false;
}

ggml_backend_t ggml_backend_gcu_init(int32_t /*device*/) {
    return nullptr;
}

ggml_backend_reg_t ggml_backend_gcu_reg(void) {
    return nullptr;
}

} // extern "C"
