// ggml-gcu: buffer-type / buffer / host-buffer interface declarations.
//
// Cross-TU surface the backend/device glue in ggml-gcu.cpp needs: the
// buffer-type context struct (the device descriptor owns one), the
// device-buffer free_buffer fn (used to identify GCU buffers by vtable
// pointer), the buffer-type vtable, and the pinned host buffer type.

#pragma once

#include "common.h"
#include "gcu_pool.h"

struct ggml_backend_gcu_buffer_type_context {
    ggml_backend_gcu_context * ctx;
};

// Unique vtable pointer used by ggml_backend_buffer_is_gcu to detect a
// GCU device buffer (mirrors ggml_backend_buffer_is_cuda).
void ggml_backend_gcu_buffer_free_buffer(ggml_backend_buffer_t buffer);

extern const ggml_backend_buffer_type_i ggml_backend_gcu_buffer_type_i;

ggml_backend_buffer_type_t ggml_backend_gcu_host_buffer_type();
