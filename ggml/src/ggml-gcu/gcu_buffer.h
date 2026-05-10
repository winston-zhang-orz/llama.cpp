// ggml-gcu: device + host buffer interfaces.
//
// Public surface needed by ggml-gcu.cpp:
//   * ggml_backend_buffer_is_gcu() — async-copy slot uses this to gate
//     whether a tensor's buffer is one of ours (so it can issue
//     topsMemcpyAsync against tensor->data).
//   * ggml_backend_gcu_buffer_type_i — used by the per-device buffer-type
//     constructor in ggml-gcu.cpp's device_context::get_buft.
//   * ggml_backend_gcu_host_buffer_type() — handed out by the device
//     interface's get_host_buffer_type slot for pinned host KV memory.

#pragma once

#include "common.h"

#include "ggml-backend-impl.h"

bool ggml_backend_buffer_is_gcu(ggml_backend_buffer_t buffer);

// The buffer-type context owned by per-device device_context; held by
// ggml-gcu.cpp's device_context::get_buft. Definition is here so the
// device_context can construct one without dragging the buffer impl in.
struct ggml_backend_gcu_buffer_type_context {
    ggml_backend_gcu_context * ctx;
};

extern const ggml_backend_buffer_type_i ggml_backend_gcu_buffer_type_i;

ggml_backend_buffer_type_t ggml_backend_gcu_host_buffer_type();
