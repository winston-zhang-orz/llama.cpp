// ggml-gcu: shared declarations.
//
// Helpers shared by every translation unit in the GCU backend: error-check
// macros, dtype mapping, tensor-descriptor builder, scratch-release helper,
// rollback flag accessors, and the cross-cutting per-op helpers
// (broadcast / aliasing / dtype gates).
//
// File layout per docs/superpowers/specs/2026-05-10-gcu-mvp5c-file-split-design.md.

#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <topsaten/topsaten.h>
#include <tops/tops_runtime.h>

// topsaten functions (Add, Mul, Linear, IndexSelect, Copy, To, ...) live in
// the topsaten:: namespace. The types (topsatenTensor, topsatenScalar_t,
// topsatenStatus_t, ...) are at global scope. Pull the namespaces in here
// since every GCU TU is a topsaten wrapper. The vllm-style op family
// (RmsNorm, RotaryEmbedding, ...) is in topsvllm::.
using namespace topsaten;
using namespace topsvllm;

#include <cstddef>
#include <cstdint>

// Forward decls — full ggml_backend_gcu_context lives in ggml-gcu.cpp until
// commit 2 lifts the pool out into its own header.
struct ggml_backend_gcu_context;

// === Error handling =================================================

[[noreturn]]
void ggml_gcu_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

const char * topsaten_status_to_str(topsatenStatus_t s);

#define TOPS_CHECK(stmt)                                                            \
    do {                                                                            \
        topsError_t err__ = (stmt);                                                 \
        if (err__ != topsSuccess) {                                                 \
            ggml_gcu_error(#stmt, __func__, __FILE__, __LINE__,                     \
                           topsGetErrorString(err__));                              \
        }                                                                           \
    } while (0)

#define TOPSATEN_CHECK(stmt)                                                        \
    do {                                                                            \
        topsatenStatus_t s__ = (stmt);                                              \
        if (s__ != TOPSATEN_STATUS_SUCCESS) {                                       \
            ggml_gcu_error(#stmt, __func__, __FILE__, __LINE__,                     \
                           topsaten_status_to_str(s__));                            \
        }                                                                           \
    } while (0)

// === Process-level topsaten init refcount ===========================
//
// topsatenInit / topsatenFinalize are documented as process-global. Wrap
// with a mutex+counter so multiple ggml_backend_gcu contexts (one per
// device) bracket lifetime correctly without re-init.

void gcu_global_init_inc();
void gcu_global_init_dec();

// === Dtype mapping & Q-typed support ================================

topsatenDataType_t ggml_to_topsaten_dtype(ggml_type t);

// MVP-3a: Q-typed weight tensors are dequantized to F16 at set_tensor
// time and stored as F16 on the device. This helper says which formats
// we accept; non-supported Q-types fall back to CPU via supports_op.
bool gcu_q_supported(ggml_type t);

// Generic dequantize-to-F32 via ggml's per-type traits (libggml-base).
// Works for any Q-type ggml supports; we use it only for those
// gcu_q_supported() accepts.
void gcu_q_dequantize_to_f32(ggml_type type, const void * src,
                             float * dst, int64_t n_elem);

// MVP-5a: BF16 added alongside F32/F16 as first-class device dtypes.
bool gcu_dtype_supported(ggml_type t);

bool gcu_all_inputs_supported_dtype(const ggml_tensor * op);

// topsaten's elementwise ops accept numpy/PyTorch-style broadcasting
// (each dim must be equal or one operand's dim is 1). ggml additionally
// allows "tiled" broadcasting which topsaten doesn't support.
bool gcu_numpy_broadcastable(const ggml_tensor * a, const ggml_tensor * b);

// topsaten's binary ops reject aliased output and lhs. Detect both the
// in-place ggml variant and the ggml-alloc memory-reuser collision case.
bool gcu_dst_aliases_src0_at_runtime(const ggml_tensor * dst);

// === Tensor descriptor builder ======================================

// Per-tensor scratch for shape/stride arrays the topsatenSize_t pointers
// must outlive. We carry them inline so the helper is self-contained.
struct gcu_tensor_dims {
    int64_t dims [GGML_MAX_DIMS];
    int64_t strs [GGML_MAX_DIMS];
};

topsatenTensor make_topsaten_tensor(const ggml_tensor * t, gcu_tensor_dims & out_dims);

// === Rollback-flag accessors ========================================

// MVP-4a: GGML_GCU_NO_ASYNC_COPY=1 forces synchronous H<->D copies on the
// async tensor slots (set_tensor_async / get_tensor_async).
bool gcu_async_disabled();

// MVP-4b: GGML_GCU_NO_QUEUED_OPS=1 keeps the pre-MVP-4b sync-and-free
// pattern (per-op topsStreamSynchronize + immediate pool.free).
bool gcu_queued_ops_disabled();

// === Scratch release ================================================

// MVP-4b: replaces every `topsStreamSynchronize + pool.free` pair inside op
// handlers. Defers the free to graph_compute's end-of-batch drain unless
// GGML_GCU_NO_QUEUED_OPS=1, in which case it restores the pre-MVP-4b
// behavior (synchronize then immediate free).
void gcu_release_scratch(ggml_backend_gcu_context * ctx, void * p, size_t sz);
