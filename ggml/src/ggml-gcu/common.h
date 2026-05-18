// ggml-gcu: shared helpers — error handling, dtype mapping, tensor wrapping.
//
// This header is the common base every translation unit in the GCU backend
// includes. It pulls in the topsaten/topsrt SDK headers, the check macros,
// and the small dtype / tensor-descriptor helpers.

#pragma once

#include "ggml-gcu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <topsaten/topsaten.h>
#include <tops/tops_runtime.h>

// topsaten functions (Add, Mul, Linear, IndexSelect, Copy, To, ...) live in
// the topsaten:: namespace. The types (topsatenTensor, topsatenScalar_t,
// topsatenStatus_t, ...) are at global scope. Pull the namespace in for
// readability since this whole backend is a topsaten wrapper. The vllm-style
// op family (RmsNorm, RotaryEmbedding, ...) is in topsvllm::.
using namespace topsaten;
using namespace topsvllm;

#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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

// === RAII wrappers for topsrt stream / event handles ================
//
// Thin owning wrappers around topsStream_t / topsEvent_t. Construction
// allocates the handle, destruction releases it (a stream synchronizes
// before destroy, matching the pre-RAII manual cleanup). Copy is deleted;
// move transfers ownership. The implicit conversion operator lets a
// wrapper be passed anywhere the raw handle was used before, so op call
// sites are unchanged.

class gcu_stream {
public:
    gcu_stream() { TOPS_CHECK(topsStreamCreate(&s_)); }
    ~gcu_stream() {
        if (s_) {
            TOPS_CHECK(topsStreamSynchronize(s_));
            TOPS_CHECK(topsStreamDestroy(s_));
        }
    }
    gcu_stream(const gcu_stream &)             = delete;
    gcu_stream & operator=(const gcu_stream &) = delete;
    gcu_stream(gcu_stream && o) noexcept : s_(o.s_) { o.s_ = nullptr; }

    operator topsStream_t() const { return s_; }

private:
    topsStream_t s_ = nullptr;
};

class gcu_event {
public:
    gcu_event() { TOPS_CHECK(topsEventCreateWithFlags(&e_, topsEventDisableTiming)); }
    ~gcu_event() {
        if (e_) {
            TOPS_CHECK(topsEventDestroy(e_));
        }
    }
    gcu_event(const gcu_event &)             = delete;
    gcu_event & operator=(const gcu_event &) = delete;
    gcu_event(gcu_event && o) noexcept : e_(o.e_) { o.e_ = nullptr; }

    operator topsEvent_t() const { return e_; }

private:
    topsEvent_t e_ = nullptr;
};

// === Tensor mapping =================================================
//
// Convert a ggml_tensor descriptor into a topsatenTensor that wraps the
// same device memory. Caller must keep the underlying ggml_tensor (and
// its buffer) alive for the duration of any op call using this wrapper.

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

// Per-tensor scratch for shape/stride arrays the topsatenSize_t pointers
// must outlive. We carry them inline so the helper is self-contained.
struct gcu_tensor_dims {
    int64_t dims [GGML_MAX_DIMS];
    int64_t strs [GGML_MAX_DIMS];
};

topsatenTensor make_topsaten_tensor(const ggml_tensor * t, gcu_tensor_dims & out_dims);

bool gcu_dtype_supported(ggml_type t);
