// ggml-gcu: shared helper implementations.
//
// Implements the declarations in common.h: error formatting, dtype mapping,
// tensor-descriptor builder, broadcast/aliasing/dtype gates, rollback-flag
// accessors, and the topsaten init refcount.

#include "common.h"

#include <cstdlib>
#include <mutex>

// === Error handling =================================================

void ggml_gcu_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    GGML_LOG_ERROR("GCU error: %s\n  in function %s at %s:%d\n  call: %s\n",
                   msg ? msg : "(no message)", func, file, line, stmt);
    GGML_ABORT("GCU error");
}

const char * topsaten_status_to_str(topsatenStatus_t s) {
    switch (s) {
        case TOPSATEN_STATUS_SUCCESS:        return "TOPSATEN_STATUS_SUCCESS";
        case TOPSATEN_STATUS_ALLOC_FAILED:   return "TOPSATEN_STATUS_ALLOC_FAILED";
        case TOPSATEN_STATUS_BAD_PARAM:      return "TOPSATEN_STATUS_BAD_PARAM";
        case TOPSATEN_STATUS_NOT_SUPPORT:    return "TOPSATEN_STATUS_NOT_SUPPORT";
        case TOPSATEN_STATUS_INTERNAL_ERROR: return "TOPSATEN_STATUS_INTERNAL_ERROR";
        case TOPSATEN_STATUS_RUNTIME_ERROR:  return "TOPSATEN_STATUS_RUNTIME_ERROR";
        case TOPSATEN_STATUS_EXECUTE_ERROR:  return "TOPSATEN_STATUS_EXECUTE_ERROR";
    }
    return "TOPSATEN_STATUS_UNKNOWN";
}

// === Process-level topsaten init refcount ===========================

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

// === Dtype mapping & Q-typed support ================================

topsatenDataType_t ggml_to_topsaten_dtype(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return TOPSATEN_DATA_FP32;
        case GGML_TYPE_F16:  return TOPSATEN_DATA_FP16;
        case GGML_TYPE_BF16: return TOPSATEN_DATA_BF16;
        case GGML_TYPE_I32:  return TOPSATEN_DATA_I32;
        default:             return TOPSATEN_DATA_NONE;
    }
}

bool gcu_q_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0 ||
           t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
           t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q3_K;
}

void gcu_q_dequantize_to_f32(ggml_type type, const void * src,
                             float * dst, int64_t n_elem) {
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    GGML_ASSERT(tt->to_float != nullptr);
    tt->to_float(src, dst, n_elem);
}

bool gcu_dtype_supported(ggml_type t) {
    // MVP-5a: BF16 added to the device's first-class activation dtypes.
    // The topsaten SDK exposes TOPSATEN_DATA_BF16 across the elementwise,
    // norm, softmax, GLU, ROPE, and matmul kernels we already use; per-op
    // gates above narrow this where a specific kernel is BF16-incompatible.
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
}

bool gcu_all_inputs_supported_dtype(const ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !gcu_dtype_supported(op->src[i]->type)) return false;
    }
    return gcu_dtype_supported(op->type);
}

bool gcu_numpy_broadcastable(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i] && a->ne[i] != 1 && b->ne[i] != 1) return false;
    }
    return true;
}

bool gcu_dst_aliases_src0_at_runtime(const ggml_tensor * dst) {
    return dst->src[0] && dst->data && dst->src[0]->data &&
           dst->data == dst->src[0]->data;
}

// === Tensor descriptor builder ======================================

topsatenTensor make_topsaten_tensor(const ggml_tensor * t, gcu_tensor_dims & out_dims) {
    GGML_ASSERT(t != nullptr);
    GGML_ASSERT(t->data != nullptr);

    topsatenDataType_t dtype = ggml_to_topsaten_dtype(t->type);
    GGML_ASSERT(dtype != TOPSATEN_DATA_NONE);

    int rank = ggml_n_dims(t);
    if (rank < 1)             rank = 1;
    if (rank > GGML_MAX_DIMS) rank = GGML_MAX_DIMS;

    // ggml stores ne[]/nb[] in slowest-last reversed-PyTorch order.
    // Build PyTorch order (slowest first), with strides in elements.
    const size_t bpe = ggml_type_size(t->type);
    for (int i = 0; i < rank; i++) {
        out_dims.dims[i] = t->ne[rank - 1 - i];
        out_dims.strs[i] = t->nb[rank - 1 - i] / (int64_t) bpe;
    }
    topsatenSize_t shape (out_dims.dims, rank);
    topsatenSize_t stride(out_dims.strs, rank);

    return topsatenTensor(shape, stride, dtype, t->data);
}

// === Rollback-flag accessors ========================================

bool gcu_async_disabled() {
    static const bool disabled = (getenv("GGML_GCU_NO_ASYNC_COPY") != nullptr);
    return disabled;
}

bool gcu_queued_ops_disabled() {
    static const bool disabled = (getenv("GGML_GCU_NO_QUEUED_OPS") != nullptr);
    return disabled;
}

// === Per-context state ==============================================

#include <cstdio>

ggml_backend_gcu_context::ggml_backend_gcu_context(int32_t dev) : device(dev), pool(dev) {
    deferred_frees.reserve(64);
    TOPS_CHECK(topsSetDevice(device));
    gcu_global_init_inc();
    TOPS_CHECK(topsStreamCreate(&compute_stream));
    TOPS_CHECK(topsStreamCreate(&copy_stream));
    TOPS_CHECK(topsEventCreateWithFlags(&last_copy_event,    topsEventDisableTiming));
    TOPS_CHECK(topsEventCreateWithFlags(&last_compute_event, topsEventDisableTiming));

    char buf[GGML_GCU_NAME_MAX];
    snprintf(buf, sizeof(buf), "GCU%d", device);
    name = buf;

    topsDeviceProp_t prop{};
    if (topsGetDeviceProperties(&prop, device) == topsSuccess) {
        description = prop.name;
    } else {
        description = "Enflame GCU";
    }
}

ggml_backend_gcu_context::~ggml_backend_gcu_context() {
    if (ones_n0) {
        pool.free(ones_n0, ones_n0_bytes);
        ones_n0 = nullptr;
        ones_n0_bytes = 0;
        ones_n0_count = 0;
    }
    if (zero_bias) {
        pool.free(zero_bias, zero_bias_bytes);
        zero_bias = nullptr;
        zero_bias_bytes = 0;
    }
    if (last_compute_event) {
        TOPS_CHECK(topsEventDestroy(last_compute_event));
        last_compute_event = nullptr;
    }
    if (last_copy_event) {
        TOPS_CHECK(topsEventDestroy(last_copy_event));
        last_copy_event = nullptr;
    }
    if (copy_stream) {
        TOPS_CHECK(topsStreamSynchronize(copy_stream));
        TOPS_CHECK(topsStreamDestroy(copy_stream));
        copy_stream = nullptr;
    }
    if (compute_stream) {
        TOPS_CHECK(topsStreamSynchronize(compute_stream));
        TOPS_CHECK(topsStreamDestroy(compute_stream));
        compute_stream = nullptr;
    }
    gcu_global_init_dec();
}

// === Scratch release ================================================

void gcu_release_scratch(ggml_backend_gcu_context * ctx, void * p, size_t sz) {
    if (!p) return;
    if (gcu_queued_ops_disabled()) {
        TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
        ctx->pool.free(p, sz);
    } else {
        ctx->defer_free(p, sz);
    }
}
