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

// topsaten functions (Add, Mul, Linear, IndexSelect, Copy, To, ...) live in
// the topsaten:: namespace. The types (topsatenTensor, topsatenScalar_t,
// topsatenStatus_t, ...) are at global scope. Pull the namespace in for
// readability since this whole file is a topsaten wrapper. The vllm-style
// op family (RmsNorm, RotaryEmbedding, ...) is in topsvllm::.
using namespace topsaten;
using namespace topsvllm;

#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// === Backend GUID ===================================================

static ggml_guid_t ggml_backend_gcu_guid() {
    // Stable 16-byte UUID — generated once for this backend; never changes.
    static ggml_guid guid = { 0x9e, 0x3f, 0x12, 0xa4, 0x77, 0x88, 0x4b, 0xc1,
                              0x90, 0x2d, 0xe5, 0x06, 0xf4, 0x18, 0x21, 0x33 };
    return &guid;
}

// === Error handling =================================================

[[noreturn]]
static void ggml_gcu_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    GGML_LOG_ERROR("GCU error: %s\n  in function %s at %s:%d\n  call: %s\n",
                   msg ? msg : "(no message)", func, file, line, stmt);
    GGML_ABORT("GCU error");
}

static const char * topsaten_status_to_str(topsatenStatus_t s) {
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

// Forward declarations: defined in the tensor-mapping section, used
// earlier in the buffer-type code (init_tensor / get_alloc_size /
// set_tensor).
static bool gcu_q_supported(ggml_type t);
static void gcu_q_dequantize_to_f32(ggml_type type, const void * src,
                                    float * dst, int64_t n_elem);

// === Process-level topsaten init refcount ===========================
//
// topsatenInit / topsatenFinalize are documented as process-global. Wrap
// with a mutex+counter so multiple ggml_backend_gcu contexts (one per
// device) bracket lifetime correctly without re-init.

static std::mutex g_init_mu;
static int        g_init_refcount = 0;

static void gcu_global_init_inc() {
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_init_refcount++ == 0) {
        TOPSATEN_CHECK(topsatenInit());
    }
}

static void gcu_global_init_dec() {
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (--g_init_refcount == 0) {
        TOPSATEN_CHECK(topsatenFinalize());
    }
}

// === LIFO size-keyed pool allocator =================================

// Recurring intermediate-tensor allocations during a forward pass have a
// small set of distinct sizes. Cache freed slabs by rounded size so we
// avoid topsMalloc/topsFree on the hot path. Cap the cache at 4 GiB to
// bound retained device memory.

class gcu_pool {
public:
    explicit gcu_pool(int32_t device, size_t cap_bytes = (size_t) 4 << 30)
        : device_(device), cap_(cap_bytes) {}

    ~gcu_pool() { drain(); }

    void * alloc(size_t size) {
        size_t key = round_up(size);
        std::lock_guard<std::mutex> lk(mu_);
        auto it = free_lists_.find(key);
        if (it != free_lists_.end() && !it->second.empty()) {
            void * p = it->second.back();
            it->second.pop_back();
            cached_bytes_ -= key;
            return p;
        }
        void * p = nullptr;
        TOPS_CHECK(topsSetDevice(device_));
        TOPS_CHECK(topsMalloc(&p, key));
        return p;
    }

    void free(void * p, size_t size) {
        if (!p) return;
        size_t key = round_up(size);
        std::lock_guard<std::mutex> lk(mu_);
        if (cached_bytes_ + key > cap_) {
            TOPS_CHECK(topsSetDevice(device_));
            TOPS_CHECK(topsFree(p));
            return;
        }
        free_lists_[key].push_back(p);
        cached_bytes_ += key;
    }

private:
    static size_t round_up(size_t n) {
        constexpr size_t G = 256;
        return (n + G - 1) & ~(G - 1);
    }
    void drain() {
        std::lock_guard<std::mutex> lk(mu_);
        TOPS_CHECK(topsSetDevice(device_));
        for (auto & kv : free_lists_) {
            for (void * p : kv.second) {
                TOPS_CHECK(topsFree(p));
            }
        }
        free_lists_.clear();
        cached_bytes_ = 0;
    }

    int32_t device_;
    size_t  cap_;
    size_t  cached_bytes_ = 0;
    std::unordered_map<size_t, std::vector<void*>> free_lists_;
    std::mutex mu_;
};

// === Per-context state ==============================================

#define GGML_GCU_NAME_MAX 64

struct ggml_backend_gcu_context {
    int32_t      device      = 0;
    std::string  name;          // "GCU0", "GCU1", ...
    std::string  description;   // populated from topsGetDeviceProperties
    topsStream_t stream        = nullptr;
    gcu_pool     pool;          // declared after stream so it's destroyed first

    // Per-context zero-filled scratch used as bias for topsatenLinear's
    // mandatory bias parameter. Grows on demand to the largest output row
    // count seen so far, in F32 (largest dtype we use), and reinterpreted
    // for F16 calls. Protected by the same single stream as compute, so
    // no lock is needed.
    void *  zero_bias       = nullptr;
    size_t  zero_bias_bytes = 0;

    // Per-context F32 ones buffer used as the gamma argument to
    // topsvllmRmsNorm (which requires a real weight tensor).
    void *  ones_n0         = nullptr;
    size_t  ones_n0_bytes   = 0;
    int64_t ones_n0_count   = 0;

    explicit ggml_backend_gcu_context(int32_t dev) : device(dev), pool(dev) {
        TOPS_CHECK(topsSetDevice(device));
        gcu_global_init_inc();
        TOPS_CHECK(topsStreamCreate(&stream));

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

    ~ggml_backend_gcu_context() {
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
        if (stream) {
            TOPS_CHECK(topsStreamSynchronize(stream));
            TOPS_CHECK(topsStreamDestroy(stream));
            stream = nullptr;
        }
        gcu_global_init_dec();
    }

    ggml_backend_gcu_context(const ggml_backend_gcu_context &) = delete;
    ggml_backend_gcu_context & operator=(const ggml_backend_gcu_context &) = delete;
};

// === Device buffer ==================================================

struct ggml_backend_gcu_buffer_context {
    ggml_backend_gcu_context * ctx;
    void *  base;
    size_t  size;
};

static void ggml_backend_gcu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    bctx->ctx->pool.free(bctx->base, bctx->size);
    delete bctx;
}

static void * ggml_backend_gcu_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    return bctx->base;
}

static enum ggml_status ggml_backend_gcu_buffer_init_tensor(ggml_backend_buffer_t /*buffer*/, ggml_tensor * /*tensor*/) {
    // Tensors are views into the slab; ggml-alloc has already set tensor->data.
    // Q-typed weight tensors store F16 on the device but we deliberately do
    // not rewrite nb[] here: test-backend-ops re-runs the same ggml_tensor
    // on CPU for comparison, and CPU MUL_MAT asserts nb[0] matches the
    // declared Q4 type size. gcu_op_mul_mat derives F16 strides locally.
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_gcu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));
    TOPS_CHECK(topsMemset((char *) tensor->data + offset, value, size));
}

static void ggml_backend_gcu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));

    // MVP-3a: Q-typed full-tensor uploads are dequantized to F16 on host
    // before transfer. Q packing is per-row (and per super-block for
    // Q4_K) so partial writes don't translate cleanly; ggml's model
    // loader uses full-tensor writes for weights, which is what we
    // assert here.
    if (gcu_q_supported(tensor->type)) {
        GGML_ASSERT(offset == 0);
        const int64_t n_elem    = ggml_nelements(tensor);
        const size_t  expect_sz = ggml_row_size(tensor->type, n_elem);
        GGML_ASSERT(size == expect_sz);

        std::vector<float>       host_f32(n_elem);
        std::vector<ggml_fp16_t> host_f16(n_elem);
        gcu_q_dequantize_to_f32(tensor->type, data, host_f32.data(), n_elem);
        ggml_fp32_to_fp16_row(host_f32.data(), host_f16.data(), n_elem);

        TOPS_CHECK(topsMemcpy(tensor->data, host_f16.data(),
                              (size_t) n_elem * sizeof(ggml_fp16_t),
                              topsMemcpyHostToDevice));
        return;

    }

    TOPS_CHECK(topsMemcpy((char *) tensor->data + offset, data, size, topsMemcpyHostToDevice));
}

static void ggml_backend_gcu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));
    TOPS_CHECK(topsMemcpy(data, (const char *) tensor->data + offset, size, topsMemcpyDeviceToHost));
}

static bool ggml_backend_gcu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_buffer_is_host(src->buffer) && src->buffer->buft == dst->buffer->buft) {
        auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
        TOPS_CHECK(topsSetDevice(bctx->ctx->device));
        TOPS_CHECK(topsMemcpy(dst->data, src->data, ggml_nbytes(src), topsMemcpyDeviceToDevice));
        return true;
    }
    return false;
}

static void ggml_backend_gcu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));
    TOPS_CHECK(topsMemset(bctx->base, value, bctx->size));
}

static const ggml_backend_buffer_i ggml_backend_gcu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_gcu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_gcu_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_gcu_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_gcu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_gcu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_gcu_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_gcu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_gcu_buffer_clear,
    /* .reset           = */ nullptr,
};

// === Buffer type ====================================================

struct ggml_backend_gcu_buffer_type_context {
    ggml_backend_gcu_context * ctx;
};

static const char * ggml_backend_gcu_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * btctx = (ggml_backend_gcu_buffer_type_context *) buft->context;
    return btctx->ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_gcu_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * btctx = (ggml_backend_gcu_buffer_type_context *) buft->context;
    void * base  = btctx->ctx->pool.alloc(size);

    auto * bctx = new ggml_backend_gcu_buffer_context{ btctx->ctx, base, size };
    return ggml_backend_buffer_init(buft, ggml_backend_gcu_buffer_i, bctx, size);
}

static size_t ggml_backend_gcu_buffer_type_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    return 256;
}

static size_t ggml_backend_gcu_buffer_type_get_max_size(ggml_backend_buffer_type_t /*buft*/) {
    return SIZE_MAX;
}

// MVP-3a: Q-typed weight tensors are stored as F16 on the device.
// Over-report the allocation size so the slab is large enough.
static size_t ggml_backend_gcu_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t /*buft*/, const ggml_tensor * t) {
    if (gcu_q_supported(t->type)) {
        return (size_t) ggml_nelements(t) * sizeof(uint16_t);
    }
    return ggml_nbytes(t);
}

// === Host (pinned) buffer type ======================================
//
// Provides pinned host memory via topsHostMalloc. llama.cpp's KV cache
// allocator picks this up when offered by the device (via the
// get_host_buffer_type slot), giving faster H<->D copies for the
// CPU-resident cache when running with -nkvo. Pattern matches CUDA's
// ggml_backend_cuda_host_buffer_type — wrap pinned memory in a normal
// CPU buffer and only override free.

static void ggml_backend_gcu_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer->context) {
        TOPS_CHECK(topsHostFree(buffer->context));
    }
}

static void * gcu_host_malloc(size_t size) {
    if (getenv("GGML_GCU_NO_PINNED") != nullptr) {
        return nullptr;
    }
    void * ptr = nullptr;
    topsError_t err = topsHostMalloc(&ptr, size, topsHostMallocDefault);
    if (err != topsSuccess) {
        GGML_LOG_DEBUG("%s: topsHostMalloc(%.2f MiB) failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, topsGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

static const char * ggml_backend_gcu_host_buffer_type_name(ggml_backend_buffer_type_t /*buft*/) {
    return "GCU_Host";
}

static ggml_backend_buffer_t ggml_backend_gcu_host_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = gcu_host_malloc(size);
    if (!ptr) {
        // fall back to a normal CPU buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft             = buft;
    buffer->iface.free_buffer = ggml_backend_gcu_host_buffer_free_buffer;
    return buffer;
}

static ggml_backend_buffer_type_t ggml_backend_gcu_host_buffer_type() {
    static ggml_backend_buffer_type bt = {
        /* .iface    = */ {
            /* .get_name        = */ ggml_backend_gcu_host_buffer_type_name,
            /* .alloc_buffer    = */ ggml_backend_gcu_host_buffer_type_alloc_buffer,
            /* .get_alignment   = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size    = */ nullptr,
            /* .get_alloc_size  = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host         = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ nullptr,   // patched by device.get_host_buffer_type
        /* .context  = */ nullptr,
    };
    return &bt;
}

static const ggml_backend_buffer_type_i ggml_backend_gcu_buffer_type_i = {
    /* .get_name        = */ ggml_backend_gcu_buffer_type_name,
    /* .alloc_buffer    = */ ggml_backend_gcu_buffer_type_alloc_buffer,
    /* .get_alignment   = */ ggml_backend_gcu_buffer_type_get_alignment,
    /* .get_max_size    = */ ggml_backend_gcu_buffer_type_get_max_size,
    /* .get_alloc_size  = */ ggml_backend_gcu_buffer_type_get_alloc_size,
    /* .is_host         = */ nullptr,    // device buffer (not host-accessible)
};

// === ggml_backend_i (mostly stubs for now) ==========================

static const char * ggml_backend_gcu_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_gcu_free(ggml_backend_t backend) {
    // Backend's context is non-owning — the actual ggml_backend_gcu_context
    // lives in the registry's device descriptor and outlives the backend
    // wrapper. Only delete the wrapper.
    delete backend;
}

static void ggml_backend_gcu_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
}

// Forward declaration: gcu_compute_node lives in the Op dispatch section
// further down in the file.
static bool gcu_compute_node(ggml_backend_gcu_context * ctx, ggml_tensor * node);

static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) continue;
        if (!gcu_compute_node(ctx, node)) {
            GGML_LOG_ERROR("%s: op %s not implemented or failed\n",
                           __func__, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }
    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_gcu_i = {
    /* .get_name             = */ ggml_backend_gcu_name,
    /* .free                 = */ ggml_backend_gcu_free,
    /* .set_tensor_async     = */ nullptr,
    /* .get_tensor_async     = */ nullptr,
    /* .set_tensor_2d_async  = */ nullptr,
    /* .get_tensor_2d_async  = */ nullptr,
    /* .cpy_tensor_async     = */ nullptr,
    /* .synchronize          = */ ggml_backend_gcu_synchronize,
    /* .graph_plan_create    = */ nullptr,
    /* .graph_plan_free      = */ nullptr,
    /* .graph_plan_update    = */ nullptr,
    /* .graph_plan_compute   = */ nullptr,
    /* .graph_compute        = */ ggml_backend_gcu_graph_compute,
    /* .event_record         = */ nullptr,
    /* .event_wait           = */ nullptr,
    /* .graph_optimize       = */ nullptr,
};

// === Tensor mapping =================================================
//
// Convert a ggml_tensor descriptor into a topsatenTensor that wraps the
// same device memory. Caller must keep the underlying ggml_tensor (and
// its buffer) alive for the duration of any op call using this wrapper.

static topsatenDataType_t ggml_to_topsaten_dtype(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return TOPSATEN_DATA_FP32;
        case GGML_TYPE_F16: return TOPSATEN_DATA_FP16;
        case GGML_TYPE_I32: return TOPSATEN_DATA_I32;
        default:            return TOPSATEN_DATA_NONE;
    }
}

// MVP-3a: Q-typed weight tensors are dequantized to F16 at set_tensor
// time and stored as F16 on the device. This helper says which formats
// we accept; non-supported Q-types fall back to CPU via supports_op.
static bool gcu_q_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K;
}

// Generic dequantize-to-F32 via ggml's per-type traits (libggml-base).
// Works for any Q-type ggml supports; we use it only for those
// gcu_q_supported() accepts.
static void gcu_q_dequantize_to_f32(ggml_type type, const void * src,
                                    float * dst, int64_t n_elem) {
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    GGML_ASSERT(tt->to_float != nullptr);
    tt->to_float(src, dst, n_elem);
}

// Per-tensor scratch for shape/stride arrays the topsatenSize_t pointers
// must outlive. We carry them inline so the helper is self-contained.
struct gcu_tensor_dims {
    int64_t dims [GGML_MAX_DIMS];
    int64_t strs [GGML_MAX_DIMS];
};

static topsatenTensor make_topsaten_tensor(const ggml_tensor * t, gcu_tensor_dims & out_dims) {
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

// === Op dispatch =====================================================

static bool gcu_dtype_supported(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16;
}

static bool gcu_all_inputs_supported_dtype(const ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !gcu_dtype_supported(op->src[i]->type)) return false;
    }
    return gcu_dtype_supported(op->type);
}

// topsaten's elementwise ops accept numpy/PyTorch-style broadcasting
// (each dim must be equal or one operand's dim is 1). ggml additionally
// allows "tiled" broadcasting (a->ne[i] is a multiple of b->ne[i]) which
// topsaten does not support; refuse those cases so the scheduler keeps
// them on CPU.
static bool gcu_numpy_broadcastable(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i] && a->ne[i] != 1 && b->ne[i] != 1) return false;
    }
    return true;
}

// topsaten's binary ops reject aliased output and lhs. This happens
// for ggml's in-place variants (dst->view_src == src[0]) AND for
// ordinary non-inplace ops when ggml-alloc's memory reuser places
// dst's slab over src[0]'s slab. Detect both at compute time via
// data-pointer comparison and route through a scratch copy.
static bool gcu_dst_aliases_src0_at_runtime(const ggml_tensor * dst) {
    return dst->src[0] && dst->data && dst->src[0]->data &&
           dst->data == dst->src[0]->data;
}

static bool gcu_op_add(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * lhs_t = dst->src[0];
    ggml_tensor * rhs_t = dst->src[1];

    // If dst aliases lhs (in-place op or memory reuse), copy lhs into a
    // scratch slab and use that as the topsatenAdd lhs. The scratch is
    // returned to the pool after the op completes on the stream.
    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->stream));
        lhs_data = scratch;
    }

    gcu_tensor_dims dout, dlhs, drhs;
    topsatenTensor out = make_topsaten_tensor(dst,   dout);

    // Build lhs manually using lhs_data (may be the scratch pointer).
    {
        const size_t bpe = ggml_type_size(lhs_t->type);
        int rank = ggml_n_dims(lhs_t); if (rank < 1) rank = 1;
        for (int i = 0; i < rank; i++) {
            dlhs.dims[i] = lhs_t->ne[rank - 1 - i];
            dlhs.strs[i] = lhs_t->nb[rank - 1 - i] / (int64_t) bpe;
        }
        topsatenSize_t shape (dlhs.dims, rank);
        topsatenSize_t stride(dlhs.strs, rank);
        topsatenTensor lhs(shape, stride, ggml_to_topsaten_dtype(lhs_t->type), lhs_data);
        topsatenTensor rhs = make_topsaten_tensor(rhs_t, drhs);

        topsatenScalar_t alpha;
        alpha.dtype = TOPSATEN_DATA_FP32;
        alpha.fval  = 1.0;
        TOPSATEN_CHECK(topsatenAdd(out, lhs, rhs, alpha, ctx->stream));
    }

    if (scratch) {
        // Synchronize before returning the scratch to the pool: the op is
        // queued on the stream and reading from scratch must complete.
        TOPS_CHECK(topsStreamSynchronize(ctx->stream));
        ctx->pool.free(scratch, scratch_bytes);
    }
    return true;
}

static bool gcu_op_mul(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * lhs_t = dst->src[0];
    ggml_tensor * rhs_t = dst->src[1];

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->stream));
        lhs_data = scratch;
    }

    gcu_tensor_dims dout, dlhs, drhs;
    topsatenTensor out = make_topsaten_tensor(dst, dout);

    {
        const size_t bpe = ggml_type_size(lhs_t->type);
        int rank = ggml_n_dims(lhs_t); if (rank < 1) rank = 1;
        for (int i = 0; i < rank; i++) {
            dlhs.dims[i] = lhs_t->ne[rank - 1 - i];
            dlhs.strs[i] = lhs_t->nb[rank - 1 - i] / (int64_t) bpe;
        }
        topsatenSize_t shape (dlhs.dims, rank);
        topsatenSize_t stride(dlhs.strs, rank);
        topsatenTensor lhs(shape, stride, ggml_to_topsaten_dtype(lhs_t->type), lhs_data);
        topsatenTensor rhs = make_topsaten_tensor(rhs_t, drhs);

        TOPSATEN_CHECK(topsatenMul(out, lhs, rhs, ctx->stream));
    }

    if (scratch) {
        TOPS_CHECK(topsStreamSynchronize(ctx->stream));
        ctx->pool.free(scratch, scratch_bytes);
    }
    return true;
}

static bool gcu_op_scale(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    float params[2];
    memcpy(params, dst->op_params, sizeof(params));
    const float scale = params[0];
    // bias != 0 is rejected upstream by supports_op for MVP-1.

    ggml_tensor * lhs_t = dst->src[0];

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->stream));
        lhs_data = scratch;
    }

    gcu_tensor_dims dout, dlhs;
    topsatenTensor out = make_topsaten_tensor(dst, dout);

    {
        const size_t bpe = ggml_type_size(lhs_t->type);
        int rank = ggml_n_dims(lhs_t); if (rank < 1) rank = 1;
        for (int i = 0; i < rank; i++) {
            dlhs.dims[i] = lhs_t->ne[rank - 1 - i];
            dlhs.strs[i] = lhs_t->nb[rank - 1 - i] / (int64_t) bpe;
        }
        topsatenSize_t shape (dlhs.dims, rank);
        topsatenSize_t stride(dlhs.strs, rank);
        topsatenTensor lhs(shape, stride, ggml_to_topsaten_dtype(lhs_t->type), lhs_data);

        topsatenScalar_t s;
        s.dtype = TOPSATEN_DATA_FP32;
        s.fval  = scale;
        TOPSATEN_CHECK(topsatenMul(out, lhs, s, ctx->stream));
    }

    if (scratch) {
        TOPS_CHECK(topsStreamSynchronize(ctx->stream));
        ctx->pool.free(scratch, scratch_bytes);
    }
    return true;
}

// Returns a device pointer to at least n_bytes of zero memory. Grows the
// per-context zero buffer on demand. Assumes single-stream serialization
// so no extra synchronization is needed between resize and use.
static void * gcu_get_zero_bias(ggml_backend_gcu_context * ctx, size_t n_bytes) {
    if (ctx->zero_bias_bytes < n_bytes) {
        if (ctx->zero_bias) {
            ctx->pool.free(ctx->zero_bias, ctx->zero_bias_bytes);
        }
        ctx->zero_bias       = ctx->pool.alloc(n_bytes);
        ctx->zero_bias_bytes = n_bytes;
        TOPS_CHECK(topsMemsetAsync(ctx->zero_bias, 0, n_bytes, ctx->stream));
    }
    return ctx->zero_bias;
}

// Returns a device pointer to a F32 ones buffer of at least `count` elements.
// Grows on demand. Always F32; callers cast to other dtypes via topsatenTo.
static void * gcu_get_ones_f32(ggml_backend_gcu_context * ctx, int64_t count) {
    if (ctx->ones_n0_count >= count) return ctx->ones_n0;
    if (ctx->ones_n0) {
        ctx->pool.free(ctx->ones_n0, ctx->ones_n0_bytes);
    }
    const size_t bytes = (size_t) count * sizeof(float);
    ctx->ones_n0       = ctx->pool.alloc(bytes);
    ctx->ones_n0_bytes = bytes;
    ctx->ones_n0_count = count;

    // Materialize ones: zero the buffer, then add scalar 1.0 broadcast.
    TOPS_CHECK(topsMemsetAsync(ctx->ones_n0, 0, bytes, ctx->stream));
    int64_t dims[1] = { count };
    int64_t strs[1] = { 1 };
    topsatenTensor t(topsatenSize_t(dims, 1), topsatenSize_t(strs, 1),
                     TOPSATEN_DATA_FP32, ctx->ones_n0);
    topsatenScalar_t one_lhs; one_lhs.dtype = TOPSATEN_DATA_FP32; one_lhs.fval = 1.0;
    topsatenScalar_t alpha;   alpha.dtype   = TOPSATEN_DATA_FP32; alpha.fval   = 1.0;
    // t = 1.0 + 1.0 * t (where t is currently 0) → t becomes 1.0
    TOPSATEN_CHECK(topsatenAdd(t, one_lhs, t, alpha, ctx->stream));
    return ctx->ones_n0;
}

// RMS_NORM. ggml_rms_norm: dst = x / sqrt(mean(x^2) + eps). No fused weight
// (the multiply happens via a downstream MUL op), but topsvllmRmsNorm
// requires a gamma weight argument — we pass an ones tensor of size ne[0].
static bool gcu_op_rms_norm(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t hidden_size = src->ne[0];

    // F32 ones buffer for gamma. If src is F16, cast to F16 first into a
    // per-call scratch (small — at most a few KiB).
    void * ones_f32 = gcu_get_ones_f32(ctx, hidden_size);
    void * gamma_data = ones_f32;
    size_t cast_bytes = 0;
    void * cast_buf   = nullptr;
    topsatenDataType_t gamma_dtype = TOPSATEN_DATA_FP32;
    if (src->type == GGML_TYPE_F16) {
        cast_bytes = (size_t) hidden_size * sizeof(uint16_t);
        cast_buf   = ctx->pool.alloc(cast_bytes);
        int64_t gd[1] = { hidden_size };
        int64_t gs[1] = { 1 };
        topsatenTensor f32_t(topsatenSize_t(gd, 1), topsatenSize_t(gs, 1),
                             TOPSATEN_DATA_FP32, ones_f32);
        topsatenTensor f16_t(topsatenSize_t(gd, 1), topsatenSize_t(gs, 1),
                             TOPSATEN_DATA_FP16, cast_buf);
        topsatenDataType_t target = TOPSATEN_DATA_FP16;
        TOPSATEN_CHECK(topsatenTo(f16_t, f32_t, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->stream));
        gamma_data  = cast_buf;
        gamma_dtype = TOPSATEN_DATA_FP16;
    }

    int64_t gamma_d[1] = { hidden_size };
    int64_t gamma_s[1] = { 1 };
    topsatenTensor gamma_t(topsatenSize_t(gamma_d, 1), topsatenSize_t(gamma_s, 1),
                           gamma_dtype, gamma_data);

    // topsteRmsNormFwd requires input rank in [2, 4]. ggml's RMS_NORM is
    // applied along the innermost dim, with shape [hidden_size, n_rows, 1, 1]
    // typically — collapse the outer dims into one and present as rank 2.
    const int64_t n_rows = src->ne[1] * src->ne[2] * src->ne[3];
    int64_t io_d[2] = { n_rows, hidden_size };
    int64_t io_s[2] = { hidden_size, 1 };
    topsatenTensor in_t (topsatenSize_t(io_d, 2), topsatenSize_t(io_s, 2),
                         ggml_to_topsaten_dtype(src->type), src->data);
    topsatenTensor out_t(topsatenSize_t(io_d, 2), topsatenSize_t(io_s, 2),
                         ggml_to_topsaten_dtype(dst->type), dst->data);

    topsatenScalar_t eps_s; eps_s.dtype = TOPSATEN_DATA_FP32; eps_s.fval = eps;

    TOPSATEN_CHECK(topsvllmRmsNorm(out_t, in_t, gamma_t, eps_s, ctx->stream));

    if (cast_buf) {
        TOPS_CHECK(topsStreamSynchronize(ctx->stream));
        ctx->pool.free(cast_buf, cast_bytes);
    }
    return true;
}

// MUL_MAT.
//
// ggml's MUL_MAT semantics: dst = src[0]^T @ src[1]
//   src[0] is the weight matrix in [K, M] ggml-shape (ne[0]=K, ne[1]=M).
//   src[1] is the input             in [K, N] ggml-shape (ne[0]=K, ne[1]=N).
//   dst                             in [M, N] ggml-shape (ne[0]=M, ne[1]=N).
//
// PyTorch Linear: out = x @ W^T + b, where x:[N,K], W:[M,K], b:[M].
// topsatenLinear requires lhs/rhs to be rank-2; we always build rank-2
// tensors explicitly (ggml_n_dims trims trailing 1s, which would give
// rank-1 for a [K, 1] weight and topsaten rejects that).
static bool gcu_op_mul_mat(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * w = dst->src[0];
    ggml_tensor * x = dst->src[1];

    const ggml_type wt = w->type;
    const ggml_type xt = x->type;
    const ggml_type ot = dst->type;

    // MVP-3a: Q-typed weights live on the device as F16 (Phase B/C). For
    // every dtype branch downstream, treat the weight as F16. Phase B
    // already rewrote w->nb[] to F16 strides at init_tensor time.
    const ggml_type wt_eff = gcu_q_supported(wt) ? GGML_TYPE_F16 : wt;

    auto build_2d = [](const ggml_tensor * t, int64_t (& d)[2], int64_t (& s)[2]) {
        const size_t bpe = ggml_type_size(t->type);
        d[0] = t->ne[1];
        d[1] = t->ne[0];
        s[0] = (int64_t) (t->nb[1] / bpe);
        s[1] = (int64_t) (t->nb[0] / bpe);
    };

    // For Q-typed weights, the device buffer holds F16 bytes (MVP-3a) but
    // ggml's nb[] still describes the Q4 packing. Build F16-stride dims
    // locally for the weight only.
    auto build_2d_q_as_f16 = [](const ggml_tensor * t, int64_t (& d)[2], int64_t (& s)[2]) {
        d[0] = t->ne[1];
        d[1] = t->ne[0];
        s[0] = t->ne[0];   // F16-element stride for next row
        s[1] = 1;
    };

    // All-F32 fast path.
    if (wt == GGML_TYPE_F32 && xt == GGML_TYPE_F32 && ot == GGML_TYPE_F32) {
        int64_t lhs_d[2], lhs_s[2], rhs_d[2], rhs_s[2], out_d[2], out_s[2];
        build_2d(x,   lhs_d, lhs_s);
        build_2d(w,   rhs_d, rhs_s);
        build_2d(dst, out_d, out_s);

        topsatenTensor lhs(topsatenSize_t(lhs_d, 2), topsatenSize_t(lhs_s, 2),
                           ggml_to_topsaten_dtype(xt), x->data);
        topsatenTensor rhs(topsatenSize_t(rhs_d, 2), topsatenSize_t(rhs_s, 2),
                           ggml_to_topsaten_dtype(wt), w->data);
        topsatenTensor out(topsatenSize_t(out_d, 2), topsatenSize_t(out_s, 2),
                           ggml_to_topsaten_dtype(ot), dst->data);

        const int64_t M = dst->ne[0];
        const size_t  bias_bytes = (size_t) M * ggml_type_size(ot);
        void * bias_dev = gcu_get_zero_bias(ctx, bias_bytes);
        int64_t bias_d[1] = { M };
        int64_t bias_s[1] = { 1 };
        topsatenTensor bias(topsatenSize_t(bias_d, 1), topsatenSize_t(bias_s, 1),
                            ggml_to_topsaten_dtype(ot), bias_dev);
        TOPSATEN_CHECK(topsatenLinear(out, lhs, rhs, bias, ctx->stream));
        return true;
    }

    // F16-weight path: cast input to F16 if needed, run F16 Linear, cast
    // output back to dst dtype. Q-typed weights take this path with their
    // device bytes interpreted as F16 (wt_eff == F16).
    GGML_ASSERT(wt_eff == GGML_TYPE_F16);
    const int64_t M = dst->ne[0];
    const int64_t K = w->ne[0];
    const int64_t N = x->ne[1];

    void * x_data = x->data;
    void * x_cast = nullptr;
    size_t x_cast_bytes = 0;
    if (xt == GGML_TYPE_F32) {
        x_cast_bytes = (size_t) N * K * sizeof(uint16_t);
        x_cast       = ctx->pool.alloc(x_cast_bytes);

        int64_t xd[2] = { N, K };
        int64_t xs[2] = { K, 1 };
        topsatenTensor x_f32(topsatenSize_t(xd, 2), topsatenSize_t(xs, 2),
                             TOPSATEN_DATA_FP32, x->data);
        topsatenTensor x_f16(topsatenSize_t(xd, 2), topsatenSize_t(xs, 2),
                             TOPSATEN_DATA_FP16, x_cast);
        topsatenDataType_t target = TOPSATEN_DATA_FP16;
        TOPSATEN_CHECK(topsatenTo(x_f16, x_f32, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->stream));
        x_data = x_cast;
    }

    // F16 output scratch.
    const size_t y_f16_bytes = (size_t) N * M * sizeof(uint16_t);
    void * y_f16 = ctx->pool.alloc(y_f16_bytes);

    // F16 zero bias. Reuse zero_bias buffer (it's zero-filled and sized in
    // bytes, so the F16-interpretation of the leading bytes is also zero).
    const size_t bias_bytes = (size_t) M * sizeof(uint16_t);
    void * bias_dev = gcu_get_zero_bias(ctx, bias_bytes);

    int64_t lhs_d[2] = { N, K }, lhs_s[2] = { K, 1 };
    int64_t rhs_d[2] = { M, K }, rhs_s[2] = { K, 1 };
    int64_t out_d[2] = { N, M }, out_s[2] = { M, 1 };
    int64_t bias_d[1] = { M };  int64_t bias_s[1] = { 1 };

    // For Q-typed weights, override rhs strides to match the F16 layout
    // we actually stored on the device.
    if (gcu_q_supported(wt)) {
        build_2d_q_as_f16(w, rhs_d, rhs_s);
    }

    topsatenTensor lhs_f16(topsatenSize_t(lhs_d, 2), topsatenSize_t(lhs_s, 2),
                           TOPSATEN_DATA_FP16, x_data);
    topsatenTensor rhs_f16(topsatenSize_t(rhs_d, 2), topsatenSize_t(rhs_s, 2),
                           TOPSATEN_DATA_FP16, w->data);
    topsatenTensor out_f16(topsatenSize_t(out_d, 2), topsatenSize_t(out_s, 2),
                           TOPSATEN_DATA_FP16, y_f16);
    topsatenTensor bias_f16(topsatenSize_t(bias_d, 1), topsatenSize_t(bias_s, 1),
                            TOPSATEN_DATA_FP16, bias_dev);

    TOPSATEN_CHECK(topsatenLinear(out_f16, lhs_f16, rhs_f16, bias_f16, ctx->stream));

    // Cast output F16 -> dst dtype if needed.
    if (ot == GGML_TYPE_F32) {
        int64_t od[2] = { N, M }, os[2] = { M, 1 };
        topsatenTensor out_f32(topsatenSize_t(od, 2), topsatenSize_t(os, 2),
                               TOPSATEN_DATA_FP32, dst->data);
        topsatenDataType_t target = TOPSATEN_DATA_FP32;
        TOPSATEN_CHECK(topsatenTo(out_f32, out_f16, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->stream));
    } else {
        TOPS_CHECK(topsMemcpyAsync(dst->data, y_f16, y_f16_bytes,
                                   topsMemcpyDeviceToDevice, ctx->stream));
    }

    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
    if (x_cast) ctx->pool.free(x_cast, x_cast_bytes);
    ctx->pool.free(y_f16, y_f16_bytes);
    return true;
}

// CPY/DUP/CONT. Same dtype + contiguous → fast topsMemcpyAsync(D2D).
// Different dtype (F32↔F16, both contiguous) → topsatenTo cast.
static bool gcu_op_cpy(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));

    if (src->type == dst->type) {
        GGML_ASSERT(ggml_nbytes(src) == ggml_nbytes(dst));
        TOPS_CHECK(topsMemcpyAsync(dst->data, src->data, ggml_nbytes(src),
                                   topsMemcpyDeviceToDevice, ctx->stream));
        return true;
    }

    // dtype-converting path (F32 ↔ F16). Same shape, same nelements.
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    gcu_tensor_dims dout, din;
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);
    topsatenTensor in_t  = make_topsaten_tensor(src, din);
    topsatenDataType_t target = ggml_to_topsaten_dtype(dst->type);
    TOPSATEN_CHECK(topsatenTo(out_t, in_t, target,
                              /*non_blocking=*/false, /*copy=*/true,
                              TOPSATEN_MEMORY_CONTIGUOUS, ctx->stream));
    return true;
}

// SET_ROWS: dst[idx[i]] = src[i].
//
// ggml's ggml_set_rows packs args unusually: result is a view of `a`
// (destination), and the node's slots are:
//   src[0] = b   (source rows)
//   src[1] = c   (row indices)
//   src[2] = a   (destination — written in place via the view)
//
// We flatten both src and dst to 2D [n_rows, row_size] where row_size is
// the innermost ggml dim (ne[0]) and n_rows is the product of the rest.
// This handles KV-cache shapes like [head_dim, n_heads, max_kv] where
// the cache is logically a 2D table of (n_heads * max_kv) rows of
// head_dim elements. Requires the tensor to be contiguous.
//
// llama.cpp stores K/V in F16 caches but produces F32 K/V vectors during
// compute, so src->type != dst->type is the common case. topsatenIndexPut
// requires matching dtypes, so we cast src into a per-context scratch
// when types differ.
static bool gcu_op_set_rows(ggml_backend_gcu_context * ctx, ggml_tensor * node) {
    ggml_tensor * src = node->src[0];
    ggml_tensor * idx = node->src[1];
    ggml_tensor * dst = node->src[2];

    // MVP-3b strategy: bypass topsatenIndexPut (which rejects ggml's KV
    // cache shapes at runtime) and write each row with a manual D2D
    // memcpy. The indices live on device so we read them to host first,
    // then issue n_rows memcpyAsync calls. Cheap for typical token counts
    // (1 for decode, ~prompt_len for prefill).

    const int64_t n_rows   = idx->ne[0];
    const size_t  row_size = (size_t) dst->ne[0] * ggml_type_size(dst->type);

    // If src and dst differ in dtype, cast src to dst dtype into a
    // scratch first; then we memcpy from the scratch.
    void * cast_buf = nullptr;
    size_t cast_bytes = 0;
    const void * src_data = src->data;
    if (src->type != dst->type) {
        const int64_t n_elem = n_rows * dst->ne[0];
        cast_bytes = (size_t) n_elem * ggml_type_size(dst->type);
        cast_buf   = ctx->pool.alloc(cast_bytes);

        int64_t  v_d[2] = { n_rows, dst->ne[0] };
        int64_t  v_s[2] = { dst->ne[0], 1 };
        topsatenTensor src_view (topsatenSize_t(v_d, 2), topsatenSize_t(v_s, 2),
                                 ggml_to_topsaten_dtype(src->type), src->data);
        topsatenTensor cast_view(topsatenSize_t(v_d, 2), topsatenSize_t(v_s, 2),
                                 ggml_to_topsaten_dtype(dst->type), cast_buf);
        topsatenDataType_t target = ggml_to_topsaten_dtype(dst->type);
        TOPSATEN_CHECK(topsatenTo(cast_view, src_view, target,
                                  /*non_blocking=*/false, /*copy=*/true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->stream));
        src_data = cast_buf;
    }

    // Read indices to host. Both I32 and I64 supported; convert to int64
    // uniformly for the addressing math.
    std::vector<int64_t> idx_host(n_rows);
    if (idx->type == GGML_TYPE_I32) {
        std::vector<int32_t> idx_i32(n_rows);
        TOPS_CHECK(topsMemcpy(idx_i32.data(), idx->data,
                              (size_t) n_rows * sizeof(int32_t),
                              topsMemcpyDeviceToHost));
        for (int64_t i = 0; i < n_rows; i++) idx_host[i] = (int64_t) idx_i32[i];
    } else {
        TOPS_CHECK(topsMemcpy(idx_host.data(), idx->data,
                              (size_t) n_rows * sizeof(int64_t),
                              topsMemcpyDeviceToHost));
    }

    char * dst_base = (char *) dst->data;
    const char * src_base = (const char *) src_data;
    for (int64_t i = 0; i < n_rows; i++) {
        const int64_t dst_row = idx_host[i];
        TOPS_CHECK(topsMemcpyAsync(dst_base + (size_t) dst_row * row_size,
                                   src_base + (size_t) i * row_size,
                                   row_size,
                                   topsMemcpyDeviceToDevice, ctx->stream));
    }

    if (cast_buf) {
        TOPS_CHECK(topsStreamSynchronize(ctx->stream));
        ctx->pool.free(cast_buf, cast_bytes);
    }
    return true;
}

// MVP-1: GET_ROWS handles only the unbatched case (in is effectively 2D
// [n, m], idx is 1D [r]). Batched GET_ROWS (be1>1 or be2>1) goes to CPU.
static bool gcu_op_get_rows(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * in_t  = dst->src[0];
    ggml_tensor * idx_t = dst->src[1];

    // Treat in as 2D (rank-trim trailing 1s) → topsaten dim 0 picks along the
    // ggml "rows" axis (ne[1]).
    int64_t in_dims[2]  = { in_t->ne[1], in_t->ne[0] };       // PyTorch order: [m, n]
    int64_t in_strs[2]  = { (int64_t) (in_t->nb[1] / ggml_type_size(in_t->type)),
                            (int64_t) (in_t->nb[0] / ggml_type_size(in_t->type)) };
    topsatenSize_t in_shape (in_dims, 2);
    topsatenSize_t in_stride(in_strs, 2);
    topsatenTensor in_tt(in_shape, in_stride, ggml_to_topsaten_dtype(in_t->type), in_t->data);

    int64_t out_dims[2] = { dst->ne[1], dst->ne[0] };
    int64_t out_strs[2] = { (int64_t) (dst->nb[1] / ggml_type_size(dst->type)),
                            (int64_t) (dst->nb[0] / ggml_type_size(dst->type)) };
    topsatenSize_t out_shape (out_dims, 2);
    topsatenSize_t out_stride(out_strs, 2);
    topsatenTensor out_tt(out_shape, out_stride, ggml_to_topsaten_dtype(dst->type), dst->data);

    int64_t idx_dims[1] = { idx_t->ne[0] };
    int64_t idx_strs[1] = { 1 };
    topsatenSize_t idx_shape (idx_dims, 1);
    topsatenSize_t idx_stride(idx_strs, 1);
    topsatenTensor idx_tt(idx_shape, idx_stride, TOPSATEN_DATA_I32, idx_t->data);

    TOPSATEN_CHECK(topsatenIndexSelect(out_tt, in_tt, /*dim=*/0, idx_tt, ctx->stream));
    return true;
}

// Builds an interleaved [max_pos, n_dims] cos/sin table per topsvllm
// convention: row p contains cos values for theta_i*p in the first n_dims/2
// slots followed by sin values in the next n_dims/2 slots.
//
// theta_i = freq_base^(-2i / n_dims) * freq_scale
static void gcu_build_rope_cos_sin_host(int n_dims, int max_pos,
                                        float freq_base, float freq_scale,
                                        std::vector<float> & out) {
    const int half = n_dims / 2;
    out.assign((size_t) max_pos * (size_t) n_dims, 0.0f);
    for (int p = 0; p < max_pos; p++) {
        float * row = out.data() + (size_t) p * n_dims;
        for (int i = 0; i < half; i++) {
            const float theta = std::pow(freq_base, -2.0f * (float) i / (float) n_dims) * freq_scale;
            const float angle = (float) p * theta;
            row[i]        = std::cos(angle);
            row[i + half] = std::sin(angle);
        }
    }
}

// ROPE. ggml mode 0 only.
//
// Inputs: x [head_dim, n_heads, n_tokens, 1] F32/F16; pos [n_tokens] I32.
// Maps to topsvllmRotaryEmbedding(query, key, positions, cos_sin_cache,
//   head_size, is_neox=false, stream).
// We treat x as the query; pass a small zero-filled scratch as key
// (topsvllm rotates both; we ignore the dummy key result).
// Positions get cast to I64 on host. cos_sin table is precomputed on host
// per call.
//
// In-place: ggml_rope returns a view of `a`, so dst->data == x->data already.
static bool gcu_op_rope(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * x   = dst->src[0];
    ggml_tensor * pos = dst->src[1];

    const int32_t n_dims = ((const int32_t *) dst->op_params)[1];
    float freq_base, freq_scale;
    memcpy(&freq_base,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const int32_t *) dst->op_params + 6, sizeof(float));

    const int64_t head_dim = x->ne[0];
    const int64_t n_heads  = x->ne[1];
    const int64_t n_tokens = x->ne[2];
    GGML_ASSERT(n_dims <= head_dim);
    GGML_ASSERT(pos->ne[0] == n_tokens);

    // Read positions to host so we can determine max_pos and cast to I64.
    std::vector<int32_t> pos_host(n_tokens);
    TOPS_CHECK(topsMemcpy(pos_host.data(), pos->data,
                          (size_t) n_tokens * sizeof(int32_t),
                          topsMemcpyDeviceToHost));

    int max_pos = 0;
    for (int32_t p : pos_host) max_pos = std::max(max_pos, (int) p);
    max_pos += 1;

    std::vector<float> cs_host;
    gcu_build_rope_cos_sin_host(n_dims, max_pos, freq_base, freq_scale, cs_host);

    // The SDK requires cos_sin_cache.dtype == query.dtype on this arch
    // (see op_vllm_rotary_embedding.h:188 — the FP32-cs override only
    // applies when the runtime check IS satisfied, but we observed NaN
    // empirically for F16 query + F32 cs). Pack to F16 if query is F16.
    const bool   cs_is_f16 = (x->type == GGML_TYPE_F16);
    const size_t cs_bpe    = cs_is_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t cs_bytes  = cs_host.size() * cs_bpe;
    const size_t pos_bytes = (size_t) n_tokens * sizeof(int64_t);
    void * cs_dev  = ctx->pool.alloc(cs_bytes);
    void * pos_dev = ctx->pool.alloc(pos_bytes);

    if (cs_is_f16) {
        std::vector<ggml_fp16_t> cs_f16(cs_host.size());
        ggml_fp32_to_fp16_row(cs_host.data(), cs_f16.data(), (int64_t) cs_host.size());
        TOPS_CHECK(topsMemcpyAsync(cs_dev, cs_f16.data(), cs_bytes,
                                   topsMemcpyHostToDevice, ctx->stream));
    } else {
        TOPS_CHECK(topsMemcpyAsync(cs_dev, cs_host.data(), cs_bytes,
                                   topsMemcpyHostToDevice, ctx->stream));
    }
    std::vector<int64_t> pos_i64(n_tokens);
    for (int64_t i = 0; i < n_tokens; i++) pos_i64[i] = (int64_t) pos_host[i];
    TOPS_CHECK(topsMemcpyAsync(pos_dev, pos_i64.data(), pos_bytes,
                               topsMemcpyHostToDevice, ctx->stream));

    // topsvllmRotaryEmbedding rotates query in-place. For ggml's non-inplace
    // ROPE (dst is a fresh tensor distinct from x), copy x into dst first so
    // we can rotate dst safely without touching x.
    if (dst->data != x->data) {
        TOPS_CHECK(topsMemcpyAsync(dst->data, x->data, ggml_nbytes(x),
                                   topsMemcpyDeviceToDevice, ctx->stream));
    }

    // Dummy key tensor — small, zero, just to satisfy topsvllmRotaryEmbedding's
    // dual-tensor signature. Shape [n_tokens, head_dim].
    const size_t dummy_bytes = (size_t) n_tokens * head_dim * ggml_type_size(x->type);
    void * dummy_key_dev = ctx->pool.alloc(dummy_bytes);
    TOPS_CHECK(topsMemsetAsync(dummy_key_dev, 0, dummy_bytes, ctx->stream));

    // query: [n_tokens, n_heads * head_dim] over dst's memory (rotated in place).
    int64_t q_d[2] = { n_tokens, n_heads * head_dim };
    int64_t q_s[2] = { (int64_t)(dst->nb[2] / ggml_type_size(dst->type)), 1 };
    topsatenTensor q_tt(topsatenSize_t(q_d, 2), topsatenSize_t(q_s, 2),
                        ggml_to_topsaten_dtype(dst->type), dst->data);

    int64_t k_d[2] = { n_tokens, head_dim };
    int64_t k_s[2] = { head_dim, 1 };
    topsatenTensor k_tt(topsatenSize_t(k_d, 2), topsatenSize_t(k_s, 2),
                        ggml_to_topsaten_dtype(x->type), dummy_key_dev);

    int64_t pos_d[1] = { n_tokens };
    int64_t pos_s[1] = { 1 };
    topsatenTensor pos_tt(topsatenSize_t(pos_d, 1), topsatenSize_t(pos_s, 1),
                          TOPSATEN_DATA_I64, pos_dev);

    int64_t cs_d[2] = { max_pos, n_dims };
    int64_t cs_s[2] = { n_dims, 1 };
    topsatenTensor cs_tt(topsatenSize_t(cs_d, 2), topsatenSize_t(cs_s, 2),
                         cs_is_f16 ? TOPSATEN_DATA_FP16 : TOPSATEN_DATA_FP32, cs_dev);

    TOPSATEN_CHECK(topsvllmRotaryEmbedding(q_tt, k_tt, pos_tt, cs_tt,
                                           (int) head_dim, /*is_neox=*/false,
                                           ctx->stream));

    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
    ctx->pool.free(cs_dev,        cs_bytes);
    ctx->pool.free(pos_dev,       pos_bytes);
    ctx->pool.free(dummy_key_dev, dummy_bytes);
    return true;
}

// SOFT_MAX. out = softmax(scale * x + mask, dim=-1). max_bias must be 0.
// Plan: scale x into a scratch slab (mul by scalar), optionally add mask,
// then topsatenSoftmaxForward along the last PyTorch dim (= ggml ne[0]).
static bool gcu_op_soft_max(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * x    = dst->src[0];
    ggml_tensor * mask = dst->src[1];   // may be nullptr
    float scale, max_bias;
    memcpy(&scale,    (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)dst->op_params + 1, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f);

    const size_t bytes = ggml_nbytes(dst);
    void * scratch = ctx->pool.alloc(bytes);

    // Build a topsatenTensor view over scratch with dst's shape/strides.
    gcu_tensor_dims d_dst, d_x, d_mask;
    topsatenTensor out_t = make_topsaten_tensor(dst, d_dst);
    topsatenTensor x_t   = make_topsaten_tensor(x,   d_x);
    topsatenTensor scratch_t;
    {
        const size_t bpe = ggml_type_size(dst->type);
        int rank = ggml_n_dims(dst); if (rank < 1) rank = 1;
        for (int i = 0; i < rank; i++) {
            d_dst.dims[i] = dst->ne[rank - 1 - i];
            d_dst.strs[i] = dst->nb[rank - 1 - i] / (int64_t) bpe;
        }
        scratch_t = topsatenTensor(topsatenSize_t(d_dst.dims, rank),
                                   topsatenSize_t(d_dst.strs, rank),
                                   ggml_to_topsaten_dtype(dst->type), scratch);
    }

    // scratch = x * scale
    {
        topsatenScalar_t s; s.dtype = TOPSATEN_DATA_FP32; s.fval = scale;
        TOPSATEN_CHECK(topsatenMul(scratch_t, x_t, s, ctx->stream));
    }

    // scratch += mask (with alpha=1)
    if (mask) {
        topsatenTensor mask_t = make_topsaten_tensor(mask, d_mask);
        topsatenScalar_t alpha; alpha.dtype = TOPSATEN_DATA_FP32; alpha.fval = 1.0;
        TOPSATEN_CHECK(topsatenAdd(scratch_t, scratch_t, mask_t, alpha, ctx->stream));
    }

    // out = softmax(scratch, dim=last)
    int rank = ggml_n_dims(dst); if (rank < 1) rank = 1;
    TOPSATEN_CHECK(topsatenSoftmaxForward(out_t, scratch_t, rank - 1, ctx->stream));

    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
    ctx->pool.free(scratch, bytes);
    return true;
}

// SILU. y = x * sigmoid(x). Same dtype in/out, contiguous.
static bool gcu_op_silu(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, dlhs;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, dlhs);
    TOPSATEN_CHECK(topsatenSilu(out, in, ctx->stream));
    return true;
}

static bool gcu_compute_node(ggml_backend_gcu_context * ctx, ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;        // metadata-only, nothing to launch
        case GGML_OP_ADD:
            return gcu_op_add(ctx, node);
        case GGML_OP_MUL:
            return gcu_op_mul(ctx, node);
        case GGML_OP_SCALE:
            return gcu_op_scale(ctx, node);
        case GGML_OP_GET_ROWS:
            return gcu_op_get_rows(ctx, node);
        case GGML_OP_SET_ROWS:
            return gcu_op_set_rows(ctx, node);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return gcu_op_cpy(ctx, node);
        case GGML_OP_MUL_MAT:
            return gcu_op_mul_mat(ctx, node);
        case GGML_OP_SOFT_MAX:
            return gcu_op_soft_max(ctx, node);
        case GGML_OP_RMS_NORM:
            return gcu_op_rms_norm(ctx, node);
        case GGML_OP_ROPE:
            return gcu_op_rope(ctx, node);
        case GGML_OP_UNARY: {
            const enum ggml_unary_op uop = ggml_get_unary_op(node);
            switch (uop) {
                case GGML_UNARY_OP_SILU: return gcu_op_silu(ctx, node);
                default: return false;
            }
        }
        default:
            return false;
    }
}

// === Device interface ================================================

#include <memory>

struct ggml_backend_gcu_device_context {
    int32_t      device;
    std::string  name;
    std::string  description;

    // Lazily-built (one per process per device).
    std::unique_ptr<ggml_backend_gcu_context>             ctx;
    std::unique_ptr<ggml_backend_buffer_type>             buft;
    std::unique_ptr<ggml_backend_gcu_buffer_type_context> buft_ctx;
    std::mutex   mu;

    ggml_backend_gcu_context * get_ctx() {
        std::lock_guard<std::mutex> lk(mu);
        if (!ctx) {
            ctx.reset(new ggml_backend_gcu_context(device));
        }
        return ctx.get();
    }

    ggml_backend_buffer_type_t get_buft() {
        std::lock_guard<std::mutex> lk(mu);
        if (!buft) {
            // Construct ctx (without re-locking) by inlining.
            if (!ctx) {
                ctx.reset(new ggml_backend_gcu_context(device));
            }
            buft_ctx.reset(new ggml_backend_gcu_buffer_type_context{ ctx.get() });
            buft.reset(new ggml_backend_buffer_type{
                /* .iface   = */ ggml_backend_gcu_buffer_type_i,
                /* .device  = */ nullptr,    // patched by get_buffer_type below
                /* .context = */ buft_ctx.get(),
            });
        }
        return buft.get();
    }
};

static const char * ggml_backend_gcu_device_get_name(ggml_backend_dev_t dev) {
    auto * d = (ggml_backend_gcu_device_context *) dev->context;
    return d->name.c_str();
}

static const char * ggml_backend_gcu_device_get_description(ggml_backend_dev_t dev) {
    auto * d = (ggml_backend_gcu_device_context *) dev->context;
    return d->description.c_str();
}

static void ggml_backend_gcu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * d = (ggml_backend_gcu_device_context *) dev->context;
    ggml_backend_gcu_get_device_memory(d->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_gcu_device_get_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static ggml_backend_buffer_type_t ggml_backend_gcu_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    auto * buft = ggml_backend_gcu_host_buffer_type();
    if (buft) buft->device = dev;
    return buft;
}

static ggml_backend_buffer_type_t ggml_backend_gcu_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * d = (ggml_backend_gcu_device_context *) dev->context;
    auto * buft = d->get_buft();
    if (buft) buft->device = dev;
    return buft;
}

static void ggml_backend_gcu_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_gcu_device_get_name(dev);
    props->description = ggml_backend_gcu_device_get_description(dev);
    props->type        = ggml_backend_gcu_device_get_type(dev);
    ggml_backend_gcu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_gcu_device_init_backend(ggml_backend_dev_t dev, const char * /*params*/) {
    auto * d = (ggml_backend_gcu_device_context *) dev->context;
    ggml_backend_t b = ggml_backend_gcu_init(d->device);
    if (b) b->device = dev;
    return b;
}

// Resolve a device id to the shared context owned by the registry's
// device descriptor. Defined here (after device_context) so it can call
// dctx->get_ctx().
static ggml_backend_gcu_context * gcu_get_shared_ctx_for_device(int32_t device) {
    ggml_backend_reg_t reg = ggml_backend_gcu_reg();
    size_t n = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        auto * d = (ggml_backend_gcu_device_context *) dev->context;
        if (d->device == device) {
            return d->get_ctx();
        }
    }
    GGML_ABORT("ggml-gcu: unknown device %d", device);
}

static bool ggml_backend_gcu_device_supports_op(ggml_backend_dev_t /*dev*/, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return gcu_all_inputs_supported_dtype(op) &&
                   gcu_numpy_broadcastable(op->src[0], op->src[1]);
        case GGML_OP_SCALE: {
            float params[2];
            memcpy(params, op->op_params, sizeof(params));
            const float bias = params[1];
            // MVP-1: only support pure scale (bias = 0). scale_bias goes to CPU.
            return gcu_all_inputs_supported_dtype(op) && bias == 0.0f;
        }
        case GGML_OP_GET_ROWS: {
            const ggml_tensor * in  = op->src[0];
            const ggml_tensor * idx = op->src[1];
            if (!in || !idx) return false;
            if (idx->type != GGML_TYPE_I32) return false;
            // MVP-1: only F32 in/out. topsatenIndexSelect's docs claim F16
            // support but it returns BAD_PARAM at runtime on this SDK
            // version; revisit in MVP-2.
            if (in->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            // Unbatched only (in is effectively 2D, idx is 1D, both contiguous).
            if (in->ne[2] != 1 || in->ne[3] != 1) return false;
            if (idx->ne[1] != 1 || idx->ne[2] != 1 || idx->ne[3] != 1) return false;
            if (!ggml_is_contiguous(idx)) return false;
            return true;
        }
        case GGML_OP_SET_ROWS: {
            const ggml_tensor * src = op->src[0];
            const ggml_tensor * idx = op->src[1];
            const ggml_tensor * dst = op->src[2];
            if (!src || !idx || !dst) return false;
            if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) return false;
            if (!gcu_dtype_supported(src->type) || !gcu_dtype_supported(dst->type)) return false;
            // MVP-2 explicit decision: F16 destination is the KV cache.
            // The MVP-3b probe (manual D2D memcpy loop bypassing
            // topsatenIndexPut) was 2-5x slower than -nkvo because of
            // per-call sync H2D index transfer. Refuse F16 dst here so
            // KV cache stays on CPU; users pass -nkvo to opt in. F32 dst
            // stays accepted for the small number of test-backend-ops
            // cases that exercise it.
            if (dst->type != GGML_TYPE_F32) return false;
            if (src->type != dst->type) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) return false;
            if (src->ne[0] != dst->ne[0]) return false;
            if (idx->ne[1] != 1 || idx->ne[2] != 1 || idx->ne[3] != 1) return false;
            if (!ggml_is_contiguous(idx)) return false;
            if (src->ne[1] * src->ne[2] * src->ne[3] != idx->ne[0]) return false;
            return true;
        }
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT: {
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type) || !gcu_dtype_supported(op->type)) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            // Same dtype OR F32↔F16 dtype conversion via topsatenTo.
            if (src->type != op->type) {
                bool ok = (src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F16) ||
                          (src->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F32);
                if (!ok) return false;
            }
            return true;
        }
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * w = op->src[0];
            const ggml_tensor * x = op->src[1];
            if (!w || !x) return false;
            // Supported dtype combos:
            //   (F32, F32, F32)        all-F32 fast path
            //   (F16, F32 or F16, F32 or F16)   F16 weight + cast input/output
            bool ok = false;
            if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ok = true;
            if (w->type == GGML_TYPE_F16 &&
                (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;
            // MVP-3a: Q-typed weight, stored on device as F16 via dequant-on-load.
            if (gcu_q_supported(w->type) &&
                (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;
            if (!ok) return false;
            // 2D unbatched matmul, contiguous, shared K.
            if (w->ne[2] != 1 || w->ne[3] != 1) return false;
            if (x->ne[2] != 1 || x->ne[3] != 1) return false;
            if (w->ne[0] != x->ne[0]) return false;
            if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x)) return false;
            return true;
        }
        case GGML_OP_UNARY: {
            const enum ggml_unary_op uop = ggml_get_unary_op(op);
            if (uop != GGML_UNARY_OP_SILU) return false;
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_RMS_NORM: {
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_ROPE: {
            // MVP-3c: F32 + F16 supported. F16 builds the cos/sin table in
            // F16 to satisfy the SDK's same-dtype requirement
            // (op_vllm_rotary_embedding.h:188).
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) return false;
            const int32_t mode = ((const int32_t *) op->op_params)[2];
            float ext_factor, attn_factor;
            memcpy(&ext_factor,  (const int32_t *) op->op_params + 7, sizeof(float));
            memcpy(&attn_factor, (const int32_t *) op->op_params + 8, sizeof(float));
            if (mode != 0) return false;
            if (ext_factor != 0.0f) return false;
            if (attn_factor != 0.0f && attn_factor != 1.0f) return false;
            if (op->src[2]) return false;            // freq factors (yarn) not supported
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            if (op->src[0]->ne[3] != 1) return false;
            return true;
        }
        case GGML_OP_SOFT_MAX: {
            float max_bias;
            memcpy(&max_bias, (const float *)op->op_params + 1, sizeof(float));
            if (max_bias != 0.0f) return false;
            if (op->src[2] != nullptr) return false;   // softmax sinks: MVP-3
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (op->src[1] && !gcu_dtype_supported(op->src[1]->type)) return false;
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        default:
            return false;
    }
}

static bool ggml_backend_gcu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_gcu_device_get_buffer_type(dev);
}

static const ggml_backend_device_i ggml_backend_gcu_device_i = {
    /* .get_name              = */ ggml_backend_gcu_device_get_name,
    /* .get_description       = */ ggml_backend_gcu_device_get_description,
    /* .get_memory            = */ ggml_backend_gcu_device_get_memory,
    /* .get_type              = */ ggml_backend_gcu_device_get_type,
    /* .get_props             = */ ggml_backend_gcu_device_get_props,
    /* .init_backend          = */ ggml_backend_gcu_device_init_backend,
    /* .get_buffer_type       = */ ggml_backend_gcu_device_get_buffer_type,
    /* .get_host_buffer_type  = */ ggml_backend_gcu_device_get_host_buffer_type,
    /* .buffer_from_host_ptr  = */ nullptr,
    /* .supports_op           = */ ggml_backend_gcu_device_supports_op,
    /* .supports_buft         = */ ggml_backend_gcu_device_supports_buft,
    /* .offload_op            = */ nullptr,
    /* .event_new             = */ nullptr,
    /* .event_free            = */ nullptr,
    /* .event_synchronize     = */ nullptr,
};

// === Registry =========================================================

struct ggml_backend_gcu_reg_context {
    std::vector<std::unique_ptr<ggml_backend_gcu_device_context>> dev_ctxs;
    std::vector<std::unique_ptr<ggml_backend_device>>             devs;
};

static const char * ggml_backend_gcu_reg_get_name(ggml_backend_reg_t /*reg*/) {
    return GGML_GCU_NAME;
}

static size_t ggml_backend_gcu_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * rctx = (ggml_backend_gcu_reg_context *) reg->context;
    return rctx->devs.size();
}

static ggml_backend_dev_t ggml_backend_gcu_reg_get_device(ggml_backend_reg_t reg, size_t idx) {
    auto * rctx = (ggml_backend_gcu_reg_context *) reg->context;
    return rctx->devs[idx].get();
}

static const ggml_backend_reg_i ggml_backend_gcu_reg_i = {
    /* .get_name         = */ ggml_backend_gcu_reg_get_name,
    /* .get_device_count = */ ggml_backend_gcu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gcu_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

// Forward decl so ggml_backend_gcu_init can locate the device's shared ctx.
// Definition lives below ggml_backend_gcu_device_context.
static ggml_backend_gcu_context * gcu_get_shared_ctx_for_device(int32_t device);

// === Stubs (filled in subsequent phases) =============================

extern "C" {

int32_t ggml_backend_gcu_get_device_count(void) {
    int count = 0;
    TOPS_CHECK(topsGetDeviceCount(&count));
    return count;
}

void ggml_backend_gcu_get_device_description(int32_t device, char * description, size_t description_size) {
    topsDeviceProp_t prop{};
    if (topsGetDeviceProperties(&prop, device) == topsSuccess) {
        snprintf(description, description_size, "%s", prop.name);
    } else {
        snprintf(description, description_size, "Enflame GCU device %d", device);
    }
}

void ggml_backend_gcu_get_device_memory(int32_t device, size_t * free, size_t * total) {
    int prev = 0;
    TOPS_CHECK(topsGetDevice(&prev));
    TOPS_CHECK(topsSetDevice(device));
    TOPS_CHECK(topsMemGetInfo(free, total));
    TOPS_CHECK(topsSetDevice(prev));
}

bool ggml_backend_is_gcu(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(backend->guid, ggml_backend_gcu_guid());
}

ggml_backend_t ggml_backend_gcu_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_gcu_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    // Locate the matching ggml_backend_dev_t in the registry so that
    // backend->device is populated even when callers invoke this function
    // directly (e.g., the smoke test) instead of going through the device
    // interface's init_backend.
    ggml_backend_reg_t reg     = ggml_backend_gcu_reg();
    ggml_backend_dev_t dev_ptr = nullptr;
    size_t n_dev = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t d = ggml_backend_reg_dev_get(reg, i);
        auto * dctx = (ggml_backend_gcu_device_context *) d->context;
        if (dctx->device == device) { dev_ptr = d; break; }
    }

    // Use the device's shared context. Sharing means buffer copies and
    // graph_compute use the same stream, so set_tensor / get_tensor
    // ordering is correct without extra event coordination.
    ggml_backend_gcu_context * ctx = gcu_get_shared_ctx_for_device(device);

    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_gcu_guid(),
        /* .iface   = */ ggml_backend_gcu_i,
        /* .device  = */ dev_ptr,
        /* .context = */ ctx,
    };
    return backend;
}

ggml_backend_reg_t ggml_backend_gcu_reg(void) {
    // The registry, its device descriptors, and each device's lazy
    // ggml_backend_gcu_context are intentionally heap-leaked. C++ static
    // destructors otherwise run during process exit AFTER topsrt has torn
    // down its own runtime, and any topsStreamSynchronize/topsStreamDestroy
    // call from our destructors then segfaults inside libefrt. Letting the
    // OS reclaim the memory at exit is the standard pattern for SDK-backed
    // singletons (CUDA/CANN do the same).
    static ggml_backend_reg *             s_reg  = nullptr;
    static ggml_backend_gcu_reg_context * s_rctx = nullptr;
    static std::once_flag                 once;

    std::call_once(once, [] {
        s_reg  = new ggml_backend_reg{};
        s_rctx = new ggml_backend_gcu_reg_context{};

        int n = ggml_backend_gcu_get_device_count();
        for (int i = 0; i < n; i++) {
            char name_buf[GGML_GCU_NAME_MAX];
            snprintf(name_buf, sizeof(name_buf), "GCU%d", i);

            std::string desc = "Enflame GCU";
            topsDeviceProp_t prop{};
            if (topsGetDeviceProperties(&prop, i) == topsSuccess) {
                desc = prop.name;
            }

            auto dctx = std::unique_ptr<ggml_backend_gcu_device_context>(
                new ggml_backend_gcu_device_context());
            dctx->device      = i;
            dctx->name        = name_buf;
            dctx->description = desc;

            auto dev = std::unique_ptr<ggml_backend_device>(new ggml_backend_device{
                /* .iface   = */ ggml_backend_gcu_device_i,
                /* .reg     = */ s_reg,
                /* .context = */ dctx.get(),
            });
            s_rctx->devs.push_back(std::move(dev));
            s_rctx->dev_ctxs.push_back(std::move(dctx));
        }

        *s_reg = ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_gcu_reg_i,
            /* .context     = */ s_rctx,
        };
    });
    return s_reg;
}

} // extern "C"
