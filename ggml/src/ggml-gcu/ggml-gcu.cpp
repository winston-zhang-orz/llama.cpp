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
// readability since this whole file is a topsaten wrapper.
using namespace topsaten;

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

static const ggml_backend_buffer_type_i ggml_backend_gcu_buffer_type_i = {
    /* .get_name        = */ ggml_backend_gcu_buffer_type_name,
    /* .alloc_buffer    = */ ggml_backend_gcu_buffer_type_alloc_buffer,
    /* .get_alignment   = */ ggml_backend_gcu_buffer_type_get_alignment,
    /* .get_max_size    = */ ggml_backend_gcu_buffer_type_get_max_size,
    /* .get_alloc_size  = */ nullptr,    // default = ggml_nbytes
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

static bool gcu_compute_node(ggml_backend_gcu_context * ctx, ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;        // metadata-only, nothing to launch
        case GGML_OP_ADD:
            return gcu_op_add(ctx, node);
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
            return gcu_all_inputs_supported_dtype(op) &&
                   gcu_numpy_broadcastable(op->src[0], op->src[1]);
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
    /* .get_host_buffer_type  = */ nullptr,
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

    // Use the device's shared context. Sharing means buffer copies and
    // graph_compute use the same stream, so set_tensor / get_tensor
    // ordering is correct without extra event coordination.
    ggml_backend_gcu_context * ctx = gcu_get_shared_ctx_for_device(device);

    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_gcu_guid(),
        /* .iface   = */ ggml_backend_gcu_i,
        /* .device  = */ nullptr,    // patched by device.init_backend
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
