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
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_gcu_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
}

static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t /*backend*/, struct ggml_cgraph * /*cgraph*/) {
    GGML_ABORT("ggml_backend_gcu_graph_compute not implemented yet");
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

    auto * ctx = new ggml_backend_gcu_context(device);

    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_gcu_guid(),
        /* .iface   = */ ggml_backend_gcu_i,
        /* .device  = */ nullptr,         // filled in by device.init_backend (Phase C)
        /* .context = */ ctx,
    };
    return backend;
}

ggml_backend_reg_t ggml_backend_gcu_reg(void) {
    return nullptr;
}

} // extern "C"
