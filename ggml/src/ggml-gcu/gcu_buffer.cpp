// ggml-gcu: device buffer / buffer-type / pinned host-buffer interfaces.
// See gcu_buffer.h.

#include "gcu_buffer.h"

// === Device buffer ==================================================

struct ggml_backend_gcu_buffer_context {
    ggml_backend_gcu_context * ctx;
    void *  base;
    size_t  size;

    // MVP-3a layout-mismatch shim. Q-typed weight tensors are stored as F16
    // on the device (dequant-on-load) but their ggml type and `nb[]` still
    // describe the Q packing. When the scheduler asks for a CPU-side copy of
    // such a tensor (for ops it places on the CPU backend), the canonical
    // path goes through the buffer's get_tensor, which expects exactly
    // ggml_nbytes(t) bytes in the original Q layout — F16 bytes lie about
    // their layout so a naive memcpy hands the CPU side garbage. Re-running
    // a Q-quant on the F16 (per cross-backend copy) is correct but in
    // practice ~1 GiB of host-side compute per matmul fallback, which makes
    // a 25B MoE unusable.
    //
    // Workaround: stash a host-side copy of the original Q-quant bytes at
    // load time (set_tensor), keyed by the device-side tensor->data pointer.
    // get_tensor for Q-typed tensors hands the cached bytes back. Memory
    // overhead is the size of the Q-quant model on host (~16 GB for the
    // Q4_K_M target). The CPU-side mmap of the gguf file already has these
    // bytes resident in page cache, but we cannot rely on the loader's
    // source pointer outliving set_tensor (the read path uses a reusable
    // scratch buffer), so we copy.
    std::unordered_map<const void *, std::vector<uint8_t>> q_host_cache;
};

void ggml_backend_gcu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
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

        // Stash the original Q-quant bytes for later get_tensor / cross-
        // backend copy requests. See the q_host_cache comment in the buffer
        // context for the why.
        auto & cache = bctx->q_host_cache[tensor->data];
        cache.assign((const uint8_t *) data, (const uint8_t *) data + size);
        return;

    }

    TOPS_CHECK(topsMemcpy((char *) tensor->data + offset, data, size, topsMemcpyHostToDevice));
}

static void ggml_backend_gcu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));

    // MVP-3a layout mismatch: Q-typed tensors are stored as F16 on the
    // device (dequant-on-load) but their ggml type and `nb[]` still
    // describe the Q packing. A naive memcpy would hand back F16 bytes
    // interpreted as Q-type and downstream consumers (CPU MUL_MAT_ID
    // fallback for MoE routing, eval-callback dump callbacks) would see
    // garbage. This was the Gemma 4 26B-A4B failure mode: argsort + per-
    // expert get_rows force some MUL_MAT_ID nodes onto CPU, the scheduler
    // synchronously copies the Q4_K weight via this get_tensor, and the
    // F16-bytes-under-Q4_K-type bait yielded garbage logits.
    //
    // Fix: hand back the original Q-quant bytes that set_tensor stashed
    // in the per-buffer host cache. Memory cost is the Q-quant model size
    // on host; in exchange the cross-backend copy is a plain memcpy and
    // we do not lose precision through a Q→F16→F32→Q round-trip.
    if (gcu_q_supported(tensor->type)) {
        auto it = bctx->q_host_cache.find(tensor->data);
        GGML_ASSERT(it != bctx->q_host_cache.end() &&
                    "GCU buffer.get_tensor: missing Q-quant host cache entry");
        GGML_ASSERT(offset + size <= it->second.size());
        memcpy(data, it->second.data() + offset, size);
        return;
    }

    TOPS_CHECK(topsMemcpy(data, (const char *) tensor->data + offset, size, topsMemcpyDeviceToHost));
}

static bool ggml_backend_gcu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_buffer_is_host(src->buffer) && src->buffer->buft == dst->buffer->buft) {
        auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
        TOPS_CHECK(topsSetDevice(bctx->ctx->device));
        // Q-typed tensors are stored as F16 on the device (dequant-on-load).
        // ggml_nbytes(src) returns the Q-quant size, but the F16 backing
        // store is larger; copy the actual F16 byte count and replicate
        // the host-side Q-quant cache for the destination.
        size_t nbytes = ggml_nbytes(src);
        if (gcu_q_supported(src->type)) {
            nbytes = (size_t) ggml_nelements(src) * sizeof(uint16_t);
            auto * dbctx = (ggml_backend_gcu_buffer_context *) dst->buffer->context;
            auto it = bctx->q_host_cache.find(src->data);
            if (it != bctx->q_host_cache.end()) {
                dbctx->q_host_cache[dst->data] = it->second;
            }
        }
        TOPS_CHECK(topsMemcpy(dst->data, src->data, nbytes, topsMemcpyDeviceToDevice));
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

ggml_backend_buffer_type_t ggml_backend_gcu_host_buffer_type() {
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

const ggml_backend_buffer_type_i ggml_backend_gcu_buffer_type_i = {
    /* .get_name        = */ ggml_backend_gcu_buffer_type_name,
    /* .alloc_buffer    = */ ggml_backend_gcu_buffer_type_alloc_buffer,
    /* .get_alignment   = */ ggml_backend_gcu_buffer_type_get_alignment,
    /* .get_max_size    = */ ggml_backend_gcu_buffer_type_get_max_size,
    /* .get_alloc_size  = */ ggml_backend_gcu_buffer_type_get_alloc_size,
    /* .is_host         = */ nullptr,    // device buffer (not host-accessible)
};
