// ggml-gcu: Enflame GCU (TOPS) backend
//
// Spec: docs/superpowers/specs/2026-05-08-gcu-s60-backend-design.md
// Plan: docs/superpowers/plans/2026-05-08-gcu-s60-backend-mvp1.md
//
// MVP-1: skeleton with stubs for the public API. No ops yet.

#include "ggml-gcu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "common.h"
#include "gcu_pool.h"
#include "gcu_buffer.h"
#include "gcu_ops.h"

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

// === Per-context state ==============================================
//
// ggml_backend_gcu_context lives in common.h; ctor/dtor are in common.cpp.


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
    // Copy first so any pending D->H drains into host memory the caller
    // may inspect; compute second.
    TOPS_CHECK(topsStreamSynchronize(ctx->copy_stream));
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
}

// Forward declaration: gcu_compute_node lives in the Op dispatch section
// further down in the file.
static bool gcu_compute_node(ggml_backend_gcu_context * ctx, ggml_tensor * node);

static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    // MVP-4a: stitch any pending async H->D copies onto the compute stream
    // so the first kernel here sees fully-transferred input.
    if (ctx->copy_event_armed) {
        TOPS_CHECK(topsStreamWaitEvent(ctx->compute_stream,
                                       ctx->last_copy_event, 0));
        ctx->copy_event_armed = false;
    }

    ggml_status status = GGML_STATUS_SUCCESS;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) continue;
        if (!gcu_compute_node(ctx, node)) {
            GGML_LOG_ERROR("%s: op %s not implemented or failed\n",
                           __func__, ggml_op_name(node->op));
            status = GGML_STATUS_FAILED;
            break;
        }
    }

    // MVP-4a: arm the compute event so subsequent get_tensor_async can wait
    // on it without blocking the host.
    TOPS_CHECK(topsEventRecord(ctx->last_compute_event, ctx->compute_stream));
    ctx->compute_event_armed = true;
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));

    // MVP-4b: drain deferred frees on every exit path. Kernels submitted
    // before the failure must complete (ensured by the synchronize above)
    // and their scratch buffers must return to the pool, otherwise the
    // pool grows monotonically across calls under failure recovery.
    for (auto & kv : ctx->deferred_frees) {
        ctx->pool.free(kv.first, kv.second);
    }
    ctx->deferred_frees.clear();

    return status;
}

// ggml_backend_buffer_is_gcu / ggml_backend_gcu_buffer_type_i /
// ggml_backend_gcu_host_buffer_type live in gcu_buffer.{h,cpp}.

// MVP-4a/4b: async H<->D on copy_stream and queued-op scratch deferral.
// Helpers (gcu_async_disabled / gcu_queued_ops_disabled / gcu_release_scratch)
// are declared in common.h and implemented in common.cpp.

static void ggml_backend_gcu_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    if (size == 0) return;

    // Q-typed weights load via the synchronous buffer path at model init
    // and are never re-set during inference. Reaching here with a Q-tensor
    // is a logic bug — silently mis-sizing the copy would corrupt the
    // dequanted F16 destination.
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    // Reject tensors not backed by a GCU device buffer.  view_src tensors
    // inherit their storage from the source tensor, so check its buffer.
    // Mirrors the defensive pattern in ggml_backend_buffer_is_cuda.
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf && ggml_backend_buffer_is_gcu(buf) && "tensor not in GCU buffer");

    if (gcu_async_disabled()) {
        TOPS_CHECK(topsMemcpy((char *) tensor->data + offset, data, size,
                              topsMemcpyHostToDevice));
        return;
    }

    TOPS_CHECK(topsMemcpyAsync(
        (char *) tensor->data + offset, data, size,
        topsMemcpyHostToDevice, ctx->copy_stream));
    TOPS_CHECK(topsEventRecord(ctx->last_copy_event, ctx->copy_stream));
    ctx->copy_event_armed = true;
}

static void ggml_backend_gcu_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    if (size == 0) return;
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    // Reject tensors not backed by a GCU device buffer (mirrors set_tensor_async).
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf && ggml_backend_buffer_is_gcu(buf) && "tensor not in GCU buffer");

    if (gcu_async_disabled()) {
        TOPS_CHECK(topsMemcpy(data, (const char *) tensor->data + offset, size,
                              topsMemcpyDeviceToHost));
        return;
    }

    // D->H must not race still-running compute.
    // Guard with compute_event_armed: topsStreamWaitEvent on an unarmed event
    // is implementation-defined in topsrt (symmetric to copy_event_armed).
    // Task 3 will set compute_event_armed = true after recording the event.
    if (ctx->compute_event_armed) {
        TOPS_CHECK(topsStreamWaitEvent(ctx->copy_stream, ctx->last_compute_event, 0));
    }
    TOPS_CHECK(topsMemcpyAsync(
        data, (const char *) tensor->data + offset, size,
        topsMemcpyDeviceToHost, ctx->copy_stream));
}

static const ggml_backend_i ggml_backend_gcu_i = {
    /* .get_name             = */ ggml_backend_gcu_name,
    /* .free                 = */ ggml_backend_gcu_free,
    /* .set_tensor_async     = */ ggml_backend_gcu_set_tensor_async,
    /* .get_tensor_async     = */ ggml_backend_gcu_get_tensor_async,
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







// Returns a device pointer to at least n_bytes of zero memory. Grows the
// per-context zero buffer on demand. Assumes single-stream serialization
// so no extra synchronization is needed between resize and use.
static void * gcu_get_zero_bias(ggml_backend_gcu_context * ctx, size_t n_bytes) {
    if (ctx->zero_bias_bytes < n_bytes) {
        gcu_release_scratch(ctx, ctx->zero_bias, ctx->zero_bias_bytes);
        ctx->zero_bias       = ctx->pool.alloc(n_bytes);
        ctx->zero_bias_bytes = n_bytes;
        TOPS_CHECK(topsMemsetAsync(ctx->zero_bias, 0, n_bytes, ctx->compute_stream));
    }
    return ctx->zero_bias;
}

// Returns a device pointer to a F32 ones buffer of at least `count` elements.
// Grows on demand. Always F32; callers cast to other dtypes via topsatenTo.
static void * gcu_get_ones_f32(ggml_backend_gcu_context * ctx, int64_t count) {
    if (ctx->ones_n0_count >= count) return ctx->ones_n0;
    gcu_release_scratch(ctx, ctx->ones_n0, ctx->ones_n0_bytes);
    const size_t bytes = (size_t) count * sizeof(float);
    ctx->ones_n0       = ctx->pool.alloc(bytes);
    ctx->ones_n0_bytes = bytes;
    ctx->ones_n0_count = count;

    // Materialize ones: zero the buffer, then add scalar 1.0 broadcast.
    TOPS_CHECK(topsMemsetAsync(ctx->ones_n0, 0, bytes, ctx->compute_stream));
    int64_t dims[1] = { count };
    int64_t strs[1] = { 1 };
    topsatenTensor t(topsatenSize_t(dims, 1), topsatenSize_t(strs, 1),
                     TOPSATEN_DATA_FP32, ctx->ones_n0);
    topsatenScalar_t one_lhs; one_lhs.dtype = TOPSATEN_DATA_FP32; one_lhs.fval = 1.0;
    topsatenScalar_t alpha;   alpha.dtype   = TOPSATEN_DATA_FP32; alpha.fval   = 1.0;
    // t = 1.0 + 1.0 * t (where t is currently 0) → t becomes 1.0
    TOPSATEN_CHECK(topsatenAdd(t, one_lhs, t, alpha, ctx->compute_stream));
    return ctx->ones_n0;
}

// RMS_NORM. ggml_rms_norm: dst = x / sqrt(mean(x^2) + eps). No fused weight
// (the multiply happens via a downstream MUL op), but topsvllmRmsNorm
// requires a gamma weight argument — we pass an ones tensor of size ne[0].
bool gcu_op_rms_norm(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t hidden_size = src->ne[0];

    // F32 ones buffer for gamma. If src is F16/BF16, cast to that dtype
    // first into a per-call scratch (small — at most a few KiB).
    void * ones_f32 = gcu_get_ones_f32(ctx, hidden_size);
    void * gamma_data = ones_f32;
    size_t cast_bytes = 0;
    void * cast_buf   = nullptr;
    topsatenDataType_t gamma_dtype = TOPSATEN_DATA_FP32;
    if (src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_BF16) {
        topsatenDataType_t target = ggml_to_topsaten_dtype(src->type);
        cast_bytes = (size_t) hidden_size * sizeof(uint16_t);
        cast_buf   = ctx->pool.alloc(cast_bytes);
        int64_t gd[1] = { hidden_size };
        int64_t gs[1] = { 1 };
        topsatenTensor f32_t(topsatenSize_t(gd, 1), topsatenSize_t(gs, 1),
                             TOPSATEN_DATA_FP32, ones_f32);
        topsatenTensor lo_t (topsatenSize_t(gd, 1), topsatenSize_t(gs, 1),
                             target, cast_buf);
        TOPSATEN_CHECK(topsatenTo(lo_t, f32_t, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        gamma_data  = cast_buf;
        gamma_dtype = target;
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

    TOPSATEN_CHECK(topsvllmRmsNorm(out_t, in_t, gamma_t, eps_s, ctx->compute_stream));

    gcu_release_scratch(ctx, cast_buf, cast_bytes);
    return true;
}

// CONCAT along the given ggml axis. ggml's dim is innermost-first
// (dim 0 = ne[0]), topsaten/PyTorch is slowest-first, so we flip it
// to match the shape order make_topsaten_tensor produces.
bool gcu_op_concat(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * a = dst->src[0];
    ggml_tensor * b = dst->src[1];

    const int32_t ggml_dim = ((const int32_t *) dst->op_params)[0];

    gcu_tensor_dims da, db, dout;
    topsatenTensor a_t   = make_topsaten_tensor(a,   da);
    topsatenTensor b_t   = make_topsaten_tensor(b,   db);
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);

    int rank = ggml_n_dims(dst);
    if (rank < 1) rank = 1;
    const int64_t topsaten_dim = (int64_t) (rank - 1 - ggml_dim);

    std::vector<topsatenTensor> inputs;
    inputs.reserve(2);
    inputs.push_back(a_t);
    inputs.push_back(b_t);
    TOPSATEN_CHECK(topsatenCat(out_t, inputs, topsaten_dim, ctx->compute_stream));
    return true;
}

// NORM (LayerNorm without affine). y = (x - mean(x)) / sqrt(var(x) + eps).
// Maps to topsatenLayerNorm, which always applies an affine; we feed
// weight=ones / bias=zeros to recover the unscaled normalize that ggml's
// GGML_OP_NORM specifies. The follow-on weight/bias multiplications are
// separate ggml ops (MUL / ADD) that the scheduler dispatches as usual.
bool gcu_op_norm(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t hidden_size = src->ne[0];

    // F32 ones for the unit affine weight. Cast to F16/BF16 if input
    // dtype is one of those (same template as gcu_op_rms_norm).
    void * ones_f32   = gcu_get_ones_f32(ctx, hidden_size);
    void * weight_data = ones_f32;
    size_t cast_bytes  = 0;
    void * cast_buf    = nullptr;
    topsatenDataType_t affine_dtype = TOPSATEN_DATA_FP32;
    if (src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_BF16) {
        topsatenDataType_t target = ggml_to_topsaten_dtype(src->type);
        cast_bytes = (size_t) hidden_size * sizeof(uint16_t);
        cast_buf   = ctx->pool.alloc(cast_bytes);
        int64_t gd[1] = { hidden_size };
        int64_t gs[1] = { 1 };
        topsatenTensor f32_t(topsatenSize_t(gd, 1), topsatenSize_t(gs, 1),
                             TOPSATEN_DATA_FP32, ones_f32);
        topsatenTensor lo_t (topsatenSize_t(gd, 1), topsatenSize_t(gs, 1),
                             target, cast_buf);
        TOPSATEN_CHECK(topsatenTo(lo_t, f32_t, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        weight_data  = cast_buf;
        affine_dtype = target;
    }

    // Bias = zeros, sized in bytes for the chosen dtype. The shared
    // zero_bias buffer is zero-filled and oversized; we just need the
    // first hidden_size elements interpreted in our dtype.
    const bool   src_is_low = (src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_BF16);
    const size_t bias_bytes = (size_t) hidden_size *
        (src_is_low ? sizeof(uint16_t) : sizeof(float));
    void * bias_data = gcu_get_zero_bias(ctx, bias_bytes);

    int64_t affine_d[1] = { hidden_size };
    int64_t affine_s[1] = { 1 };
    topsatenTensor weight(topsatenSize_t(affine_d, 1), topsatenSize_t(affine_s, 1),
                          affine_dtype, weight_data);
    topsatenTensor bias  (topsatenSize_t(affine_d, 1), topsatenSize_t(affine_s, 1),
                          affine_dtype, bias_data);

    // Collapse outer dims so input is rank-2 [n_rows, hidden_size].
    const int64_t n_rows = src->ne[1] * src->ne[2] * src->ne[3];
    int64_t io_d[2] = { n_rows, hidden_size };
    int64_t io_s[2] = { hidden_size, 1 };
    topsatenTensor in_t (topsatenSize_t(io_d, 2), topsatenSize_t(io_s, 2),
                         ggml_to_topsaten_dtype(src->type), src->data);
    topsatenTensor out_t(topsatenSize_t(io_d, 2), topsatenSize_t(io_s, 2),
                         ggml_to_topsaten_dtype(dst->type), dst->data);

    int64_t norm_shape[1] = { hidden_size };
    topsatenScalar_t eps_s; eps_s.dtype = TOPSATEN_DATA_FP32; eps_s.fval = eps;

    TOPSATEN_CHECK(topsatenLayerNorm(out_t, in_t,
                                     topsatenSize_t(norm_shape, 1),
                                     weight, bias, eps_s, ctx->compute_stream));

    gcu_release_scratch(ctx, cast_buf, cast_bytes);
    return true;
}

// Tranche F: L2_NORM. ggml's CPU forward divides each row (along ne[0]) by
// max(L2 norm, eps). topsatenNormalize computes y = x / sqrt(sum(|x|^p) + eps),
// which differs from ggml in two ways:
//   1) eps is added inside sqrt (vs. ggml's max-clamp on the divisor),
//   2) the eps default is 1e-12 vs. ggml's typical 1e-6.
// For typical inputs where sum(x^2) > eps^2 the results agree to within
// numerical noise. This op is rarely emitted by LLM-style models; if a model
// uses it with very small magnitudes the discrepancy could matter, but the
// handler still matches ggml for the common case (eps tiny relative to L2).
bool gcu_op_l2_norm(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // Build descriptors in PyTorch order (slowest-first). Reduce dim is the
    // innermost ggml dim (ne[0]) which is the last dim in PyTorch order.
    gcu_tensor_dims din, dout;
    topsatenTensor in_t  = make_topsaten_tensor(src, din);
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);

    int rank = ggml_n_dims(src); if (rank < 1) rank = 1;
    int64_t dim_arr[1] = { (int64_t)(rank - 1) };
    topsatenSize_t dim_s(dim_arr, 1);
    TOPSATEN_CHECK(topsatenNormalize(out_t, in_t, /*p=*/2.0f, dim_s, eps,
                                     ctx->compute_stream));
    return true;
}

// Tranche F: GROUP_NORM. ggml's CPU forward applies LayerNorm-style
// (subtract mean, divide by sqrt(var+eps)) per group, where channels (ne[2])
// are partitioned into n_groups. ggml has NO affine multiply/bias step; we
// pass weight=ones / bias=zeros to topsatenGroupNorm to recover that.
//
// op_params layout: [int n_groups, float eps]
bool gcu_op_group_norm(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t n_groups = ((const int32_t *) dst->op_params)[0];
    float eps;
    memcpy(&eps, (const float *) dst->op_params + 1, sizeof(float));

    // ggml shape:  [ne0, ne1, ne2 (=channels), ne3 (=batch)]
    // Build a PyTorch-NCHW tensor: [N=ne3, C=ne2, H=ne1, W=ne0].
    const int64_t N = src->ne[3];
    const int64_t C = src->ne[2];
    const int64_t H = src->ne[1];
    const int64_t W = src->ne[0];

    const size_t bpe = ggml_type_size(src->type);
    int64_t in_d[4]  = { N, C, H, W };
    int64_t in_s[4]  = { (int64_t)(src->nb[3]/bpe), (int64_t)(src->nb[2]/bpe),
                         (int64_t)(src->nb[1]/bpe), (int64_t)(src->nb[0]/bpe) };
    int64_t out_d[4] = { N, C, H, W };
    int64_t out_s[4] = { (int64_t)(dst->nb[3]/bpe), (int64_t)(dst->nb[2]/bpe),
                         (int64_t)(dst->nb[1]/bpe), (int64_t)(dst->nb[0]/bpe) };

    topsatenTensor in_t (topsatenSize_t(in_d, 4),  topsatenSize_t(in_s, 4),
                         ggml_to_topsaten_dtype(src->type), src->data);
    topsatenTensor out_t(topsatenSize_t(out_d, 4), topsatenSize_t(out_s, 4),
                         ggml_to_topsaten_dtype(dst->type), dst->data);

    // weight = ones[C], bias = zeros[C]. Reuse the shared per-context buffers.
    void * ones_dev = gcu_get_ones_f32(ctx, C);
    void * zero_dev = gcu_get_zero_bias(ctx, (size_t) C * sizeof(float));

    int64_t affine_d[1] = { C };
    int64_t affine_s[1] = { 1 };
    topsatenTensor weight(topsatenSize_t(affine_d, 1), topsatenSize_t(affine_s, 1),
                          TOPSATEN_DATA_FP32, ones_dev);
    topsatenTensor bias  (topsatenSize_t(affine_d, 1), topsatenSize_t(affine_s, 1),
                          TOPSATEN_DATA_FP32, zero_dev);

    TOPSATEN_CHECK(topsatenGroupNorm(out_t, in_t, (int64_t) n_groups,
                                     weight, bias, (double) eps,
                                     ctx->compute_stream));
    return true;
}

// Tranche H: REPEAT. dst is `repeats[i] = ne_dst[i] / ne_src[i]` times the
// src along dim i. All ne_dst[i] must be integer multiples of ne_src[i]
// (ggml_can_repeat) — the supports_op gate enforces that. Maps directly
// onto topsatenRepeat. F32 + F16 supported.
//
// Note: this is *not* the same as ggml_repeat_4d's KV-head broadcast,
// which always materializes through MUL_MAT's broadcast path.
bool gcu_op_repeat(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    gcu_tensor_dims din, dout;
    topsatenTensor in_t  = make_topsaten_tensor(src, din);
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);

    int rank = ggml_n_dims(dst); if (rank < 1) rank = 1;
    int64_t repeats[GGML_MAX_DIMS];
    // PyTorch order (slowest-first): repeats[i] = ne_dst[rank-1-i] / ne_src[rank-1-i].
    for (int i = 0; i < rank; i++) {
        const int64_t ggml_dim = rank - 1 - i;
        repeats[i] = dst->ne[ggml_dim] / src->ne[ggml_dim];
    }
    topsatenSize_t reps(repeats, rank);
    TOPSATEN_CHECK(topsatenRepeat(out_t, in_t, reps, ctx->compute_stream));
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
bool gcu_op_mul_mat(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
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
        TOPSATEN_CHECK(topsatenLinear(out, lhs, rhs, bias, ctx->compute_stream));
        return true;
    }

    // Low-precision weight path: works for both F16 and BF16 weights, as
    // well as Q-typed weights (whose device bytes are F16 after MVP-3a's
    // dequant-on-load). The kernel runs in the weight's effective dtype
    // and we cast input/output to/from F32 as needed.
    GGML_ASSERT(wt_eff == GGML_TYPE_F16 || wt_eff == GGML_TYPE_BF16);
    const topsatenDataType_t lp_dtype = ggml_to_topsaten_dtype(wt_eff);
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
        topsatenTensor x_lp (topsatenSize_t(xd, 2), topsatenSize_t(xs, 2),
                             lp_dtype, x_cast);
        topsatenDataType_t target = lp_dtype;
        TOPSATEN_CHECK(topsatenTo(x_lp, x_f32, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        x_data = x_cast;
    } else if (xt != wt_eff) {
        // F16 input but BF16 weight (or vice-versa). topsatenLinear requires
        // matching lhs/rhs dtypes; cast input to the weight's dtype.
        x_cast_bytes = (size_t) N * K * sizeof(uint16_t);
        x_cast       = ctx->pool.alloc(x_cast_bytes);

        int64_t xd[2] = { N, K };
        int64_t xs[2] = { K, 1 };
        topsatenTensor x_in (topsatenSize_t(xd, 2), topsatenSize_t(xs, 2),
                             ggml_to_topsaten_dtype(xt), x->data);
        topsatenTensor x_lp (topsatenSize_t(xd, 2), topsatenSize_t(xs, 2),
                             lp_dtype, x_cast);
        topsatenDataType_t target = lp_dtype;
        TOPSATEN_CHECK(topsatenTo(x_lp, x_in, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        x_data = x_cast;
    }

    // Low-prec output scratch (same byte size as F16: 2 bytes/element).
    const size_t y_lp_bytes = (size_t) N * M * sizeof(uint16_t);
    void * y_lp = ctx->pool.alloc(y_lp_bytes);

    // Low-prec zero bias. Reuse zero_bias buffer — zero-filled and sized
    // in bytes; the BF16/F16 interpretation of zero-bytes is still zero.
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

    topsatenTensor lhs_lp(topsatenSize_t(lhs_d, 2), topsatenSize_t(lhs_s, 2),
                          lp_dtype, x_data);
    topsatenTensor rhs_lp(topsatenSize_t(rhs_d, 2), topsatenSize_t(rhs_s, 2),
                          lp_dtype, w->data);
    topsatenTensor out_lp(topsatenSize_t(out_d, 2), topsatenSize_t(out_s, 2),
                          lp_dtype, y_lp);
    topsatenTensor bias_lp(topsatenSize_t(bias_d, 1), topsatenSize_t(bias_s, 1),
                           lp_dtype, bias_dev);

    TOPSATEN_CHECK(topsatenLinear(out_lp, lhs_lp, rhs_lp, bias_lp, ctx->compute_stream));

    // Cast output low-prec -> dst dtype if needed.
    if (ot != wt_eff) {
        int64_t od[2] = { N, M }, os[2] = { M, 1 };
        topsatenTensor out_dst(topsatenSize_t(od, 2), topsatenSize_t(os, 2),
                               ggml_to_topsaten_dtype(ot), dst->data);
        topsatenDataType_t target = ggml_to_topsaten_dtype(ot);
        TOPSATEN_CHECK(topsatenTo(out_dst, out_lp, target, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
    } else {
        TOPS_CHECK(topsMemcpyAsync(dst->data, y_lp, y_lp_bytes,
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
    }

    gcu_release_scratch(ctx, x_cast, x_cast_bytes);
    gcu_release_scratch(ctx, y_lp,   y_lp_bytes);
    return true;
}

// MUL_MAT_ID: indirect matmul for MoE expert routing.
//
//   src[0]  as  -> [cols, rows, n_expert]            expert weight matrices
//   src[1]  b   -> [cols, n_expert_used, n_tokens]   input activations
//                                                    (n_expert_used dim may
//                                                     broadcast from 1)
//   src[2]  ids -> [n_expert_used, n_tokens] i32     expert assignments
//   dst     c   -> [rows, n_expert_used, n_tokens]   F32
//
//   c[:, e, t] = as[:, :, ids[e, t]] @ b[:, e % b->ne[1], t]
//
// Implementation: read ids host-side, then loop over (token, expert_slot)
// pairs, issuing one topsatenLinear call per pair. With MVP-4b's queued
// ops each call is one driver enqueue rather than a host round-trip, so
// the launch-overhead budget is bounded for tg (n_tokens=1, ~few experts
// per layer). Q-typed weights live as F16 on the device (MVP-3a
// dequant-on-load); the F16 path casts F32 input to F16 once for the
// whole sweep and casts each F16 output row back to F32 at the dst slot.
//
// Optimization opportunity (future): gather tokens by expert to fold
// each expert's per-token calls into a single batched matmul. Saves call
// count for prompt processing where n_tokens is large.
bool gcu_op_mul_mat_id(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[3] == 1 && x->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);

    const int64_t cols          = w->ne[0];
    const int64_t rows          = w->ne[1];
    const int64_t n_expert      = w->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t r             = x->ne[1];

    GGML_ASSERT(x->ne[0] == cols);
    GGML_ASSERT(x->ne[2] == n_tokens);
    GGML_ASSERT(n_expert_used % r == 0);

    const ggml_type wt = w->type;
    const ggml_type xt = x->type;
    const bool      wq   = gcu_q_supported(wt);
    // MVP-5a: w_lp covers F16, BF16, and Q-typed weights (Q stored as F16
    // on device after MVP-3a's dequant-on-load). The kernel runs in the
    // weight's effective low-prec dtype and we cast input to match once
    // per call sweep.
    const ggml_type wt_eff = wq ? GGML_TYPE_F16 : wt;
    const bool      w_lp = wq || wt == GGML_TYPE_F16 || wt == GGML_TYPE_BF16;
    GGML_ASSERT(w_lp || wt == GGML_TYPE_F32);
    GGML_ASSERT(xt == GGML_TYPE_F32 || xt == GGML_TYPE_F16 || xt == GGML_TYPE_BF16);
    const topsatenDataType_t lp_dtype = w_lp
        ? ggml_to_topsaten_dtype(wt_eff)
        : TOPSATEN_DATA_FP32;

    // Drain compute_stream so the i32 values in ids are fully written
    // (they may have been produced by a preceding op on the same stream),
    // then read host-side.
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
    std::vector<int32_t> ids_host((size_t) n_expert_used * (size_t) n_tokens);
    TOPS_CHECK(topsMemcpy(ids_host.data(), ids->data,
                          ids_host.size() * sizeof(int32_t),
                          topsMemcpyDeviceToHost));

    // For the low-prec weight path, cast all of x once up front and index
    // into the cast buffer in the inner loop. Avoids per-call cast overhead
    // at the cost of one [cols, r, n_tokens]-sized scratch. Cast happens
    // when xt != wt_eff (covers F32 input vs F16/BF16 weight, and the
    // F16-input vs BF16-weight or BF16-input vs F16-weight mixes).
    void * x_lp_buf   = nullptr;
    size_t x_lp_bytes = 0;
    if (w_lp && xt != wt_eff) {
        const int64_t x_total = cols * r * n_tokens;
        x_lp_bytes = (size_t) x_total * sizeof(uint16_t);
        x_lp_buf   = ctx->pool.alloc(x_lp_bytes);

        int64_t xd[1] = { x_total };
        int64_t xs[1] = { 1 };
        topsatenTensor x_in(topsatenSize_t(xd, 1), topsatenSize_t(xs, 1),
                            ggml_to_topsaten_dtype(xt), x->data);
        topsatenTensor x_lp(topsatenSize_t(xd, 1), topsatenSize_t(xs, 1),
                            lp_dtype, x_lp_buf);
        topsatenDataType_t target_lp = lp_dtype;
        TOPSATEN_CHECK(topsatenTo(x_lp, x_in, target_lp, false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
    }

    // Per-call low-prec output scratch. Reused across the loop — same-stream
    // ordering guarantees the previous call's cast-to-F32 finishes before
    // the next call overwrites the buffer.
    void * y_lp_buf   = nullptr;
    size_t y_lp_bytes = 0;
    if (w_lp) {
        y_lp_bytes = (size_t) rows * sizeof(uint16_t);
        y_lp_buf   = ctx->pool.alloc(y_lp_bytes);
    }

    // Zero bias sized for one row of the chosen output dtype.
    const size_t bias_bytes = w_lp
        ? (size_t) rows * sizeof(uint16_t)
        : (size_t) rows * sizeof(float);
    void * bias_dev = gcu_get_zero_bias(ctx, bias_bytes);

    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            const int32_t expert_id = ids_host[t * n_expert_used + e];
            GGML_ASSERT(expert_id >= 0 && expert_id < n_expert);

            // Weight pointer for this expert. Q-typed weights live on
            // the device as F16 with their own packed stride.
            void * w_ptr;
            if (wq) {
                const size_t f16_per_expert = (size_t) cols * rows * sizeof(uint16_t);
                w_ptr = (char *) w->data + (size_t) expert_id * f16_per_expert;
            } else {
                w_ptr = (char *) w->data + (size_t) expert_id * w->nb[2];
            }

            // Input pointer for this (e, t) slot.
            const int64_t e_b = e % r;
            void * x_ptr;
            if (w_lp && xt != wt_eff) {
                const size_t off_elems = (size_t) (t * r + e_b) * (size_t) cols;
                x_ptr = (char *) x_lp_buf + off_elems * sizeof(uint16_t);
            } else {
                x_ptr = (char *) x->data
                      + (size_t) t * x->nb[2]
                      + (size_t) e_b * x->nb[1];
            }

            // Output slot in dst.
            void * dst_ptr = (char *) dst->data
                           + (size_t) t * dst->nb[2]
                           + (size_t) e * dst->nb[1];

            int64_t lhs_d[2] = { 1, cols };
            int64_t lhs_s[2] = { cols, 1 };
            int64_t rhs_d[2] = { rows, cols };
            int64_t rhs_s[2];
            int64_t out_d[2] = { 1, rows };
            int64_t out_s[2] = { rows, 1 };
            int64_t bias_d[1] = { rows };
            int64_t bias_s[1] = { 1 };

            if (w_lp) {
                // Q-stored-as-F16 has packed F16 strides; F16 / BF16 native
                // use the declared nb[].
                if (wq) {
                    rhs_s[0] = cols;
                    rhs_s[1] = 1;
                } else {
                    rhs_s[0] = (int64_t) (w->nb[1] / sizeof(uint16_t));
                    rhs_s[1] = 1;
                }

                topsatenTensor lhs_lp(topsatenSize_t(lhs_d, 2), topsatenSize_t(lhs_s, 2),
                                      lp_dtype, x_ptr);
                topsatenTensor rhs_lp(topsatenSize_t(rhs_d, 2), topsatenSize_t(rhs_s, 2),
                                      lp_dtype, w_ptr);
                topsatenTensor out_lp(topsatenSize_t(out_d, 2), topsatenSize_t(out_s, 2),
                                      lp_dtype, y_lp_buf);
                topsatenTensor bias_lp(topsatenSize_t(bias_d, 1), topsatenSize_t(bias_s, 1),
                                       lp_dtype, bias_dev);
                TOPSATEN_CHECK(topsatenLinear(out_lp, lhs_lp, rhs_lp, bias_lp,
                                              ctx->compute_stream));

                topsatenTensor out_f32(topsatenSize_t(out_d, 2), topsatenSize_t(out_s, 2),
                                       TOPSATEN_DATA_FP32, dst_ptr);
                topsatenDataType_t target_f32 = TOPSATEN_DATA_FP32;
                TOPSATEN_CHECK(topsatenTo(out_f32, out_lp, target_f32,
                                          false, true, TOPSATEN_MEMORY_CONTIGUOUS,
                                          ctx->compute_stream));
            } else {
                rhs_s[0] = (int64_t) (w->nb[1] / sizeof(float));
                rhs_s[1] = 1;

                topsatenTensor lhs(topsatenSize_t(lhs_d, 2), topsatenSize_t(lhs_s, 2),
                                   TOPSATEN_DATA_FP32, x_ptr);
                topsatenTensor rhs(topsatenSize_t(rhs_d, 2), topsatenSize_t(rhs_s, 2),
                                   TOPSATEN_DATA_FP32, w_ptr);
                topsatenTensor out(topsatenSize_t(out_d, 2), topsatenSize_t(out_s, 2),
                                   TOPSATEN_DATA_FP32, dst_ptr);
                topsatenTensor bias(topsatenSize_t(bias_d, 1), topsatenSize_t(bias_s, 1),
                                    TOPSATEN_DATA_FP32, bias_dev);
                TOPSATEN_CHECK(topsatenLinear(out, lhs, rhs, bias, ctx->compute_stream));
            }
        }
    }

    gcu_release_scratch(ctx, x_lp_buf, x_lp_bytes);
    gcu_release_scratch(ctx, y_lp_buf, y_lp_bytes);
    return true;
}


// CPY/DUP/CONT. Same dtype + contiguous → fast topsMemcpyAsync(D2D).
// Different dtype (F32↔F16, both contiguous) → topsatenTo cast.
bool gcu_op_cpy(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));

    if (src->type == dst->type) {
        GGML_ASSERT(ggml_nbytes(src) == ggml_nbytes(dst));
        TOPS_CHECK(topsMemcpyAsync(dst->data, src->data, ggml_nbytes(src),
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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
                              TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
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
bool gcu_op_set_rows(ggml_backend_gcu_context * ctx, ggml_tensor * node) {
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
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
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
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
    }

    gcu_release_scratch(ctx, cast_buf, cast_bytes);
    return true;
}

// MVP-1: GET_ROWS handles only the unbatched case (in is effectively 2D
// [n, m], idx is 1D [r]). Batched GET_ROWS (be1>1 or be2>1) goes to CPU.
bool gcu_op_get_rows(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
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

    TOPSATEN_CHECK(topsatenIndexSelect(out_tt, in_tt, /*dim=*/0, idx_tt, ctx->compute_stream));
    return true;
}

// Builds an interleaved [max_pos, n_dims] cos/sin table per topsvllm
// convention: row p contains cos values for theta_i*p in the first n_dims/2
// slots followed by sin values in the next n_dims/2 slots.
//
// theta_i = freq_base^(-2i / n_dims) * freq_scale / freq_factors[i]
//
// `freq_factors` may be NULL (no per-frequency scaling, behaves as 1.0).
// When non-NULL it must point to at least `n_dims/2` floats; the per-pair
// theta is divided by `freq_factors[i]` for pair index i. This matches
// the CPU reference's `theta/ff` in `ggml_rope_cache_init` (ops.cpp).
static void gcu_build_rope_cos_sin_host(int n_dims, int max_pos,
                                        float freq_base, float freq_scale,
                                        const float * freq_factors,
                                        std::vector<float> & out) {
    const int half = n_dims / 2;
    out.assign((size_t) max_pos * (size_t) n_dims, 0.0f);
    for (int p = 0; p < max_pos; p++) {
        float * row = out.data() + (size_t) p * n_dims;
        for (int i = 0; i < half; i++) {
            const float ff    = freq_factors ? freq_factors[i] : 1.0f;
            const float theta = std::pow(freq_base, -2.0f * (float) i / (float) n_dims) * freq_scale / ff;
            const float angle = (float) p * theta;
            row[i]        = std::cos(angle);
            row[i + half] = std::sin(angle);
        }
    }
}

// ROPE. Supports NORMAL (mode 0), NEOX (mode 2), `freq_factors`, and
// partial rotation. Multi-axis modes (MROPE / VISION / IMROPE) are wired
// to topsvllmMRotaryEmbedding behind the `GGML_GCU_ENABLE_MROPE`
// environment variable while we verify the SDK kernel's expected
// position-tensor layout against ggml's [4 * n_tokens] axis-major layout
// (see MVP-5b/3-4 spec). With the env unset they continue to fall back
// to CPU as before MVP-5b.
// YARN's full theta-mixing path (ext_factor != 0) still falls back to CPU.
//
// Inputs: x [head_dim, n_heads, n_tokens, 1] F32/F16/BF16;
//         pos [n_tokens]  (NORMAL/NEOX) or [4*n_tokens] (MROPE/VISION/IMROPE)  I32;
//         freq_factors [n_dims/2]  optional F32.
// Maps to topsvllmRotaryEmbedding(query, key, positions, cos_sin_cache,
//   head_size, is_neox, stream) for NORMAL/NEOX, and
// topsvllmMRotaryEmbedding(..., mrope_section, mrope_interleaved, stream)
// for MROPE/VISION/IMROPE when the gate env is set. is_neox=true rotates
// split halves (NeoX/Phi style) instead of interleaved pairs.
// We treat x as the query; pass a small zero-filled scratch as key
// (topsvllm rotates both; we ignore the dummy key result).
// Positions get cast to I64 on host. cos_sin table is precomputed on host
// per call.
//
// In-place: ggml_rope returns a view of `a`, so dst->data == x->data already.
bool gcu_op_rope(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * x        = dst->src[0];
    ggml_tensor * pos      = dst->src[1];
    ggml_tensor * src_freq = dst->src[2];   // freq_factors (may be NULL)

    const int32_t n_dims = ((const int32_t *) dst->op_params)[1];
    const int32_t mode   = ((const int32_t *) dst->op_params)[2];
    const bool    is_neox  = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool    is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;   // MROPE | VISION | IMROPE
    const bool    is_vision = (mode == GGML_ROPE_TYPE_VISION);
    const bool    is_imrope = (mode == GGML_ROPE_TYPE_IMROPE);
    float freq_base, freq_scale;
    memcpy(&freq_base,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const int32_t *) dst->op_params + 6, sizeof(float));
    int32_t sections[4] = {0, 0, 0, 0};
    memcpy(sections, (const int32_t *) dst->op_params + 11, sizeof(sections));

    const int64_t head_dim = x->ne[0];
    const int64_t n_heads  = x->ne[1];
    const int64_t n_tokens = x->ne[2];
    GGML_ASSERT(n_dims <= head_dim);
    if (is_mrope) {
        // ggml lays out positions as [4 * n_tokens] axis-major
        // [T_0..T_n, H_0..H_n, W_0..W_n, E_0..E_n] (see
        // ggml-cpu/ops.cpp:ggml_compute_forward_rope_flt where
        // p_h = pos[i2 + ne2]).
        GGML_ASSERT(pos->ne[0] == n_tokens * 4);
    } else {
        GGML_ASSERT(pos->ne[0] == n_tokens);
    }

    // Read positions to host so we can determine max_pos and cast to I64.
    const int64_t pos_count = pos->ne[0];
    std::vector<int32_t> pos_host(pos_count);
    TOPS_CHECK(topsMemcpy(pos_host.data(), pos->data,
                          (size_t) pos_count * sizeof(int32_t),
                          topsMemcpyDeviceToHost));

    int max_pos = 0;
    for (int32_t p : pos_host) max_pos = std::max(max_pos, (int) p);
    max_pos += 1;

    // Pull freq_factors to host if present; the host-side cos/sin builder
    // applies the per-pair scale.
    std::vector<float> ff_host;
    const float *      ff_ptr = nullptr;
    if (src_freq) {
        GGML_ASSERT(src_freq->type == GGML_TYPE_F32);
        const int64_t ff_n = src_freq->ne[0];
        GGML_ASSERT(ff_n >= n_dims / 2);
        ff_host.resize(ff_n);
        TOPS_CHECK(topsMemcpy(ff_host.data(), src_freq->data,
                              (size_t) ff_n * sizeof(float),
                              topsMemcpyDeviceToHost));
        ff_ptr = ff_host.data();
    }

    std::vector<float> cs_host;
    gcu_build_rope_cos_sin_host(n_dims, max_pos, freq_base, freq_scale, ff_ptr, cs_host);

    // The SDK requires cos_sin_cache.dtype == query.dtype on this arch
    // (see op_vllm_rotary_embedding.h:188 — the FP32-cs override only
    // applies when the runtime check IS satisfied, but we observed NaN
    // empirically for F16 query + F32 cs). Pack to the matching low-prec
    // dtype when query is F16 or BF16.
    const bool   cs_is_f16  = (x->type == GGML_TYPE_F16);
    const bool   cs_is_bf16 = (x->type == GGML_TYPE_BF16);
    const bool   cs_is_low  = cs_is_f16 || cs_is_bf16;
    const size_t cs_bpe     = cs_is_low ? sizeof(uint16_t) : sizeof(float);
    const size_t cs_bytes   = cs_host.size() * cs_bpe;
    const size_t pos_bytes  = (size_t) pos_count * sizeof(int64_t);
    void * cs_dev  = ctx->pool.alloc(cs_bytes);
    void * pos_dev = ctx->pool.alloc(pos_bytes);

    // Host-side staging buffers must outlive the async memcpy. topsrt's
    // pageable-host->device path normally stages the bytes synchronously
    // at memcpyAsync call time, but we keep these vectors at function
    // scope so any future implementation change (e.g. true async DMA)
    // doesn't silently reintroduce a use-after-free.
    std::vector<ggml_fp16_t> cs_f16;
    std::vector<ggml_bf16_t> cs_bf16;
    if (cs_is_f16) {
        cs_f16.resize(cs_host.size());
        ggml_fp32_to_fp16_row(cs_host.data(), cs_f16.data(), (int64_t) cs_host.size());
        TOPS_CHECK(topsMemcpyAsync(cs_dev, cs_f16.data(), cs_bytes,
                                   topsMemcpyHostToDevice, ctx->compute_stream));
    } else if (cs_is_bf16) {
        cs_bf16.resize(cs_host.size());
        ggml_fp32_to_bf16_row(cs_host.data(), cs_bf16.data(), (int64_t) cs_host.size());
        TOPS_CHECK(topsMemcpyAsync(cs_dev, cs_bf16.data(), cs_bytes,
                                   topsMemcpyHostToDevice, ctx->compute_stream));
    } else {
        TOPS_CHECK(topsMemcpyAsync(cs_dev, cs_host.data(), cs_bytes,
                                   topsMemcpyHostToDevice, ctx->compute_stream));
    }
    std::vector<int64_t> pos_i64(pos_count);
    for (int64_t i = 0; i < pos_count; i++) pos_i64[i] = (int64_t) pos_host[i];
    TOPS_CHECK(topsMemcpyAsync(pos_dev, pos_i64.data(), pos_bytes,
                               topsMemcpyHostToDevice, ctx->compute_stream));
    // Synchronize before the kernel: the H->D copies above use std::vector
    // sources whose lifetime ends when this function returns. With MVP-4b
    // there is no per-op synchronize, so the kernel may run after the
    // function returns. A targeted stream sync here drains the copies
    // before the kernel queues any reads from cs_dev / pos_dev.
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));

    // topsvllmRotaryEmbedding rotates query in-place. For ggml's non-inplace
    // ROPE (dst is a fresh tensor distinct from x), copy x into dst first so
    // we can rotate dst safely without touching x.
    if (dst->data != x->data) {
        TOPS_CHECK(topsMemcpyAsync(dst->data, x->data, ggml_nbytes(x),
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
    }

    // Dummy key tensor — small, zero, just to satisfy topsvllmRotaryEmbedding's
    // dual-tensor signature. Shape [n_tokens, head_dim].
    const size_t dummy_bytes = (size_t) n_tokens * head_dim * ggml_type_size(x->type);
    void * dummy_key_dev = ctx->pool.alloc(dummy_bytes);
    TOPS_CHECK(topsMemsetAsync(dummy_key_dev, 0, dummy_bytes, ctx->compute_stream));

    // query: [n_tokens, n_heads * head_dim] over dst's memory (rotated in place).
    int64_t q_d[2] = { n_tokens, n_heads * head_dim };
    int64_t q_s[2] = { (int64_t)(dst->nb[2] / ggml_type_size(dst->type)), 1 };
    topsatenTensor q_tt(topsatenSize_t(q_d, 2), topsatenSize_t(q_s, 2),
                        ggml_to_topsaten_dtype(dst->type), dst->data);

    int64_t k_d[2] = { n_tokens, head_dim };
    int64_t k_s[2] = { head_dim, 1 };
    topsatenTensor k_tt(topsatenSize_t(k_d, 2), topsatenSize_t(k_s, 2),
                        ggml_to_topsaten_dtype(x->type), dummy_key_dev);

    int64_t pos_d[1] = { pos_count };
    int64_t pos_s[1] = { 1 };
    topsatenTensor pos_tt(topsatenSize_t(pos_d, 1), topsatenSize_t(pos_s, 1),
                          TOPSATEN_DATA_I64, pos_dev);

    int64_t cs_d[2] = { max_pos, n_dims };
    int64_t cs_s[2] = { n_dims, 1 };
    topsatenTensor cs_tt(topsatenSize_t(cs_d, 2), topsatenSize_t(cs_s, 2),
                         ggml_to_topsaten_dtype(x->type), cs_dev);

    if (is_mrope) {
        // MROPE / VISION / IMROPE — multi-axis (T, H, W, E) rotary.
        // ggml header documents that MROPE/VISION always use NeoX
        // ordering even when the NEOX bit is unset in `mode`.
        // ggml header (ggml.h:1837): "NEOX ordering is automatically applied
        // and cannot be disabled for MROPE and VISION".
        const bool mrope_is_neox = true;
        int64_t mrope_sec[4] = {
            (int64_t) sections[0], (int64_t) sections[1],
            (int64_t) sections[2], (int64_t) sections[3]
        };
        topsatenSize_t mrope_section_t(mrope_sec, 4);
        // TODO(MVP-5b/3-4): the SDK header documents `positions: [num_tokens]`
        // but ggml passes [4 * num_tokens]. The kernel layout is in
        // dispute (axis-major vs token-major; 3 vs 4 axes). The
        // smoke under `#if 0 // MVP-5b/3-4` in test-backend-gcu.cpp
        // shows the GCU output diverging from CPU on MROPE; needs a
        // direct probe (e.g. all-zero positions on each axis to see
        // which slot the kernel reads). Until resolved, supports_op
        // refuses MROPE so this branch is unreachable in practice.
        if (is_imrope) {
            TOPSATEN_CHECK(topsvllmMRotaryEmbedding(q_tt, k_tt, pos_tt, cs_tt,
                                                    (int) head_dim, mrope_is_neox,
                                                    mrope_section_t,
                                                    /*mrope_interleaved=*/true,
                                                    ctx->compute_stream));
        } else {
            TOPSATEN_CHECK(topsvllmMRotaryEmbedding(q_tt, k_tt, pos_tt, cs_tt,
                                                    (int) head_dim, mrope_is_neox,
                                                    mrope_section_t,
                                                    ctx->compute_stream));
        }
    } else {
        TOPSATEN_CHECK(topsvllmRotaryEmbedding(q_tt, k_tt, pos_tt, cs_tt,
                                               (int) head_dim, is_neox,
                                               ctx->compute_stream));
    }

    gcu_release_scratch(ctx, cs_dev,        cs_bytes);
    gcu_release_scratch(ctx, pos_dev,       pos_bytes);
    gcu_release_scratch(ctx, dummy_key_dev, dummy_bytes);
    return true;
}

// SOFT_MAX. out = softmax(scale * x + mask, dim=-1). max_bias must be 0.
// Plan: scale x into a scratch slab (mul by scalar), optionally add mask,
// then topsatenSoftmaxForward along the last PyTorch dim (= ggml ne[0]).
bool gcu_op_soft_max(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
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
        TOPSATEN_CHECK(topsatenMul(scratch_t, x_t, s, ctx->compute_stream));
    }

    // scratch += mask (with alpha=1)
    //
    // ggml constrains mask dtype to F16 or F32, while x can also be BF16.
    // topsatenAdd refuses mixed dtypes, so when mask dtype differs from x
    // we cast mask into a per-call scratch (small — same shape as mask)
    // before the add. Real BF16 models hit the BF16 input + F16 mask path.
    void * mask_cast_buf  = nullptr;
    size_t mask_cast_bytes = 0;
    if (mask) {
        topsatenTensor mask_t = make_topsaten_tensor(mask, d_mask);
        if (mask->type != dst->type) {
            mask_cast_bytes = (size_t) ggml_nelements(mask) * ggml_type_size(dst->type);
            mask_cast_buf   = ctx->pool.alloc(mask_cast_bytes);
            // Build a contiguous descriptor for the cast destination using
            // mask's element shape (same shape, target dtype, packed strides).
            int rank_m = ggml_n_dims(mask); if (rank_m < 1) rank_m = 1;
            int64_t md[GGML_MAX_DIMS];
            int64_t ms[GGML_MAX_DIMS];
            for (int i = 0; i < rank_m; i++) md[i] = mask->ne[rank_m - 1 - i];
            ms[rank_m - 1] = 1;
            for (int i = rank_m - 2; i >= 0; i--) ms[i] = ms[i + 1] * md[i + 1];
            topsatenDataType_t target = ggml_to_topsaten_dtype(dst->type);
            topsatenTensor mask_cast_t(topsatenSize_t(md, rank_m),
                                       topsatenSize_t(ms, rank_m),
                                       target, mask_cast_buf);
            TOPSATEN_CHECK(topsatenTo(mask_cast_t, mask_t, target, false, true,
                                      TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
            mask_t = mask_cast_t;
        }
        topsatenScalar_t alpha; alpha.dtype = TOPSATEN_DATA_FP32; alpha.fval = 1.0;
        TOPSATEN_CHECK(topsatenAdd(scratch_t, scratch_t, mask_t, alpha, ctx->compute_stream));
    }

    // out = softmax(scratch, dim=last)
    int rank = ggml_n_dims(dst); if (rank < 1) rank = 1;
    TOPSATEN_CHECK(topsatenSoftmaxForward(out_t, scratch_t, rank - 1, ctx->compute_stream));

    gcu_release_scratch(ctx, scratch,        bytes);
    gcu_release_scratch(ctx, mask_cast_buf,  mask_cast_bytes);
    return true;
}

// FLASH_ATTN_EXT. Fused scaled-dot-product attention.
//
// ggml contract:
//   src[0] q    [head_dim,    n_q,  n_head,    n_batch]
//   src[1] k    [head_dim,    n_kv, n_head_kv, n_batch]   (n_head % n_head_kv == 0)
//   src[2] v    [head_dim,    n_kv, n_head_kv, n_batch]
//   src[3] mask [n_kv,        n_q,  ne32,      ne33]      F16 (or null)
//   dst         [head_dim,    n_head, n_q,     n_batch]   !! permuted output !!
//   op_params: float scale, float max_bias, float logit_softcap, int32 prec
//
// Backend: topsatenScaledDotProductAttention expects PyTorch layout
// [batch, head_num, seq_len, head_size]. ggml's reverse-fastest-first ne
// for q is [head_dim, n_q, n_head, n_batch], i.e. PyTorch
// [n_batch, n_head, n_q, head_dim] when the tensor is contiguous. The
// real call site in llama-graph.cpp passes ggml_permute(0, 2, 1, 3) views,
// so q/k/v arrive with non-contiguous strides; we materialize each into a
// PyTorch-contiguous F16 scratch via topsatenTo before SDP (topsatenTo
// folds the dtype cast and the stride collapse into a single op).
//
// The dst memory is contiguous for ne=[head_dim, n_head, n_q, n_batch],
// i.e. PyTorch [n_batch, n_q, n_head, head_dim] dense. SDP returns its
// output in [n_batch, n_head, n_q, head_dim] order; we land it in a
// contiguous scratch, then "logically permute" by re-describing the
// scratch as [n_batch, n_q, n_head, head_dim] with strides
// [n_head*n_q*head_dim, head_dim, n_q*head_dim, 1] — the same memory
// addressed in dst's [b][q][h][d] order — and feed it to topsatenCopy
// which writes into the dst layout element by element.
//
// supports_op gates out cases this handler doesn't cover (max_bias != 0
// for ALiBi, logit_softcap != 0, sinks src[4], non-F16 mask, K/V types
// other than F16, etc.). Within those gates the handler always succeeds.
bool gcu_op_flash_attn_ext(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * q    = dst->src[0];
    ggml_tensor * k    = dst->src[1];
    ggml_tensor * v    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    GGML_ASSERT(max_bias      == 0.0f);   // ALiBi: gated out by supports_op
    GGML_ASSERT(logit_softcap == 0.0f);   // softcap: gated out by supports_op

    // Tensor sizes (ggml fastest-first).
    const int64_t head_dim  = q->ne[0];
    const int64_t n_q       = q->ne[1];
    const int64_t n_head    = q->ne[2];
    const int64_t n_batch   = q->ne[3];
    const int64_t n_kv      = k->ne[1];
    const int64_t n_head_kv = k->ne[2];
    GGML_ASSERT(v->ne[0] == head_dim);
    GGML_ASSERT(v->ne[1] == n_kv);
    GGML_ASSERT(v->ne[2] == n_head_kv);
    GGML_ASSERT(v->ne[3] == n_batch);
    GGML_ASSERT(n_head % n_head_kv == 0);

    // dst ne (per ggml_flash_attn_ext): [head_dim, n_head, n_q, n_batch].
    GGML_ASSERT(dst->ne[0] == head_dim);
    GGML_ASSERT(dst->ne[1] == n_head);
    GGML_ASSERT(dst->ne[2] == n_q);
    GGML_ASSERT(dst->ne[3] == n_batch);

    // Materialize Q/K/V into PyTorch-contiguous F16 scratches. Q/K/V
    // typically come in as ggml_permute(0, 2, 1, 3) views — non-contiguous.
    // Topsaten SDP expects standard [B, H, Seq, D] contiguous layout, so we
    // collapse strides via topsatenTo (which also handles the F32→F16 cast
    // for Q in the common F32-Q decode case).
    auto materialize_f16 = [&](ggml_tensor * t,
                               int64_t B, int64_t H, int64_t S, int64_t D,
                               void *& out_scratch, size_t & out_bytes) -> topsatenTensor {
        // Build the source descriptor in [B, H, S, D] PyTorch order using
        // the tensor's raw ggml strides. We can't go through
        // make_topsaten_tensor() because it folds trailing-1 dims away
        // (via ggml_n_dims), and topsatenTo requires src and dst rank to
        // match. Build rank-4 explicitly to match dst_t below.
        const size_t bpe = ggml_type_size(t->type);
        int64_t sd[4] = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
        int64_t ss[4] = { (int64_t) (t->nb[3] / bpe), (int64_t) (t->nb[2] / bpe),
                          (int64_t) (t->nb[1] / bpe), (int64_t) (t->nb[0] / bpe) };
        GGML_ASSERT(sd[0] == B && sd[1] == H && sd[2] == S && sd[3] == D);
        topsatenTensor src_t(topsatenSize_t(sd, 4), topsatenSize_t(ss, 4),
                             ggml_to_topsaten_dtype(t->type), t->data);

        out_bytes   = (size_t) B * H * S * D * sizeof(uint16_t);
        out_scratch = ctx->pool.alloc(out_bytes);

        int64_t dd[4] = { B, H, S, D };
        int64_t ds[4] = { H * S * D, S * D, D, 1 };
        topsatenTensor dst_t(topsatenSize_t(dd, 4), topsatenSize_t(ds, 4),
                             TOPSATEN_DATA_FP16, out_scratch);
        topsatenDataType_t target = TOPSATEN_DATA_FP16;
        TOPSATEN_CHECK(topsatenTo(dst_t, src_t, target,
                                  false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        return dst_t;
    };

    void * q_buf = nullptr; size_t q_bytes = 0;
    void * k_buf = nullptr; size_t k_bytes = 0;
    void * v_buf = nullptr; size_t v_bytes = 0;
    topsatenTensor q_t = materialize_f16(q, n_batch, n_head,    n_q,  head_dim, q_buf, q_bytes);
    topsatenTensor k_t = materialize_f16(k, n_batch, n_head_kv, n_kv, head_dim, k_buf, k_bytes);
    topsatenTensor v_t = materialize_f16(v, n_batch, n_head_kv, n_kv, head_dim, v_buf, v_bytes);

    // Mask: always F16 per ggml's FLASH_ATTN_EXT contract. SDP wants
    // [batch, q_head, q_seq_len, kv_seq_len]; ggml mask ne is
    // [n_kv, n_q, ne32, ne33] (ne32/ne33 are 1 for the standard causal
    // mask, with broadcast on q-head and batch handled by stride==0 logic
    // in topsaten). Build rank-4 explicitly.
    topsatenTensor mask_t;
    bool have_mask = (mask != nullptr);
    if (have_mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        const size_t mbpe = ggml_type_size(mask->type);
        // Mask ne (ggml fastest-first): [n_kv, n_q, m_h, m_b]
        // PyTorch order [m_b, m_h, n_q, n_kv].
        int64_t md[4] = { mask->ne[3], mask->ne[2], mask->ne[1], mask->ne[0] };
        int64_t ms[4] = { (int64_t) (mask->nb[3] / mbpe),
                          (int64_t) (mask->nb[2] / mbpe),
                          (int64_t) (mask->nb[1] / mbpe),
                          (int64_t) (mask->nb[0] / mbpe) };
        mask_t = topsatenTensor(topsatenSize_t(md, 4), topsatenSize_t(ms, 4),
                                TOPSATEN_DATA_FP16, mask->data);
    }

    // SDP output scratch in F16, [B, H, n_q, D] PyTorch-contiguous.
    // We cast and permute into dst after SDP; the scratch is always
    // contiguous so SDP sees the canonical layout it expects.
    const size_t out_f16_bytes = (size_t) n_batch * n_head * n_q * head_dim * sizeof(uint16_t);
    void * out_f16 = ctx->pool.alloc(out_f16_bytes);
    int64_t od[4] = { n_batch, n_head, n_q, head_dim };
    int64_t os[4] = { n_head * n_q * head_dim, n_q * head_dim, head_dim, 1 };
    topsatenTensor sdp_out(topsatenSize_t(od, 4), topsatenSize_t(os, 4),
                           TOPSATEN_DATA_FP16, out_f16);

    topsatenScalar_t scale_s;
    scale_s.dtype = TOPSATEN_DATA_FP32;
    scale_s.fval  = (double) scale;

    // Empty-mask sentinel: pass a default-constructed tensor (.data ==
    // nullptr) when the call has no mask. Topsaten interprets that as
    // "no attn_mask".
    topsatenTensor empty_mask;
    TOPSATEN_CHECK(topsatenScaledDotProductAttention(
        sdp_out, q_t, k_t, v_t,
        have_mask ? mask_t : empty_mask,
        /* dropout_p */ 0.0,
        /* is_causal */ false,           // mask carries causal info already
        scale_s,
        ctx->compute_stream));

    // Permute SDP output [B, H, Q, D] into dst layout [B, Q, H, D].
    // dst memory is contiguous for ne=[head_dim, n_head, n_q, n_batch],
    // i.e. dense in PyTorch order [n_batch, n_q, n_head, head_dim].
    // topsatenPermute writes out[i_0, i_1, ...] = in[i_{dims[0]}, ...]; we
    // want out[b, q, h, d] = in[b, h, q, d] → dims = [0, 2, 1, 3].
    //
    // Cast dst dtype if needed (dst is typically F32; SDP scratch is F16).
    // Build a rank-4 view of dst in PyTorch order [B, n_q, H, D]. dst is
    // contiguous (gated by supports_op) for ne=[head_dim, n_head, n_q, n_batch],
    // so its dense PyTorch layout is [n_batch, n_q, n_head, head_dim].
    const size_t dst_bpe = ggml_type_size(dst->type);
    int64_t dout_d[4] = { n_batch, n_q, n_head, head_dim };
    int64_t dout_s[4] = { (int64_t) (dst->nb[3] / dst_bpe),
                          (int64_t) (dst->nb[2] / dst_bpe),
                          (int64_t) (dst->nb[1] / dst_bpe),
                          (int64_t) (dst->nb[0] / dst_bpe) };

    // Re-describe the SDP scratch in [B, n_q, n_head, head_dim] order with
    // strides that select the same memory cells as the SDP output tensor
    // viewed at [B, n_head, n_q, head_dim] dense layout. This is the
    // "logical permute" view: scratch_view[b][q][h][d] == sdp_out[b][h][q][d].
    //
    //   sdp_out strides (B, H, Q, D contig): [n_head*n_q*head_dim, n_q*head_dim, head_dim, 1]
    //   logical permute (0, 2, 1, 3):        [n_head*n_q*head_dim, head_dim, n_q*head_dim, 1]
    int64_t pd[4] = { n_batch, n_q, n_head, head_dim };
    int64_t ps[4] = { n_head * n_q * head_dim, head_dim, n_q * head_dim, 1 };

    if (dst->type == GGML_TYPE_F32) {
        // Cast F16 SDP output -> F32 in a contiguous [B, H, Q, D] scratch.
        const size_t cast_bytes = (size_t) n_batch * n_head * n_q * head_dim * sizeof(float);
        void * cast_buf = ctx->pool.alloc(cast_bytes);
        topsatenTensor cast_t(topsatenSize_t(od, 4), topsatenSize_t(os, 4),
                              TOPSATEN_DATA_FP32, cast_buf);
        topsatenDataType_t target = TOPSATEN_DATA_FP32;
        TOPSATEN_CHECK(topsatenTo(cast_t, sdp_out, target,
                                  false, true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));

        // Permute via topsatenCopy: same shape [B, n_q, n_head, head_dim]
        // on both sides, scratch view selects [b][q][h][d] = scratch[b][h][q][d].
        topsatenTensor scratch_perm(topsatenSize_t(pd, 4), topsatenSize_t(ps, 4),
                                    TOPSATEN_DATA_FP32, cast_buf);
        topsatenTensor out_t(topsatenSize_t(dout_d, 4), topsatenSize_t(dout_s, 4),
                             TOPSATEN_DATA_FP32, dst->data);
        TOPSATEN_CHECK(topsatenCopy(out_t, scratch_perm, /*non_blocking=*/false,
                                    ctx->compute_stream));

        gcu_release_scratch(ctx, cast_buf, cast_bytes);
    } else {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        topsatenTensor scratch_perm(topsatenSize_t(pd, 4), topsatenSize_t(ps, 4),
                                    TOPSATEN_DATA_FP16, out_f16);
        topsatenTensor out_t(topsatenSize_t(dout_d, 4), topsatenSize_t(dout_s, 4),
                             TOPSATEN_DATA_FP16, dst->data);
        TOPSATEN_CHECK(topsatenCopy(out_t, scratch_perm, /*non_blocking=*/false,
                                    ctx->compute_stream));
    }

    gcu_release_scratch(ctx, q_buf,   q_bytes);
    gcu_release_scratch(ctx, k_buf,   k_bytes);
    gcu_release_scratch(ctx, v_buf,   v_bytes);
    gcu_release_scratch(ctx, out_f16, out_f16_bytes);
    return true;
}














// Tranche D4: reductions (SUM / SUM_ROWS / MEAN).
//
// ggml's contract:
//   SUM        — scalar, sum over all elements;       ggml output ne = {1, 1, 1, 1}
//   SUM_ROWS   — innermost reduce, keepdim;            ggml output ne = {1, ne01, ne02, ne03}
//   MEAN       — same as SUM_ROWS but divides by ne00; ggml output ne = {1, ne01, ne02, ne03}
//
// CPU forward is F32-only for these ops (see ops.cpp); we match that
// gate so cross-backend behaviour matches.

bool gcu_op_sum(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    gcu_tensor_dims din;
    topsatenTensor in_t = make_topsaten_tensor(src, din);

    // Output: rank-1 size-1 scalar tensor backed by ggml's dst buffer.
    int64_t out_d[1] = { 1 };
    int64_t out_s[1] = { 1 };
    topsatenTensor out_t(topsatenSize_t(out_d, 1), topsatenSize_t(out_s, 1),
                         TOPSATEN_DATA_FP32, dst->data);

    // Full reduction: pass the topsatenSum overload that takes only a
    // dtype (no `dim` argument) — sums all elements.
    TOPSATEN_CHECK(topsatenSum(out_t, in_t, TOPSATEN_DATA_FP32, ctx->compute_stream));
    return true;
}

// SUM_ROWS / MEAN: reduce ggml's innermost dim (ne[0]) with keepdim.
//
// ggml stores tensors slowest-last (PyTorch reversed). make_topsaten_tensor
// flips ne[]/nb[] back to slowest-first PyTorch order using rank =
// ggml_n_dims(t), which strips trailing 1's. So a ggml ne={8,21,1,1}
// shape arrives at topsaten as a rank-2 tensor [21, 8]; reducing ggml's
// innermost dim is reducing topsaten's last dim (rank-1).
//
// Output ggml shape is keepdim — for ne={8,21,1,1} input, ggml_sum_rows
// builds output ne={1,21,1,1}. ggml_n_dims of that is 2 (ne[1]=21 is the
// last >1 dim), so the output topsaten descriptor is also rank-2 [21,1].
// dims_in stays consistent across smoke ([4096,64] -> reduce dim 1) and
// the real Gemma 4 prefill shape ([8,21] -> reduce dim 1). We use the
// dim-list overload because topsatenSum has no single-int-dim overload
// (only Mean does).

bool gcu_op_sum_rows(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    gcu_tensor_dims din, dout;
    topsatenTensor in_t  = make_topsaten_tensor(src, din);
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);

    int rank = ggml_n_dims(src); if (rank < 1) rank = 1;
    int64_t dim_arr[1] = { (int64_t) (rank - 1) };  // PyTorch last dim == ggml ne[0]
    topsatenSize_t dims(dim_arr, 1);

    TOPSATEN_CHECK(topsatenSum(out_t, in_t, dims, /*keepdims=*/true,
                               TOPSATEN_DATA_FP32, ctx->compute_stream));
    return true;
}

bool gcu_op_mean(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    gcu_tensor_dims din, dout;
    topsatenTensor in_t  = make_topsaten_tensor(src, din);
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);

    int rank = ggml_n_dims(src); if (rank < 1) rank = 1;
    // Mean has a single-int-dim overload; prefer it (one less indirection
    // than the dim-list path).
    int32_t dim = (int32_t) (rank - 1);  // PyTorch last dim == ggml ne[0]
    TOPSATEN_CHECK(topsatenMean(out_t, in_t, dim, /*keepdims=*/true,
                                TOPSATEN_DATA_FP32, ctx->compute_stream));
    return true;
}

// Tranche D5: CUMSUM. F32 only (matches ggml's CPU forward). Reduces
// along ggml's innermost dim (ne[0]); output has same shape as input.
// Use the int32_t-dim overload of topsatenCumsum.
bool gcu_op_cumsum(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    gcu_tensor_dims din, dout;
    topsatenTensor in_t  = make_topsaten_tensor(src, din);
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);

    int rank = ggml_n_dims(src); if (rank < 1) rank = 1;
    int32_t dim = (int32_t)(rank - 1);  // PyTorch last dim == ggml ne[0]
    TOPSATEN_CHECK(topsatenCumsum(out_t, in_t, dim,
                                  TOPSATEN_DATA_FP32, ctx->compute_stream));
    return true;
}


// MVP-5b: sampling helper ops (ARGMAX / TOP_K / ARGSORT).
//
// These reduce / sort along ggml's innermost dim (ne[0]) and produce I32
// indices. ggml stores ne[]/nb[] reversed-PyTorch (slowest-last); the
// make_topsaten_tensor helper flips them to PyTorch order (slowest-first),
// so ggml's "dim 0" maps to topsaten "dim rank-1".
//
// The topsaten Topk/ArgSort APIs return I64 indices (PyTorch ATen
// convention), so we allocate an I64 scratch and cast to I32 with
// topsatenTo before placing into the ggml dst buffer.

// ARGMAX. Input: rank-2 F32/F16 [ne0, ne1]. Output: rank-1 I32 [ne1].
// Reduces along innermost dim (ggml dim 0 == PyTorch last dim).
bool gcu_op_argmax(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    // PyTorch order: input is [ne1, ne0]; reduce dim 1 (== last dim).
    // Output (keepdims=false) is [ne1], rank 1.
    const int64_t ne0 = src->ne[0];
    const int64_t ne1 = src->ne[1];
    const size_t  in_bpe = ggml_type_size(src->type);

    int64_t in_d[2] = { ne1, ne0 };
    int64_t in_s[2] = { (int64_t) (src->nb[1] / in_bpe), (int64_t) (src->nb[0] / in_bpe) };
    topsatenTensor in_t(topsatenSize_t(in_d, 2), topsatenSize_t(in_s, 2),
                        ggml_to_topsaten_dtype(src->type), src->data);

    int64_t out_d[1] = { ne1 };
    int64_t out_s[1] = { 1 };
    topsatenTensor out_t(topsatenSize_t(out_d, 1), topsatenSize_t(out_s, 1),
                         TOPSATEN_DATA_I32, dst->data);

    topsatenScalar_t dim_s;
    dim_s.dtype = TOPSATEN_DATA_I64;
    dim_s.ival  = 1;  // last dim in PyTorch order
    TOPSATEN_CHECK(topsatenArgmax(out_t, in_t, dim_s, /*keepdims=*/false,
                                  ctx->compute_stream));
    return true;
}

// TOP_K. Returns I32 indices of the k largest along innermost dim.
// ggml shape: input [ne00, ne01, ne02, ne03], output [k, ne01, ne02, ne03] I32.
// Per ggml's contract: "the resulting top k indices are in no particular
// order". We pass is_sorted=true to keep behavior deterministic; ggml does
// not require a particular order.
//
// SDK note: topsatenTopk's index tensor must be I32 / U32 (not I64 — this
// is the one place its dtype contract diverges from PyTorch ATen). We
// write directly into ggml's I32 dst buffer.
bool gcu_op_top_k(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int64_t k = dst->ne[0];

    // Input descriptor in PyTorch order via the standard helper.
    gcu_tensor_dims din;
    topsatenTensor in_t = make_topsaten_tensor(src, din);
    const int rank = ggml_n_dims(src) < 1 ? 1 : ggml_n_dims(src);

    // Output value scratch (same dtype as input, same shape as dst but with
    // ne[0] = k along the reduce dim). We don't use this; ggml only wants
    // indices.
    const size_t val_bytes = (size_t) ggml_nelements(dst) * ggml_type_size(src->type);
    void * val_buf = ctx->pool.alloc(val_bytes);

    // Output shape in PyTorch order: input shape with the last dim replaced
    // by k. Strides packed (dst is fully contiguous I32 — gated in
    // supports_op — so we can use packed strides for both the value scratch
    // and the index dst).
    int64_t out_d[GGML_MAX_DIMS];
    int64_t out_s[GGML_MAX_DIMS];
    for (int i = 0; i < rank; i++) {
        out_d[i] = (i == rank - 1) ? k : din.dims[i];
    }
    out_s[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        out_s[i] = out_s[i + 1] * out_d[i + 1];
    }

    topsatenTensor val_t(topsatenSize_t(out_d, rank), topsatenSize_t(out_s, rank),
                         ggml_to_topsaten_dtype(src->type), val_buf);
    topsatenTensor idx_t(topsatenSize_t(out_d, rank), topsatenSize_t(out_s, rank),
                         TOPSATEN_DATA_I32, dst->data);

    TOPSATEN_CHECK(topsatenTopk(val_t, idx_t, in_t,
                                /*k=*/k, /*axis=*/rank - 1,
                                /*is_largest=*/true, /*is_sorted=*/true,
                                ctx->compute_stream));

    gcu_release_scratch(ctx, val_buf, val_bytes);
    return true;
}

// ARGSORT. Returns I32 indices that sort the input along innermost dim.
// ggml shape: output is same shape as input but I32 dtype.
// op_params[0] is ggml_sort_order (0=ASC, 1=DESC).
//
// We try the direct I32 dst path first; if the SDK happens to require I64
// at runtime we'd need to add a cast hop, but in this SDK version
// topsatenArgSort accepts I32 indices (matching the topsatenTopk
// constraint observed at runtime).
bool gcu_op_argsort(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int32_t order = ((const int32_t *) dst->op_params)[0];
    const bool descending = (order == GGML_SORT_ORDER_DESC);

    // Input descriptor in PyTorch order.
    gcu_tensor_dims din;
    topsatenTensor in_t = make_topsaten_tensor(src, din);
    const int rank = ggml_n_dims(src) < 1 ? 1 : ggml_n_dims(src);

    // Output shape == input shape (PyTorch order). Packed strides because
    // dst is fully contiguous I32 (gated in supports_op).
    int64_t out_d[GGML_MAX_DIMS];
    int64_t out_s[GGML_MAX_DIMS];
    for (int i = 0; i < rank; i++) out_d[i] = din.dims[i];
    out_s[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        out_s[i] = out_s[i + 1] * out_d[i + 1];
    }

    topsatenTensor idx_t(topsatenSize_t(out_d, rank), topsatenSize_t(out_s, rank),
                         TOPSATEN_DATA_I32, dst->data);

    TOPSATEN_CHECK(topsatenArgSort(idx_t, in_t,
                                   /*stable=*/false, /*dim=*/rank - 1,
                                   descending, ctx->compute_stream));
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
        case GGML_OP_ADD1:
            return gcu_op_add1(ctx, node);
        case GGML_OP_SUB:
            return gcu_op_sub(ctx, node);
        case GGML_OP_MUL:
            return gcu_op_mul(ctx, node);
        case GGML_OP_DIV:
            return gcu_op_div(ctx, node);
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
        case GGML_OP_MUL_MAT_ID:
            return gcu_op_mul_mat_id(ctx, node);
        case GGML_OP_SOFT_MAX:
            return gcu_op_soft_max(ctx, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return gcu_op_flash_attn_ext(ctx, node);
        case GGML_OP_CONCAT:
            return gcu_op_concat(ctx, node);
        case GGML_OP_NORM:
            return gcu_op_norm(ctx, node);
        case GGML_OP_RMS_NORM:
            return gcu_op_rms_norm(ctx, node);
        case GGML_OP_ROPE:
            return gcu_op_rope(ctx, node);
        case GGML_OP_GLU:
            return gcu_op_glu(ctx, node);
        case GGML_OP_ARGMAX:
            return gcu_op_argmax(ctx, node);
        case GGML_OP_TOP_K:
            return gcu_op_top_k(ctx, node);
        case GGML_OP_ARGSORT:
            return gcu_op_argsort(ctx, node);
        case GGML_OP_SQR:
            return gcu_op_sqr(ctx, node);
        case GGML_OP_SQRT:
            return gcu_op_sqrt(ctx, node);
        case GGML_OP_LOG:
            return gcu_op_log(ctx, node);
        case GGML_OP_SIN:
            return gcu_op_sin(ctx, node);
        case GGML_OP_COS:
            return gcu_op_cos(ctx, node);
        case GGML_OP_SUM:
            return gcu_op_sum(ctx, node);
        case GGML_OP_SUM_ROWS:
            return gcu_op_sum_rows(ctx, node);
        case GGML_OP_MEAN:
            return gcu_op_mean(ctx, node);
        case GGML_OP_CLAMP:
            return gcu_op_clamp(ctx, node);
        case GGML_OP_CUMSUM:
            return gcu_op_cumsum(ctx, node);
        case GGML_OP_L2_NORM:
            return gcu_op_l2_norm(ctx, node);
        case GGML_OP_GROUP_NORM:
            return gcu_op_group_norm(ctx, node);
        case GGML_OP_LEAKY_RELU:
            return gcu_op_leaky_relu(ctx, node);
        case GGML_OP_REPEAT:
            return gcu_op_repeat(ctx, node);
        case GGML_OP_UNARY: {
            const enum ggml_unary_op uop = ggml_get_unary_op(node);
            switch (uop) {
                case GGML_UNARY_OP_SILU:        return gcu_op_silu(ctx, node);
                case GGML_UNARY_OP_GELU:        return gcu_op_gelu(ctx, node);
                case GGML_UNARY_OP_GELU_QUICK:  return gcu_op_gelu_quick(ctx, node);
                case GGML_UNARY_OP_RELU:        return gcu_op_relu(ctx, node);
                case GGML_UNARY_OP_TANH:        return gcu_op_tanh(ctx, node);
                case GGML_UNARY_OP_SIGMOID:     return gcu_op_sigmoid(ctx, node);
                case GGML_UNARY_OP_HARDSWISH:   return gcu_op_hardswish(ctx, node);
                case GGML_UNARY_OP_HARDSIGMOID: return gcu_op_hardsigmoid(ctx, node);
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
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return gcu_all_inputs_supported_dtype(op) &&
                   gcu_numpy_broadcastable(op->src[0], op->src[1]);
        case GGML_OP_ADD1: {
            // ggml_add1: dst[...] = a[...] + b[0], b is a scalar tensor.
            // We pass it as a rank-1 size-1 tensor and let topsatenAdd
            // broadcast across a's full shape.
            const ggml_tensor * a = op->src[0];
            const ggml_tensor * b = op->src[1];
            if (!a || !b) return false;
            if (!gcu_dtype_supported(a->type) || !gcu_dtype_supported(b->type)) return false;
            if (a->type != op->type) return false;
            if (b->type != a->type) return false;
            if (!ggml_is_scalar(b)) return false;
            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        // GGML_OP_ADD_ID is intentionally left to the CPU fallback. The
        // op is an indexed per-row bias (dst[i0,i1,i2] = a[...] + b[i0,
        // ids[i1,i2]]) whose only real use site is MoE expert biases in
        // GGUFs that ship them. Implementing it on GCU requires either a
        // gather-then-add pair (topsatenIndexSelect + topsatenAdd) with a
        // shape gymnastic, or a per-row launch loop. The Gemma 4 26B A4B
        // canary doesn't ship per-expert biases, so the CPU fallback
        // costs nothing on the model we run end-to-end. Revisit when a
        // model with these biases lands in the test suite.
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
            // Same dtype OR any pairwise F32/F16/BF16 conversion via topsatenTo.
            if (src->type != op->type) {
                const ggml_type s = src->type, d = op->type;
                const bool ok =
                    (s == GGML_TYPE_F32  && (d == GGML_TYPE_F16 || d == GGML_TYPE_BF16)) ||
                    (s == GGML_TYPE_F16  && (d == GGML_TYPE_F32 || d == GGML_TYPE_BF16)) ||
                    (s == GGML_TYPE_BF16 && (d == GGML_TYPE_F32 || d == GGML_TYPE_F16));
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
            //   (BF16, F32/F16/BF16, F32/F16/BF16) BF16 weight + cast input/output
            bool ok = false;
            if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ok = true;
            if (w->type == GGML_TYPE_F16 &&
                (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;
            // MVP-5a: BF16 weight path. Mirrors the F16 path with TOPSATEN_DATA_BF16
            // and accepts F32/F16/BF16 input + output (handler casts as needed).
            if (w->type == GGML_TYPE_BF16 &&
                (x->type == GGML_TYPE_BF16 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;
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
        case GGML_OP_MUL_MAT_ID: {
            // MoE expert dispatch.
            //   src[0] (w):   [cols, rows, n_expert]   weights
            //   src[1] (x):   [cols, n_used_or_1, n_tokens]   input
            //   src[2] (ids): [n_used, n_tokens] i32   expert routing
            //   dst (c):      [rows, n_used, n_tokens] F32
            const ggml_tensor * w   = op->src[0];
            const ggml_tensor * x   = op->src[1];
            const ggml_tensor * ids = op->src[2];
            if (!w || !x || !ids) return false;
            if (ids->type != GGML_TYPE_I32) return false;
            if (op->type  != GGML_TYPE_F32) return false;

            // Same dtype matrix as MUL_MAT for the (w, x) pair: F32×F32,
            // F16×{F16,F32}, BF16×{BF16,F16,F32}, or Q-typed weight ×
            // {F16,F32}. Output is always F32 per the ggml MUL_MAT_ID contract.
            bool ok = false;
            if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32) ok = true;
            if (w->type == GGML_TYPE_F16 &&
                (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32)) ok = true;
            if (w->type == GGML_TYPE_BF16 &&
                (x->type == GGML_TYPE_BF16 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32)) ok = true;
            if (gcu_q_supported(w->type) &&
                (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32)) ok = true;
            if (!ok) return false;

            // Shape constraints from ggml_mul_mat_id():
            //   w is 3D (n_expert in ne[2]), x is 3D (n_tokens in ne[2]),
            //   ids is 2D, ne[1] == n_tokens, ne[0] is n_expert_used,
            //   shared K = w->ne[0] == x->ne[0],
            //   ids->ne[0] % x->ne[1] == 0 (broadcast on slot dim).
            if (w->ne[3] != 1 || x->ne[3] != 1) return false;
            if (ids->ne[2] != 1 || ids->ne[3] != 1) return false;
            if (w->ne[0] != x->ne[0]) return false;
            if (ids->ne[1] != x->ne[2]) return false;
            if (x->ne[1] == 0 || ids->ne[0] % x->ne[1] != 0) return false;

            // Contiguity: gcu_op_mul_mat_id assumes packed strides for
            // pointer-arithmetic into individual expert / token / slot
            // slices. ids is also read via a single linear topsMemcpy.
            if (!ggml_is_contiguous(w))   return false;
            if (!ggml_is_contiguous(x))   return false;
            if (!ggml_is_contiguous(ids)) return false;
            if (!ggml_is_contiguous(op))  return false;
            return true;
        }
        case GGML_OP_UNARY: {
            const enum ggml_unary_op uop = ggml_get_unary_op(op);
            switch (uop) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                    break;
                default:
                    return false;
            }
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
        case GGML_OP_NORM: {
            // LayerNorm without affine. Same dtype constraints as RMS_NORM.
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_GLU: {
            // GEGLU / GEGLU_QUICK / SWIGLU / REGLU. Two-source or split.
            const ggml_tensor * a = op->src[0];
            const ggml_tensor * b = op->src[1];
            if (!a) return false;
            if (!gcu_dtype_supported(a->type)) return false;
            if (a->type != op->type) return false;
            if (b && (b->type != a->type)) return false;
            // Split form: src[0] must have ne[0] == 2 * dst->ne[0].
            if (!b && a->ne[0] != 2 * op->ne[0]) return false;
            // Two-source form: src[0] and src[1] same shape as dst.
            if (b && (a->ne[0] != op->ne[0] || b->ne[0] != op->ne[0])) return false;
            // Contiguity: dst must be fully contiguous; sources need only
            // be contiguous in dim 0 (matches ggml_glu_impl's own assert).
            // The MoE FFN path passes strided views of a fused gate_up
            // tensor whose nb[1] != ne[0]*type_size — gcu_op_glu picks
            // up the original strides from src->nb[].
            if (!ggml_is_contiguous(op)) return false;
            if (!ggml_is_contiguous_1(a)) return false;
            if (b && !ggml_is_contiguous_1(b)) return false;
            const int32_t glu_op = ((const int32_t *) op->op_params)[0];
            switch (glu_op) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_REGLU:
                    break;
                default:
                    return false;
            }
            return true;
        }
        case GGML_OP_CONCAT: {
            // Same dtype, contiguous; no other constraints — topsatenCat
            // handles dim selection.
            const ggml_tensor * a = op->src[0];
            const ggml_tensor * b = op->src[1];
            if (!a || !b) return false;
            if (!gcu_dtype_supported(a->type)) return false;
            if (a->type != b->type || a->type != op->type) return false;
            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_ARGMAX: {
            // ggml's argmax is matrix-only (rank 2) and emits I32 of
            // shape [ne1]. We accept F32/F16 input.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type)) return false;
            if (op->type != GGML_TYPE_I32) return false;
            if (!ggml_is_matrix(src)) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_TOP_K: {
            // ggml emits I32 indices [k, ne01, ne02, ne03]; input must be
            // F32 or F16, contiguous, with k <= ne00.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type)) return false;
            if (op->type != GGML_TYPE_I32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            // dst keeps src's outer shape; ne[0] is k.
            if (op->ne[0] > src->ne[0]) return false;
            if (op->ne[1] != src->ne[1] || op->ne[2] != src->ne[2] || op->ne[3] != src->ne[3]) return false;
            return true;
        }
        case GGML_OP_ARGSORT: {
            // ggml emits I32 indices same shape as input. ASC and DESC
            // accepted; topsatenArgSort takes a `descending` bool.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type)) return false;
            if (op->type != GGML_TYPE_I32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            const int32_t order = ((const int32_t *) op->op_params)[0];
            if (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC) return false;
            return true;
        }
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS: {
            // Element-wise unary, same dtype in/out, contiguous. F32 + F16
            // both accepted by these topsaten kernels in this SDK version.
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN: {
            // F32 only. CPU forward also accepts F16/BF16 for SUM but the
            // dim-reduce path (SUM_ROWS / MEAN) is F32-only on CPU; we
            // keep all three on the F32 fast-path for simplicity.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (src->type != GGML_TYPE_F32) return false;
            if (op->type  != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_CLAMP: {
            // ggml's CPU forward supports F32 and F16; we match both.
            // op_params: [min, max] F32. min == -INFINITY or max ==
            // INFINITY are valid (one-sided clamp).
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type)) return false;
            if (src->type != op->type) return false;
            // ggml_clamp returns a view of src, so dst always aliases src
            // and ggml_is_contiguous(op) tracks src's contiguity. Require
            // a contiguous src; the handler then materializes a scratch
            // copy and clamps into dst.
            if (!ggml_is_contiguous(src)) return false;
            return true;
        }
        case GGML_OP_CUMSUM: {
            // F32-only on CPU; we match.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (src->type != GGML_TYPE_F32) return false;
            if (op->type  != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_L2_NORM: {
            // CPU forward is F32-only; topsatenNormalize uses sqrt(sum+eps)
            // vs ggml's max(sqrt(sum), eps), which agrees in practice when
            // sum > eps^2 (always true for non-degenerate inputs).
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (src->type != GGML_TYPE_F32) return false;
            if (op->type  != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_GROUP_NORM: {
            // CPU forward is F32-only and has no affine; we feed weight=ones,
            // bias=zeros to topsatenGroupNorm.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (src->type != GGML_TYPE_F32) return false;
            if (op->type  != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_LEAKY_RELU: {
            // Element-wise unary, same dtype, contiguous. F32 + F16.
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type)) return false;
            if (src->type != op->type) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_REPEAT: {
            // F32 + F16. Each ne_dst[i] must be a positive integer multiple
            // of ne_src[i] (ggml_can_repeat).
            const ggml_tensor * src = op->src[0];
            if (!src) return false;
            if (!gcu_dtype_supported(src->type)) return false;
            if (src->type != op->type) return false;
            if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (src->ne[i] == 0) return false;
                if (op->ne[i] % src->ne[i] != 0) return false;
            }
            return true;
        }
        // GGML_OP_COUNT_EQUAL: I32 inputs, I64 scalar output. Rare op
        // (training paths only); not implemented — falls back to CPU.
        //
        // Tranche H — old / training-path ops left on CPU:
        //   GGML_OP_DIAG_MASK_INF / GGML_OP_DIAG_MASK_ZERO — older causal-mask
        //     paths superseded by SOFT_MAX(mask=...) in modern transformers.
        //     Could be implemented via topsatenTriu + topsatenWhere, but the
        //     models we target don't emit them.
        //   GGML_OP_DIAG — vector → diagonal matrix; only used by training paths.
        //   GGML_OP_OUT_PROD — outer product; training-only path.
        // Tranche G — window/relative attention ops (SAM-style only):
        //   GGML_OP_WIN_PART     — custom windowing memory rearrange
        //   GGML_OP_WIN_UNPART   — inverse of WIN_PART
        //   GGML_OP_GET_REL_POS  — F16 gather with non-trivial index pattern
        //   GGML_OP_ADD_REL_POS  — broadcasting bias add along multiple axes
        // None of these have a clean topsaten counterpart, and they only
        // appear in vision/SAM models that aren't a near-term GCU target.
        // The default `return false` at the bottom of this switch sends
        // them to CPU, which is correct (their CPU forwards are fast
        // enough at the small shapes these ops use).
        case GGML_OP_ROPE: {
            // MVP-3c (NORMAL+NEOX), MVP-5a (F16/BF16 cs), MVP-5b/1
            // (freq_factors), MVP-5b/2 (partial rotation tested).
            // MVP-5b/3-4: multi-axis modes (MROPE / VISION / IMROPE) are
            // wired through the handler but gated behind the env var
            // GGML_GCU_ENABLE_MROPE while we resolve the SDK position-
            // layout discrepancy (see TODO in gcu_op_rope).
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) return false;
            const int32_t mode = ((const int32_t *) op->op_params)[2];
            float ext_factor, attn_factor;
            memcpy(&ext_factor,  (const int32_t *) op->op_params + 7, sizeof(float));
            memcpy(&attn_factor, (const int32_t *) op->op_params + 8, sizeof(float));
            // NORMAL (0) and NEOX (2) always supported. MROPE (8) /
            // VISION (24) / IMROPE (40) gated behind env var; default
            // off so multi-axis ROPE keeps falling back to CPU.
            const int32_t known_mask = GGML_ROPE_TYPE_NORMAL |
                                       GGML_ROPE_TYPE_NEOX   |
                                       GGML_ROPE_TYPE_MROPE  |
                                       GGML_ROPE_TYPE_VISION |
                                       GGML_ROPE_TYPE_IMROPE;
            if ((mode & ~known_mask) != 0) return false;
            const bool is_mrope_mode = (mode & GGML_ROPE_TYPE_MROPE) != 0;
            if (is_mrope_mode) {
                static const bool enable_mrope = (getenv("GGML_GCU_ENABLE_MROPE") != nullptr);
                if (!enable_mrope) return false;
            }
            // YARN's full theta-mixing path stays on CPU; freq_factors-only
            // (Gemma 4 / GPT-OSS proportional rope) is supported via the
            // host-side cos/sin builder.
            if (ext_factor != 0.0f) return false;
            if (attn_factor != 0.0f && attn_factor != 1.0f) return false;
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            if (op->src[0]->ne[3] != 1) return false;
            // freq_factors must be a contiguous F32 vector if present.
            if (op->src[2]) {
                if (op->src[2]->type != GGML_TYPE_F32) return false;
                if (!ggml_is_contiguous(op->src[2])) return false;
            }
            return true;
        }
        case GGML_OP_SOFT_MAX: {
            float max_bias;
            memcpy(&max_bias, (const float *)op->op_params + 1, sizeof(float));
            if (max_bias != 0.0f) return false;
            if (op->src[2] != nullptr) return false;   // softmax sinks: MVP-3
            if (!gcu_dtype_supported(op->src[0]->type)) return false;
            if (op->src[0]->type != op->type) return false;
            // mask dtype is constrained by ggml itself to F16 or F32 (see
            // ggml_soft_max_ext_impl); accept either here, the handler
            // casts mask -> dst dtype on the fly when they differ.
            if (op->src[1]) {
                const ggml_type mt = op->src[1]->type;
                if (mt != GGML_TYPE_F16 && mt != GGML_TYPE_F32) return false;
            }
            if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
            return true;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            // Tranche B: fused QKV softmax matmul via topsatenScaledDotProductAttention.
            //
            // Contract from ggml_flash_attn_ext (ggml.c):
            //   src[0] q     [head_dim, n_q,  n_head,    n_batch]   F32 or F16
            //   src[1] k     [head_dim, n_kv, n_head_kv, n_batch]   F16 (we gate to F16)
            //   src[2] v     [head_dim, n_kv, n_head_kv, n_batch]   F16 (we gate to F16)
            //   src[3] mask                                          F16 (or null)
            //   src[4] sinks                                         not supported
            //   op_params: f32 scale, f32 max_bias, f32 logit_softcap, i32 prec
            const ggml_tensor * q    = op->src[0];
            const ggml_tensor * k    = op->src[1];
            const ggml_tensor * v    = op->src[2];
            const ggml_tensor * mask = op->src[3];
            const ggml_tensor * sinks = op->src[4];
            if (!q || !k || !v) return false;
            if (sinks) return false;                          // attention sinks: not supported
            // dst dtype: F32 is the standard llama.cpp output; F16 also accepted.
            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) return false;
            // Q dtype: F32 (post-RoPE) or F16 (rare).
            if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) return false;
            // K and V: F16 only — that is what llama-graph.cpp emits after the
            // pre-FA cast (lines 1969-1975) for both kv-cache F16 and -nkvo F32 paths.
            if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) return false;
            if (mask) {
                if (mask->type != GGML_TYPE_F16) return false;
                if (!ggml_is_contiguous(mask)) return false;  // ggml asserts this
            }
            // ALiBi (max_bias != 0): not supported by topsatenSDP. CPU fallback.
            // logit_softcap != 0: Gemma-style softcap; not supported.
            float max_bias, logit_softcap;
            memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(float));
            memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(float));
            if (max_bias      != 0.0f) return false;
            if (logit_softcap != 0.0f) return false;
            // Shape constraints.
            if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0]) return false;
            if (k->ne[1] != v->ne[1]) return false;
            if (k->ne[2] != v->ne[2]) return false;
            if (k->ne[3] != q->ne[3] || v->ne[3] != q->ne[3]) return false;
            if (q->ne[2] % k->ne[2] != 0) return false;       // GQA broadcast
            // The handler materializes Q/K/V into PyTorch-contiguous F16
            // scratches via topsatenTo; only the innermost dim (head_dim)
            // must be unit-stride. dst is created by ggml as a fresh
            // tensor and is fully contiguous. Q/K/V routinely arrive as
            // ggml_permute(0, 2, 1, 3) views, which have non-monotone
            // outer strides; that is fine because make_topsaten_tensor
            // honors the raw nb[]. ggml_is_contiguous_1 would reject those
            // views (it checks dims >= 2 are packed), so use the weaker
            // "innermost dim is unit-stride" gate the handler actually needs.
            if (q->nb[0] != ggml_type_size(q->type)) return false;
            if (k->nb[0] != ggml_type_size(k->type)) return false;
            if (v->nb[0] != ggml_type_size(v->type)) return false;
            if (!ggml_is_contiguous(op))             return false;
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
