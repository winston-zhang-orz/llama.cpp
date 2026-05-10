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
