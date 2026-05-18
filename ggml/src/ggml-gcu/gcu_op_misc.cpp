// ggml-gcu: misc ops — CPY/DUP/CONT, GET_ROWS, SET_ROWS, CONCAT, REPEAT,
// SUM/SUM_ROWS/MEAN, CUMSUM, ARGMAX, TOP_K, ARGSORT, and the unary math
// ops (SQR/SQRT/LOG/SIN/COS). See gcu_ops.h.

#include "gcu_ops.h"

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
// MVP-6 strategy: use topsatenIndexPut as a fully-on-device scatter. Two
// SDK constraints we re-probed in 2026-05:
//   1. topsatenIndexPut REJECTS I64 indices ("indice data type not support: 11")
//      but accepts I32. ggml KV caches use I64 indices. We cast I64 -> I32
//      with topsatenTo on the compute stream (queued, async).
//   2. topsatenIndexPut REQUIRES self and value to share dtype. ggml writes
//      F32 K/V vectors into the F16 cache, so we cast value to dst's dtype
//      via topsatenTo before the scatter (also queued).
// At the cache shapes we measured (F16 [4096..32768, 128] with R in [1..64])
// the op completes successfully and is fully overlapped with the rest of
// the compute stream — no per-call sync, no host roundtrip. This is the
// fast-path replacement for the MVP-3b probe (manual D2D memcpy loop)
// which was 2-5x slower than -nkvo because of the synchronous H2D index
// readback that drained the layer pipeline.
//
// Earlier IndexPut probe (MVP-3) failed only because it passed I64 indices
// directly; the F16 dst + tall self combination is supported once indices
// are I32.
bool gcu_op_set_rows(ggml_backend_gcu_context * ctx, ggml_tensor * node) {
    ggml_tensor * src = node->src[0];
    ggml_tensor * idx = node->src[1];
    ggml_tensor * dst = node->src[2];

    const int64_t n_rows  = idx->ne[0];
    const int64_t row_len = dst->ne[0];

    // --- 1. Value tensor: cast src -> dst dtype if needed. ----------------
    // topsatenIndexPut requires self.dtype == value.dtype. ggml's KV write
    // is typically F32 src into F16 dst.
    void * cast_buf = nullptr;
    size_t cast_bytes = 0;
    void * value_data = src->data;
    if (src->type != dst->type) {
        const int64_t n_elem = n_rows * row_len;
        cast_bytes = (size_t) n_elem * ggml_type_size(dst->type);
        cast_buf   = ctx->pool.alloc(cast_bytes);

        int64_t v_d[2] = { n_rows, row_len };
        int64_t v_s[2] = { row_len, 1 };
        topsatenTensor src_view (topsatenSize_t(v_d, 2), topsatenSize_t(v_s, 2),
                                 ggml_to_topsaten_dtype(src->type), src->data);
        topsatenTensor cast_view(topsatenSize_t(v_d, 2), topsatenSize_t(v_s, 2),
                                 ggml_to_topsaten_dtype(dst->type), cast_buf);
        topsatenDataType_t target = ggml_to_topsaten_dtype(dst->type);
        TOPSATEN_CHECK(topsatenTo(cast_view, src_view, target,
                                  /*non_blocking=*/false, /*copy=*/true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        value_data = cast_buf;
    }

    // --- 2. Index tensor: cast I64 -> I32 if needed (queued on compute). --
    // ggml emits I64 indices; topsatenIndexPut requires I32. The cast
    // produces n_rows*4 bytes and runs entirely on the compute stream, so
    // the index buffer is ready exactly when IndexPut consumes it.
    void * idx_buf       = idx->data;
    void * idx_cast_buf  = nullptr;
    size_t idx_cast_bytes = 0;
    topsatenDataType_t idx_dtype = TOPSATEN_DATA_I32;
    if (idx->type == GGML_TYPE_I64) {
        idx_cast_bytes = (size_t) n_rows * sizeof(int32_t);
        idx_cast_buf   = ctx->pool.alloc(idx_cast_bytes);

        int64_t id[1] = { n_rows };
        int64_t is[1] = { 1 };
        topsatenTensor src_view (topsatenSize_t(id, 1), topsatenSize_t(is, 1),
                                 TOPSATEN_DATA_I64, idx->data);
        topsatenTensor cast_view(topsatenSize_t(id, 1), topsatenSize_t(is, 1),
                                 TOPSATEN_DATA_I32, idx_cast_buf);
        topsatenDataType_t target = TOPSATEN_DATA_I32;
        TOPSATEN_CHECK(topsatenTo(cast_view, src_view, target,
                                  /*non_blocking=*/false, /*copy=*/true,
                                  TOPSATEN_MEMORY_CONTIGUOUS, ctx->compute_stream));
        idx_buf = idx_cast_buf;
    } else {
        // ggml emits I32 too in some paths; pass through.
        GGML_ASSERT(idx->type == GGML_TYPE_I32);
    }

    // --- 3. Build topsatenTensor descriptors and call IndexPut. -----------
    // Flatten dst to [n_dst_rows, row_len]; src/value to [n_rows, row_len].
    const int64_t n_dst_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    int64_t sd[2] = { n_dst_rows, row_len };
    int64_t ss[2] = { row_len, 1 };
    int64_t vd[2] = { n_rows, row_len };
    int64_t vs[2] = { row_len, 1 };
    int64_t id[1] = { n_rows };
    int64_t is[1] = { 1 };

    topsatenTensor self_t(topsatenSize_t(sd, 2), topsatenSize_t(ss, 2),
                          ggml_to_topsaten_dtype(dst->type), dst->data);
    topsatenTensor val_t (topsatenSize_t(vd, 2), topsatenSize_t(vs, 2),
                          ggml_to_topsaten_dtype(dst->type), value_data);
    topsatenTensor idx_t (topsatenSize_t(id, 1), topsatenSize_t(is, 1),
                          idx_dtype, idx_buf);

    std::vector<topsatenTensor> indices = { idx_t };
    TOPSATEN_CHECK(topsatenIndexPut(self_t, indices, val_t,
                                    /*accumulate=*/false, ctx->compute_stream));

    gcu_release_scratch(ctx, cast_buf,     cast_bytes);
    gcu_release_scratch(ctx, idx_cast_buf, idx_cast_bytes);
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

// MVP-6 / Tranche D1: element-wise unary math ops.
// Same shape as SILU template — same dtype in/out, single-source contiguous.
// Each is a thin wrapper around the corresponding topsaten kernel.
//
// SDK note: ggml's GGML_OP_SQR maps to topsatenSquare (PyTorch torch.square).

bool gcu_op_sqr(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenSquare(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_sqrt(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenSqrt(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_log(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenLog(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_sin(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenSin(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_cos(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenCos(out, in, ctx->compute_stream));
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
