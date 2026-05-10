// ggml-gcu: element-wise + unary + GLU op handlers.
//
// Per the file layout in
// docs/superpowers/specs/2026-05-10-gcu-mvp5c-file-split-design.md,
// this TU hosts:
//   * binary element-wise: ADD, ADD1, SUB, MUL, DIV, SCALE
//   * unary activations:   SILU, GELU, GELU_QUICK, RELU, TANH, SIGMOID,
//                          HARDSWISH, HARDSIGMOID, LEAKY_RELU
//   * unary math:          SQR, SQRT, LOG, SIN, COS
//   * GLU family:          REGLU, GEGLU, SWIGLU, GEGLU_QUICK
//   * misc:                CLAMP

#include "common.h"
#include "gcu_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

bool gcu_op_add(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
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
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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
        TOPSATEN_CHECK(topsatenAdd(out, lhs, rhs, alpha, ctx->compute_stream));
    }

    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

// Tranche D3: ADD1 (a + scalar). src[1] is a 1-element tensor that we
// broadcast against src[0] via the standard topsatenAdd binary path —
// keeps the scalar on device and avoids a host sync.
bool gcu_op_add1(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * lhs_t = dst->src[0];
    ggml_tensor * rhs_t = dst->src[1];

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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

        // rhs is a 1-element tensor (ggml_is_scalar). Build a rank-1 size-1
        // descriptor; topsatenAdd's broadcast handles the rest.
        const size_t r_bpe = ggml_type_size(rhs_t->type);
        int64_t r_dims[1] = { 1 };
        int64_t r_strs[1] = { (int64_t) (rhs_t->nb[0] / r_bpe) };
        topsatenTensor rhs(topsatenSize_t(r_dims, 1), topsatenSize_t(r_strs, 1),
                           ggml_to_topsaten_dtype(rhs_t->type), rhs_t->data);

        topsatenScalar_t alpha;
        alpha.dtype = TOPSATEN_DATA_FP32;
        alpha.fval  = 1.0;
        TOPSATEN_CHECK(topsatenAdd(out, lhs, rhs, alpha, ctx->compute_stream));
    }
    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

// Tranche D2: element-wise binary SUB / DIV. Same template as ADD/MUL —
// honours numpy-style broadcasting and runtime-aliased dst handling.
bool gcu_op_sub(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * lhs_t = dst->src[0];
    ggml_tensor * rhs_t = dst->src[1];

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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

        topsatenScalar_t alpha;
        alpha.dtype = TOPSATEN_DATA_FP32;
        alpha.fval  = 1.0;
        TOPSATEN_CHECK(topsatenSub(out, lhs, rhs, alpha, ctx->compute_stream));
    }
    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

bool gcu_op_div(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * lhs_t = dst->src[0];
    ggml_tensor * rhs_t = dst->src[1];

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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

        // Default rounding mode (PyTorch "true" division == element-wise /).
        TOPSATEN_CHECK(topsatenDiv(out, lhs, rhs, ctx->compute_stream));
    }
    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

bool gcu_op_mul(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * lhs_t = dst->src[0];
    ggml_tensor * rhs_t = dst->src[1];

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * lhs_data = lhs_t->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(lhs_t);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, lhs_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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

        TOPSATEN_CHECK(topsatenMul(out, lhs, rhs, ctx->compute_stream));
    }

    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

bool gcu_op_scale(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
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
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
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
        TOPSATEN_CHECK(topsatenMul(out, lhs, s, ctx->compute_stream));
    }

    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

// Tranche F: LEAKY_RELU. y = x >= 0 ? x : negative_slope * x.
// Note: ggml represents this as the top-level GGML_OP_LEAKY_RELU (not as a
// GGML_UNARY_OP_*), with the negative_slope packed in op_params[0].
bool gcu_op_leaky_relu(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);

    topsatenScalar_t slope_s; slope_s.dtype = TOPSATEN_DATA_FP32; slope_s.fval = negative_slope;
    TOPSATEN_CHECK(topsatenLeakyRelu(out, in, slope_s, ctx->compute_stream));
    return true;
}

// GLU (gated linear unit). dst = activation(gate) * up, element-wise.
//
//   GEGLU:  activation = GELU
//   SWIGLU: activation = SILU
//   REGLU:  activation = RELU
//
// Two source forms (per ggml_glu_impl):
//   - Two-source: src[0]=a, src[1]=b, same shape, dst shape = a shape.
//                 gate=a, up=b (or swapped via op_params[1]).
//   - Split:      src[1]=null, src[0] has 2*n in dim 0, dst has n in dim 0.
//                 Halves of src[0] are gate (offset 0) and up (offset n).
//                 swapped flips which half is gate.
//
// Implementation: one activation kernel into a scratch the size of dst,
// then one element-wise topsatenMul into dst. The split form passes
// non-contiguous half-views (stride 2*n in dim-0 of the original tensor)
// to the activation kernel; topsaten handles the strided source.
bool gcu_op_glu(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    const int32_t glu_op = ((const int32_t *) dst->op_params)[0];
    const bool    swapped = ((const int32_t *) dst->op_params)[1] != 0;

    const ggml_type t = src0->type;
    const size_t    bpe = ggml_type_size(t);

    // Scratch holds the activation output, same shape as dst.
    const size_t scratch_bytes = ggml_nbytes(dst);
    void * scratch = ctx->pool.alloc(scratch_bytes);

    // Build outer-first dims for dst (same shape as gate/up after split).
    int rank = ggml_n_dims(dst);
    if (rank < 1) rank = 1;
    int64_t out_dims[GGML_MAX_DIMS];
    int64_t out_strs_contig[GGML_MAX_DIMS];
    for (int i = 0; i < rank; i++) {
        out_dims[i] = dst->ne[rank - 1 - i];
    }
    out_strs_contig[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        out_strs_contig[i] = out_strs_contig[i + 1] * out_dims[i + 1];
    }

    // Source view builders. For the two-source form, gate/up are full
    // tensors; for split, we build views with the same shape as dst but
    // the original tensor's strides (so dim-0 step skips the other half).
    void * gate_data = nullptr;
    void * up_data   = nullptr;
    int64_t src_dims[GGML_MAX_DIMS];
    int64_t src_strs[GGML_MAX_DIMS];

    if (src1 != nullptr) {
        // Two-source: same shape as dst, normal strides.
        ggml_tensor * gate_t = swapped ? src1 : src0;
        ggml_tensor * up_t   = swapped ? src0 : src1;
        for (int i = 0; i < rank; i++) {
            src_dims[i] = gate_t->ne[rank - 1 - i];
            src_strs[i] = (int64_t) (gate_t->nb[rank - 1 - i] / bpe);
        }
        gate_data = gate_t->data;
        up_data   = up_t->data;
    } else {
        // Split form. Halves are non-contiguous views of src0.
        const int64_t n_half = src0->ne[0] / 2;
        const size_t  gate_off = swapped ? (size_t) n_half * bpe : 0;
        const size_t  up_off   = swapped ? 0 : (size_t) n_half * bpe;
        for (int i = 0; i < rank; i++) {
            const int gd = rank - 1 - i;
            src_dims[i] = (gd == 0) ? n_half : src0->ne[gd];
            src_strs[i] = (int64_t) (src0->nb[gd] / bpe);
        }
        gate_data = (char *) src0->data + gate_off;
        up_data   = (char *) src0->data + up_off;
    }

    topsatenDataType_t dtype = ggml_to_topsaten_dtype(t);
    topsatenTensor gate(topsatenSize_t(src_dims, rank), topsatenSize_t(src_strs, rank),
                        dtype, gate_data);
    topsatenTensor up  (topsatenSize_t(src_dims, rank), topsatenSize_t(src_strs, rank),
                        dtype, up_data);
    // scratch and dst are contiguous with dst shape.
    topsatenTensor scratch_t(topsatenSize_t(out_dims, rank), topsatenSize_t(out_strs_contig, rank),
                             dtype, scratch);
    int64_t dst_strs[GGML_MAX_DIMS];
    for (int i = 0; i < rank; i++) {
        dst_strs[i] = (int64_t) (dst->nb[rank - 1 - i] / bpe);
    }
    topsatenTensor dst_t(topsatenSize_t(out_dims, rank), topsatenSize_t(dst_strs, rank),
                         dtype, dst->data);

    switch (glu_op) {
        case GGML_GLU_OP_GEGLU:
            TOPSATEN_CHECK(topsatenGelu(scratch_t, gate, "none", ctx->compute_stream));
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            TOPSATEN_CHECK(topsatenGelu(scratch_t, gate, "tanh", ctx->compute_stream));
            break;
        case GGML_GLU_OP_SWIGLU:
            TOPSATEN_CHECK(topsatenSilu(scratch_t, gate, ctx->compute_stream));
            break;
        case GGML_GLU_OP_REGLU:
            TOPSATEN_CHECK(topsatenRelu(scratch_t, gate, ctx->compute_stream));
            break;
        default:
            // Unknown / unsupported GLU op type — caller should have
            // gated supports_op accordingly.
            gcu_release_scratch(ctx, scratch, scratch_bytes);
            return false;
    }

    TOPSATEN_CHECK(topsatenMul(dst_t, scratch_t, up, ctx->compute_stream));
    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

// SILU. y = x * sigmoid(x). Same dtype in/out, contiguous.
bool gcu_op_silu(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, dlhs;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, dlhs);
    TOPSATEN_CHECK(topsatenSilu(out, in, ctx->compute_stream));
    return true;
}

// MVP-5a: additional unary activations. All follow the SILU template —
// same-dtype contiguous in/out, single topsaten op call. The dispatch
// (gcu_compute_node) and the GGML_OP_UNARY supports_op gate carry the
// per-op routing.

bool gcu_op_gelu(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    // approximate="none" matches ggml's GELU_ERF formula and the standard
    // Gemma / Phi / BERT GELU.
    TOPSATEN_CHECK(topsatenGelu(out, in, "none", ctx->compute_stream));
    return true;
}

bool gcu_op_gelu_quick(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    // approximate="tanh" matches ggml's tanh-form GELU approximation
    // (Gemma 3, GPT-2 fast path).
    TOPSATEN_CHECK(topsatenGelu(out, in, "tanh", ctx->compute_stream));
    return true;
}

bool gcu_op_relu(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenRelu(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_tanh(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenTanh(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_sigmoid(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenSigmoid(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_hardswish(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenHardswish(out, in, ctx->compute_stream));
    return true;
}

bool gcu_op_hardsigmoid(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    gcu_tensor_dims dout, din;
    topsatenTensor out = make_topsaten_tensor(dst, dout);
    topsatenTensor in  = make_topsaten_tensor(src, din);
    TOPSATEN_CHECK(topsatenHardsigmoid(out, in, ctx->compute_stream));
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

// Tranche D5: CLAMP. ggml_clamp returns a view of `a`, so dst always
// aliases src — copy src to a scratch buffer and clamp into dst (==a).
// op_params holds [min, max] as F32. Used by Gemma 4 MoE weight
// normalization to clamp weights_sum away from zero before DIV.
bool gcu_op_clamp(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == src->type);

    float clamp_min, clamp_max;
    memcpy(&clamp_min, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&clamp_max, (const float *) dst->op_params + 1, sizeof(float));

    void * scratch = nullptr;
    size_t scratch_bytes = 0;
    void * in_data = src->data;
    if (gcu_dst_aliases_src0_at_runtime(dst)) {
        scratch_bytes = ggml_nbytes(src);
        scratch       = ctx->pool.alloc(scratch_bytes);
        TOPS_CHECK(topsMemcpyAsync(scratch, in_data, scratch_bytes,
                                   topsMemcpyDeviceToDevice, ctx->compute_stream));
        in_data = scratch;
    }

    gcu_tensor_dims dout, din;
    topsatenTensor out_t = make_topsaten_tensor(dst, dout);
    {
        const size_t bpe = ggml_type_size(src->type);
        int rank = ggml_n_dims(src); if (rank < 1) rank = 1;
        for (int i = 0; i < rank; i++) {
            din.dims[i] = src->ne[rank - 1 - i];
            din.strs[i] = src->nb[rank - 1 - i] / (int64_t) bpe;
        }
        topsatenSize_t shape (din.dims, rank);
        topsatenSize_t stride(din.strs, rank);
        topsatenTensor in_t(shape, stride, ggml_to_topsaten_dtype(src->type), in_data);

        topsatenScalar_t lo, hi;
        lo.dtype = TOPSATEN_DATA_FP32; lo.fval = clamp_min;
        hi.dtype = TOPSATEN_DATA_FP32; hi.fval = clamp_max;
        TOPSATEN_CHECK(topsatenClamp(out_t, in_t, lo, hi, ctx->compute_stream));
    }
    gcu_release_scratch(ctx, scratch, scratch_bytes);
    return true;
}

