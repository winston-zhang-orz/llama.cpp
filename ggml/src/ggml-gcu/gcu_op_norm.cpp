// ggml-gcu: norm op handlers — RMS_NORM, NORM, L2_NORM, GROUP_NORM.
//
// Per the file layout in
// docs/superpowers/specs/2026-05-10-gcu-mvp5c-file-split-design.md.
// All four norms route through topsvllm or topsaten LayerNorm/RmsNorm
// kernels; the gamma/beta scratch buffers (gcu_get_ones_f32 /
// gcu_get_zero_bias) live in common.{h,cpp} since matmul shares them.

#include "common.h"
#include "gcu_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

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

