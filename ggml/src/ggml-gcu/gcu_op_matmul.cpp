// ggml-gcu: matmul ops — MUL_MAT, MUL_MAT_ID. See gcu_ops.h.

#include "gcu_ops.h"

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
