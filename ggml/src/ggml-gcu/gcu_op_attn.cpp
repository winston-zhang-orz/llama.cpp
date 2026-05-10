// ggml-gcu: attention op handlers — SOFT_MAX, ROPE, FLASH_ATTN_EXT.
//
// Per the file layout in
// docs/superpowers/specs/2026-05-10-gcu-mvp5c-file-split-design.md.
// ROPE supports NORMAL/NEOX, freq_factors, partial rotation, and (gated
// behind GGML_GCU_ENABLE_MROPE) the multi-axis MROPE/VISION/IMROPE
// scaffolding from MVP-5b/3-4. SOFT_MAX includes the broadcast mask cast
// path. FLASH_ATTN_EXT carries the F16 Q/K/V conversion scratch.

#include "common.h"
#include "gcu_ops.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

