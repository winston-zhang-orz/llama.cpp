// ggml-gcu: op-handler declarations + shared op infrastructure.
//
// One bool gcu_op_*(ctx, dst) per ggml op the backend implements. Each
// handler enqueues topsaten/topsvllm kernels on ctx->compute_stream and
// returns true on success (false routes the op back to CPU). The handlers
// are grouped by family across gcu_op_*.cpp; gcu_compute_node dispatches.

#pragma once

#include "common.h"
#include "gcu_pool.h"

// === Shared op infrastructure =======================================

// MVP-4a/4b rollback switches (env-gated, latched once).
bool gcu_async_disabled();
bool gcu_queued_ops_disabled();

// MVP-4b: replaces every `topsStreamSynchronize + pool.free` pair inside op
// handlers. Defers the free to graph_compute's end-of-batch drain unless
// GGML_GCU_NO_QUEUED_OPS=1, in which case we restore the pre-MVP-4b
// behavior (synchronize then immediate free).
void gcu_release_scratch(ggml_backend_gcu_context * ctx, void * p, size_t sz);

// Per-context scratch helpers (zero bias for topsatenLinear, ones gamma
// for topsvllmRmsNorm / unit affine).
void * gcu_get_zero_bias(ggml_backend_gcu_context * ctx, size_t n_bytes);
void * gcu_get_ones_f32 (ggml_backend_gcu_context * ctx, int64_t count);

// topsaten's binary ops reject aliased output and lhs. Detect at compute
// time via data-pointer comparison and route through a scratch copy.
bool gcu_dst_aliases_src0_at_runtime(const ggml_tensor * dst);

// === Op handlers ====================================================

// Elementwise / activations / GLU / clamp (gcu_op_elementwise.cpp).
bool gcu_op_add        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_add1       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sub        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_div        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_mul        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_scale      (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_leaky_relu (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_glu        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_silu       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_gelu       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_gelu_quick (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_relu       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_tanh       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sigmoid    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_hardswish  (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_hardsigmoid(ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_clamp      (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Normalization (gcu_op_norm.cpp).
bool gcu_op_rms_norm   (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_norm       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_l2_norm    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_group_norm (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Matmul (gcu_op_matmul.cpp).
bool gcu_op_mul_mat    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_mul_mat_id (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Attention (gcu_op_attn.cpp).
bool gcu_op_soft_max       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_rope           (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_flash_attn_ext (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Misc: copy/rows/concat/reduce/sort/unary-math (gcu_op_misc.cpp).
bool gcu_op_cpy        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_set_rows   (ggml_backend_gcu_context * ctx, ggml_tensor * node);
bool gcu_op_get_rows   (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_concat     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_repeat     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sqr        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sqrt       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_log        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sin        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_cos        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sum        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sum_rows   (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_mean       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_cumsum     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_argmax     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_top_k      (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_argsort    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
