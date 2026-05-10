// ggml-gcu: op handler declarations.
//
// Every gcu_op_<name>(ctx, dst) handler that implements a single ggml op
// is declared here. Implementations are split across:
//
//   gcu_op_elementwise.cpp  ADD, ADD1, SUB, MUL, DIV, SCALE, CLAMP,
//                           GLU, LEAKY_RELU, REPEAT, all unary activations
//                           (SILU/GELU/GELU_QUICK/RELU/TANH/SIGMOID/
//                           HARDSWISH/HARDSIGMOID), and the simple unary
//                           SQR/SQRT/LOG/SIN/COS.
//   gcu_op_norm.cpp         NORM, RMS_NORM, L2_NORM, GROUP_NORM.
//   gcu_op_matmul.cpp       MUL_MAT, MUL_MAT_ID.
//   gcu_op_attn.cpp         SOFT_MAX, ROPE, FLASH_ATTN_EXT.
//   gcu_op_misc.cpp         CPY/DUP/CONT, GET_ROWS, SET_ROWS, CONCAT,
//                           SUM, SUM_ROWS, MEAN, CUMSUM, ARGMAX, TOP_K,
//                           ARGSORT.

#pragma once

#include "common.h"

// Element-wise + unary + GLU + scale + clamp + repeat + leaky_relu
bool gcu_op_add        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_add1       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sub        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_mul        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_div        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_scale      (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_clamp      (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_glu        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_repeat     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_leaky_relu (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_silu       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_gelu       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_gelu_quick (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_relu       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_tanh       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sigmoid    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_hardswish  (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_hardsigmoid(ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sqr        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sqrt       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_log        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sin        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_cos        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Norms
bool gcu_op_norm       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_rms_norm   (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_l2_norm    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_group_norm (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Matmul
bool gcu_op_mul_mat    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_mul_mat_id (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Attention
bool gcu_op_soft_max       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_rope           (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_flash_attn_ext (ggml_backend_gcu_context * ctx, ggml_tensor * dst);

// Misc
bool gcu_op_cpy        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_get_rows   (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_set_rows   (ggml_backend_gcu_context * ctx, ggml_tensor * node);
bool gcu_op_concat     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sum        (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_sum_rows   (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_mean       (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_cumsum     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_argmax     (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_top_k      (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
bool gcu_op_argsort    (ggml_backend_gcu_context * ctx, ggml_tensor * dst);
