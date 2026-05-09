# GCU Op Coverage Roadmap

**Date:** 2026-05-09
**Branch:** `feat/ggml-gcu`
**Status:** roadmap — feeds individual MVP specs/plans

## Purpose

ggml's op enum has **95 ops** (excluding `GGML_OP_NONE`/`GGML_OP_COUNT`). The GCU backend currently dispatches **19** of them (after `MUL_MAT_ID` landed). This document categorises the remaining ~76 ops into priority tranches so we can pick the next MVP without re-deriving the analysis each time.

## What "100% support" means

"100%" depends on the workload:

- **100% of inference for current GGUF LLMs:** ~40 ops needed (current 19 + ~20–25 more). The rest are conv/vision (multimodal ports), training-only (`*_BACK`, `OPT_STEP_*`), or rare custom hooks.
- **100% of every ggml op:** not a coherent goal. `CUSTOM` and `MAP_CUSTOM*` are user-defined hooks. Conv ops require a different SDK kernel surface than topsaten. Training back-passes need autodiff plumbing we don't have.

The roadmap below targets the first definition.

## What we have (19 ops, MVP-1 through MVP-4b)

`ADD`, `MUL`, `SCALE`, `MUL_MAT`, **`MUL_MAT_ID`**, `SOFT_MAX`, `RMS_NORM`, `ROPE` (mode 0), `SILU`, `GET_ROWS`, `SET_ROWS`, `CPY`, `DUP`, `CONT`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `NONE`.

## Tranches (priority-ordered)

### Tranche A — Common LLM ops (~10 ops, MVP-5a)

Highest impact; unlocks Gemma family, Phi family, GPT-2 family, and most modern LLM variants beyond LLaMA-style.

| Op | Notes |
|---|---|
| `NORM` | LayerNorm (used by Gemma, GPT-2, BERT-derived). topsaten has `topsatenLayerNorm` or equivalent — verify before commit. |
| `UNARY:GELU` | Standard GELU; topsaten `topsatenGelu`. Used by almost every modern transformer. |
| `UNARY:GELU_QUICK` | Approximate GELU (sigmoid form). Same kernel surface; flag-only switch. |
| `UNARY:RELU` | Trivial; topsaten `topsatenRelu`. Legacy and some FFN variants. |
| `UNARY:TANH` | topsaten `topsatenTanh`. Some attention gates. |
| `UNARY:SIGMOID` | topsaten `topsatenSigmoid`. GLU gates. |
| `UNARY:HARDSWISH`, `UNARY:HARDSIGMOID` | Mobile / Phi-3.5 family. |
| `GLU` | Gated activation (FFN_MULT). Most modern FFNs use it. Source ops: SILU+MUL or GEGLU+MUL — check ggml's exact GLU contract before implementing. |
| `CONCAT` | KV cache assembly in some models, also used in feature concatenation. |
| `ROPE` mode 1 (`NEOX`) | NeoX-style rope (rotate first/last halves). Used by Phi, GPT-NeoX, some variants. |

**Estimated effort:** 1-2 days. Most of these are direct topsaten op wraps.

### Tranche B — Fused attention (1 op, MVP-5b)

| Op | Notes |
|---|---|
| `FLASH_ATTN_EXT` | Fused QKV softmax matmul. Major perf op for any model that emits it (most modern). topsaten likely has a fused attention op (`topsatenFlashAttention` / `topsatenSDP`); needs investigation against the SDK. If no fused op, can decompose into existing matmul + softmax + matmul (slower but correct). |

**Estimated effort:** 1-3 days depending on whether topsaten fused op exists.

### Tranche C — Sampling helpers (~3 ops, MVP-5c)

Often run on CPU anyway since they're cheap relative to compute, but moving them to GCU avoids host round-trips.

| Op | Notes |
|---|---|
| `ARGMAX` | Greedy sampling. topsaten `topsatenArgmax`. |
| `TOP_K` | Top-k sampling. topsaten `topsatenTopk`. |
| `ARGSORT` | Sorting indices for top-p. topsaten `topsatenArgsort`. |

**Estimated effort:** half a day. All trivial wraps.

### Tranche D — Element-wise + reductions (~10 ops, MVP-5d)

Used by various model architectures; mostly mechanical wraps.

| Op | Notes |
|---|---|
| `SUB`, `DIV` | Element-wise; topsaten direct. |
| `SQR`, `SQRT`, `LOG` | Element-wise unary; topsaten direct. |
| `SIN`, `COS` | Used by some position encoders. |
| `SUM`, `SUM_ROWS`, `MEAN` | Reductions; topsaten direct. |
| `CUMSUM` | Sometimes used in attention masking. |
| `ADD1`, `ADD_ID` | Variants of ADD; trivial wraps. |
| `CLAMP` | Activation clipping. |
| `COUNT_EQUAL` | Used in some training paths; low priority for inference. |

**Estimated effort:** 1-2 days for the inference-relevant subset.

### Tranche E — Alternative architectures (~6 ops, MVP-5e)

Each pulls in a distinct model family. Implement only when a target model needs it.

| Op | Notes |
|---|---|
| `SSM_CONV`, `SSM_SCAN` | Mamba state-space models. Custom kernel surface; substantial. |
| `RWKV_WKV6`, `RWKV_WKV7` | RWKV models. Custom kernel surface. |
| `GATED_LINEAR_ATTN`, `GATED_DELTA_NET` | Newer linear-attention variants. Niche. |

**Estimated effort:** 1-2 weeks total. Each family is its own investigation.

### Tranche F — Norm variants (~3 ops, MVP-5f)

| Op | Notes |
|---|---|
| `L2_NORM` | Used by some models for projections. |
| `GROUP_NORM` | Vision/diffusion mostly; some LLM variants. |
| `LEAKY_RELU` | UNARY-family extension. |

**Estimated effort:** half a day.

### Tranche G — Window/relative attention (~4 ops, MVP-5g)

| Op | Notes |
|---|---|
| `WIN_PART`, `WIN_UNPART` | Windowed attention partitioning. |
| `GET_REL_POS`, `ADD_REL_POS` | Relative position bias. |

**Estimated effort:** 1 day. Mostly memory-layout shuffles.

### Tranche H — Edge cases / older attention (~5 ops, optional)

| Op | Notes |
|---|---|
| `DIAG_MASK_INF`, `DIAG_MASK_ZERO`, `DIAG` | Older causal-mask impls. Modern models bake this into FLASH_ATTN_EXT or SOFT_MAX masks. |
| `REPEAT` | Sometimes used for KV head replication (when n_kv_head < n_head). |
| `OUT_PROD` | Outer product. Rare. |

**Estimated effort:** half a day for the ones we hit in practice.

## Out of scope for this backend

These ops will continue to fall back to CPU. None of them are needed for text-only LLM inference on the GGUF models we care about.

### Vision / diffusion (~14 ops)

`CONV_2D`, `CONV_3D`, `CONV_2D_DW`, `CONV_TRANSPOSE_1D`, `CONV_TRANSPOSE_2D`, `IM2COL`, `IM2COL_3D`, `POOL_1D`, `POOL_2D`, `UPSCALE`, `PAD`, `PAD_REFLECT_1D`, `ROLL`, `ARANGE`, `TIMESTEP_EMBEDDING`.

These need a topsaten convolution kernel surface that isn't part of the LLM-focused subset we've used so far. Possible future MVP if multimodal LLMs (e.g. Gemma 3 vision) become a target.

### Training back-passes (~13 ops)

`*_BACK` variants (`SILU_BACK`, `RMS_NORM_BACK`, `SOFT_MAX_BACK`, `ROPE_BACK`, `GET_ROWS_BACK`, `IM2COL_BACK`, `POOL_2D_BACK`, `FLASH_ATTN_BACK`, `REPEAT_BACK`), `OPT_STEP_ADAMW`, `OPT_STEP_SGD`, `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_BACK`.

llama.cpp inference doesn't issue these. They'd be needed only for on-device fine-tuning, which isn't a current target.

### Custom / utility (~6 ops)

`CUSTOM`, `MAP_CUSTOM1/2/3`, `TRI`, `FILL`, `SOLVE_TRI`, `SET`, `ACC`.

User-defined hooks and rare utility ops — implement on demand when a real model needs one.

## Acceptance for "100% modern-LLM inference"

Tranches A through G implemented = **~40 dispatched ops total** (current 19 + ~20 from A-G). At that point any modern GGUF LLM in the LLaMA / Gemma / Phi / Mixtral / MoE family should run end-to-end on GCU with at most CPU-fallback for sampling helpers (which are negligible).

Tranches A and B (~11 ops) cover the highest-impact gaps. C through G are filling in coverage for less common architectures.

## Rough effort summary

| Tranche | Ops | Est. effort | Cumulative ops dispatched |
|---|---|---|---|
| A — Common LLM ops              | ~10 | 1-2 days  | 29 |
| B — Fused attention             | 1   | 1-3 days  | 30 |
| C — Sampling helpers            | 3   | 0.5 day   | 33 |
| D — Element-wise + reductions   | ~10 | 1-2 days  | 43 |
| E — Alternative architectures   | 6   | 1-2 weeks | 49 |
| F — Norm variants               | 3   | 0.5 day   | 52 |
| G — Window/relative attention   | 4   | 1 day     | 56 |
| H — Edge cases                  | 5   | 0.5 day   | 61 |
| **Out of scope (CPU fallback)** | ~33 | -         | (61 of 95 ggml ops) |

Tranche A is the natural next MVP after the Gemma 4 26B A4B bench validates the current Q4-MoE path.

## Cross-cutting work needed in parallel

These aren't ops but unblock several tranches:

1. **`gcu_dtype_supported` / `gcu_q_supported` extension.** Currently only F32, F16, Q4_0, Q8_0, Q4_K. Adding Q5_K, Q6_K, Q3_K is mechanical (existing `gcu_q_dequantize_to_f32` handles them via type traits) but each needs a smoke check. Independent of op work.
2. **Buffer-type `get_alloc_size` rework** if any new op needs a non-element-aligned dtype.
3. **Real-model regression suite.** As coverage grows, we need a small fast-bench script that exercises a representative subset of model families to catch routing regressions.

## Next concrete step

After Gemma 4 26B A4B bench validates `MUL_MAT_ID`, brainstorm Tranche A as MVP-5a (~10 ops, mostly mechanical topsaten wraps). Tranche A is high-ROI: Gemma 1/2/3 and Phi families immediately become first-class GCU citizens with no per-model special casing.
