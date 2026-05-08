# Enflame GCU Backend — MVP-2 Design

**Date:** 2026-05-08
**Status:** Draft (pending review)
**Branch base:** `feat/ggml-gcu` (MVP-1 + 2 incremental SET_ROWS commits, 24 total)
**Reference:** Topsop SDK source at `/Users/root1/gitlab/topsop` (newly available — MVP-1 was designed against headers only).

## 1. Goal & non-goals

### Goal

Add a second tier of operators to the `ggml-gcu` backend so it covers the surface needed for transformer-block compute. Each op passes `test-backend-ops` on the S60. Real-model loading does not crash and produces no NaN/Inf in GCU-routed ops.

### Non-goals (deferred)

- End-to-end correct text generation from a real model. Scope is **op coverage**, not inference. Probing a real model is part of the negative test, not a pass criterion.
- Native quantized matmul (Q4_K, Q8_0, …). Quantized weights stay on CPU.
- KV cache offload to GCU. The cache lives on CPU. Attention math runs on GCU; cache writes/reads cross the H↔D boundary via ggml's scheduler. **Explicit design decision** — see §4.
- Performance work: pinned host memory, multi-stream copy/compute overlap, custom kernels, fused-op pattern matching.
- Multi-device.
- BF16, Q-types, F64.
- ROPE modes other than the basic mode 0 (yarn, neox, ext_factor are MVP-3).
- SOFT_MAX with `max_bias != 0` (alibi).

## 2. Op coverage (the 7 new ops + tightening)

| ggml op | topsaten target (header) | Notes |
|---|---|---|
| `SET_ROWS` (revisit) | existing `topsatenIndexPut` | Tighten `supports_op` to refuse ggml-style 4D KV cache shapes. KV cache writes route to CPU. |
| `RMS_NORM` | `topsvllmRmsNorm` (`topsaten_vllm.h:2113`) | Plan-time: confirm whether it requires a fused weight; if yes, supply identity weight. |
| `ROPE` | `topsvllmRotaryEmbedding` (`topsaten_vllm.h:69`) | Mode 0 only. Plan-time: bridge ggml's `freq_base/freq_scale` → cos/sin tables (precomputed per call into a scratch slab). |
| `SOFT_MAX` | `topsatenSoftmaxForward` (`topsaten_ops.h`) | With optional mask via prior ADD; `max_bias == 0` only. |
| `SILU` | `topsatenSilu` (`topsaten_ops.h:3015`) | Direct map. |
| `MUL_MAT` (mixed dtype) | existing `topsatenLinear` + `topsatenTo` casts | F16 weight + F32 input + F32 output path: cast input F32→F16, run F16 Linear, cast output F16→F32. |
| `CPY/DUP` (dtype conv) | `topsatenTo` for F32↔F16 | Same-dtype contiguous keeps `topsMemcpyAsync(D2D)` fast path. |

The full dispatch table after MVP-2 (existing + new):
`NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE, ADD, MUL, SCALE, MUL_MAT, GET_ROWS, SET_ROWS, CPY, DUP, CONT, RMS_NORM, ROPE, SOFT_MAX, SILU.`

## 3. Architecture (no changes from MVP-1)

- Single file: `ggml/src/ggml-gcu/ggml-gcu.cpp`. Pre-MVP-2 size: 1186 LOC. Expected after MVP-2: ~1900 LOC. Still under the 2k threshold the MVP-1 spec set for splitting.
- Same shared-context-per-device, same heap-leaked registry, same LIFO pool, same `TOPS_CHECK`/`TOPSATEN_CHECK` error model, same single-stream eager dispatch.
- Per-context state: existing `zero_bias` field (MUL_MAT bias). MVP-2 may add **at most one** new field, `ones_n0`, lazily-grown to the largest `ne[0]` seen, used only if `topsvllmRmsNorm` requires a fused identity weight (decided at plan-writing — see §6). A potential `rope_cos_sin_cache` is left as MVP-3.

## 4. KV cache strategy: stays on CPU

### Why this is an explicit design decision

The post-MVP-1 probe with Qwen 0.5B Q4_K_M failed at `topsatenIndexPut(F16 self [128, 32768], F16 value, I64 idx)` returning `NOT_SUPPORT` at runtime. Investigation against the topsop source revealed:

- **`topsatenIndexPut`**: runtime constraints not documented in headers. Empirically rejects the ggml KV cache shape on this SDK version.
- **`topsatenScatter`** (`op_aten_scatter.cc`): requires `output.rank == index.rank == source.rank`. ggml's `set_rows` uses a 1-D index against a higher-rank dst, so scatter would force broadcasting the index by `head_dim×`. Wasteful and not the SDK author's intent.
- **`topsvllmReshapeAndCache`**: purpose-built for KV cache writes, but requires the **vLLM block-paged cache layout** `[num_blocks, block_size, num_heads, head_size]`. ggml uses a flat contiguous cache. Adopting block-paged means rewriting llama.cpp's cache management — out of scope.

### What we do

`ggml_backend_gcu_device_supports_op` for `SET_ROWS` returns `true` only for the shape patterns we have positive `test-backend-ops` coverage for:

```
src->ne[2] == 1 && src->ne[3] == 1   // not a 4D batched cache shape
&& dst->ne[2] == 1 && dst->ne[3] == 1
&& src and dst contiguous
&& idx is 1D, I32 or I64
```

ggml's scheduler then routes `SET_ROWS` for cache-shaped tensors to the next backend that supports it (CPU). Because cache buffer placement follows where its writes can run, the cache buffer itself is allocated on CPU. **Net effect**: cache writes/reads cross H↔D. Attention math (RMS_NORM, ROPE, SOFT_MAX, MUL_MAT) still runs on GCU.

### Cost we're accepting

Per-token KV writes go through host memory. Per-layer this is two H↔D copies (K and V, ~head_dim*n_heads*element_size each). For Qwen 0.5B that's ~3 KiB per layer per token. At hundreds of tokens/s the bandwidth is small but non-zero. Acceptable for "op coverage" scope; revisit when we tackle quantized matmul in MVP-3 and have actual e2e latency to optimize.

## 5. Mixed-dtype MUL_MAT design

### Confirmed by source (not headers)

`/Users/root1/gitlab/topsop/topsop/lib/ops/aten/op_aten_matmul/op_aten_matmul.h` line 167-174:

> matmul, lhs.dtype, rhs.dtype and out.dtype should be equal

**Both `topsatenMatmul` and `topsatenLinear` require `lhs.dtype == rhs.dtype == out.dtype`.** The earlier doc-string suggesting Matmul accepts mixed dtypes was misleading.

### Strategy

Cast **input to weight dtype**, run matmul in weight dtype, cast **output to caller's dtype** if different.

For the dominant case `F16 weight × F32 input → F32 output`:

1. Allocate two F16 scratch slabs: `x_f16` (size = `ggml_nelements(x) * 2`) and `y_f16` (size = `ggml_nelements(dst) * 2`).
2. `topsatenTo(x_f16, x, FP16, false, true, CONTIGUOUS, stream)`.
3. `topsatenLinear(y_f16, x_f16, w, zero_bias_f16, stream)` (existing path, F16 dtypes).
4. `topsatenTo(dst, y_f16, FP32, false, true, CONTIGUOUS, stream)`.
5. `topsStreamSynchronize` and free both scratch slabs back to the pool.

### Why cast input not weight

- Weight is the larger tensor (a [4096, 11008] FFN gate weight in F16 = 90 MiB; in F32 = 180 MiB). Per-call cast wastes compute every layer; caching an F32 copy doubles weight memory.
- Casting input is cheap (input size is `n_tokens * d_model`, much smaller than weight per call) and matches the standard inference-backend pattern (CUDA, Metal, CANN).
- Numerical loss in the F32→F16 input cast is tolerated by every other inference backend; ggml's reference tolerance for F16-matmul cases reflects this.

### Cases handled

- All-F32: existing fast path, unchanged.
- F32 weight + F16 input: refuse (uncommon; not in MVP-2 scope).
- F16 weight + F32 input + F32 output: above flow (the common case).
- F16 weight + F16 input + F32 output: skip the input cast (already F16); still cast output.
- F16 everywhere: existing path, F16 dtypes throughout.
- Quantized weights: refuse → CPU.

## 6. Per-op design notes

### `SET_ROWS` — gate tightening only

Existing implementation (flat-2D + dtype-cast scratch) stays. `supports_op` adds the constraints documented in §4. No code change to the compute path.

### `RMS_NORM`

`ggml_rms_norm(x, eps)`: `out = x / sqrt(mean(x^2) + eps)`, no learned weight (the multiply happens via a separate downstream `MUL` in ggml's graph).

- Inputs: `x` (F32 or F16). `eps = ((float*)op_params)[0]`.
- topsaten target: `topsvllmRmsNorm(out, x, ?, eps, stream)`. **Open at plan-writing**: whether topsvllmRmsNorm requires a fused weight tensor.
  - If yes: pass an ones tensor of size `ne[0]`. Allocate per-call from the pool (the tensor is at most a few KiB; pool reuse means after the first call the same slab is recycled). The fill is a one-time `topsMemset` + `topsatenAdd` with scalar 1.0, or simpler: maintain a small per-context `ones_n0` buffer that grows when we see a larger `ne[0]`. **This is a new context field**, listed under §3 below.
  - If no: pass nullptr/none for the weight slot.
  - Fallback: implement via primitives (`MUL` (square), `MEAN`, `ADD eps`, `SQRT`, `DIV`) if `topsvllmRmsNorm` is unavailable or has unfit constraints.
- `supports_op`: F32/F16 inputs; ne[2,3] == 1 (single-batch, single-channel — inner-dim-only normalization, the ggml semantic).

### `ROPE` — mode 0 only

`ggml_rope(x, pos, n_dims, mode, freq_base, freq_scale, ...)`: rotary position encoding.

- Inputs: `x` (F32 or F16, queries or keys); `pos` (I32, position indices, length = n_tokens).
- ggml's RoPE has 13+ `op_params` fields; **MVP-2 only handles `mode == 0`** with default extension parameters. Other modes (mode 1 = NEOX, mode 2 = MROPE; ext_factor != 0; non-default attn_factor) → refuse → CPU.
- topsaten target: `topsvllmRotaryEmbedding`.
- **Bridge to ggml's representation**: topsvllm typically expects precomputed `cos/sin` tables; ggml passes `freq_base, freq_scale, n_dims, theta, attn_factor` and computes per-call. MVP-2 strategy:
  1. On host (CPU), compute `cos/sin` tables for the position range from ggml's params: `theta_i = freq_base^(-2i / n_dims) * freq_scale`.
  2. Allocate cos and sin scratch slabs from the GCU pool.
  3. `topsMemcpy(H→D)` to upload cos/sin to GCU.
  4. Call `topsvllmRotaryEmbedding` with the uploaded tables.
  5. Free both scratch slabs after stream sync.
  - This is per-call recomputation, slow but correct. MVP-3 may cache the tables keyed by `(freq_base, freq_scale, max_pos)`.
- `supports_op`: F32/F16 inputs; mode == 0; default extension params.

### `SOFT_MAX` (with optional mask)

`ggml_soft_max(x, mask, scale, max_bias)`:
- Computes `out = softmax(x * scale + mask, dim=-1)` (mask added if present).
- `op_params`: `[0]=scale`, `[1]=max_bias`.

MVP-2 supports `max_bias == 0` only. ggml's `max_bias != 0` adds a per-row alibi term — refused → CPU.

Implementation:
- If mask present: `topsatenAdd(scratch, x, mask, alpha=scale)` then `topsatenSoftmaxForward(out, scratch, dim=-1)`.
- If no mask: `topsatenMul(scratch, x, scalar=scale)` then `topsatenSoftmaxForward(out, scratch, dim=-1)`.
- Open at plan-writing: whether `topsatenSoftmaxForward` accepts a fused scale/mask in any signature variant. If yes, prefer that.
- Allocates one scratch slab the size of `x`'s data.
- `supports_op`: F32/F16 inputs; `max_bias == 0`.

### `SILU`

`ggml_silu(x)`: `out = x * sigmoid(x)`. Maps directly to `topsatenSilu(out, x, stream)`.

- Inputs/output: F32 or F16. No params.
- `supports_op`: F32/F16; in/out same dtype.

### Mixed-dtype `MUL_MAT`

See §5. Adds a dtype-branch in `gcu_op_mul_mat` and broadens `supports_op` to accept the F16-weight cases. The all-F32 fast path stays unchanged.

### `CPY/DUP` dtype conversion

Existing same-dtype contiguous fast path stays. New branch when src/dst dtypes differ but both are in {F32, F16}: build wrapper tensors with their respective dtypes and call `topsatenTo(dst, src, target_dtype, false, true, CONTIGUOUS, stream)`. `supports_op` lifts the same-dtype restriction for this dtype pair.

## 7. Testing

### Primary correctness gate: `test-backend-ops`

```
./build/bin/test-backend-ops -b GCU0 -o ADD,MUL,MUL_MAT,SCALE,CPY,DUP,CONT,GET_ROWS,SET_ROWS,RMS_NORM,ROPE,SOFT_MAX,SILU,RESHAPE,VIEW,PERMUTE,TRANSPOSE
```

Pass criterion: **0 FAILs**. Cases the spec explicitly excludes (quant types, batched cache, alibi, non-mode-0 RoPE, F32 weight with F16 input) report "not supported" — that is correct, not a failure.

### Smoke-test extension (`tests/test-backend-gcu.cpp`)

New sub-tests, alphabetical in the file:

| Sub-test | Shape | Tolerance |
|---|---|---|
| `test_mul_mat_mixed` | F16 weight [1024, 2048], F32 input [1024, 1024], F32 output [2048, 1024] | atol=1e-1, rtol=1e-2 |
| `test_rms_norm` | F32 [1024, 64], eps=1e-6 | atol=1e-4, rtol=1e-4 |
| `test_rope` | F32 [64, 8, 16] (head_dim=64, n_heads=8, n_tokens=16), positions [0..15], mode=0, freq_base=10000 | atol=1e-3, rtol=1e-3 |
| `test_silu` | F32 [4096] | atol=1e-5, rtol=1e-5 |
| `test_softmax` | F32 [16, 1024] with mask, scale=0.125 | atol=1e-5, rtol=1e-5 |

Tolerances tuned to match `tests/test-backend-ops.cpp` for similar ops; final values confirmed during plan-writing.

### Build verification

`cmake -B build -DGGML_GCU=ON && cmake --build build -j` succeeds clean. `./build/bin/test-backend-gcu` exits 0.

### Negative test

```
./build/bin/llama-cli -m /home/agent/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --device GCU0 -p "Hello" -n 8
```

Pass criterion: **loads + runs to completion without abort, no NaN/Inf in any GCU-routed op output.** Text quality is not evaluated (Q4 weights stay on CPU; KV cache stays on CPU; only attention math + activations run on GCU). The point is "doesn't crash."

The KV-cache-on-CPU choice means a real model with `--device GCU0` will work as long as the scheduler routes SET_ROWS/GET_ROWS for cache buffers back to CPU and we don't introduce regressions in the already-working ops.

### What MVP-2 does NOT need

- F32 weight × F16 input MUL_MAT
- Non-mode-0 ROPE
- SOFT_MAX with alibi
- Native quantized matmul
- Pinned host buffers, multi-stream
- Performance numbers
- Multi-device
- BF16 / Q-types / F64

## 8. Phases

Sequential, MVP-1-style. One phase = one logical unit, committed independently with verification on real hardware before the next.

| Phase | What | Rough size | Verification |
|---|---|---|---|
| **A** | SET_ROWS gate tightening (no compute change) | ~30 LOC | -o SET_ROWS still 2 OK / 0 FAIL |
| **B** | CPY/DUP dtype-converting path via `topsatenTo` | ~40 LOC | -o CPY,DUP picks up F32↔F16 cases |
| **C** | SILU via `topsatenSilu` | ~30 LOC | -o SILU ≥ 1 OK / 0 FAIL |
| **D** | SOFT_MAX (no-mask, then with-mask) | ~80 LOC | -o SOFT_MAX ≥ 4 OK / 0 FAIL; alibi cases skipped |
| **E** | RMS_NORM | ~50 LOC | -o RMS_NORM ≥ 4 OK / 0 FAIL |
| **F** | ROPE mode 0 with per-call cos/sin upload | ~120 LOC | -o ROPE ≥ 2 OK / 0 FAIL on mode-0; other modes skipped |
| **G** | Mixed-dtype MUL_MAT (F16 weight + F32 input) | ~80 LOC | -o MUL_MAT picks up F16-weight + F32-input cases as new OKs |
| **H** | Smoke-test additions (5 new sub-tests) | ~150 LOC | `test-backend-gcu` exits 0 |
| **I** | Negative test on Qwen 0.5B Q4_K_M | 0 (verification only) | `llama-cli --device GCU0` loads + runs |
| **J** | docs/build.md update for MVP-2 op set | ~20 LOC | docs reflect new op coverage |

Total expected scope: ~580 LOC backend + ~150 LOC tests + ~20 LOC docs = ~750 LOC delta over MVP-1.

### Ordering rationale

- A first (no new compute path; just tightens an existing gate).
- B before G (G reuses dtype-cast infrastructure introduced by B).
- C/D/E/F are independent unary/binary ops; alphabetical for predictability.
- G after the standalone ops (mixed-dtype path is the highest-risk new code; running on a stable baseline lets us bisect easily).
- H after C-G (each smoke-test sub-test depends on its op landing).
- I after H (real model probe is the integration check).
- J last (docs reflect what actually shipped).

## 9. Risks accepted in MVP-2

- **`topsvllmRmsNorm` may need a fused weight or have constraints we hit at runtime.** Phase E has a fallback path (manual primitive composition) documented above.
- **`topsvllmRotaryEmbedding` cos/sin convention may not match ggml's.** Phase F may take longer than 120 LOC if the bridge is non-trivial. If unworkable, ROPE goes to CPU for MVP-2 and Phase F shifts to MVP-3.
- **`topsatenSoftmaxForward` may not accept a fused mask.** Spec says "ADD then softmax" path; if softmax has its own mask param, we use it.
- **F16-everywhere accumulation in MUL_MAT loses precision** beyond the smoke test's tolerance for some models. Caller's responsibility — we don't emulate F32 accumulation.
- **Per-call ROPE table upload dominates per-token latency for very small models.** MVP-2 accepts this; MVP-3 caches.
- **KV cache on CPU caps inference throughput** at the H↔D copy bandwidth. Acceptable for correctness-first MVP-2.

## 10. Verification points carried into plan-writing

These are TBD details that don't change the design but must be resolved before code is written. All are local lookups against the topsop source or runtime probes on the S60.

1. `topsvllmRmsNorm` — required arguments and weight handling.
2. `topsvllmRotaryEmbedding` — exact parameter set and cos/sin table layout.
3. `topsatenSoftmaxForward` — whether a fused mask/scale variant exists.
4. `topsatenSilu` — whether it's destructive on input or takes separate output.
5. `topsatenTo` — confirm signature handles F32↔F16 cleanly and the `format` parameter doesn't constrain us.
6. `test-backend-ops` tolerance constants for each new op (sed in `tests/test-backend-ops.cpp`).
