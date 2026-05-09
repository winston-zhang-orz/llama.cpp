# GCU MVP-3a — Native Quantized MUL_MAT (dequant-to-F16)

**Date:** 2026-05-09
**Status:** Draft
**Branch base:** `feat/ggml-gcu` (MVP-2 complete + perf doc, 36 commits)
**Reference:** Topsop SDK source at `/Users/root1/gitlab/topsop`. ggml's host dequantizers in `ggml/src/ggml-quants.c`.

## 1. Goal & non-goals

### Goal

Q4 GGUF models run **faster on GCU than on CPU baseline** with `--device GCU0 -nkvo`. Specifically: `llama-bench` for Qwen 2.5 0.5B-Instruct Q4_K_M shows token-generation ≥ 70 t/s on GCU (current Q4 CPU baseline). MUL_MAT for Q-typed weights runs on GCU instead of falling back to CPU.

### Non-goals (deferred)

- **Native `topsatenLinearQuant`**: ggml's Q4_K block layout differs from topsaten's quant tensor layout. Converting between them is an MVP-3a' optimization once dequant-to-F16 ships and we measure if perf is enough.
- **Quantized GET_ROWS** (embedding lookup on Q-typed tensors). Embeddings are usually F32/F16 in real models; refuse Q-typed GET_ROWS.
- **Q5_K, Q6_K, Q3_K, Q2_K, Q4_1, Q5_1, IQ-types**. MVP-3a covers Q4_0, Q8_0, Q4_K. Other Q-types stay on CPU.
- **KV cache offload, F16 ROPE, SOFT_MAX with sinks** — those are separate sub-projects (MVP-3b, c).
- **Performance work**: pinned host buffers, async copy/compute overlap, multi-stream — MVP-3d.

## 2. Strategy: dequant-on-load, store F16 on GCU

When ggml's `set_tensor` uploads a Q-typed weight to a GCU buffer:

1. The buffer-type's `get_alloc_size` over-reports — returns the **F16-equivalent byte count** instead of Q-packed bytes.
2. The buffer's `set_tensor` detects Q-type input data, runs ggml's existing host dequantizer (`dequantize_row_q4_0` etc.) to produce F16 in a host scratch, then `topsMemcpy` uploads the F16 bytes to the device buffer.
3. The tensor's `nb[]` strides are rewritten at `init_tensor` time to F16 strides so downstream ops can index correctly.
4. The compute path (`gcu_op_mul_mat`) reinterprets the weight as F16 by ignoring `t->type` in the dtype branch — Q-typed weights take the existing F16-weight path.

After load, the GPU has a `[K, M]` F16 tensor that ggml believes is Q4. As long as we never let any other op compute on this tensor, the lie is harmless. Only MUL_MAT sees Q-typed weights, and we control that path.

### Cost we accept

- **2-4× weight memory on device.** Q4 (0.5 bytes/param) → F16 (2 bytes/param) is 4×. Q8 (1.0 bytes/param) → F16 is 2×. For Qwen 0.5B Q4_K_M (~315 MB Q4) the F16 expansion is ~1.26 GB. Fits comfortably on the S60's 41 GB. For 7B+ models the doubling is more painful but still fits — the S60 has plenty of memory.
- **One-time host-side dequant during model load.** ggml already does this on CPU when models load to CPU; we're just routing the result to a GCU buffer instead.

## 3. Files

| File | Change |
|---|---|
| `ggml/src/ggml-gcu/ggml-gcu.cpp` | All MVP-3a code lives here. |
| `tests/test-backend-gcu.cpp` | Add a `test_q4_k_round_trip` smoke test verifying that uploading Q4_K data and reading it back through GCU compute matches a CPU reference. |
| `docs/build.md` | Update operator coverage and perf table after benchmarking. |

## 4. Component design

### 4.1 Q-format support table

Three formats land in MVP-3a. Each has an existing ggml dequantizer:

| Q-format | block_size | bytes/block | ggml dequantizer | Real-world models |
|---|---|---|---|---|
| `GGML_TYPE_Q4_0` | 32 | 18 (one F16 scale + 32 nibbles) | `dequantize_row_q4_0` | Older quants, common |
| `GGML_TYPE_Q8_0` | 32 | 34 (one F16 scale + 32 int8) | `dequantize_row_q8_0` | High-quality quants |
| `GGML_TYPE_Q4_K` | 256 | 144 (super-block: 12-byte mins/scales + 128 nibbles) | `dequantize_row_q4_K` | Most common in modern GGUFs (Q4_K_M is default for many models) |

Other Q-types: `gcu_q_supported(type)` returns false → MUL_MAT falls back to CPU and the buffer never gets allocated on GCU (because `supports_op` says no for these tensors).

### 4.2 Buffer-type changes

```cpp
// New: returns F16-equivalent bytes for Q-typed weight tensors.
static size_t ggml_backend_gcu_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t /*buft*/, const ggml_tensor * t) {
    if (gcu_q_supported(t->type)) {
        // Store as F16 on device.
        return ggml_nelements(t) * sizeof(uint16_t);
    }
    return ggml_nbytes(t);  // default for non-Q types
}
```

Wire into the existing `ggml_backend_gcu_buffer_type_i` (the `get_alloc_size` slot is currently `nullptr`).

### 4.3 Buffer changes (set_tensor and init_tensor)

```cpp
// In ggml_backend_gcu_buffer_init_tensor:
// If tensor is Q-typed, rewrite nb[] to F16 strides so the device buffer
// indexing matches what we actually stored.
static enum ggml_status ggml_backend_gcu_buffer_init_tensor(
        ggml_backend_buffer_t /*buffer*/, ggml_tensor * tensor) {
    if (gcu_q_supported(tensor->type)) {
        const int64_t bpe_f16 = sizeof(uint16_t);
        tensor->nb[0] = bpe_f16;
        tensor->nb[1] = tensor->ne[0] * bpe_f16;
        tensor->nb[2] = tensor->ne[1] * tensor->nb[1];
        tensor->nb[3] = tensor->ne[2] * tensor->nb[2];
    }
    return GGML_STATUS_SUCCESS;
}
```

```cpp
// In set_tensor: detect Q-typed source data, dequantize on host, upload F16.
static void ggml_backend_gcu_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));

    if (gcu_q_supported(tensor->type) && offset == 0 && size == ggml_nbytes(tensor)) {
        // Full-tensor upload of Q-typed weight: dequantize to F16 on host,
        // then upload F16 bytes.
        const int64_t n = ggml_nelements(tensor);
        std::vector<float>       host_f32(n);
        std::vector<ggml_fp16_t> host_f16(n);

        // Use ggml's row-by-row dequantizer for the relevant Q type.
        const int64_t row_size_q  = ggml_row_size(tensor->type, tensor->ne[0]);
        const int64_t n_rows      = n / tensor->ne[0];
        for (int64_t r = 0; r < n_rows; r++) {
            const void * src_row = (const char *) data + r * row_size_q;
            float      * dst_row = host_f32.data() + r * tensor->ne[0];
            gcu_q_dequantize_row(tensor->type, src_row, dst_row, tensor->ne[0]);
        }
        ggml_fp32_to_fp16_row(host_f32.data(), host_f16.data(), n);

        TOPS_CHECK(topsMemcpy(tensor->data, host_f16.data(),
                              n * sizeof(ggml_fp16_t),
                              topsMemcpyHostToDevice));
        return;
    }

    // Default: byte-for-byte upload (existing behavior).
    TOPS_CHECK(topsMemcpy((char *) tensor->data + offset, data, size,
                          topsMemcpyHostToDevice));
}
```

`gcu_q_dequantize_row` is a small switch over `tensor->type` calling the right ggml dequantizer (`dequantize_row_q4_0`, etc.). These functions are already exported by `libggml-base.so` (used internally by ggml-cpu).

### 4.4 MUL_MAT changes

```cpp
// In supports_op for MUL_MAT:
case GGML_OP_MUL_MAT: {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (!w || !x) return false;

    // Existing combos:
    bool ok = false;
    if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ok = true;
    if (w->type == GGML_TYPE_F16 &&
        (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
        (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;

    // NEW: Q-typed weight (interpreted as F16 in GCU memory).
    if (gcu_q_supported(w->type) &&
        (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
        (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;

    if (!ok) return false;
    /* … existing 2D / contiguous / shared K checks … */
    return true;
}
```

```cpp
// In gcu_op_mul_mat: when w->type is Q-typed, reinterpret as F16 and take
// the F16-weight path.
static bool gcu_op_mul_mat(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * w = dst->src[0];
    ggml_tensor * x = dst->src[1];

    const ggml_type wt_effective = gcu_q_supported(w->type) ? GGML_TYPE_F16 : w->type;

    // … existing branches now key on wt_effective instead of w->type.
    // The weight tensor passes its data pointer through unchanged;
    // GGML stride fields are already F16-correct because init_tensor
    // rewrote nb[] for Q-typed tensors.
}
```

The Q-aware branches piggyback on the existing F16-weight mixed-dtype path that already handles `(F16 weight, F32 input)`, `(F16 weight, F16 input)`, etc.

### 4.5 Refusal for non-MUL_MAT Q-tensor ops

GET_ROWS, SET_ROWS, CPY/DUP, etc. on Q-typed tensors keep returning false in `supports_op`. The scheduler routes those operations to CPU. This means an embedding tensor (if it's Q-typed) goes to CPU, with copies in/out — same as MVP-2.

Real-model-impact note: most modern Qwen / Llama GGUFs keep `token_embd` as F32 or F16 even when `q4_k_m` quantizes the rest. The GET_ROWS embedding lookup typically doesn't hit the Q path, so this is fine.

## 5. Testing

### Primary correctness gate: `test-backend-gcu` smoke

Add `test_q4_k_round_trip`:

1. Build a Q4_K weight on host using `ggml_quantize_chunk(GGML_TYPE_Q4_K, ...)` over a known F32 input.
2. Build a `ggml_tensor` with type Q4_K, allocate on GCU buffer (which over-reports as F16 size), `set_tensor` with the Q4 bytes.
3. Build a F32 activation, allocate on GCU.
4. Run `ggml_mul_mat(weight_q4, x_f32)` → result.
5. Run the same MUL_MAT on CPU as ground truth (with the original F32 weight, not the requantized one — Q4 lossy compression dominates the comparison).
6. Compare with a slack tolerance (atol=0.5, rtol=0.05) — Q4 round-trip noise is real, ~5% relative error is normal.

### `test-backend-ops` impact

Some `MUL_MAT` test cases use Q4_0 / Q8_0 / Q4_K weights. They currently report "not supported" on GCU. After MVP-3a they should run on GCU and pass `test-backend-ops`'s built-in tolerance (which already accounts for Q4 quantization noise). **Acceptance: the Q-weight cases that currently report "not supported" must transition to OK with no FAILs.**

### End-to-end: real model bench

```bash
./build/bin/llama-bench -m /home/agent/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --device GCU0 -nkvo 1 -p 64,128,512 -n 16,64 -r 3
```

**Pass:** tg32 (and tg64) ≥ 70 t/s — beating Q4 CPU baseline of 70.6 t/s. pp64 should also exceed Q4 CPU's 463 t/s now that MUL_MAT runs on GCU.

## 6. Phases

| Phase | What | Approx LOC |
|---|---|---|
| **A** | Add `gcu_q_supported`, `gcu_q_dequantize_row` helpers + wire ggml's dequantizer prototypes (extern decl). | ~40 |
| **B** | Wire `get_alloc_size` and `init_tensor` to overprovision F16 storage for Q-typed weights. | ~30 |
| **C** | Extend `set_tensor` with the dequant-then-upload path. | ~50 |
| **D** | Extend MUL_MAT `supports_op` and `gcu_op_mul_mat` to treat Q-typed weights as F16. | ~30 |
| **E** | Add `test_q4_k_round_trip` smoke test. | ~80 |
| **F** | Real-model bench + docs update. | ~20 |

Total ~250 LOC. Smaller than MVP-2 because we're piggybacking on the existing F16-weight path.

## 7. Risks (and what we'd do)

- **R1: ggml's `nb[]` recomputation reasserts Q4 strides** between our `init_tensor` call and the actual compute. If ggml-alloc or some other code path resets `nb[]` based on `type`, our F16 strides get clobbered. Mitigation: re-rewrite `nb[]` defensively at the top of `gcu_op_mul_mat` for Q-typed weights, AND watch test-backend-ops for stride-related failures.
- **R2: ggml-cpu dequantizers not exported.** They're in `ggml-cpu` not `ggml-base`. We'll need to either include `ggml-quants.h` and link `ggml-cpu`, or re-export them. Mitigation: `target_link_libraries(ggml-gcu PRIVATE ggml-cpu)` — but that creates a dependency cycle (cpu loads gcu via dl). Alternative: copy the Q4_0 / Q8_0 dequant routines (small) into our backend OR add ggml-base to expose them publicly. **Decision: link ggml-cpu PRIVATE; if cycle issue arises, use object-file dependency instead.**
- **R3: F16 storage doubles memory** — bounded for our test target (Qwen 0.5B), but document.
- **R4: Embedding via GET_ROWS on Q-typed tensor** if the model uses a Q4 embedding. Mitigation: refuse in supports_op (already done), document in user-facing limitations.

## 8. Verification points carried into plan-writing

1. ggml dequantizer header location and link layer: `ggml-quants.h` is in `ggml/src/`; the `dequantize_row_*` symbols are in `libggml-cpu.so`.
2. `ggml_quantize_chunk` is exported (used in tests) — confirm signature.
3. `ggml_fp32_to_fp16_row` is in `libggml-base.so` — confirmed (we already use it in MVP-2 smoke test).
4. Whether ggml-alloc's tensor placement re-runs after our `init_tensor` and clobbers `nb[]`. Probe during plan-writing by setting `nb[]` then reading it back at compute time.
