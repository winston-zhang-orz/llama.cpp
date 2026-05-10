# GCU MVP-5b — ROPE: MROPE / IMROPE / Vision / partial rotation / freq_factors (Design)

**Date:** 2026-05-10
**Branch:** `feat/ggml-gcu`
**References:** CANN commits `eeb5605de` (MROPE+IMROPE+Vision), `ca709e427` (partial rotation)

## Goal

Extend `gcu_op_rope` from NORMAL + NEOX to cover the remaining mode bits ggml emits, plus partial rotation and `freq_factors`. Unlocks Qwen2-VL, Gemma 3 / 4 vision-tower paths, GPT-OSS proportional rope, and removes the CPU fallback for Gemma 4's full-attention layers (which currently use `freq_factors` and route to CPU per the existing supports_op gate).

## Mode taxonomy

`ggml/include/ggml.h` defines five mode types as bit flags:

| Mode | Value | Semantics |
|---|---|---|
| `GGML_ROPE_TYPE_NORMAL` | 0 | Interleaved pairs (LLaMA / Qwen 1) |
| `GGML_ROPE_TYPE_NEOX` | 2 | Split halves (GPT-NeoX / Phi) |
| `GGML_ROPE_TYPE_MROPE` | 8 | Multi-axis: rotate (T, H, W) sub-segments separately (Qwen2-VL) |
| `GGML_ROPE_TYPE_VISION` | 24 (= NEOX \| MROPE) | Vision-tower variant (Qwen2-VL vision) |
| `GGML_ROPE_TYPE_IMROPE` | 40 (= NEOX \| MROPE \| extra bit) | Inverted-MROPE (newer Qwen variants) |

Plus two orthogonal features any of the modes can combine with:
- **Partial rotation:** `n_dims < head_dim`. Only the first `n_dims` lanes of each head get rotated; the rest pass through unchanged. Already partially supported by our handler (the `n_dims` arg is read from op_params[1]) but untested.
- **`freq_factors`:** `op->src[2]` is non-null and provides per-frequency scale factors (proportional rope; Gemma 4 full attention).

## Architecture

### supports_op extension

```cpp
case GGML_OP_ROPE: {
    if (!gcu_dtype_supported(op->src[0]->type)) return false;
    if (op->src[0]->type != op->type) return false;
    if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) return false;

    const int32_t mode = ((const int32_t *) op->op_params)[2];
    // Accept NORMAL, NEOX, MROPE, VISION, IMROPE. Reject any other bits.
    const int32_t known_mask = GGML_ROPE_TYPE_NORMAL | GGML_ROPE_TYPE_NEOX |
                               GGML_ROPE_TYPE_MROPE  | GGML_ROPE_TYPE_VISION |
                               GGML_ROPE_TYPE_IMROPE;
    if ((mode & ~known_mask) != 0) return false;

    float ext_factor, attn_factor;
    memcpy(&ext_factor,  (const int32_t *) op->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params + 8, sizeof(float));
    if (ext_factor != 0.0f) return false;
    if (attn_factor != 0.0f && attn_factor != 1.0f) return false;

    // freq_factors (op->src[2]) now SUPPORTED via the host-side cos/sin table.
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) return false;
    if (op->src[0]->ne[3] != 1) return false;
    return true;
}
```

### Handler additions

The existing `gcu_op_rope` builds cos/sin on the host and calls `topsvllmRotaryEmbedding(query, key, positions, cos_sin_cache, head_size, is_neox)`. Extend it as follows:

1. **`freq_factors` support** — when `dst->src[2]` is non-null, read it host-side (small, head_dim/2 floats) and divide each `theta_i` by `freq_factors[i]` in `gcu_build_rope_cos_sin_host`. This is a 5-line change to that helper.

2. **Partial rotation** — when `n_dims < head_dim`, build the cos/sin table sized `[max_pos, n_dims]` (already does this) and ensure the kernel only rotates the first `n_dims` lanes per head. `topsvllmRotaryEmbedding` accepts `head_size = n_dims` for this; the remaining lanes pass through because we copy `dst = x` first.

3. **MROPE / IMROPE / Vision** — these split the per-token position into multi-axis sections (typically `[T, H, W, extra]`). The `pos` tensor has 4×n_tokens entries instead of n_tokens. Sections come from `op_params[10..13]`.

   Strategy: probe `topsvllmRotaryEmbedding` (or any `topsvllm*` extension) for a multi-axis variant. If absent, decompose into per-section ROPE calls — for each of up to 4 sections, call the existing rotary kernel on a sub-range of the head_dim with that section's position vector. This is a host-side dispatch loop, no new SDK call.

   For VISION / IMROPE specifically, check whether the position layout differs from MROPE — `ggml-cpu`'s reference (`ggml/src/ggml-cpu/ops.cpp`'s `ggml_compute_forward_rope_f32`) is the source of truth.

### Cos/sin table build

`gcu_build_rope_cos_sin_host` needs to accept an optional `freq_factors` pointer. One-line argument extension; the helper's per-i loop divides theta by the per-i factor when provided.

## Testing

1. **L1 small smoke** — add per-mode tests:
   - `ROPE_FREQ_FACTORS` — NORMAL mode + non-null freq_factors. Reference uses divided thetas.
   - `ROPE_PARTIAL` — NORMAL mode, `n_dims = head_dim/2`. Reference passes the upper half through unchanged.
   - `ROPE_MROPE` — MROPE mode with a 4-section position layout. Reference computed per-section.
   - `ROPE_VISION`, `ROPE_IMROPE` — same skeleton with the right mode bits.
2. **L2 dtype variants** — F16 for each mode (mirror the existing `test_rope_f16` pattern). BF16 if MVP-5a has landed.
3. **L3 real-model shape** — head_dim=128, n_head=16, n_tokens=8 — Gemma 4 / Qwen2-VL decode shapes.
4. **L4** — N/A.
5. **L5 real-model end-to-end** — Gemma 4 26B A4B canary must still produce "Paris" (this MVP moves the full-attention `freq_factors` ROPE *off* CPU fallback and onto GCU, so any wrong output here is a real regression). Llama 1B Q4_K_M canary unchanged. If a Qwen2-VL or Gemma 3 vision GGUF is on the S60, run a representative prompt as L5.
6. **All four rollback flag combos** still pass smoke.

## Acceptance

- supports_op accepts NORMAL, NEOX, MROPE, VISION, IMROPE plus `freq_factors` plus partial rotation.
- Per-mode smoke passes at HEAD under all 4 flag combos.
- Gemma 4 still produces "Paris" with `freq_factors` ROPE now on GCU.
- `docs/build.md` ROPE coverage line updated to list all five modes + freq_factors + partial.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `topsvllmRotaryEmbedding` doesn't support multi-axis layouts | Decompose into per-section calls host-side; no SDK change needed |
| `freq_factors` divides theta in the wrong direction (× vs ÷) compared to ggml-cpu | Match the CPU reference exactly; smoke against random factors with bit-exact tolerance |
| A mode's mask collision (e.g. VISION = NEOX \| MROPE) makes the `is_neox=true` flag mean different things | Read mode bits explicitly: `is_neox = (mode & GGML_ROPE_TYPE_NEOX) != 0` and check MROPE bit separately |
| Position tensor shape changes for MROPE (4× larger) breaks the `pos->ne[0] == n_tokens` assert | Update assert to handle the multi-axis layout per the CPU reference |

## Open questions

- Probe needed: does the SDK expose a multi-axis rotary kernel, or do we decompose? Decision is at L1 implementation time.

## Out of scope

- YARN-style rotary scaling (`ext_factor != 0`) — separate MVP if a target model uses it.
- Long-context extensions beyond the position cap built into the host-side cos/sin table (currently sized `max_pos = max(positions) + 1` per call — fine for any context length the model can hold).
