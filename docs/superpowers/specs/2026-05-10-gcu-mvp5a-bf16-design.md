# GCU MVP-5a — BF16 dtype on core ops (Design)

**Date:** 2026-05-10
**Branch:** `feat/ggml-gcu`
**Reference:** CANN commit `1af9dab32` (#20152)

## Goal

Add BF16 as a first-class device dtype across the core ops the GCU backend already implements. Unlocks modern BF16-only GGUFs (Llama 3.1 / 3.2 / 3.3, Mistral-Nemo, Phi-3.5-MoE, others that ship BF16 weights without an F16 alternative). Today these models either fail to load or fall back to CPU per-tensor.

## Architecture

### Dtype registration

- Extend `gcu_dtype_supported(ggml_type)` to accept `GGML_TYPE_BF16`.
- Extend `ggml_to_topsaten_dtype()` to map `GGML_TYPE_BF16` → `TOPSATEN_DATA_BF16` (verify exact enum name on the SDK first; `topsaten_define.h` is the source of truth).

### Per-op gate updates

For each op below, expand the dtype matrix in `supports_op` to allow BF16 where the topsaten kernel handles it. Implementations should be a one-line dtype swap — no algorithmic changes.

| Op | BF16 path |
|---|---|
| `ADD`, `MUL`, `SCALE`, `SUB`, `DIV` | element-wise, dtype-pass-through |
| `MUL_MAT` | BF16 weight × BF16/F32 input → BF16/F32 output (mirror existing F16-weight cast pattern) |
| `MUL_MAT_ID` | same as MUL_MAT, including the per-(token, expert) fused gate_up loop |
| `RMS_NORM`, `NORM` | dtype-pass-through; affine cast same way as F16 path |
| `SOFT_MAX`, `SILU`, all unary activations | dtype-pass-through |
| `ROPE` (NORMAL + NEOX) | already builds cos/sin in matching dtype; extend to BF16 |
| `CPY`/`DUP`/`CONT` | extend the F32↔F16 cast path to also handle F32↔BF16 and F16↔BF16 |
| `GLU` (GEGLU/SWIGLU/REGLU/GEGLU_QUICK) | dtype-pass-through |
| `CONCAT`, `GET_ROWS`, `SET_ROWS` | dtype-pass-through where the SDK accepts BF16 |
| `FLASH_ATTN_EXT` | BF16 K/V common case; keep current F16 path, add BF16 in parallel |

### Q-typed weight dequant target

Q-typed weights currently dequant-on-load to F16. For BF16 models, the runtime input/activation dtype is BF16 — keep the F16-on-device storage to avoid model-load doubling, and rely on topsatenLinear's mixed-dtype support to handle BF16 input × F16 weight. If topsatenLinear rejects that combination, fall back to a per-call F16 cast of the BF16 input (matches the existing F32→F16 pre-cast path in `gcu_op_mul_mat_id`).

### Buffer-type alloc size

`ggml_backend_gcu_buffer_type_get_alloc_size` is unchanged: `nelements * 2` for both F16 and BF16 (same byte size).

## Testing

Per the `llama-backend-op-development` skill's test pyramid:

1. **L1 / L2 small smoke** — add BF16 variants for the same op set the F16 variants cover today: `ADD_BF16`, `MUL_BF16`, `SILU_BF16`, `RMS_NORM_BF16`, `SOFT_MAX_BF16`, `ROPE_BF16`, `MUL_MAT_BF16` (full BF16 path), `MUL_MAT_MIXED_BF16w` (BF16 weight × F32 input), `MUL_MAT_ID_BF16`, `GEGLU_BF16` (split form). Reference computed in F32 from BF16-rounded inputs (mirror existing F16 reference style — there is a host-side BF16↔F32 helper in `ggml-base`).
2. **L3 real-model shape** — add a BF16 MUL_MAT at K=4096 / M=4096 / N=1 (decode shape).
3. **L4** — N/A (BF16 isn't Q-typed; no roundtrip test needed beyond L1).
4. **L5 real-model end-to-end** — run `llama-completion` with a BF16 GGUF on GCU and CPU; outputs at temp 0 must match. Check whether a BF16 model exists at `/home/agent/models/` first; if not, **skip L5** but document the gap in the commit message and the docs/build.md update. The Gemma 4 (Q4_K_M) and Llama 1B (Q4_K_M) canaries must still produce "Paris".
5. **All four rollback flag combos** still pass smoke.

## Acceptance

- BF16 added to `gcu_dtype_supported` and reflected in every op's supports_op gate.
- Smoke for at least the 9 BF16-variant tests above is green at HEAD under all 4 flag combos.
- Existing canaries (Gemma 4, Llama 1B) still produce "Paris".
- `docs/build.md` operator-coverage section updated to list F32 + F16 + **BF16** as the supported activation dtypes.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Topsaten BF16 enum name differs from `TOPSATEN_DATA_BF16` | Probe `topsaten_define.h` first, use whatever the SDK exposes |
| A specific topsaten kernel rejects BF16 inputs | Gate that single op out of BF16 support with a one-line comment; fall back to CPU |
| BF16-rounded reference values diverge from F16-rounded values, causing existing F16 tests to look broken | They're separate test fixtures; no shared state; existing F16 tests untouched |
| Model loader's BF16 → F16 promote (`ggml-cpu` does this for some types) trips us up | Verify ggml-base preserves BF16 dtype when targeting GCU buffer; if not, document and use F16 path |

## Open questions

- None — the design is gated on SDK probing for the exact BF16 enum and topsatenLinear's BF16 support matrix. If either is missing, gate accordingly and document.

## Out of scope

- BF16 native quant matmul (BF16 weight × Q-typed × BF16 — multi-dtype kernels are a deeper SDK investigation).
- BF16 KV cache (KV cache stays on CPU with `-nkvo`; this MVP is about activation/weight dtypes only).
