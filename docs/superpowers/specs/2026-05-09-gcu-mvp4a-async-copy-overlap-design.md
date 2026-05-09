# GCU MVP-4a — Async H↔D / Compute Overlap (Design)

**Date:** 2026-05-09
**Branch:** `feat/ggml-gcu`
**Status:** approved (sections §1–§5 acked in brainstorming)

## Goal

Eliminate the H↔D bandwidth bottleneck that the pinned-host-buffer optimization (MVP-3d) exposed. Move host↔device transfers onto a dedicated topsrt stream so they overlap with compute on the existing stream. Direct payoff target: another tg uplift on top of the +5–10% (Llama 1B Q4_K_M) / +18–23% (Qwen 0.5B F16) the pinned buffer delivered.

This is **Scope-1** of a two-step plan. Scope-2 (drop per-op compute synchronize so kernels themselves pipeline) is a separate MVP that will land afterwards.

## Non-Goals

- Compute pipelining within a single graph_compute. Per-op `topsStreamSynchronize` stays in Scope-1.
- Native on-device dequant for Q-typed tensors during async copy. Q-typed weights stay on the synchronous load path.
- Multi-device async coordination (events crossing devices). Per-device contexts isolated; cross-device sync is MVP-5.
- D2D inter-buffer copy optimization (already async on the compute stream — no change).

## Architecture

### Stream model

Each `gcu_device_ctx` owns two topsrt streams:

| Stream            | Owns                                                                  |
|-------------------|-----------------------------------------------------------------------|
| `compute_stream`  | All kernel launches (`topsatenXxx`), op-internal D2D scratch copies.  |
| `copy_stream`     | All async H↔D from `set_tensor_async` / `get_tensor_async`.           |

Both are plain `topsStreamCreate` (no priority hints). `compute_stream` is the rename of today's `ctx->stream`; the new `copy_stream` is added alongside.

### Event handling

Two reusable events on the device context, created at device init, destroyed at teardown:

| Event                  | Recorded by                          | Waited on by                          |
|------------------------|--------------------------------------|---------------------------------------|
| `last_copy_event`      | `set_tensor_async` after H→D enqueue | `compute_stream` at `graph_compute` entry |
| `last_compute_event`   | `compute_stream` at `graph_compute` exit | `copy_stream` before `get_tensor_async` D→H |

A `bool copy_event_armed` flag guards the first wait — if no async copy has happened since the last graph_compute, skip the wait (recording-without-arming is undefined in the SDK contract).

Re-recording is legal in topsrt (matches CUDA): `topsEventRecord` re-stamps the event; we never grow event count.

Wait API: `topsStreamWaitEvent(stream, event, 0)` — flags arg matches `tops_runtime_api.h:3257`.

### `set_tensor_async` / `get_tensor_async`

Both live on the **backend** i-table (`ggml_backend_i`), not the buffer i-table. The scheduler routes activation/KV traffic here when one side is GCU and the other is host.

i-table signatures (verified in `ggml/src/ggml-backend-impl.h:111-112`) return `void`. Both must always handle the call — there is no scheduler fallback hook. The contract:

`set_tensor_async`:
1. Early return if `size == 0`.
2. `GGML_ASSERT` tensor is **not** Q-typed. Q-typed weights load via the buffer-level synchronous `set_tensor` at model init and are never re-set during inference, so this codepath is unreachable for Q tensors. An assert here surfaces logic bugs immediately rather than silently mis-sizing the copy (Q4_0 source bytes ≠ F16 destination bytes after dequant-on-load).
3. `topsMemcpyAsync(H→D, copy_stream)`.
4. `topsEventRecord(last_copy_event, copy_stream)`; set `copy_event_armed = true`.

`get_tensor_async`:
1. Early return if `size == 0`.
2. `GGML_ASSERT` tensor is **not** Q-typed (same reasoning).
3. `topsStreamWaitEvent(copy_stream, last_compute_event, 0)` — D→H must not race still-running compute.
4. `topsMemcpyAsync(D→H, copy_stream)`.

The existing synchronous `set_tensor` (buffer iface) is unchanged: model load — including Q-typed dequant-on-load — keeps its current shape.

### `graph_compute` integration

```cpp
if (ctx->copy_event_armed) {
    TOPS_CHECK(topsStreamWaitEvent(ctx->compute_stream,
                                   ctx->last_copy_event, 0));
    ctx->copy_event_armed = false;
}

// run ops as today; each still self-syncs (Scope-1)

TOPS_CHECK(topsEventRecord(ctx->last_compute_event, ctx->compute_stream));
TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
return GGML_STATUS_SUCCESS;
```

`backend->synchronize` syncs both streams in order: copy_stream first (so any pending D→H drains into host memory the caller may inspect), then compute_stream.

### Rollback switch

`GGML_GCU_NO_ASYNC_COPY=1` — fall back to synchronous `topsMemcpy` in `set_tensor_async`/`get_tensor_async` and skip event arming. Mirrors `GGML_GCU_NO_PINNED` from MVP-3d. Bisection takes one env var.

## Testing

- **Smoke test extension** — add `ASYNC_OVERLAP` to `tests/test-backend-gcu.cpp`. Build a 2-tensor graph; populate via `ggml_backend_tensor_set_async`; run chained ops; read back via `ggml_backend_tensor_get_async`; assert numerical equality vs the synchronous baseline. Catches event-ordering regressions.
- **Existing 7-test smoke suite** stays green (correctness gate).
- **Real-model bench** — Llama 1B Q4_K_M and Qwen 0.5B F16 with `--device GCU0 -nkvo 1`, r=5, before/after. Compare tg32/tg64/pp512 against the pinned-buffer baseline already in `docs/build.md`.
- **Multi-device readiness** — the second-stream slot is per `gcu_device_ctx`; design does not paint us into a corner for MVP-5.

## Risks & Mitigations

| Risk                                                                    | Mitigation                                                       |
|-------------------------------------------------------------------------|------------------------------------------------------------------|
| Event-ordering bug allows D→H before compute finishes                  | `last_compute_event` wait in `get_tensor_async`; smoke sub-test  |
| `last_copy_event` recorded but never waited (race)                     | `copy_event_armed` flag, asserted on entry to `graph_compute`    |
| Real model regresses unexpectedly                                       | `GGML_GCU_NO_ASYNC_COPY=1` env var falls back to sync `topsMemcpy` |
| topsrt event API differs from CUDA in subtle ways                       | Smoke test exercises full triplet (record / wait / sync)         |

## Open Questions

None — design fully scoped.

## Acceptance

- All 7 existing smoke tests + new `ASYNC_OVERLAP` sub-test pass on S60.
- llama-bench Llama 1B Q4_K_M `-nkvo` shows non-regressing tg, ideally improved over the pinned-only baseline (+5–10% target).
- llama-bench Qwen 0.5B F16 `-nkvo` shows non-regressing tg, ideally improved over the pinned-only baseline (+10%+ target).
- `GGML_GCU_NO_ASYNC_COPY=1` returns the backend to the pre-MVP-4a path (verifiable by re-running the same bench under that env var).
