# GCU MVP-4b — Queued Ops (Drop Per-Op Compute Sync) (Design)

**Date:** 2026-05-09
**Branch:** `feat/ggml-gcu`
**Status:** approved (sections §1–§3 acked in brainstorming)

## Goal

Eliminate the per-op `topsStreamSynchronize(compute_stream)` that ends every GCU op handler. Today, after each kernel launch, the host blocks until completion before issuing the next op — preventing the topsrt driver from queueing kernels and serializing the GPU at the op boundary. After this MVP, ops queue and the host only synchronizes at `graph_compute` boundaries.

This is **Scope-2** of the two-step plan (Scope-1 = MVP-4a, async H↔D, already landed). The MVP-4a investigation found that the ggml scheduler doesn't route cross-backend copies through the async backend interface, so MVP-4a's per-token uplift is bounded; MVP-4b attacks the orthogonal "compute serialized at op level" lever, which is the most likely cause of the flat Llama Q4 result.

## Non-Goals

- Cross-backend async copy routing (would require ggml core changes).
- Multi-stream compute (kernels still execute in submission order on `compute_stream`).
- Native quantized matmul or KV-cache-on-device (separate MVPs, both blocked at SDK level).
- Changing the per-op handlers' kernel-launch logic — only the trailing `synchronize` + `pool.free` are touched.

## Architecture

### Scratch lifetime model

Today every op handler ends with:

```cpp
TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
ctx->pool.free(scratch, scratch_bytes);
```

The synchronize forces the host to wait for the kernel; the immediate `pool.free` returns the buffer to a free-list that the next op's `pool.alloc` can reuse. With the synchronize removed, the kernel may still be reading `scratch` when the next op tries to overwrite it — a race.

The fix is a per-`gcu_device_ctx` "deferred free" vector drained at `graph_compute` end:

```cpp
struct ggml_backend_gcu_context {
    // ... existing fields ...
    std::vector<std::pair<void *, size_t>> deferred_frees;

    void defer_free(void * p, size_t sz) {
        if (p) deferred_frees.emplace_back(p, sz);
    }
};
```

The vector is reserved to 64 entries at context construction (graphs have ~hundreds of ops; 64 covers the typical scratch count without mid-graph reallocation).

### Call-site shape

A single helper replaces every `topsStreamSynchronize + pool.free` pair:

```cpp
static void gcu_release_scratch(ggml_backend_gcu_context * ctx, void * p, size_t sz) {
    if (gcu_queued_ops_disabled()) {
        TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
        ctx->pool.free(p, sz);
    } else {
        ctx->defer_free(p, sz);
    }
}
```

Each call site becomes one line: `gcu_release_scratch(ctx, p, sz);` replacing the two-line block. Rollback semantics are obvious from the helper body.

`gcu_queued_ops_disabled()` mirrors the MVP-4a `gcu_async_disabled()` pattern — a function-local `static const bool` that reads `GGML_GCU_NO_QUEUED_OPS` once at first call.

### Sites changed

- 8 op-handler scratch-free sites (lines 783-784, 823-824, 869-870, 969-970, 1113-1115, 1225-1226, 1387-1390, 1442-1443 in `ggml/src/ggml-gcu/ggml-gcu.cpp` as of MVP-4a HEAD).
- 2 grow-on-demand frees for `zero_bias` (line 881) and `ones_n0` (line 895). These need the same treatment because a previous op's kernel could still be reading the smaller buffer when the current op decides to resize.

### `graph_compute` integration

```cpp
ggml_status status = GGML_STATUS_SUCCESS;

if (ctx->copy_event_armed) {
    TOPS_CHECK(topsStreamWaitEvent(ctx->compute_stream,
                                   ctx->last_copy_event, 0));
    ctx->copy_event_armed = false;
}

for (int i = 0; i < cgraph->n_nodes; i++) {
    ggml_tensor * node = cgraph->nodes[i];
    if (ggml_is_empty(node) || node->op == GGML_OP_NONE) continue;
    if (!gcu_compute_node(ctx, node)) {
        GGML_LOG_ERROR("%s: op %s not implemented or failed\n",
                       __func__, ggml_op_name(node->op));
        status = GGML_STATUS_FAILED;
        break;
    }
}

TOPS_CHECK(topsEventRecord(ctx->last_compute_event, ctx->compute_stream));
ctx->compute_event_armed = true;
TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));

for (auto & kv : ctx->deferred_frees) {
    ctx->pool.free(kv.first, kv.second);
}
ctx->deferred_frees.clear();

return status;
```

Two notes:

- The drain runs on every exit path including failure. Kernels submitted before the failure must complete (or the next call's first op runs against stale state). `topsStreamSynchronize` itself surfaces kernel-crash errors via `TOPS_CHECK` — matches today's behavior.
- `deferred_frees.clear()` keeps the underlying vector capacity, so subsequent graphs don't pay for re-allocation.

### Out-of-graph teardown

The dtor's existing `pool.free(zero_bias)` / `pool.free(ones_n0)` paths are unchanged — those run after the final `graph_compute` synchronize during normal teardown, with no in-flight work. The `deferred_frees` vector itself is empty at teardown (drained at the end of every `graph_compute`); no extra cleanup needed.

## Testing

- **Existing 8-test smoke suite** is the regression gate. The chained ops in those tests (e.g., MUL_MAT_MIXED's transpose+cast+matmul) already exercise multi-op queueing; if scratch lifetime is wrong, those tests will produce numerical mismatches or device errors.
- No new sub-test. The smoke suite covers the new behavior implicitly.
- **Combined-rollback runs** — verify smoke passes under all four flag combinations:
  1. defaults (both optimizations on)
  2. `GGML_GCU_NO_ASYNC_COPY=1`
  3. `GGML_GCU_NO_QUEUED_OPS=1`
  4. `GGML_GCU_NO_ASYNC_COPY=1 GGML_GCU_NO_QUEUED_OPS=1`
- **Real-model bench** — Llama 1B Q4_K_M and Qwen 0.5B F16 with `-nkvo`, r=5, before/after via `GGML_GCU_NO_QUEUED_OPS=1` toggle. Hypothesis: Llama tg moves up (compute serialized at op level was the prior investigation's standing hypothesis for the flat MVP-4a result). Qwen may or may not move — its decode is already thin.

## Risks & Mitigations

| Risk                                                                                            | Mitigation                                                       |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| Scratch reuse race between op N's deferred-freed buffer and op N+1's pool.alloc                | Defer-and-batch model: pool.free deferred to end of graph_compute, after the synchronize that drains all in-flight kernels. |
| Failure mid-graph leaks `deferred_frees` vector entries                                         | Drain runs unconditionally at graph_compute end (success and failure paths share the drain block). |
| Real model regresses unexpectedly                                                               | `GGML_GCU_NO_QUEUED_OPS=1` reverts each call site to the pre-MVP-4b sync-and-free pattern. |
| Per-op sync removal exposes a latent kernel-data-dependency bug in topsop                      | Smoke regression catches; rollback flag isolates. |
| `zero_bias` / `ones_n0` grow path frees a buffer the previous op's kernel is still reading     | Both grow-frees go through `gcu_release_scratch` (defer path). |

## Open Questions

None — design fully scoped.

## Acceptance

- All 8 existing smoke tests pass on S60 under all four flag combinations.
- llama-bench Llama 1B Q4_K_M `-nkvo` shows non-regressing tg under defaults vs `GGML_GCU_NO_QUEUED_OPS=1`.
- llama-bench Qwen 0.5B F16 `-nkvo` shows non-regressing tg under defaults vs `GGML_GCU_NO_QUEUED_OPS=1`.
- `GGML_GCU_NO_QUEUED_OPS=1` cleanly reverts to pre-MVP-4b behavior (verifiable by re-running the same bench under that flag).
- Honest perf framing in `docs/build.md`: report measured numbers; if Llama still shows flat tg, document the new finding (e.g., kernel launch overhead vs op time, or compute itself is the bottleneck) — don't oversell.
