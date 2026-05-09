# GCU MVP-4b — Queued Ops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop the per-op `topsStreamSynchronize(compute_stream)` so kernels queue and the GCU driver can pipeline them; defer scratch frees to `graph_compute` end so the previous op's kernel can't be racing against the next op's `pool.alloc` reuse.

**Architecture:** Add `deferred_frees` vector + `defer_free` method to `gcu_device_ctx`. Single helper `gcu_release_scratch(ctx, p, sz)` replaces every `topsStreamSynchronize + pool.free` pair inside op handlers — defaults to deferring; reverts to the old sync-and-free pattern when `GGML_GCU_NO_QUEUED_OPS=1`. `graph_compute` restructures its early-return into a `break` + unconditional drain so failure paths don't leak.

**Tech Stack:** topsrt (`topsStreamSynchronize`), ggml-gcu pool free-list, single-file backend.

**Spec:** `docs/superpowers/specs/2026-05-09-gcu-mvp4b-queued-ops-design.md`

**Build/test conventions (project-specific — apply to every task):**

- Edits land on the **macOS canonical** copy at `/Users/root1/github/llama.cpp.claude`.
- Build & run on the **S60 sandbox** at `agent@10.12.111.158:~/claude` (passwordless ssh).
- Sync: `rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/`
- Build: `cmake --build build --target ggml-gcu test-backend-gcu -j8`
- Smoke: `./build/bin/test-backend-gcu`

**Line numbers reference HEAD `4501447a4`** (MVP-4a + spec commits). Tasks may shift these — verify current state before each edit.

---

## File Structure

| File                              | Responsibility                                                              |
|-----------------------------------|-----------------------------------------------------------------------------|
| `ggml/src/ggml-gcu/ggml-gcu.cpp`  | All changes. `defer_free` field on ctx, helpers, op-handler conversion, `graph_compute` restructure. |
| `docs/build.md`                   | Append MVP-4b perf table next to the existing MVP-4a table.                |

No new files. Smoke test (`tests/test-backend-gcu.cpp`) is unchanged — the existing 8-test suite covers the new behavior implicitly via chained ops.

---

## Task 1: Plumbing + op-handler conversion

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

This task lands the infrastructure (helper, env flag, deferred-free vector, graph_compute drain) **and** converts all 8 op-handler scratch sites. Keeping helper + first batch of callers in one commit avoids dead-code in trunk and gives a single bisection point for "did dropping per-op sync break anything in op-handler-allocated scratch?". Task 2 separately handles the 2 grow-on-demand sites (`zero_bias`, `ones_n0`) which have a slightly different shape.

- [ ] **Step 1: Add `deferred_frees` field + `defer_free` method to `ggml_backend_gcu_context`**

Find the struct around line 180:

```cpp
    topsEvent_t  last_copy_event    = nullptr;
    topsEvent_t  last_compute_event = nullptr;
    bool         copy_event_armed   = false;
    bool         compute_event_armed = false;
};
```

Add the field and method just before the closing brace of the struct (still inside the struct):

```cpp
    topsEvent_t  last_copy_event    = nullptr;
    topsEvent_t  last_compute_event = nullptr;
    bool         copy_event_armed   = false;
    bool         compute_event_armed = false;

    // MVP-4b: scratch frees deferred to end of graph_compute so kernels
    // queue without per-op host synchronize and the next op's pool.alloc
    // can't reuse a buffer the previous op's kernel is still reading.
    std::vector<std::pair<void *, size_t>> deferred_frees;

    void defer_free(void * p, size_t sz) {
        if (p) deferred_frees.emplace_back(p, sz);
    }
};
```

In the constructor body (around line 201, after `pool(dev)` and before the `TOPS_CHECK` calls), reserve the vector:

```cpp
explicit ggml_backend_gcu_context(int32_t dev) : device(dev), pool(dev) {
    deferred_frees.reserve(64);
    TOPS_CHECK(topsSetDevice(device));
    // ... rest unchanged
```

- [ ] **Step 2: Add `gcu_queued_ops_disabled()` env helper**

Find `gcu_async_disabled()` (somewhere around line 569 — after the buffer-type code, just above the async dispatch functions). Add the new helper directly below it:

```cpp
static bool gcu_async_disabled() {
    static const bool disabled = (getenv("GGML_GCU_NO_ASYNC_COPY") != nullptr);
    return disabled;
}

// MVP-4b: when set, op handlers keep their pre-MVP-4b sync-and-free
// pattern (per-op topsStreamSynchronize + immediate pool.free). Used for
// bisection if a real model regresses with queued ops.
static bool gcu_queued_ops_disabled() {
    static const bool disabled = (getenv("GGML_GCU_NO_QUEUED_OPS") != nullptr);
    return disabled;
}
```

- [ ] **Step 3: Add `gcu_release_scratch()` helper**

Just below `gcu_queued_ops_disabled()`, add:

```cpp
// MVP-4b: replaces every `topsStreamSynchronize + pool.free` pair inside op
// handlers. Defers the free to graph_compute's end-of-batch drain unless
// GGML_GCU_NO_QUEUED_OPS=1, in which case we restore the pre-MVP-4b
// behavior (synchronize then immediate free).
static void gcu_release_scratch(ggml_backend_gcu_context * ctx, void * p, size_t sz) {
    if (!p) return;
    if (gcu_queued_ops_disabled()) {
        TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
        ctx->pool.free(p, sz);
    } else {
        ctx->defer_free(p, sz);
    }
}
```

- [ ] **Step 4: Restructure `graph_compute` — break + unconditional drain**

Find `ggml_backend_gcu_graph_compute` (around line 521 after MVP-4a):

```cpp
static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

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
            return GGML_STATUS_FAILED;
        }
    }

    TOPS_CHECK(topsEventRecord(ctx->last_compute_event, ctx->compute_stream));
    ctx->compute_event_armed = true;
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
    return GGML_STATUS_SUCCESS;
}
```

Replace with:

```cpp
static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    if (ctx->copy_event_armed) {
        TOPS_CHECK(topsStreamWaitEvent(ctx->compute_stream,
                                       ctx->last_copy_event, 0));
        ctx->copy_event_armed = false;
    }

    ggml_status status = GGML_STATUS_SUCCESS;
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

    // MVP-4b: drain deferred frees on every exit path. Kernels submitted
    // before the failure must complete (ensured by the synchronize above)
    // and their scratch buffers must return to the pool, otherwise the
    // pool grows monotonically across calls under failure recovery.
    for (auto & kv : ctx->deferred_frees) {
        ctx->pool.free(kv.first, kv.second);
    }
    ctx->deferred_frees.clear();

    return status;
}
```

- [ ] **Step 5: Convert the 8 op-handler scratch sites**

Each follows the same pattern. For each line range below, find the two-line block:

```cpp
        TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
        ctx->pool.free(scratch, scratch_bytes);
```

(or whatever `pool.free` argument names the local site uses — common variants: `scratch`, `cast_buf`, `x_cast`, `y_f16`, `cs_dev`, `pos_dev`, `dummy_key_dev`)

Replace with the matching one-line `gcu_release_scratch` call (using the same arguments). For sites that have multiple consecutive `pool.free` calls under one synchronize (lines 1387-1390 and 1113-1115), expand into multiple `gcu_release_scratch` calls.

The 8 sites at HEAD `4501447a4`:

| Site lines | Pattern |
|---|---|
| 783-784   | `gcu_release_scratch(ctx, scratch, scratch_bytes);` |
| 823-824   | `gcu_release_scratch(ctx, scratch, scratch_bytes);` |
| 869-870   | `gcu_release_scratch(ctx, scratch, scratch_bytes);` |
| 969-970   | `gcu_release_scratch(ctx, cast_buf, cast_bytes);` |
| 1113-1115 | `gcu_release_scratch(ctx, x_cast, x_cast_bytes);` then `gcu_release_scratch(ctx, y_f16, y_f16_bytes);` (drop the standalone `topsStreamSynchronize` and the surrounding `if (x_cast)` since `gcu_release_scratch` no-ops on null) |
| 1225-1226 | `gcu_release_scratch(ctx, cast_buf, cast_bytes);` |
| 1387-1390 | three calls: `gcu_release_scratch(ctx, cs_dev, cs_bytes);`, `gcu_release_scratch(ctx, pos_dev, pos_bytes);`, `gcu_release_scratch(ctx, dummy_key_dev, dummy_bytes);` |
| 1442-1443 | `gcu_release_scratch(ctx, scratch, bytes);` |

Verify each substitution by reading the surrounding 5-10 lines to confirm the variable names match. Some sites guard with `if (x_cast)` etc.; the helper handles null itself, so drop the guard when present.

- [ ] **Step 6: Sync and build**

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu test-backend-gcu -j8 2>&1 | tail -3'
```

Expected: `[100%] Built target test-backend-gcu`.

- [ ] **Step 7: Run smoke under defaults (both optimizations on)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/test-backend-gcu 2>&1 | tail -12'
```

Expected: 8 sub-tests `ok` (incl. ASYNC_OVERLAP), final `test-backend-gcu: PASSED`.

- [ ] **Step 8: Run smoke under `GGML_GCU_NO_QUEUED_OPS=1` (rollback)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_QUEUED_OPS=1 ./build/bin/test-backend-gcu 2>&1 | tail -12'
```

Expected: 8 sub-tests `ok`, final `test-backend-gcu: PASSED`.

- [ ] **Step 9: Run smoke under both flags**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_ASYNC_COPY=1 GGML_GCU_NO_QUEUED_OPS=1 ./build/bin/test-backend-gcu 2>&1 | tail -12'
```

Expected: 8 sub-tests `ok`, final `test-backend-gcu: PASSED`.

- [ ] **Step 10: Run smoke under `GGML_GCU_NO_ASYNC_COPY=1` only (regression check that MVP-4a rollback still works)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_ASYNC_COPY=1 ./build/bin/test-backend-gcu 2>&1 | tail -12'
```

Expected: 8 sub-tests `ok`, final `test-backend-gcu: PASSED`.

- [ ] **Step 11: Commit**

```bash
git add ggml/src/ggml-gcu/ggml-gcu.cpp
git commit -m "$(cat <<'EOF'
ggml-gcu : drop per-op compute sync, defer scratch frees (MVP-4b/1)

Replaces the per-op topsStreamSynchronize + pool.free pattern in 8 op
handlers with gcu_release_scratch(), which defers the free to a
deferred_frees vector drained at graph_compute end. Kernels now queue
on compute_stream without host round-trips between ops; the
end-of-graph synchronize is the only host wait.

graph_compute's early-return on failure becomes a break + drain so the
deferred_frees vector can't leak under partial failure.

GGML_GCU_NO_QUEUED_OPS=1 reverts each call site to the pre-MVP-4b
sync-and-free pattern (mirrors GGML_GCU_NO_PINNED / NO_ASYNC_COPY).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Convert the 2 grow-on-demand sites

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

The `zero_bias` and `ones_n0` resize-on-demand frees (lines 881 and 895 at HEAD) are subtler than the per-op scratch sites: they free a buffer that previous ops may still be reading. Same fix as Task 1 — route through `gcu_release_scratch` so the actual free defers.

- [ ] **Step 1: Convert `gcu_get_zero_bias`'s grow path**

Find around line 878:

```cpp
static void * gcu_get_zero_bias(ggml_backend_gcu_context * ctx, size_t n_bytes) {
    if (ctx->zero_bias_bytes < n_bytes) {
        if (ctx->zero_bias) {
            ctx->pool.free(ctx->zero_bias, ctx->zero_bias_bytes);
        }
```

Replace the inner two-line block:

```cpp
        if (ctx->zero_bias) {
            ctx->pool.free(ctx->zero_bias, ctx->zero_bias_bytes);
        }
```

with:

```cpp
        gcu_release_scratch(ctx, ctx->zero_bias, ctx->zero_bias_bytes);
```

(`gcu_release_scratch` no-ops on null, so the `if` guard goes away.)

- [ ] **Step 2: Convert `gcu_get_ones_f32`'s grow path**

Find around line 892:

```cpp
static void * gcu_get_ones_f32(ggml_backend_gcu_context * ctx, int64_t count) {
    if (ctx->ones_n0_count >= count) return ctx->ones_n0;
    if (ctx->ones_n0) {
        ctx->pool.free(ctx->ones_n0, ctx->ones_n0_bytes);
    }
```

Replace the two-line `if (ctx->ones_n0) { ... }` block with:

```cpp
    gcu_release_scratch(ctx, ctx->ones_n0, ctx->ones_n0_bytes);
```

- [ ] **Step 3: Sync and build**

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu test-backend-gcu -j8 2>&1 | tail -3'
```

Expected: clean build.

- [ ] **Step 4: Run smoke under all 4 flag combinations**

```bash
ssh agent@10.12.111.158 'cd ~/claude && \
    ./build/bin/test-backend-gcu 2>&1 | tail -2 && \
    GGML_GCU_NO_QUEUED_OPS=1 ./build/bin/test-backend-gcu 2>&1 | tail -2 && \
    GGML_GCU_NO_ASYNC_COPY=1 ./build/bin/test-backend-gcu 2>&1 | tail -2 && \
    GGML_GCU_NO_ASYNC_COPY=1 GGML_GCU_NO_QUEUED_OPS=1 ./build/bin/test-backend-gcu 2>&1 | tail -2'
```

Expected: four `test-backend-gcu: PASSED` lines.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-gcu/ggml-gcu.cpp
git commit -m "$(cat <<'EOF'
ggml-gcu : route zero_bias / ones_n0 grow-frees through release_scratch (MVP-4b/2)

Both grow-on-demand helpers (gcu_get_zero_bias, gcu_get_ones_f32) free
a previous buffer when the requested size grows. Without per-op
synchronize, that previous buffer may still be referenced by an
in-flight kernel. Routing through gcu_release_scratch defers the
free to graph_compute end, matching the op-handler scratch sites.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Bench and document

**Files:**
- Modify: `docs/build.md` (append MVP-4b table next to the existing MVP-4a paragraph in the "Known limitations (MVP-2)" section).

- [ ] **Step 1: Bench Llama 1B Q4_K_M defaults**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/llama-bench \
    -m /home/agent/models/llama3.2-1b-q4km.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

Capture pp512 / tg32 / tg64 mean ± stddev.

- [ ] **Step 2: Bench Llama 1B Q4_K_M with `GGML_GCU_NO_QUEUED_OPS=1`**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_QUEUED_OPS=1 ./build/bin/llama-bench \
    -m /home/agent/models/llama3.2-1b-q4km.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

- [ ] **Step 3: Bench Qwen 0.5B F16 defaults**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/llama-bench \
    -m /home/agent/models/qwen2.5-0.5b-instruct-fp16.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

- [ ] **Step 4: Bench Qwen 0.5B F16 with `GGML_GCU_NO_QUEUED_OPS=1`**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_QUEUED_OPS=1 ./build/bin/llama-bench \
    -m /home/agent/models/qwen2.5-0.5b-instruct-fp16.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

- [ ] **Step 5: Acceptance check**

For each model, the MVP-4b-active numbers must be ≥ MVP-4b-disabled (within ~1 stddev). The acceptance bar from the spec is **non-regressing tg** under defaults. Compute uplift percentages signed.

If any tg number regresses outside ~1 stddev, **stop and report BLOCKED** with the regression details. Likely culprit: a missing `gcu_release_scratch` site, scratch reuse race, or kernel data-dependency the smoke didn't catch. The user will diagnose.

- [ ] **Step 6: Append a perf table to `docs/build.md`**

Find the existing MVP-4a paragraph that ends with "Multi-device support is MVP-5 work." and the "Honest read of MVP-4a's payoff" paragraph that follows. Append a new paragraph immediately after the honest-read one:

```markdown
  **MVP-4b — queued ops (drop per-op compute sync).** Replaces every op handler's `topsStreamSynchronize` + `pool.free` with a deferred-free routed through `gcu_release_scratch`. Kernels now queue on `compute_stream` without per-op host round-trips; scratch returns to the pool at `graph_compute` end after a single drain synchronize. Llama 3.2 1B Q4_K_M (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4b active   | MVP-4b disabled | uplift |
  |-------|-----------------|-----------------|--------|
  | tg32  | <FILL>          | <FILL>          | <FILL> |
  | tg64  | <FILL>          | <FILL>          | <FILL> |
  | pp512 | <FILL>          | <FILL>          | <FILL> |

  Qwen 2.5 0.5B F16 (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4b active   | MVP-4b disabled | uplift |
  |-------|-----------------|-----------------|--------|
  | tg32  | <FILL>          | <FILL>          | <FILL> |
  | tg64  | <FILL>          | <FILL>          | <FILL> |
  | pp512 | <FILL>          | <FILL>          | <FILL> |

  Set `GGML_GCU_NO_QUEUED_OPS=1` to revert each call site to the pre-MVP-4b sync-and-free pattern.
```

Replace each `<FILL>` with the actual measured value, formatted `XX.XX ± Y.YY t/s` for throughput and `+X.X%` / `-X.X%` (signed, one decimal) for uplift.

If the bench reveals a **new finding** (e.g., Llama tg moves up — confirming compute serialization was the real bottleneck — or stays flat — pointing to compute itself as the ceiling), add a one-paragraph honest framing note below the tables, similar to MVP-4a's "Honest read" paragraph. Don't oversell.

- [ ] **Step 7: Commit**

```bash
git add docs/build.md
git commit -m "$(cat <<'EOF'
docs : record MVP-4b queued-ops perf numbers

llama-bench --device GCU0 -nkvo 1 -r 5 with and without
GGML_GCU_NO_QUEUED_OPS=1, both models. Targets the compute-
serialized-at-op-level bottleneck the MVP-4a investigation
pointed to.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final state

- 3 commits on `feat/ggml-gcu`, each independently buildable and smoke-passing.
- 8-test smoke suite green under all four flag combinations.
- `docs/build.md` carries the MVP-4b perf table with honest framing.
- Branch kept as-is per the standing inter-MVP preference.
