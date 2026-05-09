# GCU MVP-4a — Async H↔D / Compute Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move host↔device transfers onto a dedicated topsrt stream so they overlap with compute on the existing stream, building on the pinned-host-buffer optimization (MVP-3d).

**Architecture:** Add a second `topsStream_t` ("copy stream") to each `gcu_device_ctx` alongside the renamed `compute_stream`. Two reusable `topsEvent_t`s (`last_copy_event`, `last_compute_event`) mediate cross-stream ordering. Implement the `set_tensor_async` / `get_tensor_async` slots on `ggml_backend_i` so the scheduler can route activation/KV traffic onto the copy stream while compute keeps running.

**Tech Stack:** topsrt (`topsStream*`, `topsEvent*` from `tops_runtime_api.h`), ggml-backend i-table, llama-bench.

**Spec:** `docs/superpowers/specs/2026-05-09-gcu-mvp4a-async-copy-overlap-design.md`

**Build/test conventions (project-specific — apply to every task):**

- Edits land on the **macOS canonical** copy at `/Users/root1/github/llama.cpp.claude`.
- Build & run happen on the **S60 sandbox** at `agent@10.12.111.158:~/claude` (passwordless ssh).
- Sync via: `rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/`
- Build target on remote: `cmake --build build --target ggml-gcu -j8` (and `test-backend-gcu` for the smoke binary).

---

## File Structure

| File                                              | Responsibility                                                              |
|---------------------------------------------------|-----------------------------------------------------------------------------|
| `ggml/src/ggml-gcu/ggml-gcu.cpp`                  | Single-file backend; all stream/event code, async i-table slots, fallback.  |
| `tests/test-backend-gcu.cpp`                      | Smoke test; new `ASYNC_OVERLAP` sub-test asserts numerical equality.        |
| `docs/build.md`                                   | MVP-4a perf table appended to the GCU known-limitations section.            |

No new files. All changes are localised to the existing single-file backend and its smoke test.

---

## Task 1: Stream + event plumbing on `gcu_device_ctx`

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp:180-235` (struct definition, ctor, dtor)
- Mechanical rename throughout the file: `ctx->stream` → `ctx->compute_stream` (41 occurrences confirmed)

This task only adds plumbing. No async dispatch yet — the smoke test must remain green afterwards.

- [ ] **Step 1: Rename `ctx->stream` → `ctx->compute_stream` everywhere**

```bash
# Sanity check first
grep -c 'ctx->stream\b' ggml/src/ggml-gcu/ggml-gcu.cpp   # expect 41
```

Use the Edit tool with `replace_all: true` against the file (old: `ctx->stream`, new: `ctx->compute_stream`).

Also rename the field declaration on line 184:

```cpp
//  before
topsStream_t stream        = nullptr;
//  after
topsStream_t compute_stream = nullptr;
```

And the matching references in the ctor (line 204) and dtor (lines 230-233):

```cpp
//  ctor (~line 204)
TOPS_CHECK(topsStreamCreate(&compute_stream));

//  dtor (~lines 230-233)
if (compute_stream) {
    TOPS_CHECK(topsStreamSynchronize(compute_stream));
    TOPS_CHECK(topsStreamDestroy(compute_stream));
    compute_stream = nullptr;
}
```

- [ ] **Step 2: Add the new fields to `ggml_backend_gcu_context` (ggml-gcu.cpp:180)**

```cpp
struct ggml_backend_gcu_context {
    int32_t      device      = 0;
    std::string  name;
    std::string  description;
    topsStream_t compute_stream = nullptr;
    topsStream_t copy_stream    = nullptr;
    gcu_pool     pool;

    // ... existing zero_bias / ones_n0 fields ...

    // MVP-4a: async H<->D plumbing.
    // last_copy_event:    recorded on copy_stream after each set_tensor_async H->D enqueue
    // last_compute_event: recorded on compute_stream at end of graph_compute
    // copy_event_armed:   guards the first wait — recording-without-arming is undefined SDK behavior
    topsEvent_t  last_copy_event    = nullptr;
    topsEvent_t  last_compute_event = nullptr;
    bool         copy_event_armed   = false;
};
```

- [ ] **Step 3: Create the new resources in the ctor (after the existing `topsStreamCreate(&compute_stream)`)**

```cpp
TOPS_CHECK(topsStreamCreate(&compute_stream));
TOPS_CHECK(topsStreamCreate(&copy_stream));
TOPS_CHECK(topsEventCreate(&last_copy_event));
TOPS_CHECK(topsEventCreate(&last_compute_event));
```

- [ ] **Step 4: Destroy the new resources in the dtor (before the existing `compute_stream` teardown)**

```cpp
if (last_copy_event) {
    TOPS_CHECK(topsEventDestroy(last_copy_event));
    last_copy_event = nullptr;
}
if (last_compute_event) {
    TOPS_CHECK(topsEventDestroy(last_compute_event));
    last_compute_event = nullptr;
}
if (copy_stream) {
    TOPS_CHECK(topsStreamSynchronize(copy_stream));
    TOPS_CHECK(topsStreamDestroy(copy_stream));
    copy_stream = nullptr;
}
if (compute_stream) {
    TOPS_CHECK(topsStreamSynchronize(compute_stream));
    TOPS_CHECK(topsStreamDestroy(compute_stream));
    compute_stream = nullptr;
}
```

- [ ] **Step 5: Sync to remote and build**

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu test-backend-gcu -j8 2>&1 | tail -3'
```

Expected: `[100%] Built target test-backend-gcu` (or similar — link step succeeds).

- [ ] **Step 6: Run the existing smoke test (regression gate)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/test-backend-gcu 2>&1 | tail -10'
```

Expected: all 7 sub-tests `ok`, final line `test-backend-gcu: PASSED`.

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-gcu/ggml-gcu.cpp
git commit -m "$(cat <<'EOF'
ggml-gcu : add copy stream + events to device context (MVP-4a/1)

Pure plumbing for the upcoming async H<->D path. Adds copy_stream,
last_copy_event, last_compute_event, copy_event_armed to gcu_device_ctx;
renames the existing stream field to compute_stream throughout. No
behavior change yet — smoke suite still green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement `set_tensor_async` and `get_tensor_async`

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp:498-510` (i-table slots + new functions just above the i-table)

- [ ] **Step 1: Add the async dispatch functions just above `static const ggml_backend_i ggml_backend_gcu_i = {`**

```cpp
// MVP-4a: async H<->D on copy_stream. The synchronous buffer-level
// set_tensor (which carries the Q-typed dequant path) is left intact —
// only F32/F16 activation and KV traffic flows through here.
//
// GGML_GCU_NO_ASYNC_COPY=1 falls back to a synchronous topsMemcpy and
// skips event arming, mirroring the GGML_GCU_NO_PINNED rollback switch.

static bool gcu_async_disabled() {
    return getenv("GGML_GCU_NO_ASYNC_COPY") != nullptr;
}

static void ggml_backend_gcu_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    if (size == 0) return;

    // Q-typed weights load via the synchronous buffer path at model init
    // and are never re-set during inference. Reaching here with a Q-tensor
    // is a logic bug — silently mis-sizing the copy would corrupt the
    // dequanted F16 destination.
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    if (gcu_async_disabled()) {
        TOPS_CHECK(topsMemcpy((char *) tensor->data + offset, data, size,
                              topsMemcpyHostToDevice));
        return;
    }

    TOPS_CHECK(topsMemcpyAsync(
        (char *) tensor->data + offset, data, size,
        topsMemcpyHostToDevice, ctx->copy_stream));
    TOPS_CHECK(topsEventRecord(ctx->last_copy_event, ctx->copy_stream));
    ctx->copy_event_armed = true;
}

static void ggml_backend_gcu_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    if (size == 0) return;
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    if (gcu_async_disabled()) {
        TOPS_CHECK(topsMemcpy(data, (const char *) tensor->data + offset, size,
                              topsMemcpyDeviceToHost));
        return;
    }

    // D->H must not race still-running compute.
    TOPS_CHECK(topsStreamWaitEvent(ctx->copy_stream, ctx->last_compute_event, 0));
    TOPS_CHECK(topsMemcpyAsync(
        data, (const char *) tensor->data + offset, size,
        topsMemcpyDeviceToHost, ctx->copy_stream));
}
```

- [ ] **Step 2: Wire the slots in the i-table (ggml-gcu.cpp:498)**

```cpp
//  before
/* .set_tensor_async     = */ nullptr,
/* .get_tensor_async     = */ nullptr,
//  after
/* .set_tensor_async     = */ ggml_backend_gcu_set_tensor_async,
/* .get_tensor_async     = */ ggml_backend_gcu_get_tensor_async,
```

Leave `set_tensor_2d_async`, `get_tensor_2d_async`, and `cpy_tensor_async` as `nullptr` — out of scope for this MVP.

- [ ] **Step 3: Sync and build**

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu test-backend-gcu -j8 2>&1 | tail -3'
```

Expected: clean build.

- [ ] **Step 4: Run smoke (correctness)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/test-backend-gcu 2>&1 | tail -10'
```

Expected: all 7 sub-tests `ok`, final line `test-backend-gcu: PASSED`. (The existing tests use the synchronous `ggml_backend_tensor_set/get` path so they don't yet exercise async — the new ASYNC_OVERLAP test in Task 4 closes that gap. This step is a regression check.)

- [ ] **Step 5: Run smoke under the rollback flag (verifies the env-var path compiles + runs)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_ASYNC_COPY=1 ./build/bin/test-backend-gcu 2>&1 | tail -10'
```

Expected: all 7 sub-tests `ok`, final line `test-backend-gcu: PASSED`.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-gcu/ggml-gcu.cpp
git commit -m "$(cat <<'EOF'
ggml-gcu : implement set_tensor_async / get_tensor_async (MVP-4a/2)

set_tensor_async issues topsMemcpyAsync(H->D, copy_stream) and records
last_copy_event. get_tensor_async waits on last_compute_event before
issuing topsMemcpyAsync(D->H, copy_stream). Q-typed tensors and size==0
calls assert / early-return — Q-tensor traffic must go through the
synchronous buffer-level set_tensor that carries the dequant path.

GGML_GCU_NO_ASYNC_COPY=1 falls back to synchronous topsMemcpy and skips
event arming, mirroring the GGML_GCU_NO_PINNED rollback switch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire events into `graph_compute` and `synchronize`

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp:472-475` (`ggml_backend_gcu_synchronize`)
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp:481-496` (`ggml_backend_gcu_graph_compute`)

- [ ] **Step 1: Update `ggml_backend_gcu_synchronize` to drain both streams**

```cpp
//  before (lines 472-475)
static void ggml_backend_gcu_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsStreamSynchronize(ctx->stream));
}

//  after
static void ggml_backend_gcu_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    // Copy first so any pending D->H drains into host memory the caller
    // may inspect; compute second.
    TOPS_CHECK(topsStreamSynchronize(ctx->copy_stream));
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
}
```

(Note: after Task 1's rename, the existing line already says `ctx->compute_stream` — only the added `copy_stream` line is new.)

- [ ] **Step 2: Update `ggml_backend_gcu_graph_compute` to wait on copy event in, record compute event out**

```cpp
//  before (lines 481-496)
static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) continue;
        if (!gcu_compute_node(ctx, node)) {
            GGML_LOG_ERROR("%s: op %s not implemented or failed\n",
                           __func__, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
    return GGML_STATUS_SUCCESS;
}

//  after
static enum ggml_status ggml_backend_gcu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_gcu_context *) backend->context;
    TOPS_CHECK(topsSetDevice(ctx->device));

    // Stitch any pending async H->D copies onto the compute stream.
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

    // Arm the compute event so subsequent get_tensor_async can wait on
    // it without blocking the host.
    TOPS_CHECK(topsEventRecord(ctx->last_compute_event, ctx->compute_stream));
    TOPS_CHECK(topsStreamSynchronize(ctx->compute_stream));
    return GGML_STATUS_SUCCESS;
}
```

- [ ] **Step 3: Sync and build**

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu test-backend-gcu -j8 2>&1 | tail -3'
```

Expected: clean build.

- [ ] **Step 4: Run smoke (regression — async wiring must not break existing path)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/test-backend-gcu 2>&1 | tail -10'
```

Expected: all 7 sub-tests `ok`, final line `test-backend-gcu: PASSED`.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-gcu/ggml-gcu.cpp
git commit -m "$(cat <<'EOF'
ggml-gcu : wire async events into graph_compute / synchronize (MVP-4a/3)

graph_compute waits on last_copy_event at entry (if armed) and records
last_compute_event at exit. synchronize drains copy_stream then
compute_stream so callers see fully-drained device state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `ASYNC_OVERLAP` smoke sub-test

**Files:**
- Modify: `tests/test-backend-gcu.cpp` (add new `test_async_overlap` function before `int main()` at line 462; register call in main)

- [ ] **Step 1: Add `test_async_overlap` just above `int main()` (around line 461)**

```cpp
// Numerical-equality test for the async H<->D path. Builds an
// (a + b) * c chain on GCU; populates the inputs via
// ggml_backend_tensor_set_async, runs graph_compute, reads back via
// ggml_backend_tensor_get_async. Asserts the result matches the CPU
// reference within F32 tolerance.
static int test_async_overlap(ggml_backend_t gcu) {
    const int64_t M = 512;
    const int64_t N = 1024;
    const size_t  n = (size_t) M * N;

    auto buft = ggml_backend_get_default_buffer_type(gcu);

    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * c   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_mul(ctx, sum, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        fprintf(stderr, "ASYNC_OVERLAP: failed to allocate tensors on GCU\n");
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> ha(n), hb(n), hc(n), hout(n), expected(n);
    fill_random_f32(ha.data(), n, 11);
    fill_random_f32(hb.data(), n, 22);
    fill_random_f32(hc.data(), n, 33);
    for (size_t i = 0; i < n; i++) expected[i] = (ha[i] + hb[i]) * hc[i];

    // Async H->D for all three inputs. Each call records last_copy_event
    // on the copy stream; graph_compute waits on the latest recording.
    ggml_backend_tensor_set_async(gcu, a, ha.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set_async(gcu, b, hb.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set_async(gcu, c, hc.data(), 0, n * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_status s = ggml_backend_graph_compute(gcu, graph);
    if (s != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ASYNC_OVERLAP: graph_compute returned %d\n", (int) s);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    // Async D->H — must wait on last_compute_event internally.
    ggml_backend_tensor_get_async(gcu, out, hout.data(), 0, n * sizeof(float));
    ggml_backend_synchronize(gcu);   // drain both streams before reading

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hout[i], expected[i], 1e-5f, 1e-5f)) {
            if (bad < 4) {
                fprintf(stderr, "ASYNC_OVERLAP: mismatch at %zu: got %f want %f\n",
                        i, hout[i], expected[i]);
            }
            bad++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (bad) {
        fprintf(stderr, "ASYNC_OVERLAP: %d/%zu mismatched\n", bad, n);
        return 1;
    }
    printf("ASYNC_OVERLAP: ok (%zu elements)\n", n);
    return 0;
}
```

- [ ] **Step 2: Register the new test in `main()` (insert after the existing `test_mul_mat_mixed` line, ~line 484)**

```cpp
//  before
rc |= test_mul_mat_mixed(gcu);

ggml_backend_free(gcu);

//  after
rc |= test_mul_mat_mixed(gcu);
rc |= test_async_overlap(gcu);

ggml_backend_free(gcu);
```

- [ ] **Step 3: Sync and build**

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target test-backend-gcu -j8 2>&1 | tail -3'
```

Expected: clean build, `[100%] Built target test-backend-gcu`.

- [ ] **Step 4: Run smoke — expect 8 sub-tests `ok`**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/test-backend-gcu 2>&1 | tail -12'
```

Expected: 7 existing sub-tests `ok`, plus `ASYNC_OVERLAP: ok (524288 elements)`, final line `test-backend-gcu: PASSED`.

- [ ] **Step 5: Run smoke under rollback flag (verify both paths produce identical results)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_ASYNC_COPY=1 ./build/bin/test-backend-gcu 2>&1 | tail -12'
```

Expected: same 8 `ok` lines, `test-backend-gcu: PASSED`.

- [ ] **Step 6: Commit**

```bash
git add tests/test-backend-gcu.cpp
git commit -m "$(cat <<'EOF'
ggml-gcu : add ASYNC_OVERLAP smoke sub-test (MVP-4a/4)

Builds an (a+b)*c chain, populates inputs via
ggml_backend_tensor_set_async, reads output via
ggml_backend_tensor_get_async, and asserts numerical equality with
the CPU reference. Covers both async paths and the cross-stream
event handoff. Verified to pass under both default and
GGML_GCU_NO_ASYNC_COPY=1 modes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Bench and document

**Files:**
- Modify: `docs/build.md` (the MVP-2 limitations section already carries the MVP-3d pinned-buffer table; append a sibling MVP-4a table immediately below it).

- [ ] **Step 1: Bench Llama 1B Q4_K_M with MVP-4a active (default)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/llama-bench \
    -m /home/agent/models/llama3.2-1b-q4km.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

Capture pp512 / tg32 / tg64 numbers.

- [ ] **Step 2: Bench Llama 1B Q4_K_M with MVP-4a disabled (rollback baseline)**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_ASYNC_COPY=1 ./build/bin/llama-bench \
    -m /home/agent/models/llama3.2-1b-q4km.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

Capture pp512 / tg32 / tg64 numbers.

- [ ] **Step 3: Bench Qwen 0.5B F16 with MVP-4a active**

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/llama-bench \
    -m /home/agent/models/qwen2.5-0.5b-instruct-fp16.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

Capture pp512 / tg32 / tg64 numbers.

- [ ] **Step 4: Bench Qwen 0.5B F16 with MVP-4a disabled**

```bash
ssh agent@10.12.111.158 'cd ~/claude && GGML_GCU_NO_ASYNC_COPY=1 ./build/bin/llama-bench \
    -m /home/agent/models/qwen2.5-0.5b-instruct-fp16.gguf --device GCU0 -nkvo 1 \
    -n 32,64 -r 5 2>&1 | tail -5'
```

Capture pp512 / tg32 / tg64 numbers.

- [ ] **Step 5: Acceptance check**

For both models, the MVP-4a-active numbers must be ≥ the MVP-4a-disabled numbers (within noise). The acceptance bar from the spec:

- Llama 1B Q4_K_M `-nkvo`: tg non-regressing, target +5–10% over the disabled run
- Qwen 0.5B F16 `-nkvo`: tg non-regressing, target +10%+ over the disabled run

If either regresses outside noise (>1 stddev): **stop and investigate**, do not proceed to commit. Likely culprits: missing `topsSetDevice` before async issue, event recorded on the wrong stream, or rollback path hit unexpectedly.

- [ ] **Step 6: Append a perf table to `docs/build.md` (next to the existing MVP-3d pinned-buffer table)**

Find the line `Set \`GGML_GCU_NO_PINNED=1\` to fall back to a normal CPU buffer.` and add a sibling MVP-4a paragraph immediately after it:

```markdown
  Async H↔D / compute overlap (`topsMemcpyAsync` on a dedicated copy stream, event-mediated handoff with the compute stream — MVP-4a) stacks on top of pinned memory. Llama 3.2 1B Q4_K_M (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4a active     | MVP-4a disabled   | uplift |
  |-------|-------------------|-------------------|--------|
  | tg32  | <fill from Step 1> | <fill from Step 2> | <%>   |
  | tg64  | <fill from Step 1> | <fill from Step 2> | <%>   |
  | pp512 | <fill from Step 1> | <fill from Step 2> | <%>   |

  Qwen 2.5 0.5B F16 (`--device GCU0 -nkvo 1`, r=5):

  | test  | MVP-4a active     | MVP-4a disabled   | uplift |
  |-------|-------------------|-------------------|--------|
  | tg32  | <fill from Step 3> | <fill from Step 4> | <%>   |
  | tg64  | <fill from Step 3> | <fill from Step 4> | <%>   |
  | pp512 | <fill from Step 3> | <fill from Step 4> | <%>   |

  Set `GGML_GCU_NO_ASYNC_COPY=1` to fall back to synchronous `topsMemcpy`.
```

Replace the `<fill from Step N>` and `<%>` placeholders with the captured numbers and computed uplift percentages.

- [ ] **Step 7: Commit**

```bash
git add docs/build.md
git commit -m "$(cat <<'EOF'
docs : record MVP-4a async H<->D / compute overlap perf numbers

llama-bench --device GCU0 -nkvo 1 -r 5 with and without
GGML_GCU_NO_ASYNC_COPY=1, both models. Stacks on top of MVP-3d
pinned host buffers — the H<->D bandwidth that pinning revealed
is now overlapped with compute.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final state

- 5 commits on `feat/ggml-gcu`, each independently buildable and smoke-passing.
- 8-test smoke suite green under both default and `GGML_GCU_NO_ASYNC_COPY=1` modes.
- `docs/build.md` carries the MVP-4a perf table.
- Branch kept as-is (no merge / PR), per the user's standing preference between MVPs.
