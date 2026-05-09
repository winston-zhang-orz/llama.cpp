# GCU MVP-3a Implementation Plan — Native Quantized MUL_MAT (dequant-to-F16)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Q-typed weight tensors (Q4_0, Q8_0, Q4_K) load to GCU as F16 (one-time host dequant on `set_tensor`); MUL_MAT for those tensors runs on GCU via the existing F16-weight path. End-to-end Q4_K_M model on GCU beats CPU baseline at token generation.

**Architecture:** All changes in `ggml/src/ggml-gcu/ggml-gcu.cpp`. Buffer-type's `get_alloc_size` over-reports as F16 size for Q-typed tensors; buffer's `init_tensor` rewrites `nb[]` to F16 strides; `set_tensor` runs `ggml_get_type_traits(type)->to_float()` on host then `ggml_fp32_to_fp16_row` and uploads F16 bytes. `gcu_op_mul_mat` treats Q-typed weights as F16 by reading `t->data` with F16 strides.

**Tech Stack:** Existing — C++17, ggml backend interface, topsaten + topsrt SDK on `/opt/tops`, S60 build/run via `agent@10.12.111.158:~/claude`. New API used: `ggml_get_type_traits(type)->to_float` (in libggml-base, no extra link).

**Spec:** `docs/superpowers/specs/2026-05-09-gcu-mvp3a-design.md`.

---

## Conventions

Same as MVP-2:

```bash
# Sync to remote
rsync -a --delete --exclude=".git" --exclude="build/" --exclude="*.o" ./ agent@10.12.111.158:~/claude/

# Incremental rebuild
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3'

# Test one op
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/test-backend-ops -b GCU0 -o MUL_MAT > /tmp/mm.log 2>&1; echo OK=$(grep -cE "OK\b" /tmp/mm.log) FAIL=$(grep -cE "FAIL\b|ERR =" /tmp/mm.log)'
```

The `~/.ssh/config` entry from earlier suppresses the PQ-KEX warning so output is clean.

llama.cpp commit-message convention: `area : short imperative summary`. Write your own messages — the suggestions below are starting points.

---

## Resolved verification points (from spec §8)

These came from reading the ggml headers before plan-writing. Refer back here in the relevant phase.

1. **Generic dequantizer entry**: `ggml_get_type_traits(type)->to_float` (in `ggml.h:2800`). Exported from `libggml-base.so`. No `libggml-cpu` link required — eliminates the dependency-cycle risk in spec §7 R2.
2. **`ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix)`** (in `ggml.h:2764`): exported from `libggml-base.so`. Used in Phase E to build a Q4_K test tensor.
3. **`ggml_fp32_to_fp16_row(src, dst, n)`** (in `ggml.h`): already used in MVP-2's smoke test (`test_mul_mat_mixed`).
4. **Per-type dequantizers also available** (`dequantize_row_q4_0`, etc. in `ggml-quants.h`) — but we use the generic `to_float` indirection for cleaner code.

---

## File structure

| File | Change | Reason |
|---|---|---|
| `ggml/src/ggml-gcu/ggml-gcu.cpp` | Modify (every phase A-D) | Single-file backend |
| `tests/test-backend-gcu.cpp` | Modify (Phase E) | Add `test_q4_k_round_trip` |
| `docs/build.md` | Modify (Phase F) | Refresh op coverage + perf |

No new files.

---

## Phase A — Q-format support helpers

By the end of Phase A, the backend file has `gcu_q_supported(type)` and `gcu_q_dequantize_to_f32(type, src, dst, n)` helpers. They aren't used yet but compile.

### Task A1: Add helpers

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

- [ ] **Step 1:** Add the helpers in the `// === Tensor mapping ===` section (right after `ggml_to_topsaten_dtype`, before `make_topsaten_tensor`):

```cpp
// MVP-3a: Q-typed weight tensors are dequantized to F16 at set_tensor
// time and stored as F16 on the device. This helper says which formats
// we accept; non-supported Q-types fall back to CPU via supports_op.
static bool gcu_q_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K;
}

// Generic dequantize-to-F32 via ggml's per-type traits. Works for any Q-type
// ggml supports; we use it only for those gcu_q_supported() accepts.
static void gcu_q_dequantize_to_f32(ggml_type type, const void * src,
                                    float * dst, int64_t n_elem) {
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    GGML_ASSERT(tt->to_float != nullptr);
    tt->to_float(src, dst, n_elem);
}
```

- [ ] **Step 2:** Sync + build to confirm compile. The helpers are not yet used so this is a syntax check.

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3'
```
Expected: `[100%] Built target ggml-gcu` with no errors.

- [ ] **Step 3:** Commit (suggested: `ggml-gcu : add Q-format support helpers`)

---

## Phase B — Buffer-type allocation + tensor stride rewrite

By the end of Phase B, allocating a Q4_K weight tensor on the GCU buffer reserves F16-equivalent bytes and rewrites the tensor's `nb[]` to F16 strides. No upload path yet.

### Task B1: get_alloc_size for Q-types

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

- [ ] **Step 1:** Add the over-allocation function. Locate the `// === Buffer type ===` section (around `ggml_backend_gcu_buffer_type_get_alignment`) and add right after it:

```cpp
// For Q-typed weight tensors we store F16 on the device. Over-report the
// allocation size accordingly so the slab is large enough to hold the
// dequantized F16 bytes.
static size_t ggml_backend_gcu_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t /*buft*/, const ggml_tensor * t) {
    if (gcu_q_supported(t->type)) {
        return (size_t) ggml_nelements(t) * sizeof(uint16_t);
    }
    return ggml_nbytes(t);  // default: byte-for-byte
}
```

- [ ] **Step 2:** Wire it into the buffer-type i-table. Find:

```cpp
static const ggml_backend_buffer_type_i ggml_backend_gcu_buffer_type_i = {
    /* .get_name        = */ ggml_backend_gcu_buffer_type_name,
    /* .alloc_buffer    = */ ggml_backend_gcu_buffer_type_alloc_buffer,
    /* .get_alignment   = */ ggml_backend_gcu_buffer_type_get_alignment,
    /* .get_max_size    = */ ggml_backend_gcu_buffer_type_get_max_size,
    /* .get_alloc_size  = */ nullptr,    // default = ggml_nbytes
    /* .is_host         = */ nullptr,    // device buffer (not host-accessible)
};
```

Replace the `nullptr` for `.get_alloc_size` with `ggml_backend_gcu_buffer_type_get_alloc_size`.

- [ ] **Step 3:** Build + verify

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3 && ./build/bin/test-backend-ops -b GCU0 -o MUL_MAT > /tmp/mm.log 2>&1; echo MUL_MAT OK=$(grep -cE "OK\b" /tmp/mm.log) FAIL=$(grep -cE "FAIL\b|ERR =" /tmp/mm.log)'
```
Expected: still `OK=42 FAIL=0` from MVP-2. Q-typed cases still report "not supported" because `supports_op` hasn't been updated. The over-alloc function is now wired but only triggers when an op accepts a Q-typed tensor (which it doesn't yet).

- [ ] **Step 4:** Commit (suggested: `ggml-gcu : over-allocate F16 size for Q-typed tensors`)

### Task B2: init_tensor rewrites nb[] to F16 strides

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

- [ ] **Step 1:** Replace the existing `ggml_backend_gcu_buffer_init_tensor` (currently a no-op) with a stride-rewriting version. Locate it (search for `ggml_backend_gcu_buffer_init_tensor`) and replace:

```cpp
static enum ggml_status ggml_backend_gcu_buffer_init_tensor(
        ggml_backend_buffer_t /*buffer*/, ggml_tensor * tensor) {
    if (gcu_q_supported(tensor->type)) {
        // Stored as F16 on device; strides must reflect F16 layout so that
        // ggml's downstream stride math (used by gcu_op_mul_mat etc.)
        // indexes correctly. Element count and ne[] are unchanged.
        const int64_t bpe_f16 = (int64_t) sizeof(uint16_t);
        tensor->nb[0] = bpe_f16;
        tensor->nb[1] = tensor->ne[0] * tensor->nb[0];
        tensor->nb[2] = tensor->ne[1] * tensor->nb[1];
        tensor->nb[3] = tensor->ne[2] * tensor->nb[2];
    }
    return GGML_STATUS_SUCCESS;
}
```

- [ ] **Step 2:** Build + verify

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3 && ./build/bin/test-backend-ops -b GCU0 -o MUL_MAT > /tmp/mm.log 2>&1; echo MUL_MAT OK=$(grep -cE "OK\b" /tmp/mm.log) FAIL=$(grep -cE "FAIL\b|ERR =" /tmp/mm.log)'
```
Expected: still `OK=42 FAIL=0`. Q-typed cases still skipped. We're staging the storage layout; supports_op hasn't been opened yet.

- [ ] **Step 3:** Commit (suggested: `ggml-gcu : rewrite nb[] to F16 strides for Q-typed tensors`)

---

## Phase C — set_tensor dequant-then-upload path

By the end of Phase C, uploading a Q-typed tensor's bytes via `ggml_backend_tensor_set` runs the host dequantizer and uploads F16 to the device. Still no `supports_op` change.

### Task C1: Dequant-and-upload in set_tensor

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

- [ ] **Step 1:** Replace the existing `ggml_backend_gcu_buffer_set_tensor` with a Q-aware version. Locate it (search for `ggml_backend_gcu_buffer_set_tensor`) and replace:

```cpp
static void ggml_backend_gcu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    auto * bctx = (ggml_backend_gcu_buffer_context *) buffer->context;
    TOPS_CHECK(topsSetDevice(bctx->ctx->device));

    // Q-typed full-tensor upload: dequantize to F16 on host, upload F16.
    if (gcu_q_supported(tensor->type)) {
        // We require a full-tensor write here because Q packing is per-row
        // (and per super-block for Q4_K), so partial writes don't translate
        // cleanly. ggml's model loader uses full-tensor writes for weights.
        GGML_ASSERT(offset == 0);
        GGML_ASSERT(size == ggml_row_size(tensor->type, ggml_nelements(tensor)));

        const int64_t n_elem = ggml_nelements(tensor);

        std::vector<float>       host_f32(n_elem);
        std::vector<ggml_fp16_t> host_f16(n_elem);

        gcu_q_dequantize_to_f32(tensor->type, data, host_f32.data(), n_elem);
        ggml_fp32_to_fp16_row(host_f32.data(), host_f16.data(), n_elem);

        TOPS_CHECK(topsMemcpy(tensor->data, host_f16.data(),
                              (size_t) n_elem * sizeof(ggml_fp16_t),
                              topsMemcpyHostToDevice));
        return;
    }

    // Default: byte-for-byte upload (existing behavior).
    TOPS_CHECK(topsMemcpy((char *) tensor->data + offset, data, size,
                          topsMemcpyHostToDevice));
}
```

- [ ] **Step 2:** Build + verify

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3 && ./build/bin/test-backend-ops -b GCU0 -o MUL_MAT > /tmp/mm.log 2>&1; echo MUL_MAT OK=$(grep -cE "OK\b" /tmp/mm.log) FAIL=$(grep -cE "FAIL\b|ERR =" /tmp/mm.log)'
```
Expected: still `OK=42 FAIL=0`. The dequant path exists but isn't exercised yet (supports_op still rejects).

- [ ] **Step 3:** Commit (suggested: `ggml-gcu : dequantize Q-typed tensors to F16 at set_tensor`)

---

## Phase D — MUL_MAT supports_op + compute reinterprets Q as F16

By the end of Phase D, MUL_MAT with a Q-typed weight runs on GCU, treating the weight bytes as F16. test-backend-ops Q-typed MUL_MAT cases transition from "not supported" to OK.

### Task D1: Open the supports_op gate for Q-typed weight

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

- [ ] **Step 1:** Find the MUL_MAT case in `ggml_backend_gcu_device_supports_op` and extend the dtype-combo check. The existing block looks like:

```cpp
case GGML_OP_MUL_MAT: {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (!w || !x) return false;
    bool ok = false;
    if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ok = true;
    if (w->type == GGML_TYPE_F16 &&
        (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
        (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;
    if (!ok) return false;
    /* … shape checks … */
}
```

Add a third clause before `if (!ok) return false;`:

```cpp
    // MVP-3a: Q-typed weight, stored on device as F16 via dequant-on-load.
    if (gcu_q_supported(w->type) &&
        (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32) &&
        (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32)) ok = true;
```

- [ ] **Step 2:** Build but don't test yet — `gcu_op_mul_mat` will assert on the new path until D2.

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3'
```

(Don't run test-backend-ops between this step and D2.)

### Task D2: Reinterpret Q-typed weight as F16 in gcu_op_mul_mat

**Files:**
- Modify: `ggml/src/ggml-gcu/ggml-gcu.cpp`

- [ ] **Step 1:** Find `gcu_op_mul_mat`. The existing function has two main branches — all-F32 fast path and the F16-weight mixed-dtype path that ends with `GGML_ASSERT(wt == GGML_TYPE_F16);`. Update the assert and the branch test to accept Q-typed weights as F16.

The current top of the function:

```cpp
static bool gcu_op_mul_mat(ggml_backend_gcu_context * ctx, ggml_tensor * dst) {
    ggml_tensor * w = dst->src[0];
    ggml_tensor * x = dst->src[1];

    const ggml_type wt = w->type;
    const ggml_type xt = x->type;
    const ggml_type ot = dst->type;
    /* … */
}
```

Add a normalization pass right after the type reads:

```cpp
    // For Q-typed weights, the device buffer holds F16 bytes (per spec
    // MVP-3a). Treat them as F16 for everything downstream.
    const ggml_type wt_eff = gcu_q_supported(wt) ? GGML_TYPE_F16 : wt;
```

Then replace every later use of `wt` in branch tests and tensor-construction (only in this function) with `wt_eff`. Specifically:

1. The all-F32 fast-path test uses `wt == GGML_TYPE_F32` — unchanged (Q-types don't satisfy it).
2. The `GGML_ASSERT(wt == GGML_TYPE_F16);` line becomes `GGML_ASSERT(wt_eff == GGML_TYPE_F16);`
3. The `topsatenTensor rhs_f16(... ggml_to_topsaten_dtype(wt) ...)` (if any) → use `TOPSATEN_DATA_FP16` literal or `ggml_to_topsaten_dtype(wt_eff)`.

Concretely, here is the F16-weight branch as it should look after the change:

```cpp
    // F16-weight path: cast input to F16 if needed, run F16 Linear, cast
    // output back to dst dtype. Q-typed weights take this path with their
    // device bytes interpreted as F16.
    GGML_ASSERT(wt_eff == GGML_TYPE_F16);
    const int64_t M = dst->ne[0];
    const int64_t K = w->ne[0];
    const int64_t N = x->ne[1];

    // … the rest of the existing F16 path is unchanged: cast x to F16 if
    // F32, run topsatenLinear with F16 weight (using w->data which holds
    // F16 bytes for Q-typed weights), cast output back if needed.
```

The key insight: `w->nb[]` was rewritten to F16 strides at `init_tensor` time, and `w->data` holds F16 bytes. So the existing F16 branch reads it correctly without further changes once we use `wt_eff` for the dtype tag.

- [ ] **Step 2:** Build + run `test-backend-ops -o MUL_MAT`

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target ggml-gcu -j8 2>&1 | tail -3 && ./build/bin/test-backend-ops -b GCU0 -o MUL_MAT > /tmp/mm.log 2>&1; echo MUL_MAT OK=$(grep -cE "OK\b" /tmp/mm.log) FAIL=$(grep -cE "FAIL\b|ERR =" /tmp/mm.log) skip=$(grep -cE "not supported" /tmp/mm.log); grep -E "FAIL\b|ERR =" /tmp/mm.log | head -5'
```
Expected: **OK count goes up significantly** (Q4_0, Q8_0, Q4_K test cases now run on GCU). FAIL stays 0. The exact OK count depends on which Q-types test-backend-ops exercises — anything 100+ OK is good progress; the Q-types we explicitly skip (Q5_K, Q6_K, etc.) should still report "not supported".

- [ ] **Step 3:** If failures arise, the most likely culprits in rough order:
  - **Stride mismatch** from R1 in spec §7. Add a defensive nb[] rewrite at the top of gcu_op_mul_mat for Q-typed weights (mirror Phase B Task B2's logic). If this fixes it, also commit the defense.
  - **F16 weight tensor size mismatch** — if `ggml_nbytes(w)` reports the Q4 size somewhere our code uses, we'd read past valid memory. Search `ggml_nbytes(w)` in our backend for any callers in MUL_MAT paths.
  - **topsatenLinear rejecting** — unlikely (existing F16 path works for `[K, M]` weight shapes), but if it does, dump the dims/strides and check vs the F16-weight cases that were already passing.
  - Iterate until 0 FAIL.

- [ ] **Step 4:** Commit (suggested: `ggml-gcu : MUL_MAT accepts Q-typed weights via F16 reinterpretation`)

---

## Phase E — Q4_K round-trip smoke test

By the end of Phase E, `tests/test-backend-gcu` includes a sub-test that builds a Q4_K weight on host, allocates it on GCU, runs MUL_MAT, and compares with a CPU reference (allowing for Q4 quantization noise).

### Task E1: test_q4_k_round_trip

**Files:**
- Modify: `tests/test-backend-gcu.cpp`

- [ ] **Step 1:** Add the new sub-test before `int main()`:

```cpp
static int test_q4_k_round_trip(ggml_backend_t gcu) {
    // Q4_K rows must be a multiple of the super-block size (256). Use
    // K=512, M=128 so the weight has 2 super-blocks per row.
    const int64_t K = 512, M = 128, N = 64;

    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "Q4_K_RT: alloc failed\n"); ggml_free(ctx); return 1; }

    // Build random F32 weights, quantize to Q4_K bytes, upload to GCU.
    std::vector<float> ha_f32((size_t) K * M);
    fill_random_f32(ha_f32.data(), ha_f32.size(), 31);
    const size_t q4k_bytes = ggml_row_size(GGML_TYPE_Q4_K, K) * M;
    std::vector<uint8_t> ha_q4k(q4k_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, ha_f32.data(), ha_q4k.data(),
                        /*start=*/0, /*nrows=*/M, /*n_per_row=*/K, nullptr);

    // Reference: dequantize the Q4_K bytes back to F32 (so the comparison
    // is against the same lossy weight values the GCU sees).
    std::vector<float> ha_f32_dequant((size_t) K * M);
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q4_K);
    tt->to_float(ha_q4k.data(), ha_f32_dequant.data(), (int64_t) K * M);

    // Inputs.
    std::vector<float> hb((size_t) K * N), hc((size_t) M * N), expected((size_t) M * N);
    fill_random_f32(hb.data(), hb.size(), 32);

    // Reference: c = a^T @ b in F32 using the dequantized weights.
    for (int64_t nn = 0; nn < N; nn++) {
        for (int64_t mm = 0; mm < M; mm++) {
            float acc = 0.0f;
            const float * apf = ha_f32_dequant.data() + mm * K;
            const float * bpf = hb.data()             + nn * K;
            for (int64_t k = 0; k < K; k++) acc += apf[k] * bpf[k];
            expected[mm + nn * M] = acc;
        }
    }

    ggml_backend_tensor_set(a, ha_q4k.data(), 0, q4k_bytes);
    ggml_backend_tensor_set(b, hb.data(),     0, hb.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Q4_K_RT: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, hc.size() * sizeof(float));

    // Q4_K + F16 accumulation has more noise than F16-only matmul.
    // Tolerance is generous to match the SDK's typical F16 accumulation.
    int bad = 0;
    for (size_t i = 0; i < hc.size(); i++) {
        if (!close_enough(hc[i], expected[i], 0.5f, 0.05f)) {
            if (bad < 5) fprintf(stderr, "Q4_K_RT mismatch idx=%zu got=%f want=%f\n",
                                 i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "Q4_K_RT: %d mismatches\n", bad); return 1; }
    printf("Q4_K_ROUND_TRIP: ok (%zu elements)\n", hc.size());
    return 0;
}
```

- [ ] **Step 2:** Wire it into `main`. Find the existing list of `rc |= test_*(gcu);` lines and add:

```cpp
    rc |= test_q4_k_round_trip(gcu);
```

(Place after `test_mul_mat_mixed`.)

- [ ] **Step 3:** Build + run

```bash
rsync -a --delete --exclude='.git' --exclude='build/' --exclude='*.o' ./ agent@10.12.111.158:~/claude/
ssh agent@10.12.111.158 'cd ~/claude && cmake --build build --target test-backend-gcu -j8 2>&1 | tail -3 && ./build/bin/test-backend-gcu; echo exit=$?'
```
Expected: all existing sub-tests still pass, plus `Q4_K_ROUND_TRIP: ok (8192 elements)`. Final line `test-backend-gcu: PASSED`.

- [ ] **Step 4:** If the Q4_K test fails, the most likely cause is the strides issue (Phase D Step 3). Re-check, fix, then retest.

- [ ] **Step 5:** Commit (suggested: `ggml-gcu : add Q4_K round-trip smoke test`)

---

## Phase F — Real-model bench + docs

By the end of Phase F, the Qwen 0.5B Q4_K_M bench is captured and `docs/build.md` reflects MVP-3a coverage.

### Task F1: Bench

- [ ] **Step 1:** Run the bench

```bash
ssh agent@10.12.111.158 'cd ~/claude && ./build/bin/llama-bench -m /home/agent/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --device GCU0 -nkvo 1 -p 64,128,512 -n 16,64 -r 3 2>&1 | tail -10'
```

Capture the output. Compare with the MVP-2 baseline (Q4 GCU was 273 t/s pp64, 25.4 t/s tg32 — hopelessly slower than Q4 CPU's 463/70.6).

- [ ] **Step 2:** Pass criterion: tg64 ≥ 70 t/s on GCU. If yes, MVP-3a's success goal is met. If no, document the gap and the dominant bottleneck (almost certainly per-token H↔D for the KV cache, which is MVP-3b's domain).

### Task F2: Docs refresh

**Files:**
- Modify: `docs/build.md`

- [ ] **Step 1:** Update the GCU section's "Operator coverage" subsection. Find the line:

```
- Linear: `MUL_MAT` (F32×F32→F32 fast path; F16-weight × {F16,F32} → {F16,F32} via cast)
```

Replace with:

```
- Linear: `MUL_MAT` (F32×F32→F32 fast path; F16-weight × {F16,F32} → {F16,F32} via cast; Q4_0 / Q8_0 / Q4_K weights via F16 dequant-on-load)
```

- [ ] **Step 2:** Update the "Known limitations" section. Replace the bullet:

```
- Quantized weights (Q4_K, Q8_0, etc.) stay on CPU. Models load fine; compute on quantized tensors is CPU-bound.
```

with:

```
- Q4_0, Q8_0, and Q4_K weight tensors are dequantized to F16 at model-load time (one-time host cost) and stored as F16 on GCU (2-4× the on-disk size). Other Q-types (Q5_K, Q6_K, Q3_K, etc.) still stay on CPU. Native quantized matmul via topsatenLinearQuant is a future optimization.
```

- [ ] **Step 3:** Append the new bench numbers to the Performance subsection (refer to Task F1's output). Layout matches the existing F16 table.

- [ ] **Step 4:** Commit (suggested: `docs : refresh GCU section for MVP-3a quantized MUL_MAT`)

---

## Self-review

**Spec coverage check:**

- §2 strategy (dequant-on-load, F16 storage) → Phases A-D ✓
- §4.1 Q-format support table (Q4_0, Q8_0, Q4_K) → Task A1 `gcu_q_supported` enumerates exactly these ✓
- §4.2 Buffer-type changes (`get_alloc_size`) → Task B1 ✓
- §4.3 Buffer changes (`init_tensor` and `set_tensor`) → Tasks B2, C1 ✓
- §4.4 MUL_MAT changes → Tasks D1, D2 ✓
- §4.5 Refusal of non-MUL_MAT Q-tensor ops — handled implicitly (no other supports_op case mentions Q-types) ✓
- §5 Testing → Phase E (smoke), Phase F (real bench), test-backend-ops covered in Task D2 ✓
- §6 Phases — match the spec's phase letters ✓
- §7 Risks — R1 (stride clobber) addressed in Task D2 Step 3; R2 (cyclic dep) eliminated by using `ggml_get_type_traits` from libggml-base; R3 (memory) noted in docs (Task F2); R4 (Q embedding) implicit ✓
- §8 Verification points — resolved in plan header ✓

**Placeholder scan:** no "TBD" or "fill in" anywhere. All code snippets are complete; commands have explicit expected outputs.

**Type consistency:** `gcu_q_supported`, `gcu_q_dequantize_to_f32`, `wt_eff` are referenced consistently across phases. `ggml_type_traits` (struct) and `ggml_get_type_traits` (function) match the public ggml API.

Plan is ready.
