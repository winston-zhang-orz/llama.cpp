# GCU MVP-5c — File split + RAII hygiene (Design)

**Date:** 2026-05-10
**Branch:** `feat/ggml-gcu`
**Reference:** CANN's directory layout (`ggml/src/ggml-cann/{acl_tensor.h, acl_tensor.cpp, aclnn_ops.h, aclnn_ops.cpp, common.h, ggml-cann.cpp}` + RAII patterns from CANN commit `2376b7758`)

## Goal

Split the current 3,687-LoC single-file GCU backend (`ggml/src/ggml-gcu/ggml-gcu.cpp`) into focused multi-file structure that mirrors CANN's organization. Add smart-pointer RAII for topsaten/topsrt resources. **No behavior change** — this is a structural refactor that lands the codebase in upstream-reviewable shape.

This MVP is a precondition for any upstream-PR conversation (per the gap analysis: a 3,687-LoC monolith is unreviewable). It also makes future op work easier — adding a new op-family becomes "create a file" instead of "find the right insertion point in a giant switch".

## Target file layout

| File | Purpose | Approx. LoC budget |
|---|---|---|
| `ggml/src/ggml-gcu/ggml-gcu.cpp` | Backend / device interface, scheduler glue, registry singletons. | ~700 |
| `ggml/src/ggml-gcu/common.h` | TOPS_CHECK / TOPSATEN_CHECK macros, `gcu_tensor_dims` helper, `make_topsaten_tensor`, dtype mapping, `gcu_dtype_supported` / `gcu_q_supported`, error formatting. | ~250 |
| `ggml/src/ggml-gcu/common.cpp` | Implementations matching `common.h`. | ~150 |
| `ggml/src/ggml-gcu/gcu_pool.h` | The device-memory pool class. | ~80 |
| `ggml/src/ggml-gcu/gcu_pool.cpp` | Pool implementation. | ~120 |
| `ggml/src/ggml-gcu/gcu_buffer.cpp` | Buffer-type / buffer / host-buffer interfaces (alloc, set_tensor, get_tensor, cpy_tensor, free, the Q-host-cache). | ~450 |
| `ggml/src/ggml-gcu/gcu_ops.h` | Single header declaring every `gcu_op_*` handler signature. | ~80 |
| `ggml/src/ggml-gcu/gcu_op_elementwise.cpp` | ADD, MUL, SUB, DIV, SCALE, ADD1, all the unary activations, GLU, CLAMP. | ~400 |
| `ggml/src/ggml-gcu/gcu_op_norm.cpp` | NORM, RMS_NORM, L2_NORM, GROUP_NORM. | ~250 |
| `ggml/src/ggml-gcu/gcu_op_matmul.cpp` | MUL_MAT, MUL_MAT_ID. | ~400 |
| `ggml/src/ggml-gcu/gcu_op_attn.cpp` | SOFT_MAX, ROPE, FLASH_ATTN_EXT. | ~350 |
| `ggml/src/ggml-gcu/gcu_op_misc.cpp` | CPY/DUP/CONT, GET_ROWS, SET_ROWS, CONCAT, REPEAT, SUM, SUM_ROWS, MEAN, CUMSUM, ARGMAX, TOP_K, ARGSORT, sin/cos/log/sqr/sqrt. | ~450 |
| `ggml/src/ggml-gcu/CMakeLists.txt` | Update to compile the new sources. | ~30 |

Total: ~3,700 LoC across 12 files instead of 3,687 in one. Each file is independently reviewable.

## RAII hygiene

CANN commit `2376b7758` introduced smart-pointer wrappers for ACL handles. Mirror with two minimal wrappers in `common.h`:

```cpp
// Stream wrapper — destroys on scope exit.
class gcu_stream {
public:
    explicit gcu_stream(int device);
    ~gcu_stream();
    gcu_stream(const gcu_stream &) = delete;
    gcu_stream & operator=(const gcu_stream &) = delete;
    gcu_stream(gcu_stream && other) noexcept;
    operator topsStream_t() const { return s_; }
private:
    topsStream_t s_ = nullptr;
};

// Event wrapper — same shape.
class gcu_event {
public:
    explicit gcu_event(unsigned flags = topsEventDisableTiming);
    ~gcu_event();
    gcu_event(const gcu_event &) = delete;
    gcu_event & operator=(const gcu_event &) = delete;
    gcu_event(gcu_event && other) noexcept;
    operator topsEvent_t() const { return e_; }
private:
    topsEvent_t e_ = nullptr;
};
```

`ggml_backend_gcu_context` then holds these by value rather than raw `topsStream_t` / `topsEvent_t`. Construction allocates; destruction frees automatically — eliminating the manual cleanup blocks at lines ~250-280 of the current dtor.

Pool device pointers and Q-host-cache scratch don't need wrappers; they're already managed by the pool / map.

## Refactor sequence

Per the skill's "no behavior change" discipline, the refactor lands as a sequence of small, individually-verifiable commits. Each commit must keep all 4 rollback flag combos passing smoke and not regress Gemma 4 / Llama 1B canaries.

1. **Commit 1 — common.h / common.cpp.** Move TOPS_CHECK / TOPSATEN_CHECK / dtype mapping / `gcu_tensor_dims` / `make_topsaten_tensor` / `gcu_dtype_supported` / `gcu_q_supported` / `gcu_q_dequantize_to_f32` into the new files. Update `ggml-gcu.cpp` to `#include "common.h"`. Build, smoke, canary.
2. **Commit 2 — gcu_pool.{h,cpp}.** Move the `gcu_pool` class. Update includes. Build, smoke, canary.
3. **Commit 3 — gcu_buffer.cpp.** Move buffer/buffer-type/host-buffer code (including the Q-host-cache from `cf2fdb5e7`). Build, smoke, canary.
4. **Commit 4 — gcu_op_elementwise.cpp + gcu_ops.h.** Move all element-wise + unary + GLU + CLAMP handlers. Build, smoke, canary.
5. **Commit 5 — gcu_op_norm.cpp.** Move NORM / RMS_NORM / L2_NORM / GROUP_NORM. Build, smoke, canary.
6. **Commit 6 — gcu_op_matmul.cpp.** Move MUL_MAT + MUL_MAT_ID. Build, smoke, canary. (Largest individual move.)
7. **Commit 7 — gcu_op_attn.cpp.** Move SOFT_MAX / ROPE / FLASH_ATTN_EXT.
8. **Commit 8 — gcu_op_misc.cpp.** Move everything else.
9. **Commit 9 — RAII wrappers.** Add `gcu_stream` and `gcu_event` to `common.h`, swap context fields, drop manual cleanup.
10. **Commit 10 — CMakeLists.txt.** Update to glob the new sources. (May land earlier if needed for the build to pick up the new files.)

Each commit must compile and pass smoke + canary independently. If any one regresses, the failure is localized to that file move.

## Testing

This is a no-behavior-change refactor. The existing 76+ smoke tests are the entire correctness gate. Per the skill:

1. **L1-L4 smoke** — must remain green at every commit.
2. **L5 real-model** — Gemma 4 26B A4B and Llama 3.2 1B Q4_K_M canaries must produce "Paris" at every commit.
3. **All 4 rollback flag combos** must pass smoke at every commit.

No new smoke tests added. The structural change is verified by the existing test surface.

## Acceptance

- 3,687-LoC `ggml-gcu.cpp` reduced to ~700 LoC, with the rest distributed across the new files per the layout above.
- All 76+ smoke tests green under all 4 flag combos at the final commit.
- Gemma 4 26B A4B and Llama 1B Q4_K_M still produce "Paris".
- `gcu_stream` and `gcu_event` RAII wrappers replace raw handles in `ggml_backend_gcu_context`; the dtor drops the corresponding manual cleanup blocks.
- No behavior change. No perf regression (verify with one llama-bench run on Gemma 4 — tg should stay at ~14 t/s).

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Subtle linker-order issue from splitting (e.g. static initializers running differently) | One-step-at-a-time commits; build + smoke after each |
| Forgotten `static` qualifier when moving a function across files; symbol leak | Compile with `-Wmissing-prototypes` if available, or grep for `extern` after each move |
| RAII wrapper move-constructor leaving the source in invalid state | Use the rule-of-five pattern; smoke catches double-frees |
| CMake glob pattern misses a new .cpp file | Explicit file list in CMakeLists.txt rather than glob |
| Forward-declarations in gcu_ops.h drift from impls | One header included by every op file; mismatched signatures fail at link, not at runtime |

## Open questions

- Whether to mirror CANN's exact filenames (`acl_tensor.h` etc.) or use GCU-flavoured names (`gcu_tensor.h`). Recommend GCU-flavoured for clarity; the structural mirroring is what matters, not the names.
- Whether RAII wrappers (commit 9) should land before or after the file moves (1-8). Commit 9 last per the sequence above — keeps each move minimal; the RAII switch is its own clean commit.

## Out of scope

- Renaming public symbols (would break ggml-backend interface).
- Changing the public API of the backend (e.g. extra entry points, config knobs).
- Performance work — this MVP is structural only.
- Documentation polish beyond updating include-path notes if any are wrong.
