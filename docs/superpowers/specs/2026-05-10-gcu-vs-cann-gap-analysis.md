# GCU vs CANN Gap Analysis

**Date:** 2026-05-10
**Branch:** `feat/ggml-gcu`
**Status:** research report — drives next-MVP planning
**Inputs:**
- `upstream/master` HEAD `f3c3e0e9a` (fetched 2026-05-10)
- 127 commits touching `ggml/src/ggml-cann/` since 2024-07-17 (#6035)
- GCU backend at `ggml/src/ggml-gcu/ggml-gcu.cpp` (3 687 LoC, 63 commits)
- Existing roadmap: `docs/superpowers/specs/2026-05-09-gcu-op-coverage-roadmap.md`

---

## 1. Executive summary

- **GCU dispatches ~50 ggml ops**; CANN dispatches **~64** (MVP-2/3/4 work has closed most of the LLM-inference op-set gap, but `ABS / SGN / NEG / STEP / EXP / ELU / SOFTPLUS / GELU_ERF`, `GROUP_NORM` variants and the *alternative-architecture* family — `SSM_CONV`, `GATED_LINEAR_ATTN`, `OUT_PROD`, `CROSS_ENTROPY_LOSS`, `CONV_TRANSPOSE_1D`, `IM2COL`, `PAD`, `UPSCALE`, `ARANGE`, `TIMESTEP_EMBEDDING`, `POOL_2D` — are all present on CANN and absent on GCU).
- **The architectural gap is bigger than the op gap.** CANN has BF16 (`1af9dab32`), captured ACL graph + LRU graph cache (`224145325`, `28b5f190e`, `ce7a6dc0f`), async op submission (`7a395f67a`), backend-agnostic tensor parallelism integration (`d6f303004`), per-device pinned host buffers, pre-built CI matrix on two SoCs × two arches × two graph modes (`.github/workflows/build-cann.yml`), Dockerfile (`.devops/cann.Dockerfile`), build doc (`docs/backend/CANN.md`) and op-coverage CSV (`docs/ops/CANN.csv`, 18 855 rows). GCU has none of these.
- **Code organisation is a single-file 3 687 LoC monolith** vs CANN's six-file split (`acl_tensor.{h,cpp}` + `aclnn_ops.{h,cpp}` + `common.h` + `ggml-cann.cpp`). Splitting will be required for upstream review even if no other change happens.

---

## 2. Op coverage gap

CANN's `supports_op` switch (extracted from `upstream/master:ggml/src/ggml-cann/ggml-cann.cpp`) accepts the ops below. Cross-referenced against GCU's switch in `ggml/src/ggml-gcu/ggml-gcu.cpp:3034`. Only ops where CANN says yes and GCU says no (or partial) are listed.

| Op | CANN | GCU today | Roadmap tranche | Priority |
|---|---|---|---|---|
| `UNARY:ABS` | yes (#6035) | no | A (extension) | **P1** — common in residuals/loss; trivial wrap |
| `UNARY:SGN` | yes | no | A (extension) | P2 |
| `UNARY:NEG` | yes | no | A (extension) | P2 |
| `UNARY:STEP` | yes | no | A (extension) | P2 |
| `UNARY:EXP` | yes | no | A (extension) | P1 — used in some MoE routers |
| `UNARY:ELU` | yes (`6e1c4cebd`) | no | A | P2 |
| `UNARY:GELU_ERF` | yes | no | A | **P1** — distinct from existing GELU; required by GPT-J / some Falcon |
| `UNARY:SOFTPLUS` | yes | no | A | P2 |
| `OP_NORM` | yes | **partial** (in switch but check coverage of ops not flagged supported) | A | already wired — verify |
| `OP_GROUP_NORM` | yes | partial (dispatch exists at line 3232) | F | verify F16/BF16 paths |
| `OP_OUT_PROD` | yes (`064c90d84`) | no | H | P2 — used by training; rare in inference |
| `OP_CONCAT` | yes (`7066b4cce`) | yes | A | done |
| `OP_REPEAT` | yes | yes | done | already dispatched |
| `OP_ROPE` MROPE/IMROPE/Vision/partial | yes (`eeb5605de`, `ca709e427`) | **NO — only NORMAL+NEOX** (line 3437) | A | **P0** for Qwen2-VL, Gemma 3 vision, GPT-OSS routing |
| `OP_FLASH_ATTN_EXT` head_dim%16!=0 + ALiBi | yes (`07ba6d275`) | yes (basic only — verify) | B | P1 — verify head-dim coverage |
| `OP_PAD` / `PAD_REFLECT_1D` | yes (`09c7c50e6`) | no | out-of-scope | P2 |
| `OP_UPSCALE` | yes | no | out-of-scope | P2 (vision only) |
| `OP_IM2COL` (1D + 3D) | yes (`e09a800f9`) | no | out-of-scope | P2 |
| `OP_POOL_2D` | yes | no | out-of-scope | P2 |
| `OP_CONV_TRANSPOSE_1D` (kernel>255) | yes (`e68c19b0f`, `6e1c4cebd`) | no | out-of-scope | P2 |
| `OP_ARANGE` / `OP_TIMESTEP_EMBEDDING` | yes | no | out-of-scope | P2 |
| `OP_SSM_CONV` | yes (`b07cda687`) | no | E | P1 if Mamba is a target |
| `OP_GATED_LINEAR_ATTN` | yes (`baa4ba0ae`) | no | E | P1 if RWKV/GLA models targeted |
| `OP_CROSS_ENTROPY_LOSS` | yes (`97d511721`) | no | training-only | P3 |
| `OP_COUNT_EQUAL` | yes | no | D | P3 |
| `OP_DIAG_MASK_INF` / `DIAG` | yes | no | H | P2 |
| `OP_ACC` / `OP_SET` / `OP_FILL` / `OP_TRI` / `OP_SOLVE_TRI` | yes | no | utility | P3 |
| `OP_CUMSUM` | yes | yes | done | already dispatched |
| `OP_LEAKY_RELU` | yes | yes | done | already dispatched |
| `OP_L2_NORM` | yes (`655cddd17`) | yes | done | already dispatched |
| `OP_GLU` | yes (`11dd5a44e`) | yes | done | already dispatched |
| `OP_TOP_K` | not in CANN list | yes | GCU-ahead | leave |
| `OP_SET_ROWS` | yes (`204f2cf16`) | yes | done | already dispatched |

**P0 ops (model-coverage-blocking) for GCU**: ROPE MROPE+IMROPE+Vision+partial. Everything else is correctness or breadth, not net-new model unlock for the families we already run.

---

## 3. Architectural gaps

Each row references the CANN commit that introduced (or last refactored) the feature.

| Capability | CANN reference | GCU today | Gap impact |
|---|---|---|---|
| **BF16 for core ops** (`MUL_MAT`, `MUL_MAT_ID`, `GET_ROWS`, `SET_ROWS`, `CPY`, `CONT`, …) | `1af9dab32` (#20152, 2026-03-20) | F32 + F16 only (`gcu_dtype_supported` at line 817) | Locks us out of any BF16 GGUF (Llama-3.1, Mistral-Nemo, modern GPT-OSS variants without re-quantising) |
| **ACL/captured-graph execution** (record once, replay) | `224145325` (#15065) initial; LRU at `28b5f190e` (#15814); refactor `ce7a6dc0f` (#17752); recent fix `07ff00055` | none — every node dispatched eagerly per `graph_compute` | Each step pays per-op submission overhead. Not yet measured on GCU but CANN reports significant prefill speed-ups. |
| **Async operator submission** (separate submit thread) | `7a395f67a` (#12864) gated by `GGML_CANN_ASYNC_MODE` | partial — MVP-4a/4b implements async H↔D + scratch-deferred-free. No kernel-submit thread. | At small batches the thread-jump-vs-pipeline trade is unclear; revisit only if profiling demands it. |
| **Backend-agnostic tensor parallelism** (`--device CANNn,CANNm`) | `d6f303004` (#19378) — generic, also lights up CANN | GCU registers all S60 devices but `ggml_backend_gcu_offload_op` is `nullptr` (line ~3528) and there is no per-stream split logic | Single-GCU is the only path today. Multi-S60 boxes (we have one) bench at 1× until this lands. |
| **`offload_op` heuristic** (skip GCU for tiny work — batch < threshold) | `ggml_backend_cann_offload_op` (env `GGML_OP_OFFLOAD_MIN_BATCH`, default 32) | `nullptr` | Scheduler can't hint "send small ops to CPU"; we either eat the launch overhead or hard-route. |
| **Event API** (`event_record`, `event_wait`, `event_new`, `event_free`) | full (`ggml-cann.cpp:2758-2760, 2944-2946`) | all `nullptr` (line 738, 3530-3533) | The scheduler can't insert cross-backend sync; works because we serialise inside `graph_compute`, but blocks future overlap with CPU/CUDA peers. |
| **NZ weight format conversion** (Ascend-310P-only fast path) | `14c28dfc5` (#14407), `11490b367` (#14985) | n/a — GCU device might benefit from a similar weight-layout rewrite for matmul. Needs hardware investigation; not necessarily a gap. | unknown — defer until profiling says matmul memcpy dominates |
| **RoPE sin/cos cache** (precompute once, reuse per layer) | `10d8b2b6b` (#15912), `c247d06f3` (#15501) | recompute per call inside `gcu_op_rope` | Per-token recompute on GCU; small but free perf. |
| **Tensor parallel split / multi-device buffer types** | implicit in `d6f303004` + reg context | single device per backend | see TP entry above |
| **Smart-pointer ACL handle hygiene** | `2376b7758` (#17238) | manual (raw `topsatenTensor` lifetimes via RAII wrappers — actually fine) | parity already |
| **Operator fusion** (`ADD + RMS_NORM`) | `67e3f6f60` (#17512), env `GGML_CANN_OPERATOR_FUSION` | none | Optional perf; only matters if profiling identifies the pattern as a bottleneck. |
| **`set_tensor` thread safety** | `632219af7` (#20151) | look-busy on this — verify we don't have the same race | Unlikely if all set_tensor goes through `set_tensor_async` on the copy stream, but worth a 30-min audit. |
| **Per-device ND-to-NZ workspace** | `c1c354e44` (#15763) | n/a unless we do NZ |  defer |
| **`init_tensor` returns `ggml_status`** (API plumbing) | `70680c48e` (#11854) | already on this version — parity |
| **Pinned host buffer type** | `ggml_backend_cann_host_buffer_type` (#10454 era) | yes — line 534 | parity |

### Architecture-gap summary

- **P0 architectural gap:** BF16, captured graph, multi-device tensor parallelism (in that ROI order).
- **P1:** `offload_op` heuristic, full event API, RoPE sincos cache.
- **P2:** ACL-style fusion, NZ-layout weight rewrite (only if profiling justifies).

---

## 4. Test / build / CI infrastructure gap

| Asset | CANN | GCU |
|---|---|---|
| GitHub Actions matrix build | `.github/workflows/build-cann.yml` — 2 SoCs × 2 archs × 2 graph modes, openEuler-in-docker | none |
| Dockerfile | `.devops/cann.Dockerfile`, `.devops/llama-cli-cann.Dockerfile` | none |
| Op-coverage CSV (`test-backend-ops --output csv`) | `docs/ops/CANN.csv` (18 855 rows, kept current) | none — we have an ad-hoc smoke binary `test-backend-gcu` only |
| `clang-format` config | enforced (`7a50cf388` #15863) | not enforced |
| SoC auto-detect in CMake | `CMakeLists.txt` runs `npu-smi info` to auto-pick `SOC_TYPE` | static `TOPS_INSTALL_DIR` only |
| Compile-time SoC switch | `-DASCEND_310P` etc. via CMake | none — single binary |
| Pre-built backend libs (`ggml_add_backend_library` already in place) | yes | yes — parity |
| Runtime device count | yes — `aclrtGetDeviceCount` | yes — `topsGetDeviceCount` |
| In-tree x86 build CI | `54a727204` (#12950) | no |

The biggest CI gap is **no upstream-ready build job**. Without one, the GCU backend cannot be merged into upstream without a maintainer agreeing to host a runner. That blocks the long-term goal stated in CLAUDE.md.

---

## 5. Docs gap

| Doc | CANN | GCU |
|---|---|---|
| `docs/backend/<NAME>.md` (build + model + dtype matrix + troubleshooting) | `docs/backend/CANN.md` (b0dbc92...) | **missing** |
| `docs/build.md` section | yes (lines 32 / "## CANN") | none |
| `docs/ops/<NAME>.csv` | yes — 18 855 rows | **missing** |
| In-repo design specs | n/a (not how upstream works) | yes — `docs/superpowers/specs/*.md` (internal-only) |
| Per-MVP perf table | n/a upstream | yes — `docs/superpowers/specs/2026-05-09-gcu-mvp4*-design.md`, MVP-2 perf in CLAUDE.md |
| Model-support matrix | yes — extensive `Text-only` / `Multimodal` tables in `docs/backend/CANN.md` | none |
| Hardware/device table (PCI ID → Product) | yes | none |

For upstream submission we will need at minimum a `docs/backend/GCU.md`, a `docs/build.md` paragraph, and a generated `docs/ops/GCU.csv` from `test-backend-ops`.

---

## 6. Commit-history themes (87 CANN commits classifiable by subject; rough counts)

Sample = the 87 subjects with `CANN: ` prefix from `git log --oneline upstream/master -- ggml/src/ggml-cann/`.

| Theme | Approx % | Examples |
|---|---|---|
| **Op-coverage extensions** (new GGML ops dispatched) | ~30 % | `b07cda687` SSM_CONV, `baa4ba0ae` GLA, `97d511721` CEL, `655cddd17` L2_NORM, `204f2cf16` SET_ROWS, `11dd5a44e` GLU, `2d38b6e40` FA, `33d7aed4a` MUL_MAT_ID, `65cfe136a` SIN/COS/ARGMAX |
| **Op-level perf optimization** (caching, fusion, kernel pick) | ~20 % | `67e3f6f60` ADD+RMS fuse, `b9382c387` MUL_MAT_ID, `bbd57b7ea` CPY, `a0f98dd60` RMS cache, `a6d3cfe7f` rope, `1e7489745` FA mask, `c247d06f3` ROPE cache, `9bacd6b37` get_rows/dup, `b7420131b` ROPE |
| **Bug fixes (correctness, leaks, multi-device, races)** | ~20 % | `632219af7` race, `7f2cbd9a4` in-place ROPE, `9961d244f` softmax precision, `5421f63ab` 310I precision, `56fc38b96` CPU leak, `3dc7397a2` rope multi-device, `8a2234ea0` float type |
| **ACL graph (capture/replay/cache)** | ~7 % | `224145325` add, `28b5f190e` LRU, `ce7a6dc0f` refactor, `2370665e5` refactor, `aa4711d36` matching, `c0389dba4` disable on prefill, `85ca66a74` stream sync, `2f853687b` eager fallback, `01ad35e6d` define guard, `07ff00055` preload RoPE |
| **Build / packaging / CI** | ~7 % | `2860d479b` cann build pipeline (Dockerfile), `54a727204` x86 CI, `7a50cf388` clang-format, `673cfef9a` GCC13 fix, `605fa66c5` SOC_TYPE bug, `1e8659e65` SOC print |
| **Multi-device / scheduler** | ~5 % | `c1c354e44` per-device NZ, `85ca66a74` stream sync, `904837e0c` llama-bench multi-device crash, `3dc7397a2` rope cache multi-device |
| **Quant-type extensions** (Q4_0, Q8_0 paths, MoE quant) | ~6 % | `c02b0a8a4` q4_0, `c8a009092` q8_0, `52e38faf8` quant MUL_MAT_ID, `faaaff5f9` MUL_MAT_ID q8/q4 |
| **Dtype additions** (BF16, FP16 specifics) | ~3 % | `1af9dab32` BF16, `f9bc66c3e` more FP16 ops, `c18610b4e` 310P F32+F16 |
| **310P-specific (older Ascend SoC) work** | ~6 % | `14c28dfc5` NZ for 310P, `b43d89e31` 310P op support check, `c18610b4e` 310P accel, `5eae93488` 310P RoPE, `5421f63ab`, `bc4064cfe`, `a2b0fe8d3`, `f6da8cb86` |
| Refactors / renames / housekeeping | balance |  |

**Reading**: CANN has spent **~50 % of post-MVP commits on perf+correctness**, ~30 % on op coverage, and the rest on graph capture, dtype, CI, and 310P-only quirks. GCU's velocity is similarly skewed (most of the last 15 commits are MVP-2 op coverage + MVP-3a/4a/4b perf), but we lag on captured graph, dtype breadth, and the 310P-equivalent SoC-specific tuning (we have one SoC — S60 — so this is partly free for us).

---

## 7. Recommended next-MVP plan

Ordered by ROI: model-unblocking first, breadth-of-coverage second, perf+polish third, upstream-readiness as a long pole that runs alongside.

| # | MVP | Title | Rationale | Effort | Depends on |
|---|---|---|---|---|---|
| 1 | **MVP-5a** | Tranche A op extensions (`UNARY:ABS, NEG, SGN, STEP, EXP, ELU, GELU_ERF, SOFTPLUS` + verify `OP_NORM` and `OP_GROUP_NORM` paths) | Mechanical topsaten wraps; closes ~8 ops in <1 day; eliminates "unknown op" CPU-fallback noise on real models | 1 day | none |
| 2 | **MVP-5b** | **BF16 dtype on core ops** (mirror CANN `1af9dab32`: `MUL_MAT`, `MUL_MAT_ID`, `GET_ROWS`, `SET_ROWS`, `CPY`, `CONT`, `RMS_NORM`, `ADD`, `MUL`) | Unlocks Llama-3.1, Mistral-Nemo and modern BF16-only GGUFs without re-quantising; biggest model-coverage delta | 2-3 days (need topsaten BF16 support audit; if missing, scope down to f16-cast fallback) | none |
| 3 | **MVP-5c** | **ROPE: MROPE / IMROPE / Vision / partial-rotation** (mirror `eeb5605de` + `ca709e427`) | Unlocks Qwen2-VL, Gemma 3 vision, GPT-OSS, multimodal models | 2 days | none |
| 4 | **MVP-5d** | **Captured graph + LRU graph cache** (mirror `224145325` + `28b5f190e`) | Per-step submission overhead becomes free for steady-state decode; CANN reports double-digit-% prefill wins. Investigate whether topsaten/topsrt has an equivalent of `aclmdlRI*` capture/replay; if not, fall back to a "hash-of-cgraph → cached enqueue list" optimisation | 3-5 days; high spike risk on SDK-feature feasibility | none |
| 5 | **MVP-5e** | **`offload_op` heuristic + full event API** (mirror `ggml_backend_cann_offload_op`, env `GGML_OP_OFFLOAD_MIN_BATCH`; wire `event_record/wait/new/free`) | Unlocks proper scheduler placement; tiny ops stop hitting GCU; lets future tensor-parallel actually synchronise | 1-2 days | none |
| 6 | **MVP-5f** | **Multi-S60 tensor parallelism** (lean on `d6f303004`'s backend-agnostic TP; verify our `device->iface.split_buffer_type` etc.) | We have a multi-S60 box; benchmarks at 1× until this lands; large LLMs blocked by single-device VRAM | 4-7 days; need integration testing vs lifetime/race issues | MVP-5e (event API) |
| 7 | **MVP-5g** | **File split + smart-pointer hygiene** (mirror CANN: `acl_tensor.{h,cpp}` + `aclnn_ops.{h,cpp}` + `common.h` + `ggml-gcu.cpp`; pull RAII patterns from `2376b7758`) | Required before upstream PR; 3 687 LoC monolith is unreviewable. Independent of all op work above | 1-2 days | none — can be done in parallel |
| 8 | **MVP-5h** | **Upstream-readiness package**: `docs/backend/GCU.md`, `docs/build.md` paragraph, `docs/ops/GCU.csv` regen-script, `.github/workflows/build-gcu.yml` (self-hosted runner), `.devops/gcu.Dockerfile` | The ask in CLAUDE.md is "first-class upstream backend"; without these we can't open the PR | 3-5 days; CI runner negotiation with maintainers is the long pole | MVP-5g |
| 9 | **MVP-5i** | **Tranche E ops: `SSM_CONV`, `GATED_LINEAR_ATTN`** if Mamba/RWKV land on the target list | Each unlocks a model family. Defer until product asks | 3-5 days each | none |
| 10 | **MVP-5j** | **Operator fusion (ADD + RMS_NORM)** + **RoPE sincos cache** | Optional perf; only justified after profiling | 1-2 days | MVP-5d (captured graph clarifies whether this still matters) |

### Priority verdict

- **P0 (do next):** MVP-5b (BF16) — biggest model-coverage win; MVP-5c (ROPE variants) — unblocks Qwen2-VL/Gemma 3; MVP-5g (file split) — required before any upstream conversation.
- **P1:** MVP-5a (op extensions), MVP-5d (captured graph), MVP-5e (offload_op + events), MVP-5h (upstream package).
- **P2:** MVP-5f (multi-device TP), MVP-5i (Mamba/RWKV), MVP-5j (fusion).

---

## 8. Out of scope / deferred / NPU-specific that does not translate

- **Ascend 310P-specific paths** (`14c28dfc5` NZ for 310P, `c18610b4e`, `5eae93488` 310P RoPE, `b43d89e31` 310P op-support check, `5421f63ab` 310I DUO precision, `f6da8cb86` mask 310P TRANSPOSE_1D). These are a separate Huawei chip generation. The S60 has its own quirks but they will be GCU-specific, not CANN's. Skip directly.
- **NZ tensor format** (`ACL_FORMAT_FRACTAL_NZ`). Ascend-specific data layout. The S60/topsaten equivalent (if any) is its own investigation; not a port.
- **`aclnnGroupedMatmulV2 → V3` migration** (`343b6e94b`). API-version housekeeping; topsaten doesn't have this enum.
- **`update aclnnGroupedMatmulV2 to V3`, `Rename get_env`** etc. CANN-API churn that only matters once a topsaten upgrade arrives.
- **Cross-entropy loss / training back-passes** (`97d511721`). llama.cpp inference doesn't issue them; we only need them if on-device fine-tuning becomes a target — and CANN's roadmap has not committed to that either.
- **Conv / pool / im2col / upscale / pad / arange / timestep_embedding family**. Per the existing roadmap section "Out of scope for this backend", these need a topsaten convolution surface that we have not committed to. Defer until multimodal LLMs (Gemma 3 vision, Qwen2-VL) become a hard requirement.
- **`SOLVE_TRI` / `TRI` / `FILL` / `ACC` / `SET` / `MAP_CUSTOM*` / `CUSTOM`**. User-defined hooks and rare utility ops. CANN doesn't accelerate them either in many cases — when it does, we can mirror on demand.

---

## Appendix A — citation index

- `1bdd8ae19` — initial CANN backend (#6035, 2024-07-17)
- `c8a009092`, `c02b0a8a4` — Q8_0 / Q4_0 quant support (#8805, #8822)
- `9a4b79bcf` — perf push (#10454)
- `c18610b4e` — 310P SoC support (#10216)
- `33d7aed4a`, `faaaff5f9` — MUL_MAT_ID + quant MUL_MAT_ID (#13042, #13705)
- `7a395f67a` — async submit thread (#12864)
- `2d38b6e40` — Flash Attention basic (#13627)
- `14c28dfc5`, `11490b367`, `c1c354e44` — NZ weight format for 310P (#14407, #14985, #15763)
- `11dd5a44e` — GLU ops (#14884)
- `204f2cf16` — `OP_SET_ROWS` (#14943)
- `224145325`, `28b5f190e`, `ce7a6dc0f`, `2370665e5`, `aa4711d36` — ACL captured graph + LRU + refactors (#15065, #15814, #17752, #17333, #16166)
- `10d8b2b6b`, `c247d06f3`, `a0f98dd60` — RoPE / RMS sincos cache (#15912, #15501, #15419)
- `9961d244f` — softmax precision (#15730)
- `f9bc66c3e` — extra FP16 op coverage (#16251)
- `655cddd17`, `97d511721` — L2_NORM, CEL (#16856, #16886)
- `2376b7758` — smart-pointer ACL handles (#17238)
- `eeb5605de`, `ca709e427` — MROPE / IMROPE / Vision / partial RoPE (#17401, #17543)
- `b07cda687` — SSM_CONV (#17737)
- `e68c19b0f` — CONV_TRANSPOSE_1D kernel>255 (#17934)
- `67e3f6f60` — ADD+RMS_NORM fusion (#17512)
- `52e38faf8` — quantized MUL_MAT_ID for MoE (#19228)
- `1af9dab32` — BF16 for core operators (#20152)
- `07ba6d275` — FA head_dim%16 + ALiBi (#20031)
- `7f2cbd9a4` — in-place ROPE on non-contiguous f32 (#20274)
- `632219af7` — set_tensor multi-thread race (#20151)
- `07ff00055` — RoPE cache preload before ACL graph capture (#20747)
- `c3e08f470` — recent (last) op-add + perf push (#21204)
- `d6f303004` — backend-agnostic tensor parallelism (#19378, ggml-side, also lights CANN)
- `.github/workflows/build-cann.yml` — CI matrix
- `.devops/cann.Dockerfile`, `.devops/llama-cli-cann.Dockerfile` — packaging
- `docs/backend/CANN.md` — user docs
- `docs/ops/CANN.csv` — op-coverage truth table

---

**End of report.**
