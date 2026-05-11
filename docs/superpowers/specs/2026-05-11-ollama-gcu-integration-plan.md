# Ollama + GCU Integration Plan

Date: 2026-05-11
Author: research pass against `ollama/ollama` master @ commit `c2f2d9050` (2026-05-09).
Scope: read-only investigation; no code changes.

## 1. Executive Summary

- **Vendoring is rsync from a pinned upstream commit.** Ollama vendors `llama.cpp` (and its `ggml/` subtree) by `git checkout`-ing a pinned SHA into `llama/vendor/` and rsync-ing into two trees (`llama/llama.cpp/` and `ml/backend/ggml/ggml/`). The current pin is **`FETCH_HEAD=ec98e2002`** in `Makefile.sync`. A series of 36 patches (`llama/patches/0001…0036.patch`) is `git am`-ed on top.
- **Backends are runtime-loaded `.so` modules; the C++ side knows nothing about Ollama.** Ollama builds with `GGML_BACKEND_DL=ON` so each backend (CUDA, HIP, Vulkan, Metal, BLAS, CPU) is a separate `.so` placed under `lib/ollama/<runner_dir>/`. Ollama's Go side discovers backends by launching the runner process with `OLLAMA_LIBRARY_PATH` set to the chosen subdirs, the runner calls `ggml_backend_load_all_from_path()`, and Ollama queries `GET /info` to get a list of devices (`ml.DeviceInfo`) with a `Library` string set from each backend's `props.library`. There is **no Go-side topsrt probe code needed** — the runner's ggml registration is the authoritative discovery path. The Go side only needs ~one Library-name awareness branch.
- **Effort estimate: small-to-medium.** ~1 day to wire GCU into Ollama's CMake/CMakePresets and confirm the build; ~1–2 days to flip our fork's base from `master` (cf5e623ba) to Ollama's pin (ec98e2002) and rebase the GCU patch set; ~2 days end-to-end testing against `llama3.2:1b` on the S60 plus shaking out Go-side Library-string handling. **Total: ~1 week for an opt-in dev build; ~2 weeks if we want a clean upstream-able PR**.

## 2. Vendoring landscape

### 2.1 Mechanism and paths

- **Mechanism**: rsync from a git-tracked clone, not subtree/submodule. See [`Makefile.sync`](https://github.com/ollama/ollama/blob/main/Makefile.sync) (lines 1-58 of `/tmp/ollama-research/ollama/Makefile.sync`).
  - `llama/vendor/` is a real git clone of `https://github.com/ggml-org/llama.cpp.git`.
  - `make sync` rsyncs into two destinations:
    - `llama/llama.cpp/` ← LICENSE + filtered files (the C++ inference library used by the `llamarunner`).
    - `ml/backend/ggml/ggml/` ← LICENSE + `ggml/` subtree (used by the `ollamarunner` and by the CMake build for backend `.so`s).
- **Upstream pin**: `FETCH_HEAD=ec98e2002` (= upstream commit `ec98e2002 llama: fix early stop in params_fit if ctx is set (#18070)`). Line 3 of `Makefile.sync`. Documented in `llama/build-info.cpp.in` → `llama/build-info.cpp` substitution. (Verified by `git log -1 ec98e2002` on the upstream ggml-org clone.)
- **Cadence**: ad-hoc — `llama/README.md` explains the human workflow ("Updating Base Commit"). Looking at recent history, Ollama syncs every few weeks. The patch list has grown to 36 patches, indicating routine carrying of fixes across syncs.

### 2.2 Patches inventory and conflict surface

36 patches in `/tmp/ollama-research/ollama/llama/patches/`. Categorised:

**Touches `ggml/src/ggml-backend-reg.cpp` (collision risk for our GCU additions)**
- `0007-sort-devices-by-score.patch` — refactors `devices` from `vector<dev_t>` to `vector<pair<dev_t, int>>`. Conflicts with our 7-line GCU additions in the same file (one `#include` block, one `register_backend(...)` block). **Trivially mergeable** by hand.

**Touches CUDA backend (no GCU collision)**
- `0013-add-argsort-and-cuda-copy-for-i32.patch`, `0029-ggml-cuda-skip-large-batches.patch`, `0035-CUDA-get_rows-q6_k-support.patch`. CUDA-only.

**Touches Metal backend (no GCU collision)**
- `0033-ggml-metal-solve_tri.patch`, `0034-ggml-metal-guard-mul_mat_id-map0…`, `0036-backport-kernels-for-gemma4.patch`. Metal-only.

**Touches ggml core (`ggml.c`, `ggml.h`, `ggml-alloc.c`, etc.)**
- `0001-ggml-backend-malloc-and-free-using-the-same-compiler.patch` — compiler-aligned free/malloc.
- `0014-graph-memory-reporting-on-failure.patch`
- `0015-ggml-Export-GPU-UUIDs.patch` — adds `id` and PCIID fields to `ggml_backend_dev_props`. **Our GCU code must populate these in `gcu_get_dev_props()` to avoid losing identity through Ollama's dedup logic.**
- `0018-ggml-Add-batch-size-hint.patch`
- `0020-ggml-No-alloc-mode.patch` — adds an "estimate memory but don't allocate" mode. **Our GCU `init_tensor`/`set_tensor` paths must respect this for Ollama's memory probing.**
- `0022-ggml-Enable-resetting-backend-devices.patch`
- `0024-GPU-discovery-enhancements.patch` — affects how backends are enumerated.
- `0027-interleave-multi-rope.patch`
- `0032-ggml-enable-MLA-flash-attention-for-GLM-4.7-flash.patch`

**llama.cpp library-only (no ggml/backend touch)**
- `0002-pretokenizer`, `0003-clip-unicode`, `0004-solar-pro`, `0005-fix-deepseek-deseret-regex`, `0006-maintain-ordering-for-rules-for-grammar`, `0010-fix-string-arr-kv-loading`, `0011-ollama-debug-tensor`, `0012-add-ollama-vocab-for-grammar-support`, `0016-add-C-API-for-mtmd_input_text`, `0017-no-power-throttling-win32`, `0019-fix-mtmd-audio.cpp-build-on-windows`, `0021-decode-disable-output_all`, `0023-harden-uncaught-exception-registration`, `0026-report-LoadLibrary-failures`, `0030-fix-bakllava-regression`, `0031-win-exit-instead-of-abort`.

**Build/CPU (low collision risk)**
- `0008-add-phony-target-ggml-cpu-for-all-cpu-variants.patch`, `0009-remove-amx.patch`, `0025-NVML-fallback-for-unified-memory-GPUs.patch`, `0028-Add-memory-detection-using-DXGI-PDH.patch`.

**Summary of conflict surface for GCU**: only patch 0007 mechanically conflicts. Patches 0015 (UUID export) and 0020 (no-alloc mode) are **semantic obligations** — our backend must implement these to be a good citizen in Ollama's discovery/memory pipeline.

## 3. Build system — exact files and lines to touch

### 3.1 The build flow

1. **`/tmp/ollama-research/ollama/CMakeLists.txt`** is the umbrella CMake. It:
   - Sets `GGML_BACKEND_DL=ON` and `GGML_SCHED_MAX_COPIES=4` (lines 33-38).
   - Always builds CPU (line 87: `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)`).
   - Then `check_language(CUDA)` / `find_package(hip)` / `find_package(Vulkan)` gate optional backends (lines 111-195) and explicitly `add_subdirectory(...)` each one.
   - Each backend gets its own `install(TARGETS ggml-X ...)` block keyed by `COMPONENT` (CPU/CUDA/HIP/Vulkan/MLX).

2. **`/tmp/ollama-research/ollama/CMakePresets.json`** defines presets that map `(backend, version) → OLLAMA_RUNNER_DIR`. E.g. CUDA 12 → `cuda_v12`, ROCm → `rocm`, Vulkan → `vulkan`. The runner dir becomes the subdirectory under `lib/ollama/` where the `.so` lives.

3. **`/tmp/ollama-research/ollama/ml/backend/ggml/ggml/src/CMakeLists.txt`** has `ggml_add_backend(<name>)` macro (lines 296-307) and the call list (lines 354, 431-444). For backend `X` it does `add_subdirectory(ggml-x)` if `GGML_X=ON`. **Our fork already registers `ggml_add_backend(GCU)` indirectly via our own `ggml/CMakeLists.txt` patch — once vendored, the line just needs to exist in this file at line ~445.**

4. **Build invocation**: `scripts/build_windows.ps1` and `Dockerfile` call `cmake -B build/cuda_v12 --preset "CUDA 12"` then `cmake --build build/cuda_v12 --target ggml-cuda --config Release`. Linux uses `Dockerfile` via `docker buildx`; per-flavor flavors set via `--build-arg FLAVOR=rocm`.

### 3.2 What we touch to add GCU

| File | Change | Lines |
| --- | --- | --- |
| `CMakeLists.txt` (Ollama umbrella) | After the HIP block (line ~180), add a `check_language(CXX)`-guarded `if(GGML_GCU OR EXISTS /opt/tops/cmake_modules/FindTopscc.cmake)` block that calls `find_package(Tops)` (from `/opt/tops/cmake_modules/`), `add_subdirectory(...ggml-gcu)`, `target_include_directories(ggml-gcu PRIVATE ${GGML_INCLUDE_DIRS})`, and `install(TARGETS ggml-gcu ...)` with a new `COMPONENT GCU` and `DIRECTORIES /opt/tops/lib`. | ~25 lines added. |
| `CMakePresets.json` | New preset `"GCU"` setting `OLLAMA_RUNNER_DIR: "gcu"` and `GGML_GCU: ON`. New build preset `"GCU"` with `targets: ["ggml-gcu"]`. | ~15 lines. |
| `ml/backend/ggml/ggml/src/CMakeLists.txt` | Add line `ggml_add_backend(GCU)` after the existing `ggml_add_backend(CANN)` (current line 432). | 1 line. |
| `ml/backend/ggml/ggml/CMakeLists.txt` (the top-level inside vendored ggml) | Add `option(GGML_GCU "ggml: use GCU" OFF)` next to the existing GGML_CANN option. | 1 line. |
| `Dockerfile` | Add a build stage for GCU mirroring the ROCm stage, with `/opt/tops` mounted as a base image layer. | ~20 lines (only if we want CI-built artifacts; can be skipped for dev builds). |

All of the above are additive; nothing existing is rewritten.

## 4. Go-side device discovery — file/function we'd extend

### 4.1 How CUDA detection actually works

The flow is **not** a Go-side library probe. It is:

1. `discover/runner.go:GPUDevices()` (lines 34-361) enumerates `lib/ollama/*/ggml-*` directories — see line 55: `filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))`. So **the mere presence of `lib/ollama/gcu/libggml-gcu.so` is enough to put `gcu` into the candidate `libDirs` map.**
2. For each candidate libDir, `bootstrapDevices()` (line 485) calls `llm.StartRunner(true /*ollamaEngine*/, "" /*no model*/, []string{ml.LibOllamaPath, dir}, ...)` which spawns `ollama runner --ollama-engine --port <p>` with `OLLAMA_LIBRARY_PATH=lib/ollama:lib/ollama/gcu`.
3. The runner's `ml/backend/ggml/ggml/src/ggml.go:OnceLoad()` (lines 54-105) reads `OLLAMA_LIBRARY_PATH` and calls `ggml_backend_load_all_from_path()` for each dir, which dlopens every `libggml-*.so` in there. **GCU registration happens here via ggml-backend-reg.**
4. `ml.GetDevicesFromRunner()` (`ml/device.go:635-682`) does `GET http://127.0.0.1:<p>/info` and unmarshals into `[]ml.DeviceInfo`.
5. The `Library` string in each `DeviceInfo` comes from `props.library` (set in `ggml_backend_dev_get_props()`), populated at `ml/backend/ggml/ggml.go:727` and `:184/:203`.

So **device discovery for GCU is automatic** as long as we (a) ship `libggml-gcu.so` in `lib/ollama/gcu/`, and (b) set `props.library = "GCU"` in our backend's `get_dev_props()`.

### 4.2 Minimal Go-side additions

The Go code has a handful of `Library == "..."` switches that need a GCU branch. They are all in well-defined places:

| File:line | What | Action |
| --- | --- | --- |
| `ml/device.go:485-498` `FlashAttentionSupported()` | Decides FA support. | Add `gpu.Library == "GCU"` to the supportsFA disjunction (we have FLASH_ATTN_EXT working). |
| `ml/device.go:551-555` `NeedsInitValidation()` | Returns true for ROCm/CUDA. | Default false for GCU is fine for now (skip second-pass init filtering); revisit if topsrt asserts on bad devices. |
| `ml/device.go:565-573` `PreferredLibrary()` | Returns true for CUDA/ROCm; used to drop Vulkan duplicates. | Add `d.Library == "GCU"` so GCU wins over a coincident Vulkan device. |
| `ml/device.go:575-605` `updateVisibleDevicesEnv()` | Sets `CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`. | Add `case "GCU"` → set `TOPS_VISIBLE_DEVICES` or whatever env var our backend honors. |
| `discover/runner.go:97-108` | Filters libDirs by `EnableVulkan()` etc. | No change needed unless we want a `OLLAMA_GCU=1` opt-in gate (recommended for early rollout). |

Total: ~30 lines of Go in `ml/device.go` plus optionally ~5 lines in `discover/runner.go` for an env-var gate.

There is **no need** for a `discover/gcu_linux.go` file. The pattern for CUDA/ROCm proxies through ggml; only Metal has direct Objective-C code (`discover/gpu_info_darwin.m`) because it's the on-CPU integrated GPU on macOS.

## 5. Runner architecture

Two runners, both auto-load backend `.so`s the same way:

- **`runner/llamarunner/`** (`runner.go`, 1008 lines) — cgo wrapper around `libllama` (the C++ inference library). Used by default for arch-not-yet-supported-by-ollama-engine models.
- **`runner/ollamarunner/`** (`runner.go`, 1492 lines) — Go-native runner using `ml/backend/ggml/` to build compute graphs in Go. Used when `f.KV().OllamaEngineRequired()` returns true OR `envconfig.NewEngine()` is set OR a `model.NewTextProcessor()` exists for the architecture (`llm/server.go:148-158`).

Dispatch is via `runner/runner.go:Execute()` (lines 10-26): `--ollama-engine` → ollamarunner, otherwise llamarunner.

**Both runners get the GCU backend for free.** Because:
- llamarunner inherits ggml backend selection from upstream `llama.cpp`'s `llama_load_model_from_file()` → `llama_backend_init()` → registers all backends loaded via `ggml_backend_load_all_from_path()`.
- ollamarunner explicitly initializes via `ml.NewBackend()` → `ggml.New()` → `initDevices()` (`ml/backend/ggml/ggml.go:47-77`) which iterates ALL registered devices and puts GCU ones into `gpus`.

**Runner-side code changes for GCU: zero.** The `--device` arg passed to llama.cpp's parser already supports `--device GCU0` format because our backend registers a device named `GCU0` (set in `ggml-gcu.cpp`). Llama 3.2 1B Q4_K_M end-to-end runs we did on the S60 verified this already.

## 6. Vendor-swap recipe

### Path (a): Rebase our GCU commits onto Ollama's pin `ec98e2002`

- Rebase distance: our fork base is `cf5e623ba` (HEAD-ish of llama.cpp around 2026-04-01 by commit date); Ollama's pin `ec98e2002` is ~1600 commits newer on master. We don't have full upstream history locally so the exact figure may differ slightly.
- Conflict surface: tiny.
  - `ggml/src/ggml-backend-reg.cpp`: 7-line insertion likely needs reapplication after upstream churn + on top of Ollama's `0007-sort-devices-by-score.patch`. Two small textual conflicts, each a single hunk.
  - `ggml/CMakeLists.txt`: 1-line option addition — almost certainly still applies cleanly.
  - `ggml/src/CMakeLists.txt`: 1-line `ggml_add_backend(GCU)` — applies cleanly.
  - `ggml/src/ggml-gcu/CMakeLists.txt`, `ggml/include/ggml-gcu.h`, `ggml/src/ggml-gcu/ggml-gcu.cpp`: brand-new files, no conflict.
- Risk: our `ggml-gcu.cpp` uses backend-API functions. Since cf5e623ba (~Apr 2026), the upstream backend API has had additions (no-alloc mode, GPU UUIDs, batch-size hint, reset-backend). **Our backend doesn't implement these.** It will still load (the API uses function-pointer tables that default to NULL), but Ollama's memory-probing pass will get less-accurate results until we plumb no-alloc mode (patch 0020 obligation).

**Mechanical steps for path (a)**:
```bash
git fetch origin master
git checkout -b feat/ggml-gcu-rebased
git rebase --onto ec98e2002 cf5e623ba feat/ggml-gcu
# Resolve 1-2 small textual conflicts in ggml-backend-reg.cpp.
# Verify backend builds standalone:
cmake -B build -DGGML_GCU=ON && cmake --build build --target ggml-gcu
```

### Path (b): Replace whole vendored tree with our fork's HEAD

- Pros: faster initial drop; nothing to rebase.
- Cons: Ollama's 36 patches will need to re-apply against our older base. Patch 0007 et al. were written for a llama.cpp tree newer than ours. **Several patches that touch files between ec98e2002 and our cf5e623ba will fail.** We'd then either have to (i) skip those patches (losing Ollama functionality), or (ii) port them backward, which is harder than just rebasing our 6 file changes forward.
- The 36 patches touch ~25 different files in ggml/. Manual conflict resolution would be hours-to-days.

### Recommendation

**Path (a)** is unambiguously cleaner. Our fork's GCU diff is 6 files / ~3,950 lines including the (large) `ggml-gcu.cpp`, but **only ~50 lines land in existing upstream files**; the rest are brand-new files. Rebasing 50 lines onto a newer base is much easier than re-porting 36 Ollama patches onto an older base.

## 7. Step-by-step build & test plan

### 7.1 On laptop (Mac, the planning host)

```bash
# (1) Prepare GCU-on-Ollama-base branch
cd /Users/root1/github/llama.cpp.claude
git fetch origin master
git checkout -b feat/ggml-gcu-on-ollama-base
# Rebase onto Ollama's pinned upstream SHA:
git rebase --onto ec98e2002 cf5e623ba feat/ggml-gcu
# Resolve conflicts in ggml/src/ggml-backend-reg.cpp.
git push winston-zhang-orz feat/ggml-gcu-on-ollama-base

# (2) Clone Ollama (outside this repo!)
cd ~
git clone https://github.com/ollama/ollama.git
cd ollama
git remote add winston-fork git@github.com:winston-zhang-orz/ollama.git  # if we'll PR our own fork

# (3) Swap vendored tree: pull our rebased branch into llama/vendor
make -f Makefile.sync clean   # resets llama/vendor to ec98e2002
cd llama/vendor
git remote add gcu https://github.com/winston-zhang-orz/llama.cpp.git
git fetch gcu feat/ggml-gcu-on-ollama-base
git checkout feat/ggml-gcu-on-ollama-base
cd ../..

# (4) Update the pin and re-apply Ollama's patches
# Edit Makefile.sync: set FETCH_HEAD=<sha of our rebased tip>
# Then:
make -f Makefile.sync apply-patches   # 36 patches, expect manual resolve on 0007
make -f Makefile.sync sync             # rsync into llama/llama.cpp + ml/backend/ggml/ggml

# (5) Add GCU to CMake (the 4-5 file changes from §3.2)
# (commit as a separate Ollama-side change so it can be PR'd independently)
```

### 7.2 On S60 (`agent@10.12.111.158`)

```bash
ssh agent@10.12.111.158
# Sync our prepared Ollama tree:
rsync -avz --delete ~/ollama-with-gcu/ agent@10.12.111.158:~/claude/ollama/
ssh agent@10.12.111.158 << 'EOF'
cd ~/claude/ollama

# (1) Build the GCU backend module
source /opt/tops/setvars.sh   # exports TOPS_INSTALL, sets LD_LIBRARY_PATH
cmake -B build/gcu --preset "GCU" \
  -DCMAKE_PREFIX_PATH=/opt/tops/cmake_modules \
  -DGGML_GCU=ON \
  -DOLLAMA_RUNNER_DIR=gcu
cmake --build build/gcu --target ggml-gcu --config Release -j$(nproc)
cmake --install build/gcu --component GCU --strip --prefix dist

# (2) Build CPU base (always required)
cmake -B build/cpu --preset "CPU"
cmake --build build/cpu --target ggml-cpu --config Release -j$(nproc)
cmake --install build/cpu --component CPU --strip --prefix dist

# (3) Build the Go ollama binary
go build -trimpath -o dist/bin/ollama .

# (4) Smoke test discovery
OLLAMA_DEBUG=1 ./dist/bin/ollama serve &
sleep 3
# Look for "GPU bootstrap discovery took" + a device with Library=GCU in the logs.

# (5) Pull and run
./dist/bin/ollama pull llama3.2:1b
OLLAMA_DEBUG=1 ./dist/bin/ollama run llama3.2:1b "Hello, world."
# Expect to see in the runner log: "--device GCU0" or device entries with Name="GCU0".

# (6) Confirm GCU is actually used
# In another shell:
watch -n 0.5 'cat /sys/class/gcu/gcu0/utilization || /opt/tops/bin/topsop-stats'
# Expect non-zero GCU utilization during inference.
EOF
```

### 7.3 Expected log signatures

- Bootstrap discovery: `msg="GPU bootstrap discovery took"` followed by `slog.Info("gpu memory", "id", "0", "library", "GCU", ...)` from `server/sched.go:491`.
- Runner load: `ggml_backend_load_all_from_path` debug line in `ml/backend/ggml/ggml/src/ggml.go:94` mentioning the gcu dir.
- Inference: our backend prints `GCU0: dispatched <op> on GCU` lines (if our existing logging is intact).

## 8. Risks and unknowns

### 8.1 Likely-to-bite

1. **Patch 0020 (no-alloc mode) compliance.** Ollama probes memory by loading the model in a "dry-run" mode that estimates allocations without actually calling `cudaMalloc`/equivalent. If our backend's `init_tensor`/`alloc_buffer` paths don't gate on the no-alloc flag, Ollama's memory estimator will OOM-panic during scheduling. **Status: not implemented in our fork.** Mitigation: stub `init_tensor` to skip the topsMalloc when the no-alloc flag is set; return 0 from the buffer-type's `get_alloc_size` is already correct.
2. **Patch 0015 (GPU UUIDs / PCIID).** Ollama dedups devices by `PCIID`. Our backend currently doesn't populate `props.device_id`. If a future hypothetical second-backend ever enumerates the same physical chip, Ollama would treat them as separate devices. **For S60 with one backend it's a non-issue today; fix when we ever ship a second route.**
3. **The `0007-sort-devices-by-score.patch` collision.** Already discussed — minor.
4. **Rebase backend-API drift.** Our backend was written ~1600 commits ago. Newer ggml has additions (e.g. `ggml_backend_dev_props.driver_major/minor`, `device_id` ptr, the `library` ptr). We may need to populate these. Our existing code already sets `props->name` and similar; just enumerate the new fields after rebase and ensure compile.
5. **Patch 0007 sort-by-score scoring expects a `score` from each backend.** Our backend would default to 0, which Ollama might treat as "low priority" and rank below CPU. Need to verify by reading the actual patch behaviour, and set a reasonable score (e.g. 50, matching CUDA tier).

### 8.2 Build-system risks

- **Ollama's CI builds inside a Dockerfile.** Adding GCU likely needs a Topsrider toolchain layer in the Docker image — non-trivial. For a dev build, building directly on the S60 (skipping Docker) is the path of least resistance.
- **`GGML_BACKEND_DL=ON` requirement.** Our `ggml/src/ggml-gcu/CMakeLists.txt` must declare itself as a `MODULE` library (not `STATIC`/`SHARED`) when `GGML_BACKEND_DL` is on. The upstream `ggml_add_backend_library()` macro handles this automatically if we use it — **verify we do** (currently we have a hand-rolled CMakeLists; may need to migrate to the macro).
- **Multiple CPU-feature variants.** Ollama builds `ggml-cpu-skylakex`, `ggml-cpu-icelake`, etc. as separate `.so`s. The GCU build's CPU dep should be `ggml-base`, not `ggml-cpu`, to avoid pinning a specific CPU variant.

### 8.3 Licensing posture

- **Ollama**: MIT (`/tmp/ollama-research/ollama/LICENSE`).
- **Our fork**: MIT (inherits llama.cpp).
- **Topsrider SDK** (`/opt/tops/lib/libtopsrt.so`): licensed by Enflame, redistribution terms TBD. **Compatibility implication**: we cannot bundle `libtopsrt.so` into the Ollama tarball. The runner will dlopen it at runtime from `/opt/tops/lib`, requiring `LD_LIBRARY_PATH` or rpath. The CMake `install(TARGETS ggml-gcu RUNTIME_DEPENDENCIES ... DIRECTORIES /opt/tops/lib ...)` should be replaced with a doc note ("install topsrider SDK separately").

### 8.4 Upstream signal

Ollama PR landscape (per WebSearch):
- **#13648 MLX backend** — merged Jan 2026.
- **Vulkan support PR** — merged Oct 2025.
- **#16036 SYCL backend** — open draft (May 2026).
- **#11160, #10322 Intel GPU PRs** — open since mid-2025.

Conclusion: **Ollama does accept new ggml backends in principle** (MLX, Vulkan landed), but the bar is high (SYCL has been in flight for >12 months). For our purposes, **maintaining a fork branch is more realistic than upstreaming** until GCU has wider deployment.

## 9. Recommended next step (MVP)

**MVP work item: get `ollama run llama3.2:1b` to dispatch to GCU0 on the S60, building locally on the S60 (no Docker, no MLX, no PR).**

Concretely:
1. Rebase our `feat/ggml-gcu` branch onto upstream `ec98e2002` (path (a)). One commit per file at most. ~2 hours.
2. Add the 4-5 build-system file changes in Ollama (§3.2). ~1 hour.
3. Add the 30-line Go branch in `ml/device.go` to handle `Library == "GCU"`. ~30 minutes.
4. Implement the no-alloc-mode gate in our backend's `init_tensor`/`alloc_buffer` callbacks (patch 0020 obligation). ~3 hours.
5. Set `props->library = "GCU"` and `props->name = "GCU0"` in our `get_dev_props()` if not already. ~30 minutes.
6. Build on S60, run `ollama run llama3.2:1b`, capture log proving GCU dispatch. ~4 hours of iteration.

**Total budget: 1–2 working days for someone familiar with both codebases.**

**Stretch (once MVP works):**
- Plumb the remaining patch obligations (UUIDs, batch-size hint, reset-backend) so Ollama's memory estimator/scheduler produces accurate numbers.
- Add an `OLLAMA_GCU=1` env-var gate at `discover/runner.go:108` so the GCU libDir is skipped on non-S60 machines.
- Add a `CMakePresets.json` "GCU" preset and a `scripts/build_linux_gcu.sh` so CI can produce `ollama-linux-amd64-gcu.tar.zst` alongside the existing `-rocm` variant.
- Prepare an upstream Ollama PR adding the build-system hooks (the C++ side already lives in our llama.cpp fork; the Ollama PR is just CMake glue + ~30 lines of Go).
- Long-term: PR our `ggml-gcu/` backend to upstream `ggml-org/llama.cpp` so Ollama can pick it up via a normal `FETCH_HEAD` bump rather than a fork swap.
