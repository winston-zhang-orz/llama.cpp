// ggml-gcu: device-memory pool + per-context state.
//
// The LIFO size-keyed pool caches freed device slabs to avoid
// topsMalloc/topsFree on the hot path. ggml_backend_gcu_context owns one
// pool plus the compute/copy streams, async events, and the MVP-4b
// deferred-free machinery (kept here because defer_free() recycles slabs
// straight back into the pool).

#pragma once

#include "common.h"

// === Process-level topsaten init refcount ===========================
//
// topsatenInit / topsatenFinalize are documented as process-global. Wrap
// with a mutex+counter so multiple ggml_backend_gcu contexts (one per
// device) bracket lifetime correctly without re-init.

void gcu_global_init_inc();
void gcu_global_init_dec();

// === LIFO size-keyed pool allocator =================================

// Recurring intermediate-tensor allocations during a forward pass have a
// small set of distinct sizes. Cache freed slabs by rounded size so we
// avoid topsMalloc/topsFree on the hot path. Cap the cache at 4 GiB to
// bound retained device memory.

class gcu_pool {
public:
    explicit gcu_pool(int32_t device, size_t cap_bytes = (size_t) 4 << 30)
        : device_(device), cap_(cap_bytes) {}

    ~gcu_pool() { drain(); }

    void * alloc(size_t size) {
        size_t key = round_up(size);
        std::lock_guard<std::mutex> lk(mu_);
        auto it = free_lists_.find(key);
        if (it != free_lists_.end() && !it->second.empty()) {
            void * p = it->second.back();
            it->second.pop_back();
            cached_bytes_ -= key;
            return p;
        }
        void * p = nullptr;
        TOPS_CHECK(topsSetDevice(device_));
        TOPS_CHECK(topsMalloc(&p, key));
        return p;
    }

    void free(void * p, size_t size) {
        if (!p) return;
        size_t key = round_up(size);
        std::lock_guard<std::mutex> lk(mu_);
        if (cached_bytes_ + key > cap_) {
            TOPS_CHECK(topsSetDevice(device_));
            TOPS_CHECK(topsFree(p));
            return;
        }
        free_lists_[key].push_back(p);
        cached_bytes_ += key;
    }

private:
    static size_t round_up(size_t n) {
        constexpr size_t G = 256;
        return (n + G - 1) & ~(G - 1);
    }
    void drain() {
        std::lock_guard<std::mutex> lk(mu_);
        TOPS_CHECK(topsSetDevice(device_));
        for (auto & kv : free_lists_) {
            for (void * p : kv.second) {
                TOPS_CHECK(topsFree(p));
            }
        }
        free_lists_.clear();
        cached_bytes_ = 0;
    }

    int32_t device_;
    size_t  cap_;
    size_t  cached_bytes_ = 0;
    std::unordered_map<size_t, std::vector<void*>> free_lists_;
    std::mutex mu_;
};

// === Per-context state ==============================================

#define GGML_GCU_NAME_MAX 64

struct ggml_backend_gcu_context {
    int32_t      device      = 0;
    std::string  name;          // "GCU0", "GCU1", ...
    std::string  description;   // populated from topsGetDeviceProperties
    topsStream_t compute_stream = nullptr;
    topsStream_t copy_stream    = nullptr;
    gcu_pool     pool;

    // Per-context zero-filled scratch used as bias for topsatenLinear's
    // mandatory bias parameter. Grows on demand to the largest output row
    // count seen so far, in F32 (largest dtype we use), and reinterpreted
    // for F16 calls. Only ever flows through compute_stream, so no lock
    // is needed.
    void *  zero_bias       = nullptr;
    size_t  zero_bias_bytes = 0;

    // Per-context F32 ones buffer used as the gamma argument to
    // topsvllmRmsNorm (which requires a real weight tensor).
    void *  ones_n0         = nullptr;
    size_t  ones_n0_bytes   = 0;
    int64_t ones_n0_count   = 0;

    // MVP-4a: async H<->D plumbing.
    // last_copy_event:      recorded on copy_stream after each set_tensor_async H->D enqueue
    // last_compute_event:   recorded on compute_stream at end of graph_compute
    // copy_event_armed:     guards the first wait — recording-without-arming is undefined SDK behavior
    // compute_event_armed:  symmetric guard for get_tensor_async; set by graph_compute (Task 3)
    topsEvent_t  last_copy_event    = nullptr;
    topsEvent_t  last_compute_event = nullptr;
    bool         copy_event_armed   = false;
    bool         compute_event_armed = false;

    // MVP-4b: scratch frees deferred to end of graph_compute so kernels
    // queue without per-op host synchronize and the next op's pool.alloc
    // can't reuse a buffer the previous op's kernel is still reading.
    std::vector<std::pair<void *, size_t>> deferred_frees;
    size_t                                  deferred_bytes = 0;

    // Cap outstanding deferred scratch at this many bytes per graph_compute.
    // Real-model graphs stay well under this (a Llama-3.2-1B prefill peaks at
    // tens of MB of deferred FA scratch); the cap only triggers in
    // pathological graphs like test-backend-ops perf-mode, which duplicates
    // a single op thousands of times into one graph and would otherwise
    // accumulate tens of GiB of outstanding scratch before the end-of-graph
    // drain. When we hit the cap we synchronize the compute stream and
    // recycle the slabs through the pool so subsequent allocs can reuse them.
    // 512 MiB is comfortably above any real-model peak we've measured and
    // also stays clear of the 1 GiB smoke-mode pool-retention sanity check.
    static constexpr size_t deferred_bytes_cap = (size_t) 512 << 20; // 512 MiB

    void defer_free(void * p, size_t sz) {
        if (!p) return;
        deferred_frees.emplace_back(p, sz);
        deferred_bytes += sz;
        if (deferred_bytes > deferred_bytes_cap) {
            // Block until queued kernels complete, then return slabs to the
            // pool so the next op's alloc can reuse them. This is a no-op in
            // steady state for real models; it only ever fires for synthetic
            // benchmark graphs that fan a tiny op out thousands of times.
            TOPS_CHECK(topsStreamSynchronize(compute_stream));
            for (auto & kv : deferred_frees) {
                pool.free(kv.first, kv.second);
            }
            deferred_frees.clear();
            deferred_bytes = 0;
        }
    }

    explicit ggml_backend_gcu_context(int32_t dev) : device(dev), pool(dev) {
        deferred_frees.reserve(64);
        TOPS_CHECK(topsSetDevice(device));
        gcu_global_init_inc();
        TOPS_CHECK(topsStreamCreate(&compute_stream));
        TOPS_CHECK(topsStreamCreate(&copy_stream));
        TOPS_CHECK(topsEventCreateWithFlags(&last_copy_event,    topsEventDisableTiming));
        TOPS_CHECK(topsEventCreateWithFlags(&last_compute_event, topsEventDisableTiming));

        char buf[GGML_GCU_NAME_MAX];
        snprintf(buf, sizeof(buf), "GCU%d", device);
        name = buf;

        topsDeviceProp_t prop{};
        if (topsGetDeviceProperties(&prop, device) == topsSuccess) {
            description = prop.name;
        } else {
            description = "Enflame GCU";
        }
    }

    ~ggml_backend_gcu_context() {
        if (ones_n0) {
            pool.free(ones_n0, ones_n0_bytes);
            ones_n0 = nullptr;
            ones_n0_bytes = 0;
            ones_n0_count = 0;
        }
        if (zero_bias) {
            pool.free(zero_bias, zero_bias_bytes);
            zero_bias = nullptr;
            zero_bias_bytes = 0;
        }
        if (last_compute_event) {
            TOPS_CHECK(topsEventDestroy(last_compute_event));
            last_compute_event = nullptr;
        }
        if (last_copy_event) {
            TOPS_CHECK(topsEventDestroy(last_copy_event));
            last_copy_event = nullptr;
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
        gcu_global_init_dec();
    }

    ggml_backend_gcu_context(const ggml_backend_gcu_context &) = delete;
    ggml_backend_gcu_context & operator=(const ggml_backend_gcu_context &) = delete;
};
