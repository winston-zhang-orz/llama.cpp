// ggml-gcu: LIFO size-keyed device-memory pool.
//
// Recurring intermediate-tensor allocations during a forward pass have a
// small set of distinct sizes. Cache freed slabs by rounded size so we
// avoid topsMalloc/topsFree on the hot path. Cap the cache at 4 GiB to
// bound retained device memory.

#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

class gcu_pool {
public:
    explicit gcu_pool(int32_t device, size_t cap_bytes = (size_t) 4 << 30)
        : device_(device), cap_(cap_bytes) {}

    ~gcu_pool() { drain(); }

    void * alloc(size_t size);
    void   free(void * p, size_t size);

private:
    static size_t round_up(size_t n) {
        constexpr size_t G = 256;
        return (n + G - 1) & ~(G - 1);
    }
    void drain();

    int32_t device_;
    size_t  cap_;
    size_t  cached_bytes_ = 0;
    std::unordered_map<size_t, std::vector<void*>> free_lists_;
    std::mutex mu_;
};
