// ggml-gcu: LIFO size-keyed device-memory pool — implementation.

#include "common.h"
#include "gcu_pool.h"

void * gcu_pool::alloc(size_t size) {
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

void gcu_pool::free(void * p, size_t size) {
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

void gcu_pool::drain() {
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
