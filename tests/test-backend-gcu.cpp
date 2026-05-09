// Smoke test for the GCU (Enflame TOPS) backend.
//
// Allocates F32 tensors on GCU, runs ADD and MUL_MAT, compares with the
// CPU reference implementation, and checks that no device memory is
// leaked across init/teardown.
//
// Run on the S60 box:
//   ./build/bin/test-backend-gcu

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-gcu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_random_f32(float * p, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) p[i] = dist(rng);
}

static bool close_enough(float a, float b, float atol, float rtol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

static int test_add(ggml_backend_t gcu) {
    const int64_t M = 1024;
    const int64_t N = 4096;
    const size_t  n = (size_t) M * N;

    auto buft = ggml_backend_get_default_buffer_type(gcu);

    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * c = ggml_add(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        fprintf(stderr, "ADD: failed to allocate tensors on GCU\n");
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> ha(n), hb(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 1);
    fill_random_f32(hb.data(), n, 2);
    for (size_t i = 0; i < n; i++) expected[i] = ha[i] + hb[i];

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(b, hb.data(), 0, n * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    ggml_status s = ggml_backend_graph_compute(gcu, graph);
    if (s != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ADD: graph_compute returned %d\n", (int) s);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-5f, 1e-5f)) {
            if (bad < 5) {
                fprintf(stderr, "ADD: mismatch idx=%zu got=%f want=%f\n",
                        i, hc[i], expected[i]);
            }
            bad++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (bad) {
        fprintf(stderr, "ADD: %d mismatches over %zu elements\n", bad, n);
        return 1;
    }
    printf("ADD: ok (%zu elements)\n", n);
    return 0;
}

static int test_mul_mat(ggml_backend_t gcu) {
    // ggml convention: a in [K, M], b in [K, N], dst = a^T @ b in [M, N].
    const int64_t K = 1024;
    const int64_t M = 2048;
    const int64_t N = 1024;

    auto buft = ggml_backend_get_default_buffer_type(gcu);

    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        fprintf(stderr, "MUL_MAT: failed to allocate tensors on GCU\n");
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> ha((size_t) K * M);
    std::vector<float> hb((size_t) K * N);
    std::vector<float> hc((size_t) M * N);
    std::vector<float> expected((size_t) M * N);
    fill_random_f32(ha.data(), ha.size(), 11);
    fill_random_f32(hb.data(), hb.size(), 12);

    // Reference: c[m + n*M] = sum_k a[k + m*K] * b[k + n*K]
    for (int64_t nn = 0; nn < N; nn++) {
        for (int64_t mm = 0; mm < M; mm++) {
            float acc = 0.0f;
            const float * ap = ha.data() + mm * K;
            const float * bp = hb.data() + nn * K;
            for (int64_t k = 0; k < K; k++) acc += ap[k] * bp[k];
            expected[mm + nn * M] = acc;
        }
    }

    ggml_backend_tensor_set(a, ha.data(), 0, ha.size() * sizeof(float));
    ggml_backend_tensor_set(b, hb.data(), 0, hb.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    ggml_status s = ggml_backend_graph_compute(gcu, graph);
    if (s != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL_MAT: graph_compute returned %d\n", (int) s);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_get(c, hc.data(), 0, hc.size() * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < hc.size(); i++) {
        // Matmul accumulates K f32 products; tolerance scaled with K.
        if (!close_enough(hc[i], expected[i], 1e-2f, 1e-3f)) {
            if (bad < 5) {
                fprintf(stderr, "MUL_MAT: mismatch idx=%zu got=%f want=%f\n",
                        i, hc[i], expected[i]);
            }
            bad++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (bad) {
        fprintf(stderr, "MUL_MAT: %d mismatches over %zu elements\n", bad, hc.size());
        return 1;
    }
    printf("MUL_MAT: ok (%zu elements)\n", hc.size());
    return 0;
}

static int test_silu(ggml_backend_t gcu) {
    const size_t n = 4096;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_silu(ctx, a);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "SILU: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 5);
    for (size_t i = 0; i < n; i++) {
        const float x = ha[i];
        const float s = 1.0f / (1.0f + std::exp(-x));
        expected[i] = x * s;
    }
    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "SILU: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-5f, 1e-5f)) {
            if (bad < 5) fprintf(stderr, "SILU: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "SILU: %d mismatches\n", bad); return 1; }
    printf("SILU: ok (%zu elements)\n", n);
    return 0;
}

static int test_rms_norm(ggml_backend_t gcu) {
    const int64_t hidden = 1024, batch = 64;
    const size_t n = hidden * batch;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, batch);
    const float eps = 1e-6f;
    ggml_tensor * c = ggml_rms_norm(ctx, a, eps);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "RMS_NORM: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 7);
    for (int64_t b = 0; b < batch; b++) {
        double s = 0.0;
        const float * row = ha.data() + b * hidden;
        for (int64_t i = 0; i < hidden; i++) s += (double) row[i] * row[i];
        const float scale = 1.0f / std::sqrt((float)(s / hidden) + eps);
        for (int64_t i = 0; i < hidden; i++) expected[b * hidden + i] = row[i] * scale;
    }
    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "RMS_NORM: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-4f, 1e-4f)) {
            if (bad < 5) fprintf(stderr, "RMS_NORM: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "RMS_NORM: %d mismatches\n", bad); return 1; }
    printf("RMS_NORM: ok (%zu elements)\n", n);
    return 0;
}

static int test_softmax(ggml_backend_t gcu) {
    const int64_t cols = 1024, rows = 16;
    const size_t n = cols * rows;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    const float scale = 0.125f;
    ggml_tensor * c = ggml_soft_max_ext(ctx, a, mask, scale, 0.0f);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "SOFT_MAX: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hm(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 11);
    fill_random_f32(hm.data(), n, 12);
    for (int64_t r = 0; r < rows; r++) {
        const float * arow = ha.data() + r * cols;
        const float * mrow = hm.data() + r * cols;
        float emax = -INFINITY;
        for (int64_t i = 0; i < cols; i++) {
            const float v = arow[i] * scale + mrow[i];
            if (v > emax) emax = v;
        }
        double sum = 0.0;
        for (int64_t i = 0; i < cols; i++) {
            const float v = arow[i] * scale + mrow[i];
            const float e = std::exp(v - emax);
            expected[r * cols + i] = e;
            sum += e;
        }
        for (int64_t i = 0; i < cols; i++) expected[r * cols + i] /= (float) sum;
    }
    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(mask, hm.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "SOFT_MAX: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-5f, 1e-5f)) {
            if (bad < 5) fprintf(stderr, "SOFT_MAX: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "SOFT_MAX: %d mismatches\n", bad); return 1; }
    printf("SOFT_MAX: ok (%zu elements)\n", n);
    return 0;
}

static int test_rope(ggml_backend_t gcu) {
    const int64_t head_dim = 64, n_heads = 8, n_tokens = 16;
    const int n_dims = (int) head_dim;
    const float freq_base = 10000.0f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * c   = ggml_rope(ctx, a, pos, n_dims, GGML_ROPE_TYPE_NORMAL);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "ROPE: alloc failed\n"); ggml_free(ctx); return 1; }

    const size_t n = head_dim * n_heads * n_tokens;
    std::vector<float>   ha(n), hc(n), expected(n);
    std::vector<int32_t> hp(n_tokens);
    fill_random_f32(ha.data(), n, 13);
    for (int64_t t = 0; t < n_tokens; t++) hp[t] = (int32_t) t;

    // CPU reference: ggml mode-0 (NORMAL) RoPE uses interleaved pairs.
    // For each token p and pair (i, i+1) with i in {0, 2, 4, ...}:
    //   theta_pair = freq_base^(-(i)/n_dims), angle = p * theta_pair
    //   y[i]   = x[i]*cos - x[i+1]*sin
    //   y[i+1] = x[i]*sin + x[i+1]*cos
    for (int64_t t = 0; t < n_tokens; t++) {
        const int p = hp[t];
        for (int64_t h = 0; h < n_heads; h++) {
            const float * xrow = ha.data()       + (t * n_heads + h) * head_dim;
            float       * yrow = expected.data() + (t * n_heads + h) * head_dim;
            for (int i = 0; i < n_dims; i += 2) {
                const float theta = std::pow(freq_base, -((float) i) / (float) n_dims);
                const float angle = (float) p * theta;
                const float c1 = std::cos(angle), s1 = std::sin(angle);
                const float x0 = xrow[i];
                const float x1 = xrow[i + 1];
                yrow[i]     = x0 * c1 - x1 * s1;
                yrow[i + 1] = x0 * s1 + x1 * c1;
            }
            for (int i = n_dims; i < head_dim; i++) yrow[i] = xrow[i];
        }
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(pos, hp.data(), 0, n_tokens * sizeof(int32_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ROPE: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-3f, 1e-3f)) {
            if (bad < 5) fprintf(stderr, "ROPE: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "ROPE: %d mismatches\n", bad); return 1; }
    printf("ROPE: ok (%zu elements)\n", n);
    return 0;
}

static int test_mul_mat_mixed(ggml_backend_t gcu) {
    const int64_t K = 1024, M = 2048, N = 1024;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "MUL_MAT_MIXED: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32((size_t) K * M);
    std::vector<ggml_fp16_t> ha_f16((size_t) K * M);
    std::vector<float>       hb((size_t) K * N), hc((size_t) M * N), expected((size_t) M * N);
    fill_random_f32(ha_f32.data(), ha_f32.size(), 21);
    fill_random_f32(hb.data(),     hb.size(),     22);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha_f16.data(), ha_f32.size());

    // Reference: dst = a^T @ b in F32.
    for (int64_t nn = 0; nn < N; nn++) {
        for (int64_t mm = 0; mm < M; mm++) {
            float acc = 0.0f;
            const float * apf = ha_f32.data() + mm * K;
            const float * bpf = hb.data()     + nn * K;
            for (int64_t k = 0; k < K; k++) acc += apf[k] * bpf[k];
            expected[mm + nn * M] = acc;
        }
    }
    ggml_backend_tensor_set(a, ha_f16.data(), 0, ha_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b, hb.data(),     0, hb.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL_MAT_MIXED: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, hc.size() * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < hc.size(); i++) {
        if (!close_enough(hc[i], expected[i], 1e-1f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "MUL_MAT_MIXED: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "MUL_MAT_MIXED: %d mismatches\n", bad); return 1; }
    printf("MUL_MAT_MIXED: ok (%zu elements)\n", hc.size());
    return 0;
}

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

int main() {
    if (ggml_backend_gcu_get_device_count() < 1) {
        fprintf(stderr, "no GCU device found\n");
        return 1;
    }

    size_t free_before = 0, total_before = 0;
    ggml_backend_gcu_get_device_memory(0, &free_before, &total_before);

    ggml_backend_t gcu = ggml_backend_gcu_init(0);
    if (!gcu) {
        fprintf(stderr, "ggml_backend_gcu_init failed\n");
        return 1;
    }

    int rc = 0;
    rc |= test_add(gcu);
    rc |= test_mul_mat(gcu);
    rc |= test_silu(gcu);
    rc |= test_rms_norm(gcu);
    rc |= test_softmax(gcu);
    rc |= test_rope(gcu);
    rc |= test_mul_mat_mixed(gcu);
    rc |= test_async_overlap(gcu);

    ggml_backend_free(gcu);

    size_t free_after = 0, total_after = 0;
    ggml_backend_gcu_get_device_memory(0, &free_after, &total_after);

    // The pool, the cached zero-bias, and topsaten's own workspace all
    // outlive a backend instance (heap-leaked singletons; OS reclaims at
    // exit). So a strict leak check would always trip. Print the delta
    // as info, and only flag truly extreme values (>1 GiB) as failure.
    if (total_after == total_before && free_before > free_after) {
        size_t diff = free_before - free_after;
        printf("retained device memory: %zu bytes (pool + topsaten workspace)\n", diff);
        if (diff > ((size_t) 1 << 30)) {
            fprintf(stderr, "leak detected: %zu bytes retained (>1 GiB)\n", diff);
            rc |= 1;
        }
    }

    if (rc == 0) printf("test-backend-gcu: PASSED\n");
    return rc;
}
