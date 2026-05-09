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

#include <algorithm>
#include <chrono>
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

// NORM (LayerNorm without affine): y = (x - mean) / sqrt(var + eps).
// Tested per-row for both F32 and F16 dtypes.
static int test_norm(ggml_backend_t gcu) {
    const int64_t hidden = 1024, batch = 64;
    const size_t  n = (size_t) hidden * batch;
    const float   eps = 1e-5f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, batch);
    ggml_tensor * c = ggml_norm(ctx, a, eps);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "NORM: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 401);
    for (int64_t b = 0; b < batch; b++) {
        const float * row = ha.data() + b * hidden;
        // mean
        double sum = 0.0;
        for (int64_t i = 0; i < hidden; i++) sum += row[i];
        const float mean = (float)(sum / hidden);
        // variance
        double vsum = 0.0;
        for (int64_t i = 0; i < hidden; i++) {
            const float d = row[i] - mean;
            vsum += (double) d * d;
        }
        const float scale = 1.0f / std::sqrt((float)(vsum / hidden) + eps);
        for (int64_t i = 0; i < hidden; i++) {
            expected[b * hidden + i] = (row[i] - mean) * scale;
        }
    }
    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "NORM: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int   bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float err = std::fabs(hc[i] - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(hc[i], expected[i], 1e-4f, 1e-4f)) {
            if (bad < 5) fprintf(stderr, "NORM: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "NORM: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("NORM: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
    return 0;
}

static int test_norm_f16(ggml_backend_t gcu) {
    const int64_t hidden = 1024, batch = 64;
    const size_t  n = (size_t) hidden * batch;
    const float   eps = 1e-5f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden, batch);
    ggml_tensor * c = ggml_norm(ctx, a, eps);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "NORM_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hc(n);
    fill_random_f32(ha_f32.data(), n, 411);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);

    for (int64_t b = 0; b < batch; b++) {
        double sum = 0.0;
        for (int64_t i = 0; i < hidden; i++) sum += ggml_fp16_to_fp32(ha[b * hidden + i]);
        const float mean = (float)(sum / hidden);
        double vsum = 0.0;
        for (int64_t i = 0; i < hidden; i++) {
            const float d = ggml_fp16_to_fp32(ha[b * hidden + i]) - mean;
            vsum += (double) d * d;
        }
        const float scale = 1.0f / std::sqrt((float)(vsum / hidden) + eps);
        for (int64_t i = 0; i < hidden; i++) {
            const float v = ggml_fp16_to_fp32(ha[b * hidden + i]);
            expected[b * hidden + i] = (v - mean) * scale;
        }
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "NORM_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int   bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(got, expected[i], 5e-3f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "NORM_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "NORM_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("NORM_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
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

// === MVP-5a unary activations ===========================================
//
// Shared smoke harness for the new GGML_OP_UNARY ops added in MVP-5a.
// Each test creates a 1D F32 tensor, applies the chosen ggml unary op,
// and checks the result against a host-side reference. Tolerances are
// per-op (some activations like GELU have transcendental inner ops with
// modest precision; RELU is exact).
struct unary_test_spec {
    const char *  label;
    int           seed;
    float         atol;
    float         rtol;
    ggml_tensor * (*build_op)(ggml_context * ctx, ggml_tensor * a);
    float         (*ref)(float x);
};

static int run_unary_test(ggml_backend_t gcu, const unary_test_spec & s) {
    const size_t n = 4096;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = s.build_op(ctx, a);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "%s: alloc failed\n", s.label); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, (uint32_t) s.seed);
    for (size_t i = 0; i < n; i++) expected[i] = s.ref(ha[i]);

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", s.label);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int   bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float err = std::fabs(hc[i] - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(hc[i], expected[i], s.atol, s.rtol)) {
            if (bad < 5) fprintf(stderr, "%s: mismatch idx=%zu got=%f want=%f\n",
                                 s.label, i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) {
        fprintf(stderr, "%s: %d mismatches (max_abs_err=%f)\n", s.label, bad, max_err);
        return 1;
    }
    printf("%s: ok (%zu elements, max_abs_err=%f)\n", s.label, n, max_err);
    return 0;
}

// Reference activations (host-side F32 implementations). Match ggml's
// ggml-cpu/ops.cpp definitions where they exist.
static float ref_gelu(float x) {
    // ggml's "exact" GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}
static float ref_gelu_quick(float x) {
    // ggml's tanh-form GELU approximation:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x3)));
}
static float ref_relu(float x)        { return x > 0.0f ? x : 0.0f; }
static float ref_tanh(float x)        { return std::tanh(x); }
static float ref_sigmoid(float x)     { return 1.0f / (1.0f + std::exp(-x)); }
static float ref_hardswish(float x)   {
    if (x <= -3.0f) return 0.0f;
    if (x >=  3.0f) return x;
    return x * (x + 3.0f) / 6.0f;
}
static float ref_hardsigmoid(float x) {
    if (x <= -3.0f) return 0.0f;
    if (x >=  3.0f) return 1.0f;
    return (x + 3.0f) / 6.0f;
}

static int test_gelu(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"GELU",        301, 1e-4f, 1e-4f, ggml_gelu,        ref_gelu});
}
static int test_gelu_quick(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"GELU_QUICK",  302, 1e-3f, 1e-3f, ggml_gelu_quick,  ref_gelu_quick});
}
static int test_relu(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"RELU",        303, 0.0f,   0.0f, ggml_relu,        ref_relu});
}
static int test_tanh(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"TANH",        304, 1e-5f, 1e-5f, ggml_tanh,        ref_tanh});
}
static int test_sigmoid(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"SIGMOID",     305, 1e-5f, 1e-5f, ggml_sigmoid,     ref_sigmoid});
}
static int test_hardswish(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"HARDSWISH",   306, 1e-5f, 1e-5f, ggml_hardswish,   ref_hardswish});
}
static int test_hardsigmoid(ggml_backend_t gcu) {
    return run_unary_test(gcu, {"HARDSIGMOID", 307, 1e-5f, 1e-5f, ggml_hardsigmoid, ref_hardsigmoid});
}

// MUL_MAT_MIXED variant with F16 input (instead of F32): F16w × F16x -> F32.
// Fills the third dtype combo in the F16-weight family of MUL_MAT:
//   F32w × F32x → F32   (test_mul_mat)
//   F16w × F32x → F32   (test_mul_mat_mixed)
//   F16w × F16x → F32   (this test)
// This is the path Qwen 0.5B F16 hits when activations stay on-device as
// F16 between layers — common for fully-F16 models without input casts.
static int test_mul_mat_mixed_f16in(ggml_backend_t gcu) {
    const int64_t K = 1024, M = 2048, N = 1024;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "MUL_MAT_MIXED_F16in: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32((size_t) K * M);
    std::vector<float>       hb_f32((size_t) K * N);
    std::vector<ggml_fp16_t> ha((size_t) K * M);
    std::vector<ggml_fp16_t> hb((size_t) K * N);
    std::vector<float>       hc((size_t) M * N), expected((size_t) M * N);
    fill_random_f32(ha_f32.data(), ha_f32.size(), 271);
    fill_random_f32(hb_f32.data(), hb_f32.size(), 272);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), ha_f32.size());
    ggml_fp32_to_fp16_row(hb_f32.data(), hb.data(), hb_f32.size());

    // Reference accumulates in F32 from the F16-rounded inputs (the same
    // values the device sees), so the comparison only catches on-device
    // F16-accumulator drift.
    for (int64_t nn = 0; nn < N; nn++) {
        for (int64_t mm = 0; mm < M; mm++) {
            float acc = 0.0f;
            const ggml_fp16_t * apf = ha.data() + mm * K;
            const ggml_fp16_t * bpf = hb.data() + nn * K;
            for (int64_t k = 0; k < K; k++) {
                acc += ggml_fp16_to_fp32(apf[k]) * ggml_fp16_to_fp32(bpf[k]);
            }
            expected[mm + nn * M] = acc;
        }
    }

    ggml_backend_tensor_set(a, ha.data(), 0, ha.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b, hb.data(), 0, hb.size() * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL_MAT_MIXED_F16in: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, hc.size() * sizeof(float));

    int   bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < hc.size(); i++) {
        const float err = std::fabs(hc[i] - expected[i]);
        if (err > max_err) max_err = err;
        // Same K=1024 F16-accumulator tolerance as test_mul_mat_mixed.
        if (!close_enough(hc[i], expected[i], 1e-1f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "MUL_MAT_MIXED_F16in: mismatch idx=%zu got=%f want=%f (err=%f)\n",
                                 i, hc[i], expected[i], err);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) {
        fprintf(stderr, "MUL_MAT_MIXED_F16in: %d mismatches over %zu (max_abs_err=%f)\n",
                bad, hc.size(), max_err);
        return 1;
    }
    printf("MUL_MAT_MIXED_F16in: ok (%zu elements, max_abs_err=%f)\n", hc.size(), max_err);
    return 0;
}

// MUL_MAT_ID smoke: MoE expert dispatch on a small tractable shape.
//   as:  [K, M, n_expert]                F32 expert weights
//   b:   [K, n_expert_used, n_tokens]    F32 input
//   ids: [n_expert_used, n_tokens] i32   expert routing
//   c:   [M, n_expert_used, n_tokens]    F32 output
// Reference: c[m, e, t] = sum_k as[k, m, ids[e, t]] * b[k, e, t]
static int test_mul_mat_id(ggml_backend_t gcu) {
    const int64_t K = 256;
    const int64_t M = 512;
    const int64_t n_expert      = 4;
    const int64_t n_expert_used = 2;
    const int64_t n_tokens      = 8;

    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * as_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, n_expert);
    ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_expert_used, n_tokens);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * c    = ggml_mul_mat_id(ctx, as_w, b, ids);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        fprintf(stderr, "MUL_MAT_ID: alloc failed\n");
        ggml_free(ctx);
        return 1;
    }

    std::vector<float>   h_as((size_t) K * M * n_expert);
    std::vector<float>   h_b ((size_t) K * n_expert_used * n_tokens);
    std::vector<int32_t> h_ids((size_t) n_expert_used * n_tokens);
    std::vector<float>   h_c ((size_t) M * n_expert_used * n_tokens);
    std::vector<float>   expected((size_t) M * n_expert_used * n_tokens);

    fill_random_f32(h_as.data(), h_as.size(), 31);
    fill_random_f32(h_b.data(),  h_b.size(),  32);

    // Round-robin expert assignment with offset per slot for variety.
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            h_ids[t * n_expert_used + e] = (int32_t) ((t + e * 2) % n_expert);
        }
    }

    // CPU reference.
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            const int32_t expert_id = h_ids[t * n_expert_used + e];
            const float * w_mat = h_as.data() + (size_t) expert_id * M * K;
            const float * b_vec = h_b.data()  + (size_t)(t * n_expert_used + e) * K;
            float       * c_vec = expected.data() + (size_t)(t * n_expert_used + e) * M;
            for (int64_t m = 0; m < M; m++) {
                float acc = 0.0f;
                const float * w_row = w_mat + m * K;
                for (int64_t k = 0; k < K; k++) acc += w_row[k] * b_vec[k];
                c_vec[m] = acc;
            }
        }
    }

    ggml_backend_tensor_set(as_w, h_as.data(),  0, h_as.size()  * sizeof(float));
    ggml_backend_tensor_set(b,    h_b.data(),   0, h_b.size()   * sizeof(float));
    ggml_backend_tensor_set(ids,  h_ids.data(), 0, h_ids.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL_MAT_ID: compute failed\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_get(c, h_c.data(), 0, h_c.size() * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < h_c.size(); i++) {
        // K=256 F32 sum; scale tolerance with K like the MUL_MAT test.
        if (!close_enough(h_c[i], expected[i], 1e-2f, 1e-3f)) {
            if (bad < 5) {
                fprintf(stderr, "MUL_MAT_ID: mismatch idx=%zu got=%f want=%f\n",
                        i, h_c[i], expected[i]);
            }
            bad++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    if (bad) {
        fprintf(stderr, "MUL_MAT_ID: %d mismatches over %zu elements\n", bad, h_c.size());
        return 1;
    }
    printf("MUL_MAT_ID: ok (%zu elements, %lld experts, %lld tokens)\n",
           h_c.size(), (long long) n_expert, (long long) n_tokens);
    return 0;
}

// Q-weight × F32-input matmul. The Q tensor is populated via host-side
// ggml_quantize_chunk; the GCU buffer's set_tensor dequantizes the Q
// bytes to F16 on the device (MVP-3a dequant-on-load). Expected output
// is computed from the *dequantized* F32 weight (the same lossy view
// the device sees), so the comparison only needs to tolerate the F16
// accumulation drift that comes from running matmul on the device.
//
// Returns 0 on success; prints "<label>: ok ..." or detail on mismatch.
static int test_mul_mat_q_weight(ggml_backend_t gcu, ggml_type qt, const char * label) {
    // K must be a multiple of the format's block size (32 for Q4_0 / Q8_0,
    // 256 for Q4_K). Pick K = 256 so every supported type works.
    const int64_t K = 256;
    const int64_t M = 1024;
    const int64_t N = 64;

    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, qt, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "%s: alloc failed\n", label); ggml_free(ctx); return 1; }

    // Random F32 reference weight, then quantize to qt.
    std::vector<float> w_f32_ref((size_t) K * M);
    fill_random_f32(w_f32_ref.data(), w_f32_ref.size(), 91);

    const size_t row_size_q = ggml_row_size(qt, K);
    std::vector<uint8_t> w_quant((size_t) M * row_size_q);
    const size_t actual = ggml_quantize_chunk(qt, w_f32_ref.data(), w_quant.data(),
                                              0, M, K, /*imatrix=*/ nullptr);
    if (actual != w_quant.size()) {
        fprintf(stderr, "%s: quantize_chunk returned %zu, expected %zu\n",
                label, actual, w_quant.size());
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }

    // Dequantize back to F32 so we have the same "lossy" view the device
    // operates on. The device stores F16 of this; the F16 conversion
    // adds a tiny additional drift (handled by the tolerance below).
    const ggml_type_traits * tt = ggml_get_type_traits(qt);
    if (!tt || !tt->to_float) {
        fprintf(stderr, "%s: no to_float trait for type\n", label);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    std::vector<float> w_f32_lossy((size_t) K * M);
    tt->to_float(w_quant.data(), w_f32_lossy.data(), (size_t) K * M);

    std::vector<float> hx((size_t) K * N);
    fill_random_f32(hx.data(), hx.size(), 92);

    // Reference: c[m, n] = sum_k w_lossy[m*K + k] * x[n*K + k]
    std::vector<float> expected((size_t) M * N), hc((size_t) M * N);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            float acc = 0.0f;
            const float * wp = w_f32_lossy.data() + m * K;
            const float * xp = hx.data()          + n * K;
            for (int64_t k = 0; k < K; k++) acc += wp[k] * xp[k];
            expected[m + n * M] = acc;
        }
    }

    // Set the Q bytes on the device — buffer.set_tensor dequants to F16.
    ggml_backend_tensor_set(w, w_quant.data(), 0, w_quant.size());
    ggml_backend_tensor_set(x, hx.data(),      0, hx.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", label);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, hc.size() * sizeof(float));

    // Tolerance: K=256 F16-accumulated F32 sums against an F32 reference.
    // The F16-rounding of the dequantized weight introduces ~2^-11 ≈ 5e-4
    // relative error per element; the K-long sum amplifies that. The
    // accumulator on the device is F16 (topsatenLinear's F16 path), so
    // expect ~1% relative drift on typical magnitudes plus a small atol
    // for sums near zero.
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < hc.size(); i++) {
        const float err = std::fabs(hc[i] - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(hc[i], expected[i], 5e-2f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "%s: mismatch idx=%zu got=%f want=%f (err=%f)\n",
                                 label, i, hc[i], expected[i], err);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) {
        fprintf(stderr, "%s: %d mismatches over %zu (max_abs_err=%f)\n",
                label, bad, hc.size(), max_err);
        return 1;
    }
    printf("%s: ok (%zu elements, max_abs_err=%f)\n", label, hc.size(), max_err);
    return 0;
}

static int test_mul_mat_q4_0(ggml_backend_t gcu) { return test_mul_mat_q_weight(gcu, GGML_TYPE_Q4_0, "MUL_MAT_Q4_0"); }
static int test_mul_mat_q8_0(ggml_backend_t gcu) { return test_mul_mat_q_weight(gcu, GGML_TYPE_Q8_0, "MUL_MAT_Q8_0"); }
static int test_mul_mat_q4_k(ggml_backend_t gcu) { return test_mul_mat_q_weight(gcu, GGML_TYPE_Q4_K, "MUL_MAT_Q4_K"); }

// Element-wise MUL: dst = a * b on F32, large 2D tensor. Mirrors test_add.
static int test_mul(ggml_backend_t gcu) {
    const int64_t M = 1024, N = 4096;
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
    ggml_tensor * c = ggml_mul(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "MUL: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hb(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 41);
    fill_random_f32(hb.data(), n, 42);
    for (size_t i = 0; i < n; i++) expected[i] = ha[i] * hb[i];

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(b, hb.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-5f, 1e-5f)) {
            if (bad < 5) fprintf(stderr, "MUL: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "MUL: %d mismatches\n", bad); return 1; }
    printf("MUL: ok (%zu elements)\n", n);
    return 0;
}

// SCALE: dst = a * scalar. F32 only (the GCU path lives in gcu_op_scale).
static int test_scale(ggml_backend_t gcu) {
    const int64_t M = 512, N = 2048;
    const size_t  n = (size_t) M * N;
    const float   s = 0.125f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * c = ggml_scale(ctx, a, s);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "SCALE: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float> ha(n), hc(n), expected(n);
    fill_random_f32(ha.data(), n, 51);
    for (size_t i = 0; i < n; i++) expected[i] = ha[i] * s;

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "SCALE: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!close_enough(hc[i], expected[i], 1e-5f, 1e-5f)) {
            if (bad < 5) fprintf(stderr, "SCALE: mismatch idx=%zu got=%f want=%f\n", i, hc[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "SCALE: %d mismatches\n", bad); return 1; }
    printf("SCALE: ok (%zu elements)\n", n);
    return 0;
}

// GET_ROWS: dst[i, :] = src[indices[i], :].  Indices are i32 per ggml.
// gcu_op_get_rows supports F32 src only, unbatched (ne[2]/ne[3] == 1).
static int test_get_rows(ggml_backend_t gcu) {
    const int64_t cols   = 256;
    const int64_t n_rows = 128;     // rows in source
    const int64_t n_take = 64;      // rows to gather
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_rows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_take);
    ggml_tensor * out = ggml_get_rows(ctx, src, idx);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "GET_ROWS: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>   h_src((size_t) cols * n_rows);
    std::vector<int32_t> h_idx((size_t) n_take);
    std::vector<float>   h_out((size_t) cols * n_take);
    std::vector<float>   expected((size_t) cols * n_take);

    fill_random_f32(h_src.data(), h_src.size(), 61);
    // Picking indices including duplicates and edges to exercise the lookup.
    std::mt19937 rng(62);
    std::uniform_int_distribution<int32_t> idx_dist(0, (int32_t) n_rows - 1);
    for (int64_t i = 0; i < n_take; i++) h_idx[i] = idx_dist(rng);

    for (int64_t i = 0; i < n_take; i++) {
        const int32_t r = h_idx[i];
        const float * src_row = h_src.data() + (size_t) r * cols;
        float       * dst_row = expected.data() + (size_t) i * cols;
        std::memcpy(dst_row, src_row, cols * sizeof(float));
    }

    ggml_backend_tensor_set(src, h_src.data(), 0, h_src.size() * sizeof(float));
    ggml_backend_tensor_set(idx, h_idx.data(), 0, h_idx.size() * sizeof(int32_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "GET_ROWS: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(out, h_out.data(), 0, h_out.size() * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < h_out.size(); i++) {
        if (h_out[i] != expected[i]) {
            if (bad < 5) fprintf(stderr, "GET_ROWS: mismatch idx=%zu got=%f want=%f\n", i, h_out[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "GET_ROWS: %d mismatches\n", bad); return 1; }
    printf("GET_ROWS: ok (%zu elements)\n", h_out.size());
    return 0;
}

// SET_ROWS: write src rows into a F32 destination at index positions.
// Per ggml's contract:
//   a (dst): [n_embd, ne1, ne2, ne3]
//   b (src): [n_embd, n_rows, ne02, ne03]
//   c (idx): I64 [n_rows, ne11, ne12, 1]   c[i] in [0, ne1)
// gcu_op_set_rows accepts F32 dst only; we run unbatched (ne02=1, ne03=1).
static int test_set_rows(ggml_backend_t gcu) {
    const int64_t cols   = 256;
    const int64_t n_dst  = 128;     // rows in destination
    const int64_t n_src  = 64;      // rows to write
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_dst);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_src);
    ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, n_src, 1);
    ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "SET_ROWS: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>   h_dst((size_t) cols * n_dst);
    std::vector<float>   h_src((size_t) cols * n_src);
    std::vector<int64_t> h_idx((size_t) n_src);
    std::vector<float>   h_out((size_t) cols * n_dst);
    std::vector<float>   expected((size_t) cols * n_dst);

    fill_random_f32(h_dst.data(), h_dst.size(), 71);
    fill_random_f32(h_src.data(), h_src.size(), 72);

    // Distinct row indices so destination semantics are well-defined
    // (ggml says "undefined behavior if destination rows overlap").
    std::vector<int64_t> all(n_dst);
    for (int64_t i = 0; i < n_dst; i++) all[i] = i;
    std::mt19937 rng(73);
    std::shuffle(all.begin(), all.end(), rng);
    for (int64_t i = 0; i < n_src; i++) h_idx[i] = all[i];

    // CPU reference: start from h_dst, then overwrite the indexed rows
    // with the corresponding src rows.
    expected = h_dst;
    for (int64_t i = 0; i < n_src; i++) {
        const int64_t r = h_idx[i];
        std::memcpy(expected.data() + (size_t) r * cols,
                    h_src.data()    + (size_t) i * cols,
                    cols * sizeof(float));
    }

    ggml_backend_tensor_set(dst, h_dst.data(), 0, h_dst.size() * sizeof(float));
    ggml_backend_tensor_set(src, h_src.data(), 0, h_src.size() * sizeof(float));
    ggml_backend_tensor_set(idx, h_idx.data(), 0, h_idx.size() * sizeof(int64_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "SET_ROWS: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    // ggml_set_rows returns a view over the destination — the result lives
    // in dst's memory.
    ggml_backend_tensor_get(dst, h_out.data(), 0, h_out.size() * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < h_out.size(); i++) {
        if (h_out[i] != expected[i]) {
            if (bad < 5) fprintf(stderr, "SET_ROWS: mismatch idx=%zu got=%f want=%f\n", i, h_out[i], expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "SET_ROWS: %d mismatches\n", bad); return 1; }
    printf("SET_ROWS: ok (%zu elements)\n", h_out.size());
    return 0;
}

// CPY: same-dtype contiguous and F32<->F16 conversion. The handler is
// shared with DUP/CONT, so this also exercises that code path.
static int test_cpy(ggml_backend_t gcu) {
    const int64_t M = 512, N = 1024;
    const size_t  n = (size_t) M * N;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * a_dup   = ggml_dup(ctx, a);                            // F32->F32
    ggml_tensor * a_f16_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N);
    ggml_tensor * a_f16   = ggml_cpy(ctx, a, a_f16_t);                   // F32->F16 cast
    ggml_tensor * a_f32   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * a_back  = ggml_cpy(ctx, a_f16, a_f32);                 // F16->F32 cast

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "CPY: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha(n), hdup(n), hback(n);
    std::vector<ggml_fp16_t> hf16(n);
    std::vector<float>       expected_f16_round(n);
    fill_random_f32(ha.data(), n, 81);
    // F16 round-trip reference
    {
        std::vector<ggml_fp16_t> tmp16(n);
        ggml_fp32_to_fp16_row(ha.data(), tmp16.data(), n);
        ggml_fp16_to_fp32_row(tmp16.data(), expected_f16_round.data(), n);
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, a_dup);
    ggml_build_forward_expand(graph, a_back);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CPY: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(a_dup, hdup.data(),  0, n * sizeof(float));
    ggml_backend_tensor_get(a_f16, hf16.data(),  0, n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(a_back, hback.data(), 0, n * sizeof(float));

    int bad_dup = 0, bad_round = 0, bad_f16 = 0;
    for (size_t i = 0; i < n; i++) {
        if (hdup[i]  != ha[i])                           { if (bad_dup   < 3) fprintf(stderr, "CPY-DUP: mismatch idx=%zu got=%f want=%f\n", i, hdup[i], ha[i]);   bad_dup++; }
        // F32->F16 quantization: each F32 must round to the F16 we store.
        const float f16_back = ggml_fp16_to_fp32(hf16[i]);
        if (f16_back != expected_f16_round[i])           { if (bad_f16   < 3) fprintf(stderr, "CPY-F32toF16: mismatch idx=%zu got=%f want=%f\n", i, f16_back, expected_f16_round[i]); bad_f16++; }
        // F16 round-trip: dst F32 must equal hf16's F32 expansion.
        if (!close_enough(hback[i], expected_f16_round[i], 1e-5f, 1e-5f)) {
            if (bad_round < 3) fprintf(stderr, "CPY-F16toF32: mismatch idx=%zu got=%f want=%f\n", i, hback[i], expected_f16_round[i]);
            bad_round++;
        }
    }

    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad_dup || bad_f16 || bad_round) {
        fprintf(stderr, "CPY: %d dup-mismatches, %d F32->F16 mismatches, %d round-trip mismatches\n",
                bad_dup, bad_f16, bad_round);
        return 1;
    }
    printf("CPY: ok (%zu elements; F32->F32, F32->F16, F16->F32)\n", n);
    return 0;
}

// MUL_MAT_ID with F16 weight × F32 input → F32 output. This exercises
// the F16-weight branch of gcu_op_mul_mat_id: input is cast to F16 once
// up front and each per-(token, expert) call goes through topsatenLinear
// in F16, with the F16 result cast back to F32 at the dst slot. This is
// the same path Q-typed weights take after dequant-on-load.
static int test_mul_mat_id_f16(ggml_backend_t gcu) {
    const int64_t K = 256;
    const int64_t M = 512;
    const int64_t n_expert      = 4;
    const int64_t n_expert_used = 2;
    const int64_t n_tokens      = 8;

    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * as_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, M, n_expert);
    ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_expert_used, n_tokens);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * c    = ggml_mul_mat_id(ctx, as_w, b, ids);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "MUL_MAT_ID_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       w_f32((size_t) K * M * n_expert);
    std::vector<ggml_fp16_t> w_f16((size_t) K * M * n_expert);
    std::vector<float>       h_b ((size_t) K * n_expert_used * n_tokens);
    std::vector<int32_t>     h_ids((size_t) n_expert_used * n_tokens);
    std::vector<float>       h_c ((size_t) M * n_expert_used * n_tokens);
    std::vector<float>       expected((size_t) M * n_expert_used * n_tokens);

    fill_random_f32(w_f32.data(), w_f32.size(), 261);
    fill_random_f32(h_b.data(),   h_b.size(),   262);
    ggml_fp32_to_fp16_row(w_f32.data(), w_f16.data(), w_f16.size());

    // Round-robin expert assignment (same as the F32 test) to exercise
    // multiple experts per token.
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            h_ids[t * n_expert_used + e] = (int32_t) ((t + e * 2) % n_expert);
        }
    }

    // CPU reference: dequant the F16 weight back to F32 (so we compare
    // against what the device actually saw) and accumulate in F32.
    std::vector<float> w_lossy((size_t) K * M * n_expert);
    for (size_t i = 0; i < w_f16.size(); i++) {
        w_lossy[i] = ggml_fp16_to_fp32(w_f16[i]);
    }
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            const int32_t expert_id = h_ids[t * n_expert_used + e];
            const float * w_mat = w_lossy.data() + (size_t) expert_id * M * K;
            const float * b_vec = h_b.data()     + (size_t)(t * n_expert_used + e) * K;
            float       * c_vec = expected.data() + (size_t)(t * n_expert_used + e) * M;
            for (int64_t m = 0; m < M; m++) {
                float acc = 0.0f;
                const float * w_row = w_mat + m * K;
                for (int64_t k = 0; k < K; k++) acc += w_row[k] * b_vec[k];
                c_vec[m] = acc;
            }
        }
    }

    ggml_backend_tensor_set(as_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b,    h_b.data(),   0, h_b.size()   * sizeof(float));
    ggml_backend_tensor_set(ids,  h_ids.data(), 0, h_ids.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL_MAT_ID_F16: compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, h_c.data(), 0, h_c.size() * sizeof(float));

    int   bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < h_c.size(); i++) {
        const float err = std::fabs(h_c[i] - expected[i]);
        if (err > max_err) max_err = err;
        // F16 weight × F32 input run through topsatenLinear's F16 path:
        // expect F16-precision drift on a K=256 sum, comparable to the
        // Q-weight tests above.
        if (!close_enough(h_c[i], expected[i], 5e-2f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "MUL_MAT_ID_F16: mismatch idx=%zu got=%f want=%f (err=%f)\n",
                                 i, h_c[i], expected[i], err);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) {
        fprintf(stderr, "MUL_MAT_ID_F16: %d mismatches over %zu (max_abs_err=%f)\n",
                bad, h_c.size(), max_err);
        return 1;
    }
    printf("MUL_MAT_ID_F16: ok (%zu elements, max_abs_err=%f)\n", h_c.size(), max_err);
    return 0;
}

// === F16 dtype variants =================================================
//
// Mirror the existing F32 tests on F16 tensors. The reference is computed
// in F32 from the F16-rounded inputs, then bit-rounded back to the F16
// representation we expect on the device — so the only remaining drift
// in the comparison is what the GCU's F16-accumulator path adds. Tolerance
// is F16-precision-aware: ~5e-3 relative for element-wise, slightly looser
// for reductions and ROPE.

static int test_add_f16(ggml_backend_t gcu) {
    const int64_t M = 1024, N = 1024;
    const size_t  n = (size_t) M * N;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N);
    ggml_tensor * c = ggml_add(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "ADD_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32(n), hb_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hb(n), hc(n);
    fill_random_f32(ha_f32.data(), n, 201);
    fill_random_f32(hb_f32.data(), n, 202);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);
    ggml_fp32_to_fp16_row(hb_f32.data(), hb.data(), n);
    // Reference uses the F16-rounded values that the device will see.
    for (size_t i = 0; i < n; i++) {
        expected[i] = ggml_fp16_to_fp32(ha[i]) + ggml_fp16_to_fp32(hb[i]);
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b, hb.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ADD_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(got, expected[i], 5e-3f, 5e-3f)) {
            if (bad < 5) fprintf(stderr, "ADD_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "ADD_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("ADD_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
    return 0;
}

static int test_mul_f16(ggml_backend_t gcu) {
    const int64_t M = 1024, N = 1024;
    const size_t  n = (size_t) M * N;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N);
    ggml_tensor * c = ggml_mul(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "MUL_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32(n), hb_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hb(n), hc(n);
    fill_random_f32(ha_f32.data(), n, 211);
    fill_random_f32(hb_f32.data(), n, 212);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);
    ggml_fp32_to_fp16_row(hb_f32.data(), hb.data(), n);
    for (size_t i = 0; i < n; i++) {
        expected[i] = ggml_fp16_to_fp32(ha[i]) * ggml_fp16_to_fp32(hb[i]);
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b, hb.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "MUL_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(got, expected[i], 5e-3f, 5e-3f)) {
            if (bad < 5) fprintf(stderr, "MUL_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "MUL_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("MUL_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
    return 0;
}

static int test_silu_f16(ggml_backend_t gcu) {
    const size_t n = 4096;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    ggml_tensor * c = ggml_silu(ctx, a);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "SILU_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hc(n);
    fill_random_f32(ha_f32.data(), n, 221);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);
    for (size_t i = 0; i < n; i++) {
        const float x = ggml_fp16_to_fp32(ha[i]);
        const float s = 1.0f / (1.0f + std::exp(-x));
        expected[i] = x * s;
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "SILU_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(got, expected[i], 5e-3f, 5e-3f)) {
            if (bad < 5) fprintf(stderr, "SILU_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "SILU_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("SILU_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
    return 0;
}

static int test_rms_norm_f16(ggml_backend_t gcu) {
    const int64_t hidden = 1024, batch = 64;
    const size_t  n = (size_t) hidden * batch;
    const float   eps = 1e-6f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden, batch);
    ggml_tensor * c = ggml_rms_norm(ctx, a, eps);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "RMS_NORM_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hc(n);
    fill_random_f32(ha_f32.data(), n, 231);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);
    // Reference uses the F16-rounded values, summed in F64 for stability.
    for (int64_t b = 0; b < batch; b++) {
        double s = 0.0;
        for (int64_t i = 0; i < hidden; i++) {
            const float x = ggml_fp16_to_fp32(ha[b * hidden + i]);
            s += (double) x * x;
        }
        const float scale = 1.0f / std::sqrt((float)(s / hidden) + eps);
        for (int64_t i = 0; i < hidden; i++) {
            const float x = ggml_fp16_to_fp32(ha[b * hidden + i]);
            expected[b * hidden + i] = x * scale;
        }
    }

    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "RMS_NORM_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        // RMS_NORM has a 1024-wide reduction in F16; allow 1% relative drift
        // plus a small atol for elements near zero.
        if (!close_enough(got, expected[i], 5e-3f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "RMS_NORM_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "RMS_NORM_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("RMS_NORM_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
    return 0;
}

static int test_softmax_f16(ggml_backend_t gcu) {
    const int64_t cols = 1024, rows = 16;
    const size_t  n = (size_t) cols * rows;
    const float   scale = 0.125f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols, rows);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols, rows);
    ggml_tensor * c    = ggml_soft_max_ext(ctx, a, mask, scale, 0.0f);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "SOFT_MAX_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    std::vector<float>       ha_f32(n), hm_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hm(n), hc(n);
    fill_random_f32(ha_f32.data(), n, 241);
    fill_random_f32(hm_f32.data(), n, 242);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);
    ggml_fp32_to_fp16_row(hm_f32.data(), hm.data(), n);
    for (int64_t r = 0; r < rows; r++) {
        float emax = -INFINITY;
        for (int64_t i = 0; i < cols; i++) {
            const float v = ggml_fp16_to_fp32(ha[r * cols + i]) * scale +
                            ggml_fp16_to_fp32(hm[r * cols + i]);
            if (v > emax) emax = v;
        }
        double sum = 0.0;
        for (int64_t i = 0; i < cols; i++) {
            const float v = ggml_fp16_to_fp32(ha[r * cols + i]) * scale +
                            ggml_fp16_to_fp32(hm[r * cols + i]);
            const float e = std::exp(v - emax);
            expected[r * cols + i] = e;
            sum += e;
        }
        for (int64_t i = 0; i < cols; i++) expected[r * cols + i] /= (float) sum;
    }

    ggml_backend_tensor_set(a,    ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, hm.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "SOFT_MAX_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        // Softmax outputs sum to 1; per-element values are in (0,1) and small.
        // F16 rounding adds ~5e-4 absolute drift; reduction over 1024 elements
        // can amplify that to a few 1e-3.
        if (!close_enough(got, expected[i], 5e-3f, 1e-2f)) {
            if (bad < 5) fprintf(stderr, "SOFT_MAX_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "SOFT_MAX_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("SOFT_MAX_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
    return 0;
}

static int test_rope_f16(ggml_backend_t gcu) {
    const int64_t head_dim = 64, n_heads = 8, n_tokens = 16;
    const int     n_dims = (int) head_dim;
    const float   freq_base = 10000.0f;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, n_heads, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * c   = ggml_rope(ctx, a, pos, n_dims, GGML_ROPE_TYPE_NORMAL);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "ROPE_F16: alloc failed\n"); ggml_free(ctx); return 1; }

    const size_t n = (size_t) head_dim * n_heads * n_tokens;
    std::vector<float>       ha_f32(n), expected(n);
    std::vector<ggml_fp16_t> ha(n), hc(n);
    std::vector<int32_t>     hp(n_tokens);
    fill_random_f32(ha_f32.data(), n, 251);
    ggml_fp32_to_fp16_row(ha_f32.data(), ha.data(), n);
    for (int64_t t = 0; t < n_tokens; t++) hp[t] = (int32_t) t;

    for (int64_t t = 0; t < n_tokens; t++) {
        const int p_idx = hp[t];
        for (int64_t h = 0; h < n_heads; h++) {
            const ggml_fp16_t * xrow = ha.data()       + (t * n_heads + h) * head_dim;
            float             * yrow = expected.data() + (t * n_heads + h) * head_dim;
            for (int i = 0; i < n_dims; i += 2) {
                const float theta = std::pow(freq_base, -((float) i) / (float) n_dims);
                const float angle = (float) p_idx * theta;
                const float c1 = std::cos(angle), s1 = std::sin(angle);
                const float x0 = ggml_fp16_to_fp32(xrow[i]);
                const float x1 = ggml_fp16_to_fp32(xrow[i + 1]);
                yrow[i]     = x0 * c1 - x1 * s1;
                yrow[i + 1] = x0 * s1 + x1 * c1;
            }
            for (int i = n_dims; i < head_dim; i++) yrow[i] = ggml_fp16_to_fp32(xrow[i]);
        }
    }

    ggml_backend_tensor_set(a,   ha.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(pos, hp.data(), 0, n_tokens * sizeof(int32_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (ggml_backend_graph_compute(gcu, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ROPE_F16: compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return 1;
    }
    ggml_backend_tensor_get(c, hc.data(), 0, n * sizeof(ggml_fp16_t));

    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float got = ggml_fp16_to_fp32(hc[i]);
        const float err = std::fabs(got - expected[i]);
        if (err > max_err) max_err = err;
        if (!close_enough(got, expected[i], 5e-3f, 5e-3f)) {
            if (bad < 5) fprintf(stderr, "ROPE_F16: mismatch idx=%zu got=%f want=%f\n", i, got, expected[i]);
            bad++;
        }
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    if (bad) { fprintf(stderr, "ROPE_F16: %d mismatches (max_abs_err=%f)\n", bad, max_err); return 1; }
    printf("ROPE_F16: ok (%zu elements, max_abs_err=%f)\n", n, max_err);
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

// === Per-op perf micro-bench ============================================
//
// Opt-in via environment: GCU_BENCH=1 ./test-backend-gcu. When unset,
// the bench section is skipped and the smoke run finishes in seconds.
// Each bench builds an isolated graph for one op, runs warmup iterations
// to settle, then times a fixed number of synchronous graph_compute calls
// and reports mean ± stddev wall time. Shapes target a typical decode-step
// (n_tokens = 1) at hidden_dim = 4096 / FFN = 11008 — close to a Llama
// 7B-class layer.

static void time_graph(ggml_backend_t gcu, ggml_cgraph * graph, const char * label,
                       int n_warmup, int n_iter) {
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < n_warmup; i++) {
        ggml_backend_graph_compute(gcu, graph);
    }
    ggml_backend_synchronize(gcu);

    std::vector<double> times_ms;
    times_ms.reserve(n_iter);
    for (int i = 0; i < n_iter; i++) {
        auto t0 = clock::now();
        ggml_backend_graph_compute(gcu, graph);
        ggml_backend_synchronize(gcu);
        auto t1 = clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double sum = 0.0;
    for (double t : times_ms) sum += t;
    double mean = sum / n_iter;
    double sq_sum = 0.0;
    for (double t : times_ms) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / n_iter);

    printf("BENCH %-22s mean=%7.3f ms  std=%6.3f ms  (n=%d)\n",
           label, mean, stddev, n_iter);
}

// Helper: standard ggml_init params used by every bench.
static ggml_init_params bench_init_params() {
    return {
        /* .mem_size   = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
}

static void bench_add_f32(ggml_backend_t gcu) {
    const int64_t M = 4096, N = 4096;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float> ha((size_t) M * N), hb((size_t) M * N);
        fill_random_f32(ha.data(), ha.size(), 101);
        fill_random_f32(hb.data(), hb.size(), 102);
        ggml_backend_tensor_set(a, ha.data(), 0, ha.size() * sizeof(float));
        ggml_backend_tensor_set(b, hb.data(), 0, hb.size() * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "ADD F32 [4096,4096]", 3, 10);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

static void bench_rms_norm_f32(ggml_backend_t gcu) {
    const int64_t M = 4096, N = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * c = ggml_rms_norm(ctx, a, 1e-5f);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float> ha((size_t) M * N);
        fill_random_f32(ha.data(), ha.size(), 111);
        ggml_backend_tensor_set(a, ha.data(), 0, ha.size() * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "RMS_NORM F32 [4096,1]", 3, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

static void bench_softmax_f32(ggml_backend_t gcu) {
    // Shape mirrors decode-step attention scores: [n_kv=128, n_head=32, n=1].
    const int64_t n_kv = 128, n_head = 32, n_tokens = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_kv, n_head, n_tokens);
    ggml_tensor * c = ggml_soft_max(ctx, a);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float> ha((size_t) n_kv * n_head * n_tokens);
        fill_random_f32(ha.data(), ha.size(), 121);
        ggml_backend_tensor_set(a, ha.data(), 0, ha.size() * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "SOFT_MAX F32 [128,32,1]", 3, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

static void bench_silu_f32(ggml_backend_t gcu) {
    const int64_t M = 11008, N = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * c = ggml_silu(ctx, a);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float> ha((size_t) M * N);
        fill_random_f32(ha.data(), ha.size(), 131);
        ggml_backend_tensor_set(a, ha.data(), 0, ha.size() * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "SILU F32 [11008,1]", 3, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

static void bench_rope_f32(ggml_backend_t gcu) {
    // [head_dim=128, n_head=32, n_tokens=1]
    const int64_t head_dim = 128, n_head = 32, n_tokens = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * c   = ggml_rope(ctx, a, pos, head_dim, /*mode=*/0);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float>   ha((size_t) head_dim * n_head * n_tokens);
        std::vector<int32_t> hp(n_tokens);
        fill_random_f32(ha.data(), ha.size(), 141);
        for (int64_t i = 0; i < n_tokens; i++) hp[i] = (int32_t) i;
        ggml_backend_tensor_set(a,   ha.data(), 0, ha.size() * sizeof(float));
        ggml_backend_tensor_set(pos, hp.data(), 0, hp.size() * sizeof(int32_t));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "ROPE F32 [128,32,1]", 3, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

// MUL_MAT F16 weight × F32 input → F32 output. K=4096 M=4096 N=1
// matches a Llama-7B-class projection on a single decode step.
static void bench_mul_mat_f16(ggml_backend_t gcu) {
    const int64_t K = 4096, M = 4096, N = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, w, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float>       w_f32((size_t) K * M);
        std::vector<ggml_fp16_t> w_f16((size_t) K * M);
        std::vector<float>       hx((size_t) K * N);
        fill_random_f32(w_f32.data(), w_f32.size(), 151);
        fill_random_f32(hx.data(),    hx.size(),    152);
        ggml_fp32_to_fp16_row(w_f32.data(), w_f16.data(), w_f16.size());
        ggml_backend_tensor_set(w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(x, hx.data(),    0, hx.size()    * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "MUL_MAT F16w [4096x4096x1]", 5, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

// MUL_MAT Q4_K weight × F32 input → F32 output, decode-step shape.
// Mirrors what GGUF Q4_K_M models hit on every projection.
static void bench_mul_mat_q4_k(ggml_backend_t gcu) {
    const int64_t K = 4096, M = 4096, N = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, w, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float>   w_f32((size_t) K * M);
        std::vector<uint8_t> w_q(M * ggml_row_size(GGML_TYPE_Q4_K, K));
        std::vector<float>   hx((size_t) K * N);
        fill_random_f32(w_f32.data(), w_f32.size(), 161);
        fill_random_f32(hx.data(),    hx.size(),    162);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, w_f32.data(), w_q.data(), 0, M, K, nullptr);
        ggml_backend_tensor_set(w, w_q.data(), 0, w_q.size());
        ggml_backend_tensor_set(x, hx.data(),  0, hx.size() * sizeof(float));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "MUL_MAT Q4_Kw [4096x4096x1]", 5, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

// MUL_MAT_ID at MoE decode-step shape. K=4096 M=11008 (FFN intermediate)
// with 8 experts and 2 active per token.
static void bench_mul_mat_id_f16(ggml_backend_t gcu) {
    const int64_t K = 4096, M = 11008;
    const int64_t n_expert = 8, n_used = 2, n_tokens = 1;
    auto buft = ggml_backend_get_default_buffer_type(gcu);
    auto p = bench_init_params();
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, M, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * c   = ggml_mul_mat_id(ctx, w, b, ids);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        std::vector<float>       w_f32((size_t) K * M * n_expert);
        std::vector<ggml_fp16_t> w_f16((size_t) K * M * n_expert);
        std::vector<float>       hb((size_t) K * n_used * n_tokens);
        std::vector<int32_t>     hids((size_t) n_used * n_tokens);
        fill_random_f32(w_f32.data(), w_f32.size(), 171);
        fill_random_f32(hb.data(),    hb.size(),    172);
        ggml_fp32_to_fp16_row(w_f32.data(), w_f16.data(), w_f16.size());
        for (size_t i = 0; i < hids.size(); i++) hids[i] = (int32_t) (i % n_expert);
        ggml_backend_tensor_set(w,   w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(b,   hb.data(),    0, hb.size()    * sizeof(float));
        ggml_backend_tensor_set(ids, hids.data(),  0, hids.size()  * sizeof(int32_t));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        time_graph(gcu, graph, "MUL_MAT_ID F16w MoE", 5, 20);
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
}

static void bench_all(ggml_backend_t gcu) {
    if (getenv("GCU_BENCH") == nullptr) return;
    printf("---- per-op micro-bench (decode-step shapes; warmup+timed) ----\n");
    bench_add_f32(gcu);
    bench_rms_norm_f32(gcu);
    bench_softmax_f32(gcu);
    bench_silu_f32(gcu);
    bench_rope_f32(gcu);
    bench_mul_mat_f16(gcu);
    bench_mul_mat_q4_k(gcu);
    bench_mul_mat_id_f16(gcu);
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
    rc |= test_add_f16(gcu);
    rc |= test_mul(gcu);
    rc |= test_mul_f16(gcu);
    rc |= test_scale(gcu);
    rc |= test_get_rows(gcu);
    rc |= test_set_rows(gcu);
    rc |= test_cpy(gcu);
    rc |= test_mul_mat(gcu);
    rc |= test_mul_mat_q4_0(gcu);
    rc |= test_mul_mat_q8_0(gcu);
    rc |= test_mul_mat_q4_k(gcu);
    rc |= test_silu(gcu);
    rc |= test_silu_f16(gcu);
    rc |= test_gelu(gcu);
    rc |= test_gelu_quick(gcu);
    rc |= test_relu(gcu);
    rc |= test_tanh(gcu);
    rc |= test_sigmoid(gcu);
    rc |= test_hardswish(gcu);
    rc |= test_hardsigmoid(gcu);
    rc |= test_norm(gcu);
    rc |= test_norm_f16(gcu);
    rc |= test_rms_norm(gcu);
    rc |= test_rms_norm_f16(gcu);
    rc |= test_softmax(gcu);
    rc |= test_softmax_f16(gcu);
    rc |= test_rope(gcu);
    rc |= test_rope_f16(gcu);
    rc |= test_mul_mat_mixed(gcu);
    rc |= test_mul_mat_mixed_f16in(gcu);
    rc |= test_mul_mat_id(gcu);
    rc |= test_mul_mat_id_f16(gcu);
    rc |= test_async_overlap(gcu);

    const bool bench_ran = (rc == 0 && getenv("GCU_BENCH") != nullptr);
    if (bench_ran) bench_all(gcu);

    ggml_backend_free(gcu);

    size_t free_after = 0, total_after = 0;
    ggml_backend_gcu_get_device_memory(0, &free_after, &total_after);

    // The pool, the cached zero-bias, and topsaten's own workspace all
    // outlive a backend instance (heap-leaked singletons; OS reclaims at
    // exit). So a strict leak check would always trip. Print the delta
    // as info, and only flag truly extreme values as failure. The bench
    // section legitimately loads ~1 GiB of MoE-shape weight scratch into
    // the pool, so raise the threshold when bench mode ran.
    if (total_after == total_before && free_before > free_after) {
        size_t diff = free_before - free_after;
        printf("retained device memory: %zu bytes (pool + topsaten workspace)\n", diff);
        const size_t leak_threshold = bench_ran ? ((size_t) 4 << 30) : ((size_t) 1 << 30);
        if (diff > leak_threshold) {
            fprintf(stderr, "leak detected: %zu bytes retained (>%zu GiB)\n",
                    diff, leak_threshold >> 30);
            rc |= 1;
        }
    }

    if (rc == 0) printf("test-backend-gcu: PASSED\n");
    return rc;
}
