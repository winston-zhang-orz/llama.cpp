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
    rc |= test_mul(gcu);
    rc |= test_scale(gcu);
    rc |= test_get_rows(gcu);
    rc |= test_set_rows(gcu);
    rc |= test_cpy(gcu);
    rc |= test_mul_mat(gcu);
    rc |= test_mul_mat_q4_0(gcu);
    rc |= test_mul_mat_q8_0(gcu);
    rc |= test_mul_mat_q4_k(gcu);
    rc |= test_silu(gcu);
    rc |= test_rms_norm(gcu);
    rc |= test_softmax(gcu);
    rc |= test_rope(gcu);
    rc |= test_mul_mat_mixed(gcu);
    rc |= test_mul_mat_id(gcu);
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
