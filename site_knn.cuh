#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "bitonic-hubs-grid.cuh"
#include "near_project.cuh"

__global__ static void fill_iota(idx_t* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) out[i] = (idx_t)i;
}

__global__ static void compact_unfrozen_kernel(
    const unsigned char* __restrict__ frozen,
    idx_t* __restrict__ out_indices,
    int* __restrict__ out_count,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !frozen[i]) {
        int pos = atomicAdd(out_count, 1);
        out_indices[pos] = (idx_t)i;
    }
}

__global__ static void gather_float3_kernel(
    const float* __restrict__ src,
    const idx_t* __restrict__ indices,
    float* __restrict__ out,
    int n_active)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_active) {
        idx_t idx = indices[i];
        out[i * 3 + 0] = src[idx * 3 + 0];
        out[i * 3 + 1] = src[idx * 3 + 1];
        out[i * 3 + 2] = src[idx * 3 + 2];
    }
}

__global__ static void fill_iota_idx(idx_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (idx_t)i;
}

static inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b){
    float ms=0.f; cudaEventElapsedTime(&ms,a,b); return ms;
}

/** Site-site K-NN by bruteforce (mode 0). Writes K neighbors per point to d_knn (no self). */
template<int K>
static inline float run_knn_sites_bruteforce(int N, unsigned char* __restrict__ frozen,
    const float3* d_pts, idx_t* d_knn)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int block = 256;
    int grid = (N + block - 1) / block;
    knn_sites_bruteforce_kernel<K, idx_t><<<grid, block>>>(d_pts, N, frozen, d_knn);
    cudaDeviceSynchronize();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = elapsed_ms(t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

static inline float run_knn_bitonic_hubs(int N, int K, unsigned char* __restrict__ frozen, const float3* d_pts, idx_t* d_knn, float* d_dist){
    idx_t* d_queries=nullptr;

    cudaMalloc(&d_queries, (size_t)N*sizeof(idx_t));

    int T=1024;
    fill_iota<<<(N+T-1)/T, T>>>(d_queries, N);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    static std::string name="sites";
    bitonic_hubs_grid::C_and_Q<float>(
        N,
        (float*)d_pts,
        N,
        d_queries,
        K,
        frozen,
        d_knn,
        d_dist,
        name
    );

    cudaDeviceSynchronize();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = elapsed_ms(t0,t1);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_queries);
    return ms;
}

/** KNN from query points (device) to candidate points (device) using bitonic grid.
 *  Candidates = mesh vertices (dV), queries = sites or centroids (d_query).
 *  Builds grid from d_candidates; query_points are copied to host for C_and_Q. */
static inline float run_knn_bitonic_query_to_mesh(int nCand, const float3* d_candidates,
    const float3* d_query, int nQ, int K, unsigned char* frozen,
    idx_t* d_knn, float* d_dist, const char* name_for_timing)
{
    std::vector<float> h_query((size_t)nQ * 3);
    cudaMemcpy(h_query.data(), d_query, sizeof(float) * (size_t)nQ * 3, cudaMemcpyDeviceToHost);
    std::string s(name_for_timing);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    bitonic_hubs_grid::C_and_Q<float>(nCand, (float*)d_candidates, (std::size_t)nQ, nullptr,
        (std::size_t)K, frozen, d_knn, d_dist, s, h_query.data());
    cudaDeviceSynchronize();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = elapsed_ms(t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

// ─── Cached KNN: build once over mesh vertices, query many times ───
using BitonicKnnCache = bitonic_hubs_grid::BitonicKnnCache;

static inline BitonicKnnCache build_knn_mesh_cache(int nCand, const float3* d_candidates) {
    return bitonic_hubs_grid::C_build<float>(nCand, (float*)d_candidates);
}

// Scratch buffers struct for cached query (allocate once in main, reuse every call)
struct KnnQueryScratch {
    float* d_qp;     // q * 3
    idx_t* d_idx;    // q
    int*   d_count;  // 1
    int*   d_misc;   // 3 (cell_psum[2] + cell_counts[1])
    int*   d_hubs;   // q
    int*   d_pts;    // q
};

static inline KnnQueryScratch alloc_knn_query_scratch(int nQ) {
    KnnQueryScratch s;
    s.d_qp = nullptr; s.d_idx = nullptr; s.d_count = nullptr;
    s.d_misc = nullptr; s.d_hubs = nullptr; s.d_pts = nullptr;
    CUDA_CALL(cudaMalloc(&s.d_qp,    sizeof(float) * nQ * 3));
    CUDA_CALL(cudaMalloc(&s.d_idx,   sizeof(idx_t) * nQ));
    CUDA_CALL(cudaMalloc(&s.d_count, sizeof(int)));
    CUDA_CALL(cudaMalloc(&s.d_misc,  sizeof(int) * 3));
    CUDA_CALL(cudaMalloc(&s.d_hubs,  sizeof(int) * nQ));
    CUDA_CALL(cudaMalloc(&s.d_pts,   sizeof(int) * nQ));
    printf("scratch alloc: qp=%p idx=%p count=%p misc=%p hubs=%p pts=%p\n",
           s.d_qp, s.d_idx, s.d_count, s.d_misc, s.d_hubs, s.d_pts);
    return s;
}

static inline void free_knn_query_scratch(KnnQueryScratch& s) {
    cudaFree(s.d_qp);    cudaFree(s.d_idx);
    cudaFree(s.d_count);  cudaFree(s.d_misc);
    cudaFree(s.d_hubs);   cudaFree(s.d_pts);
}

static inline float run_knn_cached_query_to_mesh(
    const BitonicKnnCache& cache,
    const float3* d_query, int nQ, int K,
    unsigned char* frozen, idx_t* d_knn, float* d_dist,
    KnnQueryScratch& scratch,
    const idx_t* seed_knn = nullptr, const float* seed_dist = nullptr)
{
    auto T0 = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize(); // drain prior work
    auto T1 = std::chrono::high_resolution_clock::now();

    // Compact unfrozen queries
    cudaMemset(scratch.d_count, 0, sizeof(int));
    if (frozen) {
        int T = 1024;
        compact_unfrozen_kernel<<<(nQ + T - 1) / T, T>>>(frozen, scratch.d_idx, scratch.d_count, nQ);
    } else {
        int T = 1024;
        fill_iota_idx<<<(nQ + T - 1) / T, T>>>(scratch.d_idx, nQ);
        int h_nQ = nQ;
        cudaMemcpy(scratch.d_count, &h_nQ, sizeof(int), cudaMemcpyHostToDevice);
    }
    int h_n_active = 0;
    cudaMemcpy(&h_n_active, scratch.d_count, sizeof(int), cudaMemcpyDeviceToHost);
    auto T2 = std::chrono::high_resolution_clock::now();

    float gather_ms = 0, cq_setup_ms = 0, cq_kernel_ms = 0;
    if (h_n_active > 0) {
        // Gather
        {
            int BLK = 256;
            gather_float3_kernel<<<(h_n_active + BLK - 1) / BLK, BLK>>>(
                (const float*)d_query, scratch.d_idx, scratch.d_qp, h_n_active);
        }
        cudaDeviceSynchronize();
        auto T3 = std::chrono::high_resolution_clock::now();
        gather_ms = std::chrono::duration<float, std::milli>(T3 - T2).count();

        // C_query setup (memcpys)
        int h_qcc[1] = { h_n_active };
        int h_qcp[2] = { 0, h_n_active };
        cudaMemcpy(scratch.d_misc,     h_qcp, sizeof(int)*2, cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_misc + 2, h_qcc, sizeof(int)*1, cudaMemcpyHostToDevice);
        cudaMemset(scratch.d_hubs, 0, sizeof(int) * nQ);
        cudaMemset(scratch.d_pts,  0, sizeof(int) * nQ);
        cudaDeviceSynchronize();
        auto T4 = std::chrono::high_resolution_clock::now();
        cq_setup_ms = std::chrono::duration<float, std::milli>(T4 - T3).count();

        // Kernel launch (via C_query which does the launch + sync)
        bitonic_hubs_grid::C_query<float>(cache, (std::size_t)h_n_active, (std::size_t)K,
            d_knn, d_dist,
            scratch.d_qp, scratch.d_idx,
            scratch.d_misc, scratch.d_hubs, scratch.d_pts,
            seed_knn, seed_dist);
        cudaDeviceSynchronize();
        auto T5 = std::chrono::high_resolution_clock::now();
        cq_kernel_ms = std::chrono::duration<float, std::milli>(T5 - T4).count();
    }

    // Read back hubs_scanned and points_scanned for active queries
    float avg_hubs = 0, avg_pts = 0;
    if (h_n_active > 0 && h_n_active <= nQ) {
        // Sum on device would be better, but for diagnostics just sample first N
        int sample = std::min(h_n_active, 10000);
        std::vector<int> h_hubs(sample), h_pts(sample);
        // The counters are written at global_query_id positions (scattered).
        // With compaction, active indices are in scratch.d_idx. Read those positions.
        std::vector<idx_t> h_active_idx(sample);
        cudaMemcpy(h_active_idx.data(), scratch.d_idx, sizeof(idx_t)*sample, cudaMemcpyDeviceToHost);
        // Read hubs_scanned and points_scanned for those indices
        std::vector<int> h_hubs_all(nQ), h_pts_all(nQ);
        cudaMemcpy(h_hubs_all.data(), scratch.d_hubs, sizeof(int)*nQ, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pts_all.data(), scratch.d_pts, sizeof(int)*nQ, cudaMemcpyDeviceToHost);
        long long sum_h = 0, sum_p = 0;
        for (int i = 0; i < sample; i++) {
            sum_h += h_hubs_all[h_active_idx[i]];
            sum_p += h_pts_all[h_active_idx[i]];
        }
        avg_hubs = (float)sum_h / sample;
        avg_pts  = (float)sum_p / sample;
    }

    float drain_ms  = std::chrono::duration<float, std::milli>(T1 - T0).count();
    float compact_ms = std::chrono::duration<float, std::milli>(T2 - T1).count();
    float ms = drain_ms + compact_ms + gather_ms + cq_setup_ms + cq_kernel_ms;
    printf("  [knn_cached] n_active=%d/%d  drain=%.1f compact=%.1f gather=%.1f setup=%.1f kernel=%.1f total=%.1f  hubs=%.1f pts=%.0f\n",
           h_n_active, nQ, drain_ms, compact_ms, gather_ms, cq_setup_ms, cq_kernel_ms, ms, avg_hubs, avg_pts);
    return ms;
}

static inline void destroy_knn_mesh_cache(BitonicKnnCache& cache) {
    bitonic_hubs_grid::C_destroy(cache);
}

template<int KIN, int KOUT, typename IndexT>
__global__ void knn_drop_self_kernel(const IndexT* __restrict__ in_knn,
                                    IndexT* __restrict__ out_knn,
                                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int w = 0;
    const IndexT* src = in_knn + (size_t)i * KIN;
    IndexT* dst = out_knn + (size_t)i * KOUT;

    for (int t = 0; t < KIN && w < KOUT; ++t) {
        IndexT j = src[t];
        if ((int)j == i) continue;
        dst[w++] = j;
    }

    if (w == 0) {
        for (int k = 0; k < KOUT; ++k) dst[k] = (IndexT)i;
    } else {
        IndexT fill = dst[w - 1];
        for (; w < KOUT; ++w) dst[w] = fill;
    }
}
