#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

#include "bitonic-hubs-grid.cuh"

__global__ static void fill_iota(idx_t* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) out[i] = (idx_t)i;
}

static inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b){
    float ms=0.f; cudaEventElapsedTime(&ms,a,b); return ms;
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
