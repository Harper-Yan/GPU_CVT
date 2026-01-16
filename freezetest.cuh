#pragma once

template<int K, typename IndexT>
__global__ void freeze_test_kernel(
    const float3* __restrict__ Sold,
    const float3* __restrict__ Snew,
    const IndexT* __restrict__ knn,
    const IndexT* __restrict__ prev_knn,
    unsigned char* __restrict__ frozen,   // pass-1 output (dFreezeCand)
    float thresh2,
    int n,
    int has_prev_knn,
    int* out_counts
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    float3 d;
    d.x = Snew[i].x - Sold[i].x;
    d.y = Snew[i].y - Sold[i].y;
    d.z = Snew[i].z - Sold[i].z;
    float disp2 = d.x*d.x + d.y*d.y + d.z*d.z;
    bool low_disp = (disp2 < thresh2);

    bool same = false;
    if (has_prev_knn) {
        same = true;
        #pragma unroll
        for (int a = 0; a < K; ++a) {
            IndexT va = knn[i*(int)K + a];
            bool found = false;
            #pragma unroll
            for (int b = 0; b < K; ++b) {
                if (prev_knn[i*(int)K + b] == va) { found = true; break; }
            }
            if (!found) { same = false; break; }
        }
    }

    bool both = low_disp && same;

    if (same)     atomicAdd(&out_counts[0], 1);
    if (low_disp) atomicAdd(&out_counts[1], 1);
    if (both)     atomicAdd(&out_counts[2], 1);

    // pass-1 output flag (dFreezeCand)
    if (both) frozen[i] = 1;
}

__global__ void count_frozen_kernel(const unsigned char* frozen, int n, int* out_sum)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n && frozen[i]) atomicAdd(out_sum, 1);
}

template<int K, typename IndexT>
__global__ void freeze_test_kernel_secured(
    const float3* __restrict__ Sold,
    const float3* __restrict__ Snew,
    const IndexT* __restrict__ knn,
    const IndexT* __restrict__ prev_knn,
    unsigned char* __restrict__ frozen,         // pass-1 output (dFreezeCand)
    const unsigned char* __restrict__ secureReached,
    float thresh2,
    int n,
    int has_prev_knn,
    int* out_counts
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    float3 d;
    d.x = Snew[i].x - Sold[i].x;
    d.y = Snew[i].y - Sold[i].y;
    d.z = Snew[i].z - Sold[i].z;
    float disp2 = d.x*d.x + d.y*d.y + d.z*d.z;
    bool low_disp = (disp2 < thresh2);

    bool same = false;
    if (has_prev_knn) {
        same = true;
        #pragma unroll
        for (int a = 0; a < K; ++a) {
            IndexT va = knn[i*(int)K + a];
            bool found = false;
            #pragma unroll
            for (int b = 0; b < K; ++b) {
                if (prev_knn[i*(int)K + b] == va) { found = true; break; }
            }
            if (!found) { same = false; break; }
        }
    }

    bool both = low_disp && same;

    if (same)     atomicAdd(&out_counts[0], 1);
    if (low_disp) atomicAdd(&out_counts[1], 1);
    if (both)     atomicAdd(&out_counts[2], 1);

    // pass-1 output flag (secured-gated)
    if (both && secureReached[i]) frozen[i] = 1;
}

// Pass-2: consensus freezing.
// out_counts[3] = newly frozen by consensus this iter
template<int K, int TOPN, typename IndexT>
__global__ void freeze_consensus_pass2_kernel(
    const IndexT* __restrict__ knn,
    const unsigned char* __restrict__ pass1, // dFreezeCand
    unsigned char* __restrict__ frozen,      // dFrozen
    int n,
    int* out_counts
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    if (!pass1[i]) return; // must pass pass-1 itself

    #pragma unroll
    for (int a = 0; a < TOPN; ++a) {
        IndexT j = knn[i*(int)K + a];
        if (j < 0 || j >= n || !pass1[(int)j]) return;
    }

    if (!frozen[i]) {
        frozen[i] = 1;
        atomicAdd(&out_counts[3], 1);
    }
}

template<int K, int TOPN, typename IndexT>
__global__ void freeze_consensus_pass2_kernel_secured(
    const IndexT* __restrict__ knn,
    const unsigned char* __restrict__ pass1, // dFreezeCand
    const unsigned char* __restrict__ secureReached,
    unsigned char* __restrict__ frozen,      // dFrozen
    int n,
    int* out_counts
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    if (!secureReached[i]) return;
    if (!pass1[i]) return;

    #pragma unroll
    for (int a = 0; a < TOPN; ++a) {
        IndexT j = knn[i*(int)K + a];
        if (j < 0 || j >= n || !pass1[(int)j]) return;
    }

    if (!frozen[i]) {
        frozen[i] = 1;
        atomicAdd(&out_counts[3], 1);
    }
}
template<int K, typename IndexT>
__global__ void freeze_test_kernel_streak(
    const float3* __restrict__ S,
    const float3* __restrict__ Snew,
    const IndexT* __restrict__ knn,
    const IndexT* __restrict__ prev_knn,
    unsigned char* __restrict__ freezeCand,
    unsigned char* __restrict__ streak,
    float thresh2,
    int nV,
    int has_prev_knn,
    int freeze_monitor_iters,   // e.g. 5
    int* __restrict__ counts
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nV) return;

    // Need at least 2 iterations to compare neighbors
    if (!has_prev_knn) {
        streak[i] = 0;
        freezeCand[i] = 0;
        return;
    }

    // same neighbors?
    bool same = true;
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        if (knn[i*K + j] != prev_knn[i*K + j]) { same = false; break; }
    }

    // low displacement?
    float3 a = S[i];
    float3 b = Snew[i];
    float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    bool low = (dx*dx + dy*dy + dz*dz) <= thresh2;

    bool both = same && low;

    // counts (keep same meaning as before)
    if (same) atomicAdd(&counts[0], 1);
    if (low)  atomicAdd(&counts[1], 1);
    if (both) atomicAdd(&counts[2], 1);

    // streak update: how many consecutive successful transitions
    unsigned char s = streak[i];
    if (both) {
        if (s < 255) s++;
    } else {
        s = 0;
    }
    streak[i] = s;

    int need = freeze_monitor_iters - 1;   // transitions needed
    if (need < 1) need = 1;

    freezeCand[i] = (s >= need) ? 1 : 0;
}
template<int K, typename IndexT>
__global__ void freeze_test_kernel_secured_streak(
    const float3* __restrict__ S,
    const float3* __restrict__ Snew,
    const IndexT* __restrict__ knn,
    const IndexT* __restrict__ prev_knn,
    unsigned char* __restrict__ freezeCand,
    unsigned char* __restrict__ streak,
    const unsigned char* __restrict__ secureReached,
    float thresh2,
    int nV,
    int has_prev_knn,
    int freeze_monitor_iters,
    int* __restrict__ counts
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nV) return;

    if (!has_prev_knn) {
        streak[i] = 0;
        freezeCand[i] = 0;
        return;
    }

    bool same = true;
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        if (knn[i*K + j] != prev_knn[i*K + j]) { same = false; break; }
    }

    float3 a = S[i], b = Snew[i];
    float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    bool low = (dx*dx + dy*dy + dz*dz) <= thresh2;

    bool sec = (secureReached[i] != 0);
    bool both = same && low && sec;

    if (same) atomicAdd(&counts[0], 1);
    if (low)  atomicAdd(&counts[1], 1);
    if (same && low) atomicAdd(&counts[2], 1); // keep your old "both" meaning if you want
    // (or use `both` here if you prefer secured "both" meaning)

    unsigned char s = streak[i];
    if (both) {
        if (s < 255) s++;
    } else {
        s = 0;
    }
    streak[i] = s;

    int need = freeze_monitor_iters - 1;
    if (need < 1) need = 1;

    freezeCand[i] = (s >= need) ? 1 : 0;
}
__global__ void freeze_apply_cand_kernel(
    const unsigned char* __restrict__ freezeCand,
    unsigned char* __restrict__ frozen,
    int nV,
    int* __restrict__ counts // counts[3] = newly frozen
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nV) return;

    if (freezeCand[i]) {
        if (!frozen[i]) {
            frozen[i] = 1;
            if (counts) atomicAdd(&counts[3], 1);
        }
    }
}
