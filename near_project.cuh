
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

__host__ __device__ __forceinline__
float f3_len2(const float3 a)
{
    return f3_dot(a, a);
}

__device__ __forceinline__ float3 closest_point_tri(float3 P, float3 A, float3 B, float3 C){
    float3 AB = f3_sub(B,A);
    float3 AC = f3_sub(C,A);
    float3 AP = f3_sub(P,A);

    float d1 = f3_dot(AB, AP);
    float d2 = f3_dot(AC, AP);
    if (d1 <= 0.0f && d2 <= 0.0f) return A;

    float3 BP = f3_sub(P,B);
    float d3 = f3_dot(AB, BP);
    float d4 = f3_dot(AC, BP);
    if (d3 >= 0.0f && d4 <= d3) return B;

    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f){
        float v = d1 / (d1 - d3 + 1e-30f);
        return f3_add(A, f3_mul(AB, v));
    }

    float3 CP = f3_sub(P,C);
    float d5 = f3_dot(AB, CP);
    float d6 = f3_dot(AC, CP);
    if (d6 >= 0.0f && d5 <= d6) return C;

    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f){
        float w = d2 / (d2 - d6 + 1e-30f);
        return f3_add(A, f3_mul(AC, w));
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f){
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-30f);
        return f3_add(B, f3_mul(f3_sub(C,B), w));
    }

    float denom = 1.0f / (va + vb + vc + 1e-30f);
    float v = vb * denom;
    float w = vc * denom;
    return f3_add(A, f3_add(f3_mul(AB, v), f3_mul(AC, w)));
}
template<int KPROJ, bool EXCLUDE_SELF>
__global__ void knn_vertices_bruteforce_k(const float3* __restrict__ Vpos, int nV,
                                         const float3* __restrict__ Q, int nQ, const unsigned char* __restrict__ frozen, 
                                         int* __restrict__ out_idx){
            
    int qi = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qi >= nQ) return;

    if (frozen[qi]) return;


    float3 q = Q[qi];
    float best_d2[KPROJ];
    int   best_i[KPROJ];
    #pragma unroll
    for(int t=0;t<KPROJ;++t){ best_d2[t] = 1e30f; best_i[t] = -1; }

    for(int vi=0; vi<nV; ++vi){
        if constexpr (EXCLUDE_SELF) {
            if (vi == qi) continue;
        }

        float3 d = f3_sub(Vpos[vi], q);
        float d2 = f3_len2(d);

        if (d2 >= best_d2[KPROJ-1]) continue;
        int pos = KPROJ-1;
        best_d2[pos] = d2; best_i[pos] = vi;
        while(pos>0 && best_d2[pos] < best_d2[pos-1]){
            float td = best_d2[pos-1]; best_d2[pos-1] = best_d2[pos]; best_d2[pos] = td;
            int   ti = best_i[pos-1];  best_i[pos-1]  = best_i[pos];  best_i[pos]  = ti;
            --pos;
        }
    }

    int base = qi * KPROJ;
    #pragma unroll
    for(int t=0;t<KPROJ;++t) out_idx[base + t] = best_i[t];
}

/** Restore vertex KNN for frozen sites from previous iteration (bitonic path skips frozen). */
template<int KPROJ>
__global__ void restore_prev_knn_vertices_kernel(const unsigned char* __restrict__ frozen,
    const int* __restrict__ prev_knn, int* __restrict__ out_knn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !frozen[i]) return;
    int base = i * KPROJ;
    for (int t = 0; t < KPROJ; ++t)
        out_knn[base + t] = prev_knn[base + t];
}

template<int KPROJ, bool EXCLUDE_SELF>
__global__ void knn_vertices_bruteforce_k_active(
    const float3* __restrict__ Vpos, int nV,
    const float3* __restrict__ Q,    int nQ,
    const int*    __restrict__ active_idx, int nActive,
    int* __restrict__ out_idx)
{
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= nActive) return;

    int qi = active_idx[t];     
    if (qi >= nQ) return;

    float3 q = Q[qi];
    float best_d2[KPROJ];
    int   best_i[KPROJ];

    #pragma unroll
    for(int k=0;k<KPROJ;++k){ best_d2[k] = 1e30f; best_i[k] = -1; }

    for(int vi=0; vi<nV; ++vi){
        if constexpr (EXCLUDE_SELF) {
            if (vi == qi) continue;
        }
        float3 d = f3_sub(Vpos[vi], q);
        float d2 = f3_len2(d);

        if (d2 >= best_d2[KPROJ-1]) continue;
        int pos = KPROJ-1;
        best_d2[pos] = d2; best_i[pos] = vi;

        while(pos>0 && best_d2[pos] < best_d2[pos-1]){
            float td = best_d2[pos-1]; best_d2[pos-1] = best_d2[pos]; best_d2[pos] = td;
            int   ti = best_i[pos-1];  best_i[pos-1]  = best_i[pos];  best_i[pos]  = ti;
            --pos;
        }
    }

    int base = qi * KPROJ;      // IMPORTANT: write back to original qi slot
    #pragma unroll
    for(int k=0;k<KPROJ;++k) out_idx[base + k] = best_i[k];
}

template<int KPROJ>
__global__ void project_centroids_to_mesh(const float3* __restrict__ cent3d,
                                         const float3* __restrict__ Vpos,
                                         const int3*  __restrict__ F,
                                         const int*   __restrict__ vf_off,
                                         const int*   __restrict__ vf_faces,
                                         const int*   __restrict__ knnV,
                                         int nV, int nF, int nQ, const unsigned char* __restrict__ frozen, float3* __restrict__ Sold,
                                         float3* __restrict__ out_Snew){
    int qi = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (qi >= nQ) return;

    // testfreeze2 (lloyd_iter_sites_only): project all sites (no Snew[frozen]=S[frozen])
    float3 c = cent3d[qi];

    float best = 1e30f;
    float3 bestP = c;

    int base = qi * KPROJ;
    #pragma unroll
    for(int t=0;t<KPROJ;++t){
        int v = knnV[base + t];
        if ((unsigned)v >= (unsigned)nV) continue;

        int beg = vf_off[v];
        int end = vf_off[v+1];
        for(int ii=beg; ii<end; ++ii){
            int fi = vf_faces[ii];
            if ((unsigned)fi >= (unsigned)nF) continue;
            int3 tri = F[fi];
            float3 A = Vpos[tri.x];
            float3 B = Vpos[tri.y];
            float3 C = Vpos[tri.z];
            float3 p = closest_point_tri(c, A, B, C);
            float d2 = f3_len2(f3_sub(p, c));
            if (d2 < best){ best = d2; bestP = p; }
        }
    }

    out_Snew[qi] = bestP;
}
#include <cuda_runtime.h>

static __device__ __forceinline__ float eval_halfspace(const float2& p, float a, float b, float c)
{
    return a * p.x + b * p.y - c;
}

static __device__ __forceinline__ float2 lerp2(const float2& s, const float2& e, float t)
{
    return make_float2(s.x + t * (e.x - s.x), s.y + t * (e.y - s.y));
}
__device__ int clip_poly_halfspace_labeled_dev(
    const float2* __restrict__ inP,
    const int*    __restrict__ inL,
    int inN,
    float a, float b, float c,
    float2* __restrict__ outP,
    int*    __restrict__ outL,
    float eps,
    int maxN,
    int label_j
){
    if (inN <= 0) return 0;

    int outN = 0;
    float2 s = inP[inN - 1];
    int    ls = inL[inN - 1];
    float fs = eval_halfspace(s, a, b, c);

    for (int idx = 0; idx < inN; ++idx) {
        float2 e = inP[idx];
        int    le = inL[idx];
        float fe = eval_halfspace(e, a, b, c);

        bool ins_e = (fe <= eps);
        bool ins_s = (fs <= eps);

        if (ins_e) {
            if (!ins_s) {
                float t = fs / (fs - fe + 1e-30f);
                float2 x = lerp2(s, e, t);
                if (outN < maxN) { outP[outN] = x; outL[outN] = label_j; outN++; }
            }
            if (outN < maxN) { outP[outN] = e; outL[outN] = le; outN++; }
        } else if (ins_s) {
            float t = fs / (fs - fe + 1e-30f);
            float2 x = lerp2(s, e, t);
            if (outN < maxN) { outP[outN] = x; outL[outN] = label_j; outN++; }
        }

        s = e; ls = le; fs = fe;
    }

    return outN;
}

template<int KNEIGH, int MAX_POLY, typename IndexT>
__global__ void cell_poly2d_kernel(
    const float3* __restrict__ S,
    const float3* __restrict__ U,
    const float3* __restrict__ V,
    const IndexT* __restrict__ idxn,
    int n,
    float R,
    float eps_clip,           
    float tol_active,         
    float2* __restrict__ poly2d,
    int*    __restrict__ polyN,
    int2*   __restrict__ polyLab2   
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    float2 bufA[MAX_POLY];
    float2 bufB[MAX_POLY];
    int    labA[MAX_POLY];
    int    labB[MAX_POLY];

    int na = 4;
    bufA[0] = make_float2(-R, -R);
    bufA[1] = make_float2( R, -R);
    bufA[2] = make_float2( R,  R);
    bufA[3] = make_float2(-R,  R);
    labA[0] = labA[1] = labA[2] = labA[3] = -1;

    float3 si = S[i];
    float3 ui = U[i];
    float3 vi = V[i];

    const IndexT* neigh = idxn + (size_t)i * KNEIGH;

    // --- clipping stage (unchanged) ---
    for (int t = 0; t < KNEIGH; ++t) {
        int j = (int)neigh[t];
        if ((unsigned)j >= (unsigned)n) continue;
        if (j == i) continue;

        float3 d = make_float3(S[j].x - si.x, S[j].y - si.y, S[j].z - si.z);

        float a = d.x*ui.x + d.y*ui.y + d.z*ui.z;
        float b = d.x*vi.x + d.y*vi.y + d.z*vi.z;
        float c = 0.5f * (d.x*d.x + d.y*d.y + d.z*d.z);

        int nb = clip_poly_halfspace_labeled_dev(
            bufA, labA, na,
            a, b, c,
            bufB, labB,
            eps_clip, MAX_POLY,
            j
        );

        na = nb;
        if (na == 0) break;

        #pragma unroll
        for (int k = 0; k < MAX_POLY; ++k) {
            if (k < na) { bufA[k] = bufB[k]; labA[k] = labB[k]; }
        }
    }

    polyN[i] = na;

    float2* outP = poly2d  + (size_t)i * MAX_POLY;
    int2*   outL = polyLab2 + (size_t)i * MAX_POLY;

    // write polygon vertices
    for (int k = 0; k < na; ++k) outP[k] = bufA[k];

    // --- active-bisector stage (NEW): find 2 closest constraints for each vertex ---
    // Python computes residuals against ALL neighbors; do the same.
    for (int k = 0; k < na; ++k) {
        float2 p = bufA[k];

        float best1 = 1e30f, best2 = 1e30f;
        int   j1 = -1, j2 = -1;

        for (int t = 0; t < KNEIGH; ++t) {
            int j = (int)neigh[t];
            if ((unsigned)j >= (unsigned)n) continue;
            if (j == i) continue;

            float3 d = make_float3(S[j].x - si.x, S[j].y - si.y, S[j].z - si.z);

            float a = d.x*ui.x + d.y*ui.y + d.z*ui.z;
            float b = d.x*vi.x + d.y*vi.y + d.z*vi.z;
            float c = 0.5f * (d.x*d.x + d.y*d.y + d.z*d.z);

            // residual: a*x + b*y - c
            float s  = a * p.x + b * p.y - c;
            float as = fabsf(s);

            if (as < best1) { best2 = best1; j2 = j1; best1 = as; j1 = j; }
            else if (as < best2) { best2 = as; j2 = j; }
        }

        // require TWO active constraints (python behavior)
        if (j1 >= 0 && j2 >= 0 && j1 != j2 && best1 <= tol_active && best2 <= tol_active) {
            outL[k] = make_int2(j1, j2);
        } else {
            outL[k] = make_int2(-1, -1);
        }
    }
}

static __device__ int clip_poly_halfspace_dev(
    const float2* in_poly, int nin,
    float a, float b, float c,
    float2* out_poly,
    float eps,
    int max_out
){
    if (nin <= 0) return 0;

    float2 s = in_poly[nin - 1];
    float fs = eval_halfspace(s, a, b, c);

    int nout = 0;
    for (int k = 0; k < nin; ++k) {
        float2 e = in_poly[k];
        float fe = eval_halfspace(e, a, b, c);

        bool ins_e = (fe <= eps);
        bool ins_s = (fs <= eps);

        if (ins_e) {
            if (!ins_s) {
                float denom = (fs - fe);
                float t = fs / (denom + 1e-30f);
                float2 p = lerp2(s, e, t);
                if (nout < max_out) out_poly[nout++] = p;
            }
            if (nout < max_out) out_poly[nout++] = e;
        } else if (ins_s) {
            float denom = (fs - fe);
            float t = fs / (denom + 1e-30f);
            float2 p = lerp2(s, e, t);
            if (nout < max_out) out_poly[nout++] = p;
        }

        s = e;
        fs = fe;
    }

    return nout;
}