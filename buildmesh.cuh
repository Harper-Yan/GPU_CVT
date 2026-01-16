#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

static __host__ __device__ __forceinline__ uint64_t pack3_21(int a, int b, int c)
{
    return (uint64_t)(uint32_t)a | ((uint64_t)(uint32_t)b << 21) | ((uint64_t)(uint32_t)c << 42);
}

static __host__ __device__ __forceinline__ void sort3(int& a, int& b, int& c)
{
    if (a > b) { int t=a; a=b; b=t; }
    if (b > c) { int t=b; b=c; c=t; }
    if (a > b) { int t=a; a=b; b=t; }
}

template<int KNEIGH, int MAX_POLY, typename IndexT>
__global__ void emit_candidate_faces_kernel(
    const float3* __restrict__ S,
    const float3* __restrict__ U,
    const float3* __restrict__ V,
    const IndexT* __restrict__ idxn,
    const float2* __restrict__ poly2d,
    const int* __restrict__ polyN,
    int n,
    float tol,
    uint64_t* __restrict__ out_keys,
    int* __restrict__ out_count
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    int nv = polyN[i];
    if (nv < 3) return;

    float3 si = S[i];
    float3 ui = U[i];
    float3 vi = V[i];

    const IndexT* neigh = idxn + (size_t)i * KNEIGH;
    const float2* poly = poly2d + (size_t)i * MAX_POLY;

    for (int k = 0; k < nv; ++k) {
        float x = poly[k].x;
        float y = poly[k].y;

        int best0 = -1, best1 = -1;
        float v0 = 1e30f, v1 = 1e30f;

        for (int t = 0; t < KNEIGH; ++t) {
            int j = (int)neigh[t];
            if ((unsigned)j >= (unsigned)n) continue;
            if (j == i) continue;

            float3 d;
            d.x = S[j].x - si.x;
            d.y = S[j].y - si.y;
            d.z = S[j].z - si.z;

            float a = d.x*ui.x + d.y*ui.y + d.z*ui.z;
            float b = d.x*vi.x + d.y*vi.y + d.z*vi.z;
            float c = 0.5f * (d.x*d.x + d.y*d.y + d.z*d.z);

            float ab = fabsf(a*x + b*y - c);

            if (ab < v0) {
                v1 = v0; best1 = best0;
                v0 = ab; best0 = t;
            } else if (ab < v1) {
                v1 = ab; best1 = t;
            }
        }

        if (best0 < 0 || best1 < 0) continue;

        int j1 = (int)neigh[best0];
        int j2 = (int)neigh[best1];
        if (j1 == j2) continue;
        if (j1 == i || j2 == i) continue;

        if (v0 > tol || v1 > tol) continue;

        int a = i, b = j1, c = j2;
        if (a == b || a == c || b == c) continue;

        int pos = atomicAdd(out_count, 1);
        out_keys[pos] = pack3_21(a, b, c);
    }
}

struct Tri { int a,b,c; };

static inline void decode_keys_to_faces(const std::vector<uint64_t>& keys, std::vector<Tri>& F)
{
    F.clear();
    F.reserve(keys.size());
    for (uint64_t k : keys) {
        int a = (int)(k & ((1u<<21)-1));
        int b = (int)((k >> 21) & ((1u<<21)-1));
        int c = (int)((k >> 42) & ((1u<<21)-1));
        F.push_back({a,b,c});
    }
}

static inline void write_obj_cpu(const std::string& path, const std::vector<float3>& V, const std::vector<Tri>& F)
{
    std::ofstream os(path);
    for (auto& p : V) os << "v " << p.x << " " << p.y << " " << p.z << "\n";
    for (auto& t : F) os << "f " << (t.a+1) << " " << (t.b+1) << " " << (t.c+1) << "\n";
}
template<int KNEIGH, typename IndexT>
__global__ void precompute_abc_kernel(
    const float3* __restrict__ S,
    const float3* __restrict__ U,
    const float3* __restrict__ V,
    const IndexT* __restrict__ idxn,
    int n,
    double* __restrict__ A,
    double* __restrict__ B,
    double* __restrict__ C
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    float3 si = S[i];
    float3 ui = U[i];
    float3 vi = V[i];

    const IndexT* neigh = idxn + (size_t)i * KNEIGH;

    #pragma unroll
    for (int t = 0; t < KNEIGH; ++t) {
        int j = (int)neigh[t];
        size_t p = (size_t)i * KNEIGH + (size_t)t;

        if ((unsigned)j >= (unsigned)n || j == i) {
            A[p] = 0.0;
            B[p] = 0.0;
            C[p] = 0.0;
            continue;
        }

        float3 d;
        d.x = S[j].x - si.x;
        d.y = S[j].y - si.y;
        d.z = S[j].z - si.z;

        double a = (double)d.x*(double)ui.x + (double)d.y*(double)ui.y + (double)d.z*(double)ui.z;
        double b = (double)d.x*(double)vi.x + (double)d.y*(double)vi.y + (double)d.z*(double)vi.z;
        double c = 0.5 * ((double)d.x*(double)d.x + (double)d.y*(double)d.y + (double)d.z*(double)d.z);

        A[p] = a;
        B[p] = b;
        C[p] = c;
    }
}

__device__ __forceinline__ void sort3_int(int& a, int& b, int& c) {
    if (a > b) { int t=a; a=b; b=t; }
    if (b > c) { int t=b; b=c; c=t; }
    if (a > b) { int t=a; a=b; b=t; }
}

template<int MAX_POLY>
__global__ void emit_faces_from_labels_kernel(
    const int*  __restrict__ polyN,
    const int2* __restrict__ polyLab2,
    int n,
    uint64_t* __restrict__ out_keys,
    int* __restrict__ out_count,
    int max_out
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    int nv = polyN[i];
    if (nv < 3) return;

    const int2* lab = polyLab2 + (size_t)i * MAX_POLY;

    for (int k = 0; k < nv; ++k) {
        int b = lab[k].x;
        int c = lab[k].y;

        if (b < 0 || c < 0) continue;
        if (b == c) continue;
        if (b == i || c == i) continue;

        int a = i;
        sort3(a, b, c);

        int pos = atomicAdd(out_count, 1);
        if (pos < max_out) out_keys[pos] = pack3_21(a, b, c);
    }
}

static inline void write_pts_cpu(const std::string& path, const std::vector<float3>& P)
{
    std::ofstream os(path);
    for (auto& p : P) os << p.x << " " << p.y << " " << p.z << "\n";
}
static inline float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline float dot3(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static inline float norm3(const float3& a) {
    return sqrtf(dot3(a,a));
}
static inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

static inline bool is_degenerate_tri(const std::vector<float3>& V, int a, int b, int c, float eps) {
    if (a == b || b == c || a == c) return true;
    if ((unsigned)a >= V.size() || (unsigned)b >= V.size() || (unsigned)c >= V.size()) return true;

    const float3 A = V[(size_t)a];
    const float3 B = V[(size_t)b];
    const float3 C = V[(size_t)c];

    float e0 = norm3(sub3(B, A));
    float e1 = norm3(sub3(C, B));
    float e2 = norm3(sub3(A, C));

    float3 n = cross3(sub3(B, A), sub3(C, A));
    float area2 = norm3(n); // matches python: norm(cross(...))

    return !(e0 > eps && e1 > eps && e2 > eps && area2 > eps);
}
