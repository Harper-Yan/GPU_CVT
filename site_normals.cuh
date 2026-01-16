#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

struct int3i { int x, y, z; };

__host__ __device__ __forceinline__  float3 f3_sub(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ __forceinline__ float3 f3_cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ __forceinline__  float f3_dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__global__ void zero_normals(float3* n, int nV) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nV) n[i] = make_float3(0.f, 0.f, 0.f);
}

__global__ void accumulate_face_normals_atomic(const float3* pos, const int3i* tri, float3* nrm, int nF) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;

    int a = tri[f].x, b = tri[f].y, c = tri[f].z;
    float3 pa = pos[a], pb = pos[b], pc = pos[c];
    float3 fn = f3_cross(f3_sub(pb, pa), f3_sub(pc, pa));

    atomicAdd(&nrm[a].x, fn.x); atomicAdd(&nrm[a].y, fn.y); atomicAdd(&nrm[a].z, fn.z);
    atomicAdd(&nrm[b].x, fn.x); atomicAdd(&nrm[b].y, fn.y); atomicAdd(&nrm[b].z, fn.z);
    atomicAdd(&nrm[c].x, fn.x); atomicAdd(&nrm[c].y, fn.y); atomicAdd(&nrm[c].z, fn.z);
}

__global__ void normalize_normals(float3* n, int nV) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nV) return;
    float3 v = n[i];
    float len2 = f3_dot(v, v);
    if (len2 > 0.f) {
        float inv = rsqrtf(len2);
        n[i] = make_float3(v.x * inv, v.y * inv, v.z * inv);
    }
}

__device__ __forceinline__ float3 f3_make(float x, float y, float z) { return make_float3(x,y,z); }

__device__ __forceinline__ float3 f3_norm(float3 v) {
    float l2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (l2 > 0.f) {
        float inv = rsqrtf(l2);
        return f3_make(v.x*inv, v.y*inv, v.z*inv);
    }
    return f3_make(0.f, 0.f, 0.f);
}

__global__ inline void tangent_planes_from_normals(const float3* N, float3* U, float3* V, int nV)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nV) return;

    float3 n = f3_norm(N[i]);

    float3 h = (fabsf(n.z) > 0.9f) ? f3_make(1.f, 0.f, 0.f) : f3_make(0.f, 0.f, 1.f);
    float3 u = f3_norm(f3_cross(h, n));
    float3 v = f3_norm(f3_cross(n, u));

    U[i] = u;
    V[i] = v;
}
template<int KPROJ>
__global__ void uv_from_nearest_vertex_normal(
    const float3* __restrict__ N0,
    const int*    __restrict__ knnV,
    float3* __restrict__ U,
    float3* __restrict__ V,
    int n
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    int nn = knnV[i * KPROJ + 0];
    if (nn < 0) { U[i] = make_float3(1,0,0); V[i] = make_float3(0,1,0); return; }

    // normalize nrm (python does this)
    float3 nrm = N0[nn];
    float invn = rsqrtf(nrm.x*nrm.x + nrm.y*nrm.y + nrm.z*nrm.z + 1e-30f);
    nrm.x *= invn; nrm.y *= invn; nrm.z *= invn;

    // python: if abs(nz)>0.9 -> h=(1,0,0) else h=(0,0,1)
    float3 h = (fabsf(nrm.z) > 0.9f) ? make_float3(1.f,0.f,0.f) : make_float3(0.f,0.f,1.f);

    // u = normalize(cross(h, n))
    float3 u = make_float3(
        h.y*nrm.z - h.z*nrm.y,
        h.z*nrm.x - h.x*nrm.z,
        h.x*nrm.y - h.y*nrm.x
    );
    float invu = rsqrtf(u.x*u.x + u.y*u.y + u.z*u.z + 1e-30f);
    u.x *= invu; u.y *= invu; u.z *= invu;

    // v = normalize(cross(n, u))   (python normalizes v too)
    float3 v = make_float3(
        nrm.y*u.z - nrm.z*u.y,
        nrm.z*u.x - nrm.x*u.z,
        nrm.x*u.y - nrm.y*u.x
    );
    float invv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z + 1e-30f);
    v.x *= invv; v.y *= invv; v.z *= invv;

    U[i] = u;
    V[i] = v;
}