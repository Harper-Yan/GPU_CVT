#include <cmath>
#include <limits>

#ifndef MAX_POLY_VERTS
#define MAX_POLY_VERTS 64
#endif

__host__ __device__ __forceinline__ float3 f3_add(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ __forceinline__ float3 f3_mul(float3 a, float s)   { return make_float3(a.x*s, a.y*s, a.z*s); }

__device__ __forceinline__ bool inside_halfspace(float2 p, float a, float b, float c, float eps) {
    return (a*p.x + b*p.y - c) <= eps;
}

__device__ __forceinline__ float2 intersect_seg_halfspace(float2 s, float2 e,
                                                          float a, float b, float c) {
    float fs = a*s.x + b*s.y - c;
    float fe = a*e.x + b*e.y - c;
    float t  = fs / (fs - fe + 1e-30f);
    return make_float2(s.x + t*(e.x - s.x), s.y + t*(e.y - s.y));
}

__device__ __forceinline__ int clip_poly_halfspace(const float2* inPoly, int inN,
                                                   float2* outPoly,
                                                   float a, float b, float c,
                                                   float eps) {
    if (inN <= 0) return 0;

    float2 s = inPoly[inN - 1];
    bool ins_s = inside_halfspace(s, a, b, c, eps);

    int outN = 0;
    for (int i = 0; i < inN; ++i) {
        float2 e = inPoly[i];
        bool ins_e = inside_halfspace(e, a, b, c, eps);

        if (ins_e) {
            if (!ins_s) outPoly[outN++] = intersect_seg_halfspace(s, e, a, b, c);
            outPoly[outN++] = e;
        } else if (ins_s) {
            outPoly[outN++] = intersect_seg_halfspace(s, e, a, b, c);
        }

        s = e;
        ins_s = ins_e;

        if (outN >= MAX_POLY_VERTS) { // hard cap (keeps it safe)
            outN = MAX_POLY_VERTS;
            break;
        }
    }
    return outN;
}

__device__ __forceinline__ float2 poly_centroid_2d(const float2* poly, int n) {
    if (n < 3) return make_float2(NAN, NAN);

    float A2 = 0.0f;
    float Cx = 0.0f, Cy = 0.0f;

    for (int i = 0; i < n; ++i) {
        float2 p  = poly[i];
        float2 q  = poly[(i + 1) % n];
        float cr  = p.x*q.y - q.x*p.y;
        A2 += cr;
        Cx += (p.x + q.x) * cr;
        Cy += (p.y + q.y) * cr;
    }

    // A = 0.5*A2
    float A = 0.5f * A2;
    if (fabsf(A) < 1e-18f) {
        float mx = 0.f, my = 0.f;
        for (int i = 0; i < n; ++i) { mx += poly[i].x; my += poly[i].y; }
        return make_float2(mx / n, my / n);
    }

    float inv6A = 1.0f / (6.0f * A);
    return make_float2(Cx * inv6A, Cy * inv6A);
}

__global__ void centroids_tangent_voronoi(const float3* __restrict__ S,
                                          const float3* __restrict__ U,
                                          const float3* __restrict__ V,
                                          const idx_t*   __restrict__ knn, // [nV*K]
                                          int nV, int K,
                                          float R,
                                          const unsigned char* dFrozen,
                                          float3* __restrict__ cent3d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nV) return;

    // testfreeze2 (lloyd_iter_sites_only): centroid for ALL sites including frozen
    float3 si = S[i];
    float3 ui = U[i];
    float3 vi = V[i];

    float2 polyA[MAX_POLY_VERTS];
    float2 polyB[MAX_POLY_VERTS];

    // initial square
    polyA[0] = make_float2(-R, -R);
    polyA[1] = make_float2( R, -R);
    polyA[2] = make_float2( R,  R);
    polyA[3] = make_float2(-R,  R);
    int nPoly = 4;

    const float eps = 1e-12f;

    // clip against neighbor bisectors (approx Voronoi cell)
    for (int t = 0; t < K; ++t) {
        idx_t j = knn[(size_t)i * K + t];

        if (j >= (idx_t)nV) continue;
        float3 d = f3_sub(S[(int)j], si);
        float a = f3_dot(d, ui);
        float b = f3_dot(d, vi);
        float c = 0.5f * f3_dot(d, d);

        nPoly = clip_poly_halfspace(polyA, nPoly, polyB, a, b, c, eps);
        if (nPoly == 0) break;

        // swap poly buffers
        for (int k = 0; k < nPoly; ++k) polyA[k] = polyB[k];
    }

    float2 c2 = poly_centroid_2d(polyA, nPoly);
    if (!isfinite(c2.x) || !isfinite(c2.y)) {
        cent3d[i] = make_float3(NAN, NAN, NAN);
        return;
    }

    cent3d[i] = f3_add(si, f3_add(f3_mul(ui, c2.x), f3_mul(vi, c2.y)));
}

__global__ void secured_centroids_tangent_voronoi(const float3* __restrict__ S,
                                          const float3* __restrict__ U,
                                          const float3* __restrict__ V,
                                          const idx_t*   __restrict__ knn, // [nV*K]
                                          int nV, int K,
                                          float R,
                                          const unsigned char* __restrict__ dFrozen,
                                          float3* __restrict__ cent3d,
                                          unsigned char* __restrict__ dSecureReached) // [nV] OUT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nV) return;

    // Default: not secured
    if (dSecureReached) dSecureReached[i] = 0;

    if (dFrozen[i]) {
        // Frozen cells don't update centroid; also not considered "newly secured" here.
        return;
    }

    float3 si = S[i];
    float3 ui = U[i];
    float3 vi = V[i];

    float2 polyA[MAX_POLY_VERTS];
    float2 polyB[MAX_POLY_VERTS];

    // initial square
    polyA[0] = make_float2(-R, -R);
    polyA[1] = make_float2( R, -R);
    polyA[2] = make_float2( R,  R);
    polyA[3] = make_float2(-R,  R);
    int nPoly = 4;

    const float eps = 1e-12f;

    // Compute initial rmax from square
    float rmax = 0.0f;
    for (int k = 0; k < nPoly; ++k) {
        float x = polyA[k].x, y = polyA[k].y;
        rmax = fmaxf(rmax, sqrtf(x*x + y*y));
    }

    unsigned char secured = 0;

    // clip against neighbor bisectors (approx Voronoi cell)
    for (int t = 0; t < K; ++t) {
        idx_t j = knn[(size_t)i * K + t];
        if (j >= (idx_t)nV) continue;

        float3 d = f3_sub(S[(int)j], si);
        float a = f3_dot(d, ui);
        float b = f3_dot(d, vi);

        float dj = sqrtf(a*a + b*b);

        // If next neighbor is too far to affect current polygon bounds, we are "secured"
        if (dj > 2.0f * rmax) {
            secured = 1;
            break;
        }

        float c = 0.5f * f3_dot(d, d);

        nPoly = clip_poly_halfspace(polyA, nPoly, polyB, a, b, c, eps);
        if (nPoly == 0) {
            // Degenerate cell => not secured
            secured = 0;
            break;
        }

        // update rmax + copy polyB -> polyA
        rmax = 0.0f;
        for (int k = 0; k < nPoly; ++k) {
            polyA[k] = polyB[k];
            float x = polyA[k].x, y = polyA[k].y;
            rmax = fmaxf(rmax, sqrtf(x*x + y*y));
        }
    }

    float2 c2 = poly_centroid_2d(polyA, nPoly);
    if (!isfinite(c2.x) || !isfinite(c2.y)) {
        cent3d[i] = make_float3(NAN, NAN, NAN);
        if (dSecureReached) dSecureReached[i] = 0;
        return;
    }

    cent3d[i] = f3_add(si, f3_add(f3_mul(ui, c2.x), f3_mul(vi, c2.y)));

    if (dSecureReached) dSecureReached[i] = secured;
}
