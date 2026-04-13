// multifacet_clip.cuh
// Reimplementation of Fei et al. 2025 multi-facet clipping baseline.
// "A Remeshing Method via Adaptive Multiple Original-Facet-Clipping and CVT"
// arXiv:2505.14306
//
// Per site: find host face, build 2-ring neighborhood, select 1-3 facets
// based on curvature (normal angle thresholds), clip the Voronoi cell on
// each facet's plane, and combine via area-weighted centroid.

#pragma once
#include <cuda_runtime.h>

#ifndef MAX_CAND_FACES
#define MAX_CAND_FACES 64
#endif

#ifndef MAX_RING_VERTS
#define MAX_RING_VERTS 48
#endif

// ── per-face unit normal computation ─────────────────────────────────────────
__global__ void compute_face_normals(const float3* __restrict__ Vpos,
                                     const int3i*  __restrict__ Faces,
                                     float3* __restrict__ faceN,
                                     int nF)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    int3i tri = Faces[f];
    float3 A = Vpos[tri.x], B = Vpos[tri.y], C = Vpos[tri.z];
    float3 n = f3_cross(f3_sub(B, A), f3_sub(C, A));
    float len2 = f3_dot(n, n);
    if (len2 > 0.f) {
        float inv = rsqrtf(len2);
        n.x *= inv; n.y *= inv; n.z *= inv;
    }
    faceN[f] = n;
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Add val to arr[0..n) if not already present; return updated n.
__device__ __forceinline__ int arr_add_unique_int(int* arr, int n, int val, int cap) {
    for (int d = 0; d < n; ++d)
        if (arr[d] == val) return n;
    if (n < cap) arr[n++] = val;
    return n;
}

// Face centroid (inline, avoids repeated code).
__device__ __forceinline__ float3 face_centroid_dev(
    const float3* Vpos, const int3i* Faces, int fi)
{
    int3i t = Faces[fi];
    return make_float3(
        (Vpos[t.x].x + Vpos[t.y].x + Vpos[t.z].x) * (1.0f / 3.0f),
        (Vpos[t.x].y + Vpos[t.y].y + Vpos[t.z].y) * (1.0f / 3.0f),
        (Vpos[t.x].z + Vpos[t.y].z + Vpos[t.z].z) * (1.0f / 3.0f));
}

// Point-in-triangle via barycentric coordinates.
// Tests whether the projection of p onto the plane of triangle (A,B,C)
// falls inside the triangle.  fn = unit face normal (precomputed).
__device__ __forceinline__ bool point_in_triangle_dev(
    float3 p, float3 A, float3 B, float3 C, float3 fn)
{
    // project p onto triangle plane
    float h = f3_dot(f3_sub(p, A), fn);
    float3 pp = f3_sub(f3_sub(p, f3_mul(fn, h)), A);   // pp = proj(p) - A

    float3 e0 = f3_sub(B, A);
    float3 e1 = f3_sub(C, A);

    float d00 = f3_dot(e0, e0);
    float d01 = f3_dot(e0, e1);
    float d02 = f3_dot(e0, pp);
    float d11 = f3_dot(e1, e1);
    float d12 = f3_dot(e1, pp);

    float det = d00 * d11 - d01 * d01;
    if (fabsf(det) < 1e-30f) return false;
    float inv = 1.0f / det;
    float bu = (d11 * d02 - d01 * d12) * inv;
    float bv = (d00 * d12 - d01 * d02) * inv;

    const float eps = 1e-5f;
    return (bu >= -eps) && (bv >= -eps) && (bu + bv <= 1.0f + eps);
}

// Barycentric interpolation of vertex normals at point p on triangle (A,B,C).
__device__ __forceinline__ float3 interp_vertex_normal(
    float3 p, float3 A, float3 B, float3 C,
    float3 nA, float3 nB, float3 nC)
{
    float3 v0 = f3_sub(B, A), v1 = f3_sub(C, A), v2 = f3_sub(p, A);
    float d00 = f3_dot(v0, v0), d01 = f3_dot(v0, v1), d02 = f3_dot(v0, v2);
    float d11 = f3_dot(v1, v1), d12 = f3_dot(v1, v2);
    float det = d00 * d11 - d01 * d01;
    float inv = (fabsf(det) > 1e-30f) ? (1.0f / det) : 0.0f;
    float u = (d11 * d02 - d01 * d12) * inv;
    float v = (d00 * d12 - d01 * d02) * inv;
    float w = 1.0f - u - v;
    // clamp to triangle
    w = fmaxf(w, 0.0f); u = fmaxf(u, 0.0f); v = fmaxf(v, 0.0f);
    float s = w + u + v;
    if (s > 0.0f) { w /= s; u /= s; v /= s; }
    float3 n = make_float3(
        w * nA.x + u * nB.x + v * nC.x,
        w * nA.y + u * nB.y + v * nC.y,
        w * nA.z + u * nB.z + v * nC.z);
    return f3_norm(n);
}

// ── multi-facet Voronoi clipping kernel ──────────────────────────────────────
template<int KPROJ>
__global__ void centroids_multifacet_voronoi(
    const float3* __restrict__ S,
    const idx_t*  __restrict__ knn_sites,   // [nSites * K]
    int nSites, int K,
    const float3* __restrict__ Vpos,        // original mesh vertices
    const int3i*  __restrict__ Faces,       // original mesh faces
    const float3* __restrict__ faceN,       // per-face unit normals [nF]
    const float3* __restrict__ vertN,      // per-vertex unit normals [nMeshV]
    int nMeshV, int nF,
    const int*    __restrict__ knnV,        // site-to-mesh-vertex KNN [nSites*KPROJ]
    const int*    __restrict__ vf_off,      // vertex-to-face CSR offset [nMeshV+1]
    const int*    __restrict__ vf_faces,    // vertex-to-face list
    float R,                                // bounding half-width (same as tangent-plane Lloyd)
    float3* __restrict__ cent3d,
    double* __restrict__ energy_out,
    int* __restrict__ level_counts = nullptr) // optional [3]: atomicAdd level 1/2/3 counts
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nSites) return;

    float3 si = S[i];

    // ══════════════════════════════════════════════════════════════════════
    // 1.  Find host face f_t  (point-in-triangle containment test)
    // ══════════════════════════════════════════════════════════════════════
    int hostFace = -1;
    for (int kp = 0; kp < KPROJ && hostFace < 0; ++kp) {
        int v = knnV[i * KPROJ + kp];
        if ((unsigned)v >= (unsigned)nMeshV) continue;
        int beg = vf_off[v];
        int end = vf_off[v + 1];
        for (int ii = beg; ii < end; ++ii) {
            int fi = vf_faces[ii];
            if ((unsigned)fi >= (unsigned)nF) continue;
            int3i tri = Faces[fi];
            if (point_in_triangle_dev(si, Vpos[tri.x], Vpos[tri.y], Vpos[tri.z], faceN[fi])) {
                hostFace = fi;
                break;
            }
        }
    }
    // Fallback: if no containment found (site on edge / numerical noise),
    // pick the face with smallest perpendicular distance.
    if (hostFace < 0) {
        float bestPD = 1e30f;
        for (int kp = 0; kp < KPROJ; ++kp) {
            int v = knnV[i * KPROJ + kp];
            if ((unsigned)v >= (unsigned)nMeshV) continue;
            int beg = vf_off[v];
            int end = vf_off[v + 1];
            for (int ii = beg; ii < end; ++ii) {
                int fi = vf_faces[ii];
                if ((unsigned)fi >= (unsigned)nF) continue;
                float pd = fabsf(f3_dot(f3_sub(si, Vpos[Faces[fi].x]), faceN[fi]));
                if (pd < bestPD) { bestPD = pd; hostFace = fi; }
            }
        }
    }
    if (hostFace < 0) {
        cent3d[i] = si;
        if (energy_out) energy_out[i] = 0.0;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2.  Build twice-ring (2-ring) neighbourhood from host face
    // ══════════════════════════════════════════════════════════════════════
    int cand[MAX_CAND_FACES];
    int nCand = 0;

    int ring_v[MAX_RING_VERTS];
    int nRV = 0;

    // Ring-0: host face itself + its 3 vertices
    cand[nCand++] = hostFace;
    {
        int3i ht = Faces[hostFace];
        ring_v[nRV++] = ht.x;
        ring_v[nRV++] = ht.y;
        ring_v[nRV++] = ht.z;
    }
    int ring1_vstart = nRV;   // vertices added from ring-1 start here

    // Ring-1: faces incident to ring-0 vertices, collect new vertices
    for (int rv = 0; rv < ring1_vstart; ++rv) {
        int v = ring_v[rv];
        if ((unsigned)v >= (unsigned)nMeshV) continue;
        int beg = vf_off[v];
        int end = vf_off[v + 1];
        for (int ii = beg; ii < end && nCand < MAX_CAND_FACES; ++ii) {
            int fi = vf_faces[ii];
            if ((unsigned)fi >= (unsigned)nF) continue;
            nCand = arr_add_unique_int(cand, nCand, fi, MAX_CAND_FACES);
            // collect vertices for next ring
            int3i tri = Faces[fi];
            nRV = arr_add_unique_int(ring_v, nRV, tri.x, MAX_RING_VERTS);
            nRV = arr_add_unique_int(ring_v, nRV, tri.y, MAX_RING_VERTS);
            nRV = arr_add_unique_int(ring_v, nRV, tri.z, MAX_RING_VERTS);
        }
    }

    // Ring-2: faces incident to ring-1 vertices (those added above)
    for (int rv = ring1_vstart; rv < nRV; ++rv) {
        int v = ring_v[rv];
        if ((unsigned)v >= (unsigned)nMeshV) continue;
        int beg = vf_off[v];
        int end = vf_off[v + 1];
        for (int ii = beg; ii < end && nCand < MAX_CAND_FACES; ++ii) {
            int fi = vf_faces[ii];
            if ((unsigned)fi >= (unsigned)nF) continue;
            nCand = arr_add_unique_int(cand, nCand, fi, MAX_CAND_FACES);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 3.  d_max  +  distance filter  (≤ 2 × d_max from host-face centroid)
    // ══════════════════════════════════════════════════════════════════════
    float d_max_sq = 0.0f;
    for (int t = 0; t < K; ++t) {
        idx_t j = knn_sites[(size_t)i * K + t];
        if (j >= (idx_t)nSites) continue;
        float3 d = f3_sub(S[j], si);
        float d2 = f3_dot(d, d);
        if (d2 > d_max_sq) d_max_sq = d2;
    }
    float d_max = sqrtf(d_max_sq) + 1e-10f;

    float3 fc_host = face_centroid_dev(Vpos, Faces, hostFace);
    float dist_lim_sq = 4.0f * d_max_sq;   // (2 × d_max)²

    int nFiltered = 0;
    for (int ci = 0; ci < nCand; ++ci) {
        float3 fc_c = face_centroid_dev(Vpos, Faces, cand[ci]);
        float3 dd   = f3_sub(fc_c, fc_host);
        if (f3_dot(dd, dd) <= dist_lim_sq)
            cand[nFiltered++] = cand[ci];
    }
    nCand = nFiltered;

    if (nCand == 0) {
        cent3d[i] = si;
        if (energy_out) energy_out[i] = 0.0;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4.  Use ALL candidate triangles as clipping facets (RVD approach).
    //     Each triangle is clipped by the Voronoi bisectors, giving the
    //     exact restricted Voronoi cell on that triangle.
    // ══════════════════════════════════════════════════════════════════════

    // Sort candidates by distance to site (closest first) and cap at 10
    constexpr int MAX_CLIP = 10;
    int nClipFaces = (nCand < MAX_CLIP) ? nCand : MAX_CLIP;

    // simple selection sort of first nClipFaces by distance to site
    for (int ci = 0; ci < nClipFaces; ++ci) {
        float bestD = 1e30f;
        int   bestJ = ci;
        for (int cj = ci; cj < nCand; ++cj) {
            float3 fc_c = face_centroid_dev(Vpos, Faces, cand[cj]);
            float3 dd   = f3_sub(fc_c, si);
            float  d2   = f3_dot(dd, dd);
            if (d2 < bestD) { bestD = d2; bestJ = cj; }
        }
        int tmp = cand[ci]; cand[ci] = cand[bestJ]; cand[bestJ] = tmp;
    }

    if (level_counts) atomicAdd(&level_counts[min(nClipFaces, 3) - 1], 1);

    // ══════════════════════════════════════════════════════════════════════
    // 5.  Clip Voronoi cell on each facet triangle, area-weighted combine
    // ══════════════════════════════════════════════════════════════════════
    float3 weighted_cent = make_float3(0.f, 0.f, 0.f);
    double total_area   = 0.0;
    double total_energy = 0.0;

    for (int cf = 0; cf < nClipFaces; ++cf) {
        int   fi  = cand[cf];
        int3i tri = Faces[fi];
        float3 fA = Vpos[tri.x], fB = Vpos[tri.y], fC = Vpos[tri.z];
        float3 fn = faceN[fi];

        // tangent frame from face normal
        float3 h = (fabsf(fn.z) > 0.9f) ? make_float3(1.f, 0.f, 0.f)
                                         : make_float3(0.f, 0.f, 1.f);
        float3 u  = f3_norm(f3_cross(h, fn));
        float3 vv = f3_norm(f3_cross(fn, u));

        // project site onto facet plane
        float  hi     = f3_dot(f3_sub(si, fA), fn);
        float3 proj_i = f3_sub(si, f3_mul(fn, hi));

        // initial polygon = the mesh TRIANGLE (not [-R,R]²)
        // This bounds integration to the actual surface facet.
        float3 dA = f3_sub(fA, proj_i);
        float3 dB = f3_sub(fB, proj_i);
        float3 dC = f3_sub(fC, proj_i);

        float2 polyA[MAX_POLY_VERTS];
        float2 polyB[MAX_POLY_VERTS];

        polyA[0] = make_float2(f3_dot(dA, u), f3_dot(dA, vv));
        polyA[1] = make_float2(f3_dot(dB, u), f3_dot(dB, vv));
        polyA[2] = make_float2(f3_dot(dC, u), f3_dot(dC, vv));
        int nPoly = 3;

        const float eps = 1e-12f;

        // clip against KNN-site bisectors
        for (int t = 0; t < K; ++t) {
            idx_t j = knn_sites[(size_t)i * K + t];
            if (j >= (idx_t)nSites) continue;

            float3 d = f3_sub(S[j], si);
            float a = f3_dot(d, u);
            float b = f3_dot(d, vv);
            // bisector half-plane: a*x + b*y ≤ c
            // general form accounts for site being off-plane by hi
            float c = 0.5f * f3_dot(d, d) + hi * f3_dot(d, fn);

            nPoly = clip_poly_halfspace(polyA, nPoly, polyB, a, b, c, eps);
            if (nPoly == 0) break;
            for (int k = 0; k < nPoly; ++k) polyA[k] = polyB[k];
        }

        if (nPoly < 3) continue;

        // polygon area (shoelace)
        float area2 = 0.0f;
        for (int pi = 0; pi < nPoly; ++pi) {
            float2 p = polyA[pi];
            float2 q = polyA[(pi + 1) % nPoly];
            area2 += p.x * q.y - q.x * p.y;
        }
        float area = 0.5f * fabsf(area2);
        if (area < 1e-20f) continue;

        // 2-D centroid → 3-D
        float2 c2 = poly_centroid_2d(polyA, nPoly);
        if (!isfinite(c2.x) || !isfinite(c2.y)) continue;

        float3 cent_on_plane = f3_add(proj_i,
                                      f3_add(f3_mul(u, c2.x), f3_mul(vv, c2.y)));

        weighted_cent = f3_add(weighted_cent, f3_mul(cent_on_plane, area));
        total_area += (double)area;

        // CVT energy: ∫|x−s_i|² dA = ∫|p|² dA  +  h_i²·area
        if (energy_out) {
            double e = poly_energy_2d(polyA, nPoly) + (double)(hi * hi) * (double)area;
            total_energy += e;
        }
    }

    if (total_area > 1e-20) {
        float inv = 1.0f / (float)total_area;
        cent3d[i] = make_float3(weighted_cent.x * inv,
                                weighted_cent.y * inv,
                                weighted_cent.z * inv);
    } else {
        cent3d[i] = si;  // fallback: no valid clipping
    }

    if (energy_out) energy_out[i] = total_energy;
}
