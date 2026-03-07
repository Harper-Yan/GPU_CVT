#include <vector>
#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#ifndef HD
#define HD __host__ __device__ __forceinline__
#endif
// ----------------------------
// Evaluation structs/utilities
// ----------------------------
struct BVHNode {
    float3 bmin;
    float3 bmax;
    int left;      // -1 if leaf
    int right;     // -1 if leaf
    int triStart;  // into triIndices
    int triCount;  // 0 for internal
};

HD float3 f3min(const float3& a, const float3& b){
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
HD float3 f3max(const float3& a, const float3& b){
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
HD float3 f3add(const float3& a, const float3& b){
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
HD float3 f3sub(const float3& a, const float3& b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
HD float3 f3mul(const float3& a, float s){
    return make_float3(a.x*s, a.y*s, a.z*s);
}
HD float f3dot(const float3& a, const float3& b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// point->AABB squared distance
__host__ __device__ static inline float point_aabb_d2(const float3& p, const float3& bmin, const float3& bmax){
    float d = 0.f;
    float v;
    v = p.x; if (v < bmin.x){ float t=bmin.x-v; d += t*t; } else if (v > bmax.x){ float t=v-bmax.x; d += t*t; }
    v = p.y; if (v < bmin.y){ float t=bmin.y-v; d += t*t; } else if (v > bmax.y){ float t=v-bmax.y; d += t*t; }
    v = p.z; if (v < bmin.z){ float t=bmin.z-v; d += t*t; } else if (v > bmax.z){ float t=v-bmax.z; d += t*t; }
    return d;
}

// point->triangle squared distance (Ericson RTCD)
__host__ __device__ static inline float point_tri_d2(const float3& p, const float3& a, const float3& b, const float3& c){
    float3 ab = f3sub(b,a);
    float3 ac = f3sub(c,a);
    float3 ap = f3sub(p,a);

    float d1 = f3dot(ab, ap);
    float d2 = f3dot(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return f3dot(ap, ap);

    float3 bp = f3sub(p,b);
    float d3 = f3dot(ab, bp);
    float d4 = f3dot(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return f3dot(bp, bp);

    float vc = d1*d4 - d3*d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f){
        float v = d1 / (d1 - d3);
        float3 proj = f3add(a, f3mul(ab, v));
        float3 d = f3sub(p, proj);
        return f3dot(d,d);
    }

    float3 cp = f3sub(p,c);
    float d5 = f3dot(ab, cp);
    float d6 = f3dot(ac, cp);
    if (d6 >= 0.f && d5 <= d6) return f3dot(cp, cp);

    float vb = d5*d2 - d1*d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f){
        float w = d2 / (d2 - d6);
        float3 proj = f3add(a, f3mul(ac, w));
        float3 d = f3sub(p, proj);
        return f3dot(d,d);
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f){
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float3 bc = f3sub(c,b);
        float3 proj = f3add(b, f3mul(bc, w));
        float3 d = f3sub(p, proj);
        return f3dot(d,d);
    }

    float denom = 1.f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float3 proj = f3add(a, f3add(f3mul(ab, v), f3mul(ac, w)));
    float3 d = f3sub(p, proj);
    return f3dot(d,d);
}

// ----------------------------
// CPU BVH builder (flat nodes)
// ----------------------------
struct TriAABBInfo {
    float3 bmin, bmax, ctr;
    int tid;
};

static int build_bvh_recursive(
    std::vector<BVHNode>& nodes,
    std::vector<int>& triIndices,
    const std::vector<TriAABBInfo>& info,
    std::vector<int>& tids,
    int leafSize
){
    // compute node bounds
    float3 bmin = make_float3( 1e30f, 1e30f, 1e30f);
    float3 bmax = make_float3(-1e30f,-1e30f,-1e30f);
    for(int tid : tids){
        bmin = f3min(bmin, info[tid].bmin);
        bmax = f3max(bmax, info[tid].bmax);
    }

    BVHNode node{};
    node.bmin = bmin;
    node.bmax = bmax;
    node.left = node.right = -1;
    node.triStart = -1;
    node.triCount = 0;

    int myIndex = (int)nodes.size();
    nodes.push_back(node);

    if ((int)tids.size() <= leafSize){
        int start = (int)triIndices.size();
        triIndices.insert(triIndices.end(), tids.begin(), tids.end());
        nodes[myIndex].triStart = start;
        nodes[myIndex].triCount = (int)tids.size();
        return myIndex;
    }

    // split by longest axis of centroid bbox
    float3 cmin = make_float3( 1e30f, 1e30f, 1e30f);
    float3 cmax = make_float3(-1e30f,-1e30f,-1e30f);
    for(int tid : tids){
        cmin = f3min(cmin, info[tid].ctr);
        cmax = f3max(cmax, info[tid].ctr);
    }
    float3 ext = f3sub(cmax, cmin);
    int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);

    std::stable_sort(tids.begin(), tids.end(), [&](int a, int b){
        const float3 ca = info[a].ctr;
        const float3 cb = info[b].ctr;
        return (axis==0? ca.x : axis==1? ca.y : ca.z) < (axis==0? cb.x : axis==1? cb.y : cb.z);
    });

    int mid = (int)tids.size()/2;
    std::vector<int> L(tids.begin(), tids.begin()+mid);
    std::vector<int> R(tids.begin()+mid, tids.end());

    int left = build_bvh_recursive(nodes, triIndices, info, L, leafSize);
    int right = build_bvh_recursive(nodes, triIndices, info, R, leafSize);

    nodes[myIndex].left = left;
    nodes[myIndex].right = right;
    return myIndex;
}

static void build_bvh(
    const std::vector<float3>& V,
    const std::vector<int3i>& F,
    int leafSize,
    std::vector<BVHNode>& outNodes,
    std::vector<int>& outTriIndices
){
    const int nF = (int)F.size();
    std::vector<TriAABBInfo> info(nF);

    for(int i=0;i<nF;++i){
        int3i t = F[(size_t)i];
        float3 a = V[(size_t)t.x];
        float3 b = V[(size_t)t.y];
        float3 c = V[(size_t)t.z];
        float3 mn = f3min(a, f3min(b,c));
        float3 mx = f3max(a, f3max(b,c));
        float3 ctr = f3mul(f3add(a, f3add(b,c)), 1.0f/3.0f);
        info[i] = {mn,mx,ctr,i};
    }

    outNodes.clear();
    outTriIndices.clear();
    std::vector<int> tids(nF);
    for(int i=0;i<nF;++i) tids[i] = i;
    build_bvh_recursive(outNodes, outTriIndices, info, tids, leafSize);
}

// ----------------------------
// GPU BVH query: one thread per point
// ----------------------------
__global__ void bvh_closest_d2_kernel(
    const float3* __restrict__ points, int nP,
    const float3* __restrict__ V, const int3i* __restrict__ F, // mesh triangles
    const BVHNode* __restrict__ nodes, int nNodes,
    const int* __restrict__ triIndices,
    float* __restrict__ out_d2
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nP) return;

    float3 p = points[i];
    float best = 1e30f;

    // small stack for iterative traversal
    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // root at node 0

    while(sp){
        int ni = stack[--sp];
        BVHNode nd = nodes[ni];
        float d2box = point_aabb_d2(p, nd.bmin, nd.bmax);
        if (d2box >= best) continue;

        if (nd.triCount > 0){
            int start = nd.triStart;
            int count = nd.triCount;
            for(int k=0;k<count;++k){
                int tid = triIndices[start + k];
                int3i t = F[tid];
                float3 a = V[t.x];
                float3 b = V[t.y];
                float3 c = V[t.z];
                float d2 = point_tri_d2(p,a,b,c);
                if (d2 < best) best = d2;
            }
        } else {
            // internal
            if (nd.left >= 0) stack[sp++] = nd.left;
            if (nd.right >= 0) stack[sp++] = nd.right;
        }
    }

    out_d2[i] = best;
}

// Build deterministic probe points on GPU (V + centroids + edge midpoints)
__global__ void build_probe_points_kernel(
    const float3* __restrict__ V, int nV,
    const int3i* __restrict__ F, int nF,
    float3* __restrict__ P
){
    // Layout: [0..nV) vertices, then 4*nF triangle-derived points
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    // copy vertices
    for(int vi = tid; vi < nV; vi += (int)(gridDim.x*blockDim.x)){
        P[vi] = V[vi];
    }

    // triangle points
    for(int fi = tid; fi < nF; fi += (int)(gridDim.x*blockDim.x)){
        int3i t = F[fi];
        float3 a = V[t.x];
        float3 b = V[t.y];
        float3 c = V[t.z];
        float3 cent = f3mul(f3add(a, f3add(b,c)), 1.0f/3.0f);
        float3 m01 = f3mul(f3add(a,b), 0.5f);
        float3 m12 = f3mul(f3add(b,c), 0.5f);
        float3 m20 = f3mul(f3add(c,a), 0.5f);

        int base = nV + 4*fi;
        P[base + 0] = cent;
        P[base + 1] = m01;
        P[base + 2] = m12;
        P[base + 3] = m20;
    }
}

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>

// probe mesh (V1,F1) -> query mesh (V2,F2)
static float hausdorff_bvh_gpu(
    const float3* dV1, int nV1,
    const int3i*  dF1, int nF1,

    const float3*  dV2,
    const int3i*   dF2,
    const BVHNode* dNodes2, int nNodes2,
    const int*     dTriIdx2,

    cudaStream_t stream = 0
){
    const int nP1 = nV1 + 4 * nF1;

    // temp buffers
    float3* dP1 = nullptr;
    float*  dD2 = nullptr;
    cudaMalloc(&dP1, nP1 * sizeof(float3));
    cudaMalloc(&dD2, nP1 * sizeof(float));

    // build probe points
    {
        dim3 blk(256);
        dim3 grd((std::max(nV1, nF1) + blk.x - 1) / blk.x);
        build_probe_points_kernel<<<grd, blk, 0, stream>>>(
            dV1, nV1,
            dF1, nF1,
            dP1
        );
    }

    // BVH closest distance^2 for each probe
    {
        dim3 blk(256);
        dim3 grd((nP1 + blk.x - 1) / blk.x);
        bvh_closest_d2_kernel<<<grd, blk, 0, stream>>>(
            dP1, nP1,
            dV2,
            dF2,
            dNodes2, nNodes2,
            dTriIdx2,
            dD2
        );
    }

    // Reduce max(dD2) using CUB
    float* dMax = nullptr;
    cudaMalloc(&dMax, sizeof(float));

    void*  dTemp = nullptr;
    size_t tempBytes = 0;
    cub::DeviceReduce::Max(nullptr, tempBytes, dD2, dMax, nP1, stream);
    cudaMalloc(&dTemp, tempBytes);
    cub::DeviceReduce::Max(dTemp, tempBytes, dD2, dMax, nP1, stream);

    // Copy result back
    float maxd2 = 0.f;
    cudaMemcpyAsync(&maxd2, dMax, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // cleanup
    cudaFree(dTemp);
    cudaFree(dMax);
    cudaFree(dD2);
    cudaFree(dP1);

    return std::sqrt(maxd2);
}

// Candidate mesh -> Reference mesh (uses ONLY reference BVH)
static float hausdorff_cand_to_ref_gpu(
    const float3*  dVref,     int nVref,
    const int3i*   dFref,     int nFref,
    const BVHNode* dNodesRef, int nNodesRef,
    const int*     dTriIdxRef,

    const float3*  dVcand,    int nVcand,
    const int3i*   dFcand,    int nFcand,

    cudaStream_t stream = 0
){
    // probe = candidate (V1,F1), query/BVH = reference (V2,F2,BVH2)
    return hausdorff_bvh_gpu(
        dVcand, nVcand,
        dFcand, nFcand,

        dVref,
        dFref,
        dNodesRef, nNodesRef,
        dTriIdxRef,

        stream
    );
}

// ----------------------------
// Quality/angle metrics on CPU (cheap)
// ----------------------------
static void eval_quality_angles_cpu(
    const std::vector<float3>& V,
    const std::vector<int3i>& F,
    float& Qmin, float& Qavg,
    float& theta_min, float& theta_min_avg,
    float& theta_lt_30_pct, float& theta_gt_90_pct
){
    auto angle_deg = [](const float3& u, const float3& v)->float{
        float du = sqrtf(f3dot(u,u));
        float dv = sqrtf(f3dot(v,v));
        float denom = fmaxf(du*dv, 1e-30f);
        float cs = f3dot(u,v) / denom;
        cs = fminf(1.f, fmaxf(-1.f, cs));
        return acosf(cs) * (180.0f / 3.14159265358979323846f);
    };

    Qmin = 1e30f;
    double Qsum = 0.0;

    theta_min = 1e30f;
    double thmin_sum = 0.0;
    double cnt_lt30 = 0.0;
    double cnt_gt90 = 0.0;
    double cnt_ang = 0.0;

    for (const auto& t : F){
        float3 a = V[(size_t)t.x];
        float3 b = V[(size_t)t.y];
        float3 c = V[(size_t)t.z];

        float3 ab = f3sub(b,a);
        float3 ac = f3sub(c,a);
        float3 ba = f3sub(a,b);
        float3 bc = f3sub(c,b);
        float3 ca = f3sub(a,c);
        float3 cb = f3sub(b,c);

        float A = angle_deg(ab, ac);
        float B = angle_deg(ba, bc);
        float C = angle_deg(ca, cb);

        float tmin = fminf(A, fminf(B,C));
        theta_min = fminf(theta_min, tmin);
        thmin_sum += (double)tmin;

        cnt_lt30 += (A<30.f) + (B<30.f) + (C<30.f);
        cnt_gt90 += (A>90.f) + (B>90.f) + (C>90.f);
        cnt_ang += 3.0;

        float l0 = sqrtf(f3dot(f3sub(b,a), f3sub(b,a)));
        float l1 = sqrtf(f3dot(f3sub(c,b), f3sub(c,b)));
        float l2 = sqrtf(f3dot(f3sub(a,c), f3sub(a,c)));

        float3 cr = make_float3(
            ab.y*ac.z - ab.z*ac.y,
            ab.z*ac.x - ab.x*ac.z,
            ab.x*ac.y - ab.y*ac.x
        );
        float area = 0.5f * sqrtf(f3dot(cr,cr));

        float S = 0.5f * (l0 + l1 + l2);
        float E = fmaxf(l0, fmaxf(l1,l2));
        float denom = fmaxf(S*E, 1e-30f);
        float q = (6.0f / sqrtf(3.0f)) * (area / denom);

        Qmin = fminf(Qmin, q);
        Qsum += (double)q;
    }

    int nF = (int)F.size();
    Qavg = (nF > 0) ? (float)(Qsum / (double)nF) : 0.f;
    theta_min_avg = (nF > 0) ? (float)(thmin_sum / (double)nF) : 0.f;
    theta_lt_30_pct = (cnt_ang > 0.0) ? (float)(cnt_lt30 * 100.0 / cnt_ang) : 0.f;
    theta_gt_90_pct = (cnt_ang > 0.0) ? (float)(cnt_gt90 * 100.0 / cnt_ang) : 0.f;
}

// ----------------------------
// CSV append (eval_iters.csv)
// Step times (exclude vorpalite and evaluation). Freeze counts for low-freeze investigation.
// ----------------------------
static void append_eval_iters_csv(
    const std::string& path,
    const std::string& meshname,
    const std::string& mode,
    int iter,
    float Qmin, float Qavg,
    float theta_min, float theta_min_avg,
    float theta_lt_30_pct, float theta_gt_90_pct,
    float dH,
    float iter_remesh_ms,
    float total_remesh_ms,
    int freeze_cell_num,
    int n_vertices,
    // Step times (no vorpalite / eval)
    float knn_sites_ms,
    float knn_site_to_mesh_ms,
    float uv_from_mesh_ms,
    float centroids_ms,
    float knn_centroid_to_mesh_ms,
    float project_ms,
    float freeze_ms,
    // Freeze-condition counts: which condition blocks freezing most
    int count_same,
    int count_low_disp,
    int count_both,
    int count_newly_frozen
){
    const bool exists = std::filesystem::exists(path);

    std::ofstream out(path, std::ios::app);
    if (!out) return;

    float freeze_pct = (n_vertices > 0) ? (100.f * (float)freeze_cell_num / (float)n_vertices) : 0.f;
    int blocked_by_same = count_low_disp - count_both;  // had low disp but failed same-neighbors
    if (blocked_by_same < 0) blocked_by_same = 0;
    int blocked_by_low_disp = count_same - count_both; // had same neighbors but failed low disp
    if (blocked_by_low_disp < 0) blocked_by_low_disp = 0;

    // Write header once (in canonical order)
    if (!exists) {
        out <<
          "mesh,mode,iter,"
          "Qmin,Qavg,"
          "theta_min,theta_min_avg,"
          "theta_lt_30_pct,theta_gt_90_pct,"
          "dH,"
          "iter_remesh_ms,total_remesh_ms,"
          "freeze_cell_num,n_vertices,freeze_pct,"
          "knn_sites_ms,knn_site_to_mesh_ms,uv_from_mesh_ms,centroids_ms,knn_centroid_to_mesh_ms,project_ms,freeze_ms,"
          "count_same,count_low_disp,count_both,count_newly_frozen,blocked_by_same,blocked_by_low_disp\n";
    }

    // Write data row in the SAME order
    out << meshname << ","
        << mode << ","
        << iter << ","
        << Qmin << ","
        << Qavg << ","
        << theta_min << ","
        << theta_min_avg << ","
        << theta_lt_30_pct << ","
        << theta_gt_90_pct << ","
        << dH << ","
        << iter_remesh_ms << ","
        << total_remesh_ms << ","
        << freeze_cell_num << ","
        << n_vertices << ","
        << freeze_pct << ","
        << knn_sites_ms << ","
        << knn_site_to_mesh_ms << ","
        << uv_from_mesh_ms << ","
        << centroids_ms << ","
        << knn_centroid_to_mesh_ms << ","
        << project_ms << ","
        << freeze_ms << ","
        << count_same << ","
        << count_low_disp << ","
        << count_both << ","
        << count_newly_frozen << ","
        << blocked_by_same << ","
        << blocked_by_low_disp
        << "\n";
}

