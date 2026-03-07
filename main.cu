#include "site_normals.cuh"
#include "site_knn.cuh"
#include "2dClip.cuh"
#include "near_project.cuh"
#include "freezetest.cuh"
#include "buildmesh.cuh"
#include "evaluate.cuh"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ─── Mode 1: 5-tier (nv only). Same params as mode 2: uniform disp_thr, streak 10+5*tier ─
namespace tier1 {
    static const double TIER_THRESHOLDS[4] = {0.15, 0.35, 0.55, 0.80};
    static const float DISP_THR[5]  = {1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f};
    static const int   STREAK[5]    = {5, 10, 15, 20, 25};

    static void compute_tier_id_v1(int nV, const std::vector<double>& nv, std::vector<unsigned char>& tier_id) {
        tier_id.resize((size_t)nV);
        for (int i = 0; i < nV; ++i) {
            int t = 0;
            while (t < 4 && nv[i] >= TIER_THRESHOLDS[t]) t++;
            tier_id[i] = (unsigned char)t;
        }
    }
}

// ─── Mode 2: 6-tier (testfreeze2.py, sharp split by L) ────────────────────────
namespace tier2 {
    static const double TIER_THRESHOLDS[4] = {0.15, 0.35, 0.55, 0.80};
    static const double L_SHARP_THR = 0.80;
    static const float DISP_THR[6]  = {1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f};
    static const int   STREAK[6]    = {5, 10, 15, 20, 25, 10};  // tier 4 (sharp_B, less L) stricter; tier 5 (sharp_A, more L) looser (streak 10)

    static void normal_variation_score(int nV, int k, const float* hN, const idx_t* hKNN, std::vector<double>& nv_out) {
        nv_out.resize((size_t)nV);
        for (int i = 0; i < nV; ++i) {
            double sum_cos = 0.0;
            int count = 0;
            for (int j = 0; j < k; ++j) {
                int jj = (int)hKNN[(size_t)i * k + j];
                if (jj < 0 || jj >= nV) continue;
                double cos_ij = (double)hN[i*3+0] * (double)hN[jj*3+0]
                              + (double)hN[i*3+1] * (double)hN[jj*3+1]
                              + (double)hN[i*3+2] * (double)hN[jj*3+2];
                if (cos_ij > 1.0) cos_ij = 1.0;
                if (cos_ij < -1.0) cos_ij = -1.0;
                sum_cos += cos_ij;
                count++;
            }
            nv_out[i] = (count > 0) ? (1.0 - sum_cos / (double)count) : 0.0;
        }
    }

    // 3x3 symmetric eigensolver: returns lambda1 >= lambda2 >= lambda3.
    static void sym3_eigenvalues(const double C[9], double lam[3]) {
        double v[3] = {1.0, 0.0, 0.0};
        for (int it = 0; it < 20; ++it) {
            double w[3];
            w[0] = C[0]*v[0] + C[1]*v[1] + C[2]*v[2];
            w[1] = C[3]*v[0] + C[4]*v[1] + C[5]*v[2];
            w[2] = C[6]*v[0] + C[7]*v[1] + C[8]*v[2];
            double n = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
            if (n < 1e-30) break;
            v[0] = w[0]/n; v[1] = w[1]/n; v[2] = w[2]/n;
        }
        lam[0] = v[0]*(C[0]*v[0]+C[1]*v[1]+C[2]*v[2])
               + v[1]*(C[3]*v[0]+C[4]*v[1]+C[5]*v[2])
               + v[2]*(C[6]*v[0]+C[7]*v[1]+C[8]*v[2]);
        double C2[9];
        for (int i = 0; i < 9; ++i) C2[i] = C[i];
        C2[0] -= lam[0]*v[0]*v[0]; C2[1] -= lam[0]*v[0]*v[1]; C2[2] -= lam[0]*v[0]*v[2];
        C2[3] -= lam[0]*v[1]*v[0]; C2[4] -= lam[0]*v[1]*v[1]; C2[5] -= lam[0]*v[1]*v[2];
        C2[6] -= lam[0]*v[2]*v[0]; C2[7] -= lam[0]*v[2]*v[1]; C2[8] -= lam[0]*v[2]*v[2];
        double v2[3] = {0.0, 1.0, 0.0};
        for (int it = 0; it < 20; ++it) {
            double w[3];
            w[0] = C2[0]*v2[0] + C2[1]*v2[1] + C2[2]*v2[2];
            w[1] = C2[3]*v2[0] + C2[4]*v2[1] + C2[5]*v2[2];
            w[2] = C2[6]*v2[0] + C2[7]*v2[1] + C2[8]*v2[2];
            double n = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
            if (n < 1e-30) break;
            v2[0] = w[0]/n; v2[1] = w[1]/n; v2[2] = w[2]/n;
        }
        lam[1] = v2[0]*(C2[0]*v2[0]+C2[1]*v2[1]+C2[2]*v2[2])
               + v2[1]*(C2[3]*v2[0]+C2[4]*v2[1]+C2[5]*v2[2])
               + v2[2]*(C2[6]*v2[0]+C2[7]*v2[1]+C2[8]*v2[2]);
        lam[2] = C[0] + C[4] + C[8] - lam[0] - lam[1];
    }

    static void normal_covariance_L(int nV, int k, const float* hN, const idx_t* hKNN, std::vector<double>& L_out) {
        L_out.resize((size_t)nV, 0.0);
        for (int i = 0; i < nV; ++i) {
            double mean[3] = {0,0,0};
            int count = 0;
            for (int j = 0; j < k; ++j) {
                int jj = (int)hKNN[(size_t)i * k + j];
                if (jj < 0 || jj >= nV) continue;
                mean[0] += (double)hN[jj*3+0];
                mean[1] += (double)hN[jj*3+1];
                mean[2] += (double)hN[jj*3+2];
                count++;
            }
            if (count < 2) continue;
            mean[0] /= count; mean[1] /= count; mean[2] /= count;
            double C[9] = {0,0,0,0,0,0,0,0,0};
            for (int j = 0; j < k; ++j) {
                int jj = (int)hKNN[(size_t)i * k + j];
                if (jj < 0 || jj >= nV) continue;
                double n0 = (double)hN[jj*3+0] - mean[0];
                double n1 = (double)hN[jj*3+1] - mean[1];
                double n2 = (double)hN[jj*3+2] - mean[2];
                C[0] += n0*n0; C[1] += n0*n1; C[2] += n0*n2;
                C[3] += n1*n0; C[4] += n1*n1; C[5] += n1*n2;
                C[6] += n2*n0; C[7] += n2*n1; C[8] += n2*n2;
            }
            double lam[3];
            sym3_eigenvalues(C, lam);
            if (lam[0] > lam[1]) std::swap(lam[0], lam[1]);
            if (lam[1] > lam[2]) std::swap(lam[1], lam[2]);
            if (lam[0] > lam[1]) std::swap(lam[0], lam[1]);
            if (lam[2] > 1e-30)
                L_out[i] = (lam[2] - lam[1]) / lam[2];
        }
    }

    static void compute_tier_id_v2(int nV, int k, const float* hN, const idx_t* hKNN,
                                   std::vector<double>& nv, std::vector<double>& L_site,
                                   std::vector<unsigned char>& tier_id) {
        nv.resize((size_t)nV);
        L_site.resize((size_t)nV, 0.0);
        tier_id.resize((size_t)nV);
        normal_variation_score(nV, k, hN, hKNN, nv);
        normal_covariance_L(nV, k, hN, hKNN, L_site);
        for (int i = 0; i < nV; ++i) {
            int t = 0;
            while (t < 4 && nv[i] >= TIER_THRESHOLDS[t]) t++;
            if (t == 4) {
                if (L_site[i] > L_SHARP_THR) t = 5;
            }
            tier_id[i] = (unsigned char)t;
        }
    }
}

struct IsUnfrozen {
    __host__ __device__ bool operator()(const unsigned char f) const {
        return f == 0;
    }
};

static inline int parse_int_before_slash(const char* s, char** endp) {
    return (int)std::strtol(s, endp, 10);
}

static float load_obj_triangles(const std::string& path,
                                std::vector<float3>& V,
                                std::vector<int3i>& F)
{
    std::ifstream in(path);
    if (!in) std::exit(1);

    float3 bmin = make_float3( 1e30f,  1e30f,  1e30f);
    float3 bmax = make_float3(-1e30f, -1e30f, -1e30f);

    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;

        if (line[0] == 'v' && line[1] == ' ') {
            const char* s = line.c_str() + 2;
            char* e = nullptr;
            float x = std::strtof(s, &e); s = e;
            float y = std::strtof(s, &e); s = e;
            float z = std::strtof(s, &e);
            V.push_back(make_float3(x, y, z));

            bmin.x = fminf(bmin.x, x);
            bmin.y = fminf(bmin.y, y);
            bmin.z = fminf(bmin.z, z);
            bmax.x = fmaxf(bmax.x, x);
            bmax.y = fmaxf(bmax.y, y);
            bmax.z = fmaxf(bmax.z, z);

        } else if (line[0] == 'f' && line[1] == ' ') {
            const char* s = line.c_str() + 2;
            std::vector<int> idx;
            idx.reserve(8);

            while (*s) {
                while (*s == ' ' || *s == '\t') ++s;
                if (*s == 0) break;

                char* e = nullptr;
                int vi = parse_int_before_slash(s, &e);
                if (e == s) break;

                int nV = (int)V.size();
                if (vi < 0) vi = nV + vi + 1;
                vi -= 1;
                idx.push_back(vi);

                s = e;
                while (*s && *s != ' ' && *s != '\t') ++s;
            }

            if (idx.size() >= 3) {
                int a = idx[0];
                for (size_t k = 1; k + 1 < idx.size(); ++k) {
                    int b = idx[k];
                    int c = idx[k + 1];
                    F.push_back({a, b, c});
                }
            }
        }
    }
    float3 bbox = make_float3(bmax.x - bmin.x,
                          bmax.y - bmin.y,
                          bmax.z - bmin.z);
    float R = fmaxf(bbox.x, fmaxf(bbox.y, bbox.z));
    return R;
}

void compact_mesh(std::vector<float3>& V, std::vector<int3i>& F) {
    const int nV = (int)V.size();
    std::vector<unsigned char> used((size_t)nV, 0);

    for (const auto& t : F) {
        if ((unsigned)t.x < (unsigned)nV) used[(size_t)t.x] = 1;
        if ((unsigned)t.y < (unsigned)nV) used[(size_t)t.y] = 1;
        if ((unsigned)t.z < (unsigned)nV) used[(size_t)t.z] = 1;
    }

    std::vector<int> remap((size_t)nV, -1);
    int newN = 0;
    for (int i = 0; i < nV; ++i) if (used[(size_t)i]) remap[(size_t)i] = newN++;

    std::vector<float3> V2((size_t)newN);
    for (int i = 0; i < nV; ++i) {
        int j = remap[(size_t)i];
        if (j >= 0) V2[(size_t)j] = V[(size_t)i];
    }

    for (auto& t : F) {
        t.x = remap[(size_t)t.x];
        t.y = remap[(size_t)t.y];
        t.z = remap[(size_t)t.z];
    }

    V.swap(V2);
}

static void build_vf_adjacency(
    int nV, int nF,
    const std::vector<int3i>& F,
    std::vector<int>& vf_off,
    std::vector<int>& vf_faces
){
    vf_off.assign((size_t)nV + 1, 0);

    for (int fi = 0; fi < nF; ++fi) {
        const int3i t = F[(size_t)fi];
        vf_off[(size_t)t.x + 1]++;
        vf_off[(size_t)t.y + 1]++;
        vf_off[(size_t)t.z + 1]++;
    }

    for (int v = 0; v < nV; ++v) vf_off[(size_t)v + 1] += vf_off[(size_t)v];

    vf_faces.assign((size_t)vf_off[(size_t)nV], -1);

    std::vector<int> cur = vf_off;
    for (int fi = 0; fi < nF; ++fi) {
        const int3i t = F[(size_t)fi];
        vf_faces[(size_t)cur[(size_t)t.x]++] = fi;
        vf_faces[(size_t)cur[(size_t)t.y]++] = fi;
        vf_faces[(size_t)cur[(size_t)t.z]++] = fi;
    }
}

static inline void unpack3_21(uint64_t k, int& a, int& b, int& c)
{
    a = int((k >> 42) & ((1ull << 21) - 1));
    b = int((k >> 21) & ((1ull << 21) - 1));
    c = int((k >>  0) & ((1ull << 21) - 1));
}

static void write_frozen_log_txt(const char* path, const unsigned char* frozen, int nV)
{
    std::ofstream out(path);
    if (!out) return;
    for (int i = 0; i < nV; ++i) out << (int)frozen[(size_t)i] << "\n";
}

template<int K, typename IndexT>
__global__ void restore_prev_knn_for_frozen_kernel(
    const unsigned char* __restrict__ frozen,
    const IndexT* __restrict__ prev_knn,
    IndexT* __restrict__ knn,
    int n
){
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || !frozen[i]) return;
    for (int a = 0; a < K; ++a)
        knn[i * K + a] = prev_knn[i * K + a];
}

// Run Geogram vorpalite to reconstruct mesh from sites (same options as testfreeze/testfreeze2).
// Returns 0 on success, non-zero on failure.
static int run_vorpalite(const std::string& vorpalite_exe,
                         const std::string& sites_xyz,
                         const std::string& out_obj,
                         const char* radius = "5%",
                         int nb_neighbors = 30)
{
    std::string cmd;
    auto quote = [](const std::string& s) {
        std::string r = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') r += '\\';
            r += c;
        }
        r += '"';
        return r;
    };
    cmd += quote(vorpalite_exe) + " " + quote(sites_xyz) + " " + quote(out_obj);
    cmd += " co3ne=true pre=false remesh=false post=true";
    cmd += " co3ne:repair=true";
    cmd += " co3ne:radius="; cmd += radius;
    cmd += " co3ne:nb_neighbors="; cmd += std::to_string(nb_neighbors);
    cmd += " log:quiet=true log:pretty=false";
    int ret = std::system(cmd.c_str());
    return ret;
}

static void append_run_csv(
    const std::string& csv_path,
    const std::string& mesh_name,
    const char* mode_str,
    int final_nv,
    int final_nf,
    float converge_rate,
    int used_iters,
    float total_remesh_ms
){
    const bool exists = std::filesystem::exists(csv_path);
    std::ofstream out(csv_path, std::ios::app);
    if (!out) return;

    if (!exists) {
        out << "meshname,mode,final_nv,final_nf,converge_rate,used_iterations,total_remeshing_time\n";
    }

    out << mesh_name << ","
        << mode_str << ","
        << final_nv << ","
        << final_nf << ","
        << converge_rate << ","
        << used_iters << ","
        << total_remesh_ms
        << "\n";
}


int main(int argc, char** argv)
{

    if (argc < 2) return 1;

    std::string input_mesh = argv[1];

    std::string mesh_name =
        std::filesystem::path(input_mesh).stem().string();

    int mode = 1;
    if (argc >= 3) mode = std::atoi(argv[2]);

    // modes:
    // 0: baseline (gpu_cvt)         - dFrozen computed then cleared each iter
    // 1: freezing_cvt               - 5-tier freeze (testfreeze.py params)
    // 2: freezing_cvt_tiered       - 6-tier freeze (testfreeze2.py, sharp split by L)
    const char* mode_name = (mode == 0) ? "baseline" : (mode == 2 ? "freezing_cvt_tiered" : "freezing_cvt");
    printf("mode %d (%s)\n", mode, mode_name);

    // ---------------- Tunable constants ----------------
    constexpr int   THREADS        = 1024;
    constexpr int   MAX_POLY        = 256;
    constexpr int   KPROJ           = 5;
    constexpr int   DUMP_STRIDE     = 10;     // dump an .obj and eval row every N iters
    constexpr float DEGENERATE_EPS  = 1e-6f; // degenerate-triangle test epsilon

    constexpr int   K_NEIGH         = 32;
    constexpr int   K_SITE          = K_NEIGH + 1;

    int total_iter = 100;
    if (argc >= 4) total_iter = std::atoi(argv[3]);

    const std::string vorpalite_path = "/home/hyan/local/geogram/bin/vorpalite";
    const bool use_geogram = true;

    int freeze_monitor_iters = 5;
    // freeze_disp = squared displacement threshold, set below from mesh bbox (0.01 * max bbox edge)^2
    float freeze_disp = 0.0f;
    // ----------------------------------------------------

    int used_iters = 0;
    float final_converge_rate = 0.0f;
    int final_nf = 0;
    float last_frozenRatio = 0.0f;

    std::vector<unsigned char> hFrozenIter;

    std::vector<float3> hV;
    std::vector<int3i> hF;
    float R = load_obj_triangles(argv[1], hV, hF);
    compact_mesh(hV, hF);

    int nV = (int)hV.size();
    int nF = (int)hF.size();
    printf("V %d\n", nV);
    printf("F %d\n", nF);
    if (nV == 0 || nF == 0) return 1;

    // Same freeze_disp (squared) for all 3 modes: (0.01 * R)^2, R = max bbox edge from load_obj
    freeze_disp = 0.01f * R * 0.01f * R;
    printf("freeze_disp %.6g (squared, linear=0.01*R R=%.6g)\n", (double)freeze_disp, (double)R);

    // ---- Reference mesh (input) BVH built once ----
    std::vector<BVHNode> refNodes;
    std::vector<int> refTriIdx;
    build_bvh(hV, hF, /*leafSize=*/8, refNodes, refTriIdx);

    printf("refNodes %zu\n", refNodes.size());
    printf("refTriIdx %zu\n", refTriIdx.size());

    float3* dVref;
    int3i*  dFref;
    BVHNode* dNodesRef;
    int* dTriIdxRef;

    cudaMalloc(&dVref,    hV.size()    * sizeof(float3));
    cudaMalloc(&dFref,    hF.size()    * sizeof(int3i));
    cudaMalloc(&dNodesRef, refNodes.size() * sizeof(BVHNode));
    cudaMalloc(&dTriIdxRef, refTriIdx.size() * sizeof(int));

    cudaMemcpy(dVref,    hV.data(),    hV.size()    * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dFref,    hF.data(),    hF.size()    * sizeof(int3i),  cudaMemcpyHostToDevice);
    cudaMemcpy(dNodesRef, refNodes.data(), refNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    cudaMemcpy(dTriIdxRef, refTriIdx.data(), refTriIdx.size() * sizeof(int), cudaMemcpyHostToDevice);

    float3*  dVcand = nullptr;
    int3i*   dFcand = nullptr;
    BVHNode* dNodesCand = nullptr;
    int*     dTriIdxCand = nullptr;

    size_t dVcand_cap = 0;
    size_t dFcand_cap = 0;
    size_t dNodesCand_cap = 0;
    size_t dTriIdxCand_cap = 0;

    float3* dV = nullptr;
    int3i* dF = nullptr;
    float3* dN = nullptr;
    cudaMalloc(&dV, (size_t)nV * sizeof(float3));
    cudaMalloc(&dF, (size_t)nF * sizeof(int3i));
    cudaMalloc(&dN, (size_t)nV * sizeof(float3));
    cudaMemcpy(dV, hV.data(), (size_t)nV * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dF, hF.data(), (size_t)nF * sizeof(int3i), cudaMemcpyHostToDevice);

    cudaFree(0);
    zero_normals<<<1,1>>>(dN, 1);
    cudaDeviceSynchronize();

    zero_normals<<<(nV + THREADS - 1) / THREADS, THREADS>>>(dN, nV);
    accumulate_face_normals_atomic<<<(nF + THREADS - 1) / THREADS, THREADS>>>(dV, dF, dN, nF);
    normalize_normals<<<(nV + THREADS - 1) / THREADS, THREADS>>>(dN, nV);

    float3* du = nullptr;
    float3* dv = nullptr;
    cudaMalloc(&du, (size_t)nV * sizeof(float3));
    cudaMalloc(&dv, (size_t)nV * sizeof(float3));

    float3* dS = nullptr;
    float3* dSnew = nullptr;
    cudaMalloc(&dS, (size_t)nV * sizeof(float3));
    cudaMalloc(&dSnew, (size_t)nV * sizeof(float3));
    cudaMemcpy(dS, dV, (size_t)nV * sizeof(float3), cudaMemcpyDeviceToDevice);


    idx_t* dKNN_sites = nullptr;
    idx_t* dKNN_sites_raw = nullptr;
    idx_t* dPrevKNN_sites = nullptr;
    float* dDist_sites = nullptr;
    float* dDist_sites_raw = nullptr;
    cudaMalloc(&dKNN_sites_raw,  (size_t)nV * K_SITE  * sizeof(idx_t));
    cudaMalloc(&dDist_sites_raw, (size_t)nV * K_SITE  * sizeof(float));
    cudaMalloc(&dKNN_sites,      (size_t)nV * K_NEIGH * sizeof(idx_t));
    cudaMalloc(&dPrevKNN_sites,  (size_t)nV * K_NEIGH * sizeof(idx_t));
    cudaMalloc(&dDist_sites,     (size_t)nV * K_NEIGH * sizeof(float));

    cudaMemset(dPrevKNN_sites, 0, (size_t)nV * K_NEIGH * sizeof(idx_t));

    float2* dPoly2d = nullptr;
    int* dPolyN = nullptr;
    cudaMalloc(&dPoly2d, (size_t)nV * MAX_POLY * sizeof(float2));
    cudaMalloc(&dPolyN,  (size_t)nV * sizeof(int));

    int* dKeyCount = nullptr;
    cudaMalloc(&dKeyCount, sizeof(int));

    uint64_t* dKeys = nullptr;
    size_t maxCand = (size_t)nV * MAX_POLY;
    cudaMalloc(&dKeys, maxCand * sizeof(uint64_t));

    std::vector<float3> hSnew((size_t)nV);
    std::vector<uint64_t> hKeys;
    hKeys.reserve(maxCand);

    std::vector<Tri> hFnew;

    std::vector<float3> hV_geo;
    std::vector<int3i> hF_geo;
    int last_geogram_nf = 0;

    float3* dCent = nullptr;
    cudaMalloc(&dCent, (size_t)nV * sizeof(float3));

    std::vector<int> h_vf_off, h_vf_faces;
    build_vf_adjacency(nV, nF, hF, h_vf_off, h_vf_faces);

    int* d_vf_off = nullptr;
    int* d_vf_faces = nullptr;
    cudaMalloc(&d_vf_off, (size_t)(nV + 1) * sizeof(int));
    cudaMalloc(&d_vf_faces, (size_t)h_vf_faces.size() * sizeof(int));
    cudaMemcpy(d_vf_off, h_vf_off.data(), (size_t)(nV + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vf_faces, h_vf_faces.data(), (size_t)h_vf_faces.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* dKnnV = nullptr;
    cudaMalloc(&dKnnV, (size_t)nV * KPROJ * sizeof(int));

    unsigned char* dFrozen = nullptr;
    cudaMalloc(&dFrozen, (size_t)nV);
    cudaMemset(dFrozen, 0, (size_t)nV);
    // temp buffer: per-iteration first-pass freeze candidates (before neighbor-consensus pass)
    unsigned char* dFreezeCand = nullptr;
    cudaMalloc(&dFreezeCand, (size_t)nV);
    cudaMemset(dFreezeCand, 0, (size_t)nV);

    unsigned char* dFreezeStreak; 
    cudaMalloc(&dFreezeStreak, (size_t)nV);
    cudaMemset(dFreezeStreak, 0, (size_t)nV);

    unsigned char* d_tier_id = nullptr;
    float* d_thresh2_tier = nullptr;
    int* d_streak_tier = nullptr;
    if (mode == 1) {
        cudaMalloc(&d_tier_id, (size_t)nV);
        cudaMalloc(&d_thresh2_tier, 6 * sizeof(float));
        cudaMalloc(&d_streak_tier, 6 * sizeof(int));
        float h_thresh2[6];
        int h_streak[6];
        for (int t = 0; t < 6; ++t) {
            h_thresh2[t] = freeze_disp;  // same freeze_disp (squared) for all tiers
            h_streak[t] = (t < 5) ? tier1::STREAK[t] : tier1::STREAK[4];
        }
        cudaMemcpy(d_thresh2_tier, h_thresh2, 6 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_streak_tier, h_streak, 6 * sizeof(int), cudaMemcpyHostToDevice);
    } else if (mode == 2) {
        cudaMalloc(&d_tier_id, (size_t)nV);
        cudaMalloc(&d_thresh2_tier, 6 * sizeof(float));
        cudaMalloc(&d_streak_tier, 6 * sizeof(int));
        float h_thresh2[6];
        for (int t = 0; t < 6; ++t) {
            h_thresh2[t] = freeze_disp;  // same freeze_disp (squared) for all tiers
        }
        cudaMemcpy(d_thresh2_tier, h_thresh2, 6 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_streak_tier, tier2::STREAK, 6 * sizeof(int), cudaMemcpyHostToDevice);
    }

    std::vector<unsigned char> hFrozen(nV);

    float* dKnnV_dist = nullptr;
    int* dPrevKnnV = nullptr;
    if (mode != 0) {
        cudaMalloc(&dKnnV_dist, (size_t)nV * KPROJ * sizeof(float));
        cudaMalloc(&dPrevKnnV, (size_t)nV * KPROJ * sizeof(int));
    }

    int* dFrozenSum = nullptr;
    cudaMalloc(&dFrozenSum, sizeof(int));

    int* dCounts = nullptr;
    cudaMalloc(&dCounts, 4 * sizeof(int));


    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaMalloc(&dA, (size_t)nV * K_SITE * sizeof(double));
    cudaMalloc(&dB, (size_t)nV * K_SITE * sizeof(double));
    cudaMalloc(&dC, (size_t)nV * K_SITE * sizeof(double));

    int2* dPolyLab2;
    cudaMalloc(&dPolyLab2, (size_t)nV * MAX_POLY * sizeof(int2));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    dim3 blk(256);
    dim3 grd((nV + blk.x - 1) / blk.x);

    float total_remesh_ms = 0.0f;

    const char* root_dir = (mode == 0) ? "gpucvt" : (mode == 2 ? "freeze_tiered" : "freeze");
    const std::string output_base = "experiments/output";
    std::string out_dir = output_base + "/" + root_dir + "/" + mesh_name;
    std::filesystem::create_directories(out_dir);

    std::string eval_csv = out_dir + "/eval_iters.csv";
    {
        FILE* f = fopen(eval_csv.c_str(), "w");
        if (f) {
            fprintf(f,
            "mesh,mode,iter,Qmin,Qavg,theta_min,theta_min_avg,"
            "theta_lt_30_pct,theta_gt_90_pct,dH,iter_remesh_ms,total_remesh_ms,"
            "freeze_cell_num,n_vertices,freeze_pct,"
            "knn_sites_ms,knn_site_to_mesh_ms,uv_from_mesh_ms,centroids_ms,knn_centroid_to_mesh_ms,project_ms,freeze_ms,"
            "count_same,count_low_disp,count_both,count_newly_frozen,blocked_by_same,blocked_by_low_disp\n");
            fclose(f);
        }
    }

    if (mode == 1) {
        run_knn_bitonic_hubs(nV, K_SITE, dFrozen, dS, dKNN_sites_raw, dDist_sites_raw);
        knn_drop_self_kernel<K_SITE, K_NEIGH, idx_t><<<grd, blk>>>(dKNN_sites_raw, dKNN_sites, nV);
        cudaDeviceSynchronize();
        std::vector<idx_t> hKNN((size_t)nV * K_NEIGH);
        std::vector<float> hN((size_t)nV * 3);
        cudaMemcpy(hKNN.data(), dKNN_sites, (size_t)nV * K_NEIGH * sizeof(idx_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hN.data(), dN, (size_t)nV * sizeof(float3), cudaMemcpyDeviceToHost);
        std::vector<double> h_nv;
        tier2::normal_variation_score(nV, K_NEIGH, hN.data(), hKNN.data(), h_nv);
        std::vector<unsigned char> h_tier_id;
        tier1::compute_tier_id_v1(nV, h_nv, h_tier_id);
        cudaMemcpy(d_tier_id, h_tier_id.data(), (size_t)nV, cudaMemcpyHostToDevice);
        printf("mode 1: tier assignment done (5-tier, testfreeze.py)\n");
    } else if (mode == 2) {
        run_knn_bitonic_hubs(nV, K_SITE, dFrozen, dS, dKNN_sites_raw, dDist_sites_raw);
        knn_drop_self_kernel<K_SITE, K_NEIGH, idx_t><<<grd, blk>>>(dKNN_sites_raw, dKNN_sites, nV);
        cudaDeviceSynchronize();
        std::vector<idx_t> hKNN((size_t)nV * K_NEIGH);
        std::vector<float> hN((size_t)nV * 3);
        cudaMemcpy(hKNN.data(), dKNN_sites, (size_t)nV * K_NEIGH * sizeof(idx_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hN.data(), dN, (size_t)nV * sizeof(float3), cudaMemcpyDeviceToHost);
        std::vector<double> h_nv, h_L;
        std::vector<unsigned char> h_tier_id;
        tier2::compute_tier_id_v2(nV, K_NEIGH, hN.data(), hKNN.data(), h_nv, h_L, h_tier_id);
        cudaMemcpy(d_tier_id, h_tier_id.data(), (size_t)nV, cudaMemcpyHostToDevice);
        printf("mode 2: tier assignment done (6-tier, testfreeze2.py sharp split by L)\n");
    }

    for (int iter = 0; iter < total_iter; ++iter)
    {
        printf("\n=== iter %d ===\n", iter);

        float knn_sites_ms = run_knn_bitonic_hubs(nV, K_SITE, dFrozen, dS, dKNN_sites_raw, dDist_sites_raw);
        printf("knn_sites_ms %.3f\n", knn_sites_ms);

        knn_drop_self_kernel<K_SITE, K_NEIGH, idx_t><<<grd, blk>>>(dKNN_sites_raw, dKNN_sites, nV);
        if (mode != 0)
            restore_prev_knn_for_frozen_kernel<K_NEIGH, idx_t><<<grd, blk>>>(dFrozen, dPrevKNN_sites, dKNN_sites, nV);

        float knn_site_to_mesh_ms = 0.0f;
        float knn_centroid_to_mesh_ms = 0.0f;

        if (mode != 0) {
            cudaMemcpy(dPrevKnnV, dKnnV, (size_t)nV * KPROJ * sizeof(int), cudaMemcpyDeviceToDevice);
            knn_site_to_mesh_ms = run_knn_bitonic_query_to_mesh(nV, dV, dS, nV, KPROJ, dFrozen,
                (idx_t*)dKnnV, dKnnV_dist, "site_to_mesh");
            restore_prev_knn_vertices_kernel<KPROJ><<<grd, blk>>>(dFrozen, dPrevKnnV, dKnnV, nV);
            cudaDeviceSynchronize();
            printf("knn_site_to_mesh_ms %.3f\n", knn_site_to_mesh_ms);
        } else {
            dim3 grdK((nV + blk.x - 1) / blk.x);
            cudaEventRecord(e0);
            knn_vertices_bruteforce_k<KPROJ, false><<<grdK, blk>>>(
                dV, nV, dS, nV, dFrozen, dKnnV
            );
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            knn_site_to_mesh_ms = elapsed_ms(e0, e1);
            printf("knn_site_to_mesh_ms %.3f\n", knn_site_to_mesh_ms);
        }

        
        cudaEventRecord(e0);
        uv_from_nearest_vertex_normal<KPROJ><<<grd, blk>>>(dN, dKnnV, du, dv, nV);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float uv_from_mesh_ms = elapsed_ms(e0, e1);
        printf("uv_from_mesh_ms %.3f\n", uv_from_mesh_ms);

        cudaEventRecord(e0);
        centroids_tangent_voronoi<<<grd, blk>>>(dS, du, dv, dKNN_sites, nV, K_NEIGH, R, dFrozen, dCent);

        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float centroids_ms = elapsed_ms(e0, e1);
        printf("centroids_ms %.3f\n", centroids_ms);

        if (mode != 0) {
            cudaMemcpy(dPrevKnnV, dKnnV, (size_t)nV * KPROJ * sizeof(int), cudaMemcpyDeviceToDevice);
            knn_centroid_to_mesh_ms = run_knn_bitonic_query_to_mesh(nV, dV, dCent, nV, KPROJ, dFrozen,
                (idx_t*)dKnnV, dKnnV_dist, "centroid_to_mesh");
            restore_prev_knn_vertices_kernel<KPROJ><<<grd, blk>>>(dFrozen, dPrevKnnV, dKnnV, nV);
            cudaDeviceSynchronize();
            printf("knn_centroid_to_mesh_ms %.3f\n", knn_centroid_to_mesh_ms);
        } else {
            dim3 grdK((nV + blk.x - 1) / blk.x);
            cudaEventRecord(e0);
            knn_vertices_bruteforce_k<KPROJ, true><<<grdK, blk>>>(
                dV, nV, dCent, nV, dFrozen, dKnnV
            );
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            knn_centroid_to_mesh_ms = elapsed_ms(e0, e1);
            printf("knn_centroid_to_mesh_ms %.3f\n", knn_centroid_to_mesh_ms);
        }

        cudaEventRecord(e0);
        project_centroids_to_mesh<KPROJ><<<grd, blk>>>(
            dCent, dV, (const int3*)dF,
            d_vf_off, d_vf_faces,
            dKnnV,
            nV, nF, nV, dFrozen, dS,
            dSnew
        );

        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float project_ms = elapsed_ms(e0, e1);
        printf("project_ms %.3f\n", project_ms);

        {
            cudaMemcpy(hSnew.data(), dSnew, (size_t)nV * sizeof(float3), cudaMemcpyDeviceToHost);
            const float eps = 1e-9f;
            std::vector<float3> tmp(hSnew);
            std::sort(tmp.begin(), tmp.end(), [](const float3& a, const float3& b) {
                if (a.x != b.x) return a.x < b.x;
                if (a.y != b.y) return a.y < b.y;
                return a.z < b.z;
            });
            int unique_pos = 1;
            for (int i = 1; i < nV; ++i) {
                float dx = tmp[i].x - tmp[i-1].x, dy = tmp[i].y - tmp[i-1].y, dz = tmp[i].z - tmp[i-1].z;
                if (dx*dx + dy*dy + dz*dz > eps*eps) unique_pos++;
            }
            int duplicate_sites = nV - unique_pos;
            printf("after projection: unique_pos %d  duplicate_sites %d (of %d)\n", unique_pos, duplicate_sites, nV);
        }

        int has_prev_knn = (iter > 0) ? 1 : 0;
        cudaMemset(dCounts, 0, 4 * sizeof(int));
        cudaMemset(dFreezeCand, 0, (size_t)nV); // pass-1 candidates (this iter only)

        cudaEventRecord(e0);
        if (mode == 1 || mode == 2) {
            freeze_test_kernel_streak_tiered<K_NEIGH, idx_t><<<grd, blk>>>(
                dS, dSnew,
                dKNN_sites, dPrevKNN_sites,
                dFreezeCand,
                dFreezeStreak,
                d_tier_id,
                d_thresh2_tier,
                d_streak_tier,
                nV,
                has_prev_knn,
                dCounts
            );
        } else {
            freeze_test_kernel_streak<K_NEIGH, idx_t><<<grd, blk>>>(
                dS, dSnew,
                dKNN_sites, dPrevKNN_sites,
                dFreezeCand,
                dFreezeStreak,
                freeze_disp, nV,
                has_prev_knn,
                freeze_monitor_iters,
                dCounts
            );
        }
        freeze_apply_cand_kernel<<<grd, blk>>>(dFreezeCand, dFrozen, nV, dCounts);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float freeze_ms = elapsed_ms(e0, e1);
        printf("freeze_ms %.3f\n", freeze_ms);

        float iter_remesh_ms = knn_sites_ms + knn_site_to_mesh_ms + uv_from_mesh_ms + centroids_ms + knn_centroid_to_mesh_ms + project_ms;
        total_remesh_ms += iter_remesh_ms;
        printf("iter_remesh_ms %.3f (total_remesh_ms %.3f)\n", iter_remesh_ms, total_remesh_ms);

        int hCounts[4] = {0,0,0,0};
        cudaMemcpy(hCounts, dCounts, 4 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("same neighbors %d\n", hCounts[0]);
        printf("low displacement %d\n", hCounts[1]);
        printf("both %d\n", hCounts[2]);   // pass-1 self-pass
        printf("pass2 %d\n", hCounts[3]);  // newly frozen by consensus
        if (mode != 0) {
            int blocked_by_same = hCounts[1] - hCounts[2];  // had low disp, failed same-neighbors
            int blocked_by_low = hCounts[0] - hCounts[2];   // had same neighbors, failed low disp
            if (blocked_by_same < 0) blocked_by_same = 0;
            if (blocked_by_low < 0) blocked_by_low = 0;
            printf("freeze_investigate: blocked_by_same=%d blocked_by_low_disp=%d (larger blocks more)\n",
                blocked_by_same, blocked_by_low);
        }

        cudaMemset(dFrozenSum, 0, sizeof(int));
        count_frozen_kernel<<<grd, blk>>>(dFrozen, nV, dFrozenSum);

        int hFrozenSum = 0;
        cudaMemcpy(&hFrozenSum, dFrozenSum, sizeof(int), cudaMemcpyDeviceToHost);

        const float frozenRatio = (float)hFrozenSum / (float)nV;
        last_frozenRatio = frozenRatio;
        printf("total frozen so far %d / %d (%.2f%%)\n",
            hFrozenSum, nV, frozenRatio * 100.0f);

        const bool dump_iter = (iter % DUMP_STRIDE == 0);
        if (dump_iter) {
            hFrozenIter.resize((size_t)nV);
            cudaMemcpy(hFrozenIter.data(), dFrozen, (size_t)nV, cudaMemcpyDeviceToHost);
        }

        // Baseline mode computes dFrozen but clears it every iteration to avoid freezing.
        if (mode == 0) {
            cudaMemset(dFrozen, 0, (size_t)nV);
            cudaMemset(dFreezeStreak, 0, (size_t)nV);
        }

        // // const float EMA_ALPHA = 0.3f;
        // // frozenRatio_ema = (iter == 0) ? frozenRatio
        // //                             : EMA_ALPHA * frozenRatio + (1.0f - EMA_ALPHA) * frozenRatio_ema;

        // // float growth = frozenRatio_ema - last_frozenRatio;
        // // last_frozenRatio = frozenRatio_ema;

        // // if (fabsf(growth) < GROWTH_EPS) {
        // //     flat_counter++;
        // // } else {
        // //     flat_counter = 0;
        // // }

        // // const int FLAT_ITERS = 8;

        // // bool terminate_now = (flat_counter >= FLAT_ITERS);

        // // terminate_now = false;

        // // if (frozenRatio_ema < 0.75f and iter < 121) {
        // //     terminate_now = false;
        // // }

        // // if (terminate_now) {
        // //     printf("Termination condition met: frozen rate plateaued at %.6f\n", frozenRatio_ema);
        // // }

        cudaMemcpy(dPrevKNN_sites, dKNN_sites,
           (size_t)nV * K_NEIGH * sizeof(idx_t),
           cudaMemcpyDeviceToDevice);

        float3* tmp = dS;
        dS = dSnew;
        dSnew = tmp;

        run_knn_bitonic_hubs(nV, K_SITE, dFrozen, dS, dKNN_sites_raw, dDist_sites_raw);

        if (mode != 0) {
            cudaMemcpy(dPrevKnnV, dKnnV, (size_t)nV * KPROJ * sizeof(int), cudaMemcpyDeviceToDevice);
            run_knn_bitonic_query_to_mesh(nV, dV, dS, nV, KPROJ, dFrozen, (idx_t*)dKnnV, dKnnV_dist, "site_to_mesh");
            restore_prev_knn_vertices_kernel<KPROJ><<<grd, blk>>>(dFrozen, dPrevKnnV, dKnnV, nV);
        } else {
            knn_vertices_bruteforce_k<KPROJ, false><<<grd, blk>>>(dV, nV, dS, nV, dFrozen, dKnnV);
        }

        cudaEventRecord(e0);
        uv_from_nearest_vertex_normal<KPROJ><<<grd, blk>>>(dN, dKnnV, du, dv, nV);

        float3 hN[10];
        cudaMemcpy(hN, dN, sizeof(float3) * 10, cudaMemcpyDeviceToHost);

        float3 mn = make_float3(1e30f,1e30f,1e30f), mx = make_float3(-1e30f,-1e30f,-1e30f);
        cudaMemcpy(hSnew.data(), dS,
         (size_t)nV * sizeof(float3), cudaMemcpyDeviceToHost);

        for (int i = 0; i < nV; ++i) {
            const float3 p = hSnew[(size_t)i];
            mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
            mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
        }
        float3 bb = make_float3(mx.x-mn.x, mx.y-mn.y, mx.z-mn.z);
        float R2 = 2.0f * 0.5f * std::max(bb.x, std::max(bb.y, bb.z));

        if (use_geogram && (iter % DUMP_STRIDE == 0 || iter == total_iter - 1)) {
            char sites_xyz_path[512];
            std::snprintf(sites_xyz_path, sizeof(sites_xyz_path), "%s/iter_%03d_sites.xyz", out_dir.c_str(), iter);
            write_pts_cpu(sites_xyz_path, hSnew);
            char obj_path[512];
            std::snprintf(obj_path, sizeof(obj_path), "%s/iter_%03d.obj", out_dir.c_str(), iter);
            int vret = run_vorpalite(vorpalite_path, sites_xyz_path, obj_path);
            if (vret != 0) {
                fprintf(stderr, "vorpalite failed (exit %d)\n", vret);
                return 1;
            }
            hV_geo.clear();
            hF_geo.clear();
            load_obj_triangles(obj_path, hV_geo, hF_geo);
            last_geogram_nf = (int)hF_geo.size();
        }

        if (!use_geogram) {
            cudaMemset(dPolyN, 0, (size_t)nV * sizeof(int));

            cell_poly2d_kernel<K_NEIGH, MAX_POLY, idx_t><<<grd, blk>>>(
                dS, du, dv, dKNN_sites, nV,
                R2,
                1e-8f,
                1e-7f,
                dPoly2d, dPolyN, dPolyLab2
            );

            cudaMemset(dKeyCount, 0, sizeof(int));
            emit_faces_from_labels_kernel<MAX_POLY><<<grd, blk>>>(
                dPolyN, dPolyLab2, nV, dKeys, dKeyCount, (int)maxCand
            );

            int hCount = 0;
            cudaMemcpy(&hCount, dKeyCount, sizeof(int), cudaMemcpyDeviceToHost);
            if (hCount < 0) hCount = 0;
            if ((size_t)hCount > maxCand) hCount = (int)maxCand;

            hKeys.resize((size_t)hCount);
            cudaMemcpy(hKeys.data(), dKeys, (size_t)hCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            std::sort(hKeys.begin(), hKeys.end());
            auto it = std::unique(hKeys.begin(), hKeys.end());
            hKeys.erase(it, hKeys.end());

            decode_keys_to_faces(hKeys, hFnew);

            size_t before = hFnew.size();
            std::vector<Tri> hFfilt;
            hFfilt.reserve(before);

            for (const Tri& t : hFnew) {
                int a = t.a, b = t.b, c = t.c;
                if (!is_degenerate_tri(hSnew, a, b, c, DEGENERATE_EPS)) {
                    hFfilt.push_back(t);
                }
            }

            size_t after = hFfilt.size();
            printf("degenerate filtered: %zu removed, %zu kept (from %zu)\n",
                before - after, after, before);

            hFnew.swap(hFfilt);
        }

        if (iter % DUMP_STRIDE == 0) {

            char path[512];
            std::snprintf(path, sizeof(path), "%s/iter_%03d.obj", out_dir.c_str(), iter);

            if (!use_geogram) {
                write_obj_cpu(path, hSnew, hFnew);
            }
            // when use_geogram, vorpalite already wrote path

            char fpath[512];
            std::snprintf(fpath, sizeof(fpath), "%s/iter_%03d_frozen.txt", out_dir.c_str(), iter);

            if (!hFrozenIter.empty()) {
                write_frozen_log_txt(fpath, hFrozenIter.data(), nV);
            }

            const size_t nV_dump = use_geogram ? hV_geo.size() : hSnew.size();
            const size_t nF_dump = use_geogram ? (size_t)last_geogram_nf : hFnew.size();
            printf("iter %d V %zu faces %zu wrote %s\n",
                iter, nV_dump, nF_dump, path);

            // ---- Build candidate mesh arrays for evaluation ----
            std::vector<int3i> hFcand;
            if (use_geogram) {
                hFcand.assign(hF_geo.begin(), hF_geo.end());
            } else {
                hFcand.reserve(hFnew.size());
                for (const Tri& t : hFnew) hFcand.push_back({t.a, t.b, t.c});
            }
            std::vector<float3>& hV_dump = use_geogram ? hV_geo : hSnew;

            float Qmin, Qavg, theta_min, theta_min_avg, theta_lt_30_pct, theta_gt_90_pct;
            eval_quality_angles_cpu(hV_dump, hFcand, Qmin, Qavg, theta_min, theta_min_avg, theta_lt_30_pct, theta_gt_90_pct);

            size_t nVc = hV_dump.size();
            size_t nFc = hFcand.size();

            if (nVc > dVcand_cap) {
                if (dVcand) cudaFree(dVcand);
                cudaMalloc(&dVcand, nVc * sizeof(float3));
                dVcand_cap = nVc;
            }
            if (nFc > dFcand_cap) {
                if (dFcand) cudaFree(dFcand);
                cudaMalloc(&dFcand, nFc * sizeof(int3i));
                dFcand_cap = nFc;
            }

            cudaMemcpy(dVcand, hV_dump.data(), nVc * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(dFcand, hFcand.data(), nFc * sizeof(int3i), cudaMemcpyHostToDevice);

            float dH = hausdorff_cand_to_ref_gpu(
                dVref, (int)hV.size(),
                dFref, (int)hF.size(),
                dNodesRef, (int)refNodes.size(),
                dTriIdxRef,

                dVcand, (int)hSnew.size(),
                dFcand, (int)hFcand.size()
            );

            std::string eval_path = out_dir + "/eval_iters.csv";

            append_eval_iters_csv(
                eval_path,
                mesh_name,
                std::string(root_dir),
                iter,
                Qmin, Qavg,
                theta_min, theta_min_avg,
                theta_lt_30_pct, theta_gt_90_pct,
                dH,
                iter_remesh_ms,
                total_remesh_ms,
                hFrozenSum,
                nV,
                knn_sites_ms,
                knn_site_to_mesh_ms,
                uv_from_mesh_ms,
                centroids_ms,
                knn_centroid_to_mesh_ms,
                project_ms,
                freeze_ms,
                hCounts[0],
                hCounts[1],
                hCounts[2],
                hCounts[3]
            );

        }

        used_iters = iter + 1;
        final_converge_rate = last_frozenRatio;
        final_nf = use_geogram ? last_geogram_nf : (int)hFnew.size();

        //if (terminate_now) break;
    }

    printf("\nTotal remeshing time (mesh rebuilding excluded): %.3f ms\n", total_remesh_ms);

    {
        const char* mode_str = (mode == 0) ? "gpucvt" : (mode == 2 ? "freeze_tiered" : "freeze");
        append_run_csv("experiments/runs.csv", mesh_name, mode_str, nV, final_nf, final_converge_rate, used_iters, total_remesh_ms);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    cudaFree(dCounts);
    cudaFree(dFrozen);
    if (d_tier_id) {
        cudaFree(d_tier_id);
        cudaFree(d_thresh2_tier);
        cudaFree(d_streak_tier);
    }
    cudaFree(dFrozenSum);
    if (dKnnV_dist) cudaFree(dKnnV_dist);
    if (dPrevKnnV) cudaFree(dPrevKnnV);
    cudaFree(dKnnV);
    cudaFree(d_vf_off);
    cudaFree(d_vf_faces);
    cudaFree(dPrevKNN_sites);
    cudaFree(dCent);
    cudaFree(dS);
    cudaFree(dSnew);
    cudaFree(du);
    cudaFree(dv);
    cudaFree(dN);
    cudaFree(dV);
    cudaFree(dF);
    cudaFree(dKNN_sites_raw);
    cudaFree(dDist_sites_raw);

    cudaFree(dKeys);
    cudaFree(dKeyCount);
    cudaFree(dPoly2d);
    cudaFree(dPolyN);

    if (dVcand) cudaFree(dVcand);
    if (dFcand) cudaFree(dFcand);
    if (dNodesCand) cudaFree(dNodesCand);
    if (dTriIdxCand) cudaFree(dTriIdxCand);

    dVcand = nullptr; dFcand = nullptr; dNodesCand = nullptr; dTriIdxCand = nullptr;

    return 0;
}
