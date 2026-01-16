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
#include <filesystem>
#include <cstdio>
#include <cstring>

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
    // 0: baseline (gpu_cvt)      - keep existing behavior (dFrozen is computed then cleared each iter)
    // 1: freezing_cvt            - keep existing behavior (dFrozen accumulates across iters)
    // 2: secured_ccu             - secured centroids + secured freezing condition
    const char* mode_name = (mode == 0) ? "baseline"
                         : (mode == 1) ? "freezing_cvt"
                         : "secured_ccu";
    printf("mode %d (%s)\n", mode, mode_name);

    const int freeze_mode = (mode != 0) ? 1 : 0;

    // ---------------- Tunable constants ----------------
    constexpr int   THREADS        = 1024;
    constexpr int   MAX_POLY        = 256;
    constexpr int   KPROJ           = 5;
    constexpr int   TOP_FREEZE_NEIGH = 3;
    constexpr int   DUMP_STRIDE     = 25;     // dump an .obj every N iters
    constexpr float GROWTH_EPS      = 5e-4f;  // used by flat/growth heuristics
    constexpr float DEGENERATE_EPS  = 1e-6f; // degenerate-triangle test epsilon

    constexpr int   K_NEIGH         = 32;
    constexpr int   K_SITE          = K_NEIGH + 1;

    int total_iter = 1000;
    float freeze_disp = 1e-4f;
    int freeze_monitor_iters = 5;    
    // ----------------------------------------------------

    float thresh2 = freeze_disp * freeze_disp;
    float last_frozenRatio = 0.0f;
    float frozenRatio_ema = 0.0f;
    int   flat_counter = 0;
    int last_freeze_bucket = -1;

    int used_iters = 0;
    float final_converge_rate = 0.0f;
    int final_nf = 0;

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

    std::vector<unsigned char> hFrozen(nV);
    std::vector<int> hActiveIdx(nV);
    int* dActiveIdx = nullptr;
    cudaMalloc(&dActiveIdx, nV*sizeof(int));

    auto build_active_list_cpu = [&]()->int{
        cudaMemcpy(hFrozen.data(), dFrozen, nV, cudaMemcpyDeviceToHost);
        int nA = 0;
        for(int i=0;i<nV;++i) if (!hFrozen[i]) hActiveIdx[nA++] = i;
        cudaMemcpy(dActiveIdx, hActiveIdx.data(), nA*sizeof(int), cudaMemcpyHostToDevice);
        return nA;
    };


    // For secured_ccu (mode==2): record whether each cell reached its security radius.
    // This flag gates freezing: only freeze if (low displacement) && (same neighbor list) && (secureReached).
    unsigned char* dSecureReached = nullptr;
    cudaMalloc(&dSecureReached, (size_t)nV);
    cudaMemset(dSecureReached, 0, (size_t)nV);

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
    float first_frozenRatio = -1.0f;
    bool freeze_disp_doubled = false;

    const char* root_dir = (mode == 0) ? "gpucvt" : (mode == 1) ? "freeze" : "secured_ccu";

    std::string out_dir = std::string(root_dir) + "/" + mesh_name;
    std::filesystem::create_directories(out_dir);

    std::string eval_csv = out_dir + "/eval_iters.csv";
    {
        FILE* f = fopen(eval_csv.c_str(), "w");
        if (f) {
            fprintf(f,
            "mesh,mode,iter,Qmin,Qavg,theta_min,theta_min_avg,"
            "theta_lt_30_pct,theta_gt_90_pct,dH,iter_remesh_ms,total_remesh_ms\n");
            fclose(f);
        }
    }

    for (int iter = 0; iter < total_iter; ++iter)
    {
        printf("\n=== iter %d ===\n", iter);

        float knn_sites_ms = run_knn_bitonic_hubs(nV, K_SITE, dFrozen, dS, dKNN_sites_raw, dDist_sites_raw);
        printf("knn_sites_ms %.3f\n", knn_sites_ms);

        knn_drop_self_kernel<K_SITE, K_NEIGH, idx_t><<<grd, blk>>>(dKNN_sites_raw, dKNN_sites, nV);

        float knn_site_to_mesh_ms = 0.0f;
        float knn_centroid_to_mesh_ms = 0.0f;

        int nActive = build_active_list_cpu();

        dim3 blkA(256);
        dim3 grdA((nActive + blkA.x - 1) / blkA.x);
        if (mode != 0) {

            cudaEventRecord(e0);
            knn_vertices_bruteforce_k_active<KPROJ, false><<<grdA, blkA>>>(
                dV, nV,
                dS, nV,
                dActiveIdx, nActive,   // <-- raw device pointer
                dKnnV
            );
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            knn_site_to_mesh_ms = elapsed_ms(e0, e1);
            printf("knn_site_to_mesh_ms %.3f\n", knn_site_to_mesh_ms);
        }
        else{
                dim3 grd((nV + blk.x - 1) / blk.x);

                cudaEventRecord(e0);
                knn_vertices_bruteforce_k<KPROJ, false><<<grd, blk>>>(
                    dV, nV,
                    dS, nV,
                    dFrozen,
                    dKnnV
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
        if (mode == 2) {
            cudaMemset(dSecureReached, 0, (size_t)nV);
            // secured_centroids_tangent_voronoi is expected to set dSecureReached[i]=1 once the cell reaches
            // its security radius criterion.
            secured_centroids_tangent_voronoi<<<grd, blk>>>(
                dS, du, dv,
                dKNN_sites,
                nV, K_NEIGH,
                R,
                dFrozen,
                dCent,
                dSecureReached
            );
        } else {
            centroids_tangent_voronoi<<<grd, blk>>>(dS, du, dv, dKNN_sites, nV, K_NEIGH, R, dFrozen, dCent);
        }

        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float centroids_ms = elapsed_ms(e0, e1);
        printf("centroids_ms %.3f\n", centroids_ms);

        if (mode != 0) {
            cudaEventRecord(e0);
            knn_vertices_bruteforce_k_active<KPROJ, true><<<grdA, blkA>>>(
                dV, nV,
                dCent, nV,
                dActiveIdx, nActive,   
                dKnnV
            );
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            knn_centroid_to_mesh_ms = elapsed_ms(e0, e1);
            printf("knn_centroid_to_mesh_ms %.3f\n", knn_centroid_to_mesh_ms);
        }else{
            cudaEventRecord(e0);
            knn_vertices_bruteforce_k<KPROJ, true><<<grd, blk>>>(
                dV, nV,
                dCent, nV,
                dFrozen,
                dKnnV
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

        int has_prev_knn = (iter > 0) ? 1 : 0;
        cudaMemset(dCounts, 0, 4 * sizeof(int));
        cudaMemset(dFreezeCand, 0, (size_t)nV); // pass-1 candidates (this iter only)

        cudaEventRecord(e0);
        if (mode == 2) {
            // pass-1 (secured): template<int K, typename IndexT>
            // freeze_test_kernel_secured<K_NEIGH, idx_t><<<grd, blk>>>(
            //     dS, dSnew,
            //     dKNN_sites, dPrevKNN_sites,
            //     dFreezeCand,
            //     dSecureReached,
            //     thresh2, nV,
            //     has_prev_knn,
            //     dCounts
            // );

            freeze_test_kernel_secured_streak<K_NEIGH, idx_t><<<grd, blk>>>(
                dS, dSnew,
                dKNN_sites, dPrevKNN_sites,
                dFreezeCand,
                dFreezeStreak,
                dSecureReached,
                thresh2, nV,
                has_prev_knn,
                freeze_monitor_iters,
                dCounts
            );
            freeze_apply_cand_kernel<<<grd, blk>>>(dFreezeCand, dFrozen, nV, dCounts);
            // // pass-2 (secured): template<int K, int TOPN, typename IndexT>
            // freeze_consensus_pass2_kernel_secured<K_NEIGH, TOP_FREEZE_NEIGH, idx_t><<<grd, blk>>>(
            //     dKNN_sites,
            //     dFreezeCand,
            //     dSecureReached,
            //     dFrozen,
            //     nV,
            //     dCounts
            // );
        } else {
            // pass-1 (non-secured): template<int K, typename IndexT>
            // freeze_test_kernel<K_NEIGH, idx_t><<<grd, blk>>>(
            //     dS, dSnew,
            //     dKNN_sites, dPrevKNN_sites,
            //     dFreezeCand,
            //     thresh2, nV,
            //     has_prev_knn,
            //     dCounts
            // );

            freeze_test_kernel_streak<K_NEIGH, idx_t><<<grd, blk>>>(
                dS, dSnew,
                dKNN_sites, dPrevKNN_sites,
                dFreezeCand,
                dFreezeStreak,
                thresh2, nV,
                has_prev_knn,
                freeze_monitor_iters,
                dCounts
            );
            freeze_apply_cand_kernel<<<grd, blk>>>(dFreezeCand, dFrozen, nV, dCounts);
            // pass-2 (non-secured): template<int K, int TOPN, typename IndexT>
            // freeze_consensus_pass2_kernel<K_NEIGH, TOP_FREEZE_NEIGH, idx_t><<<grd, blk>>>(
            //     dKNN_sites,
            //     dFreezeCand,
            //     dFrozen,
            //     nV,
            //     dCounts
            // );
        }
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

        // ---- A) low-displacement ratio: ONLY for adapting freeze_disp
        const float lowDispRatio = (float)hCounts[1] / (float)nV;

        // record baseline signal at iter 0 (for the adaptation logic only)
        if (iter == 0) {
            first_frozenRatio = lowDispRatio;   // (rename this variable if you want; see note below)
        }

        // ---- B) frozen ratio from dFrozen: used for all other logic (progress, termination, etc.)
        cudaMemset(dFrozenSum, 0, sizeof(int));
        count_frozen_kernel<<<grd, blk>>>(dFrozen, nV, dFrozenSum);

        int hFrozenSum = 0;
        cudaMemcpy(&hFrozenSum, dFrozenSum, sizeof(int), cudaMemcpyDeviceToHost);

        const float frozenRatio = (float)hFrozenSum / (float)nV;
        int freeze_bucket = (int)floorf(frozenRatio * 10.0f);   // 0..10

        if (freeze_bucket > last_freeze_bucket) {
            // We crossed a new 10% boundary
            if (last_freeze_bucket >= 0) {   // skip at iter 0 initialization
                //freeze_disp *= 0.5f;         // tighten by 10%
                //thresh2 = freeze_disp * freeze_disp;
                freeze_monitor_iters += 5;

                printf("[freeze_disp] tighten at %.0f%% frozen → freeze_disp = %.6e\n",
                    freeze_bucket * 10.0f, freeze_disp);
            }

            last_freeze_bucket = freeze_bucket;
        }

        // -------------------- Adaptation of freeze_disp --------------------
        // Only based on lowDispRatio signal (NOT frozenRatio).
        // Keep your original intent: only decide after first 2 iters.
        if (iter == 1) {
            // gate: if iter0 lowDisp was already tiny and iter1 is still tiny => too strict => relax
            if (first_frozenRatio < GROWTH_EPS && lowDispRatio < GROWTH_EPS) {

                freeze_disp *= 5.0f;
                thresh2 = freeze_disp * freeze_disp;

                // reset loop + plateau detector
                flat_counter = 0;
                first_frozenRatio = -1.0f;
                iter = -1; // next ++iter becomes 0

                cudaMemset(dFreezeStreak, 0, (size_t)nV);

                {
                    FILE* f = fopen(eval_csv.c_str(), "w");
                    if (f) {
                        fprintf(f,
                            "mesh,mode,iter,Qmin,Qavg,theta_min,theta_min_avg,"
                            "theta_lt_30_pct,theta_gt_90_pct,dH,iter_remesh_ms,total_remesh_ms\n");
                        fclose(f);
                    }
                }

                continue;
            }
        }

        // -------------------- Everything else uses frozenRatio --------------------
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

        knn_vertices_bruteforce_k<KPROJ, false><<<grd, blk>>>(dV, nV, dS, nV, dFrozen, dKnnV);

        cudaEventRecord(e0);
        uv_from_nearest_vertex_normal<KPROJ><<<grd, blk>>>(dN, dKnnV, du, dv, nV);

        float3 hN[10];
        cudaMemcpy(hN, dN, sizeof(float3) * 10, cudaMemcpyDeviceToHost);

        float3 mn = make_float3(1e30f,1e30f,1e30f), mx = make_float3(-1e30f,-1e30f,-1e30f);
        cudaMemcpy(hSnew.data(), dS, (size_t)nV * sizeof(float3), cudaMemcpyDeviceToHost);

        for (int i = 0; i < nV; ++i) {
            const float3 p = hSnew[(size_t)i];
            mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
            mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
        }
        float3 bb = make_float3(mx.x-mn.x, mx.y-mn.y, mx.z-mn.z);
        float R2 = 2.0f * 0.5f * std::max(bb.x, std::max(bb.y, bb.z));

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

        if (iter % DUMP_STRIDE == 0) {

            char path[512];
            std::snprintf(path, sizeof(path),
                        "%s/%s/iter_%03d.obj",
                        root_dir, mesh_name.c_str(), iter);

            write_obj_cpu(path, hSnew, hFnew);

            char fpath[512];
            std::snprintf(fpath, sizeof(fpath),
                        "%s/%s/iter_%03d_frozen.txt",
                        root_dir, mesh_name.c_str(), iter);

            if (!hFrozenIter.empty()) {
                write_frozen_log_txt(fpath, hFrozenIter.data(), nV);
            }

            printf("iter %d V %d cand %d faces %d wrote %s\n",
                iter, nV, hCount, (int)hFnew.size(), path);

            // ---- Build candidate mesh arrays for evaluation ----
            std::vector<int3i> hFcand;
            hFcand.reserve(hFnew.size());
            for (const Tri& t : hFnew) {
                hFcand.push_back({t.a, t.b, t.c});
            }

            float Qmin, Qavg, theta_min, theta_min_avg, theta_lt_30_pct, theta_gt_90_pct;
            eval_quality_angles_cpu(hSnew, hFcand, Qmin, Qavg, theta_min, theta_min_avg, theta_lt_30_pct, theta_gt_90_pct);

            size_t nVc = hSnew.size();
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

            cudaMemcpy(dVcand, hSnew.data(), nVc * sizeof(float3), cudaMemcpyHostToDevice);
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
                total_remesh_ms
            );

        }

        used_iters = iter + 1;
        final_converge_rate = frozenRatio_ema;
        final_nf = (int)hFnew.size();

        //if (terminate_now) break;
    }

    printf("\nTotal remeshing time (mesh rebuilding excluded): %.3f ms\n", total_remesh_ms);

    {
        const char* mode_str = (mode == 0) ? "gpucvt" : (mode == 1) ? "freeze" : "secured_ccu";
        append_run_csv("runs.csv", mesh_name, mode_str, nV, final_nf, final_converge_rate, used_iters, total_remesh_ms);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    cudaFree(dCounts);
    cudaFree(dFrozen);
    cudaFree(dSecureReached);
    cudaFree(dFrozenSum);
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
