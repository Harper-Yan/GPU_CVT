#pragma once

#include <cassert>
#include <cmath>   // for INFINITY; numeric limits doesn't work
#include <curand_kernel.h> 
#include <cfloat>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <random>


#include <thread>
#include <cuda_runtime.h>

#include "cuda_util.cuh"
#include "spatial.cuh"
#include "bitonic-shared.cuh"

inline idx_t constexpr H = 1024;
inline idx_t constexpr warp_size = 32;

namespace bitonic_hubs_grid {
// Hleper function in time logging--------------------
void time_section(std::vector<float>& durations, const std::string& name,
                  std::vector<std::string>& names,
                  cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    durations.push_back(ms);
    names.push_back(name);
    cudaEventRecord(start);  
}
// Grid-related Pre-processing kernels --------------------------------------
__host__ __device__ inline int float_to_ordered_int(float val) {
#ifdef __CUDA_ARCH__
    int bits = __float_as_int(val);
#else
    int bits;
    std::memcpy(&bits, &val, sizeof(float));
#endif
    return (bits >= 0) ? bits : bits ^ 0x7FFFFFFF;
}

__host__ __device__ inline float ordered_int_to_float(int ordered_bits) {
    int bits = (ordered_bits >= 0) ? ordered_bits : ordered_bits ^ 0x7FFFFFFF;
#ifdef __CUDA_ARCH__
    return __int_as_float(bits);
#else
    float val;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
#endif
}

__global__ void compute_min_max(const float* data, int* min_vals_int, int* max_vals_int, int num_points) {
    __shared__ int local_min[3];
    __shared__ int local_max[3];

    if (threadIdx.x < 3) {
        local_min[threadIdx.x] = float_to_ordered_int(FLT_MAX);
        local_max[threadIdx.x] = float_to_ordered_int(-FLT_MAX);
    }
    __syncthreads();

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_points; i += blockDim.x * gridDim.x) {
        float x = data[i * 3 + 0];
        float y = data[i * 3 + 1];
        float z = data[i * 3 + 2];

        atomicMin(&local_min[0], float_to_ordered_int(x));
        atomicMin(&local_min[1], float_to_ordered_int(y));
        atomicMin(&local_min[2], float_to_ordered_int(z));

        atomicMax(&local_max[0], float_to_ordered_int(x));
        atomicMax(&local_max[1], float_to_ordered_int(y));
        atomicMax(&local_max[2], float_to_ordered_int(z));
    }
    __syncthreads();

    if (threadIdx.x < 3) {
        atomicMin(&min_vals_int[threadIdx.x], local_min[threadIdx.x]);
        atomicMax(&max_vals_int[threadIdx.x], local_max[threadIdx.x]);
    }
}

__global__ void assign_bins(const float* data, const float* min_vals, const float* max_vals, int* bin_ids, int* bin_counts, int num_points, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float x = data[i * 3 + 0];
    float y = data[i * 3 + 1];
    float z = data[i * 3 + 2];

    float range_x = max_vals[0] - min_vals[0] ;
    float range_y = max_vals[1] - min_vals[1] ;
    float range_z = max_vals[2] - min_vals[2] ;

    int bin_x = min((int)(((x - min_vals[0]) / range_x) * p), p - 1);
    int bin_y = min((int)(((y - min_vals[1]) / range_y) * p), p - 1);
    int bin_z = min((int)(((z - min_vals[2]) / range_z) * p), p - 1);

    int bin_id = bin_x * p * p + bin_y * p + bin_z;

    bin_ids[i] = bin_id;
    atomicAdd(&bin_counts[bin_id], 1);
}

__global__
void BucketSortPoints(int n, const float* data, float* d_data, const int* bin_ids, int* d_cell_psum,  idx_t* d_sorted_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int bin = bin_ids[idx];
    int loc = atomicAdd(&d_cell_psum[bin], 1);

    d_data[loc * 3 + 0] = data[idx * 3 + 0];
    d_data[loc * 3 + 1] = data[idx * 3 + 1];
    d_data[loc * 3 + 2] = data[idx * 3 + 2];

    d_sorted_idx[loc] = idx;
}
// End of Grid-related Pre-processing kernels --------------------------------------
// Start of Clover Construction kernels---------------------------------------------
__device__ int logPopCounts[33];
bool GenerateUniqueHubsAndUpload(
    std::size_t n,
    std::size_t H,
    idx_t* dH_i,
    cudaStream_t stream,
    int stream_index,
    unsigned int base_seed = 1234
) {
    // Basic checks
    if (H > n) return false;
    if (H == 0) return true;           // nothing to do, treat as success
    if (dH_i == nullptr) return false; // invalid destination

    // Prepare 0..n-1 and shuffle
    std::vector<idx_t> local_hubs(n);
    std::iota(local_hubs.begin(), local_hubs.end(), 0);

    std::mt19937 rng(base_seed + stream_index);
    std::shuffle(local_hubs.begin(), local_hubs.end(), rng);

    // Copy only the first H unique indices to the device
    const cudaError_t err = cudaMemcpyAsync(
        dH_i,
        local_hubs.data(),
        H * sizeof(idx_t),
        cudaMemcpyHostToDevice,
        stream
    );
    return (err == cudaSuccess);
}
template <typename R>
__global__  
__launch_bounds__(128, 4) 
void Calculate_Distances(
    idx_t b_id,
    idx_t b_size,
    idx_t n,
    const idx_t* __restrict__ dH,
    R* __restrict__ distances,
    const R* __restrict__ points,
    idx_t* __restrict__ hub_counts,
    idx_t* __restrict__ dH_assignments   
)
{
    extern __shared__ R shared_hub_coords[]; 
    R* hub_x = shared_hub_coords;
    R* hub_y = &hub_x[H];
    R* hub_z = &hub_y[H];

    idx_t local_idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx_t global_idx = local_idx + b_id * b_size;

    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        idx_t hub_idx = dH[i];
        hub_x[i] =  points[hub_idx * dim + 0];
        hub_y[i] =  points[hub_idx * dim + 1];
        hub_z[i] =  points[hub_idx * dim + 2];
    }
    __syncthreads();

    if ( global_idx >= n || local_idx >= b_size) return;

    R qx =  points[global_idx * dim + 0];
    R qy =  points[global_idx * dim + 1];
    R qz =  points[global_idx * dim + 2];

    R min_dist2 = FLT_MAX;
    idx_t best_hub = H + 1; 

    for (int h = 0; h < H; ++h) {
        R dx = hub_x[h] - qx;
        R dy = hub_y[h] - qy;
        R dz = hub_z[h] - qz;
        R dist2 = dx * dx + dy * dy + dz * dz;

        distances[h * b_size + local_idx ] = sqrtf(dist2);

        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            best_hub = h;
        }
    }

    dH_assignments[global_idx] = best_hub;
    atomicAdd(&hub_counts[best_hub], 1);
}
template < typename T >
__device__ __forceinline__
void prefix_sum_warp( T & my_val )
{
    int constexpr FULL_MASK = 0xFFFFFFFF;
    int constexpr warp_size = 32;

    for( int stride = 1; stride < warp_size; stride = stride << 1 )
    {
        __syncwarp();
        T const paired_val = __shfl_up_sync( FULL_MASK, my_val, stride );
        if( threadIdx.x >= stride )
        {
            my_val += paired_val;
        }
    }
}

template < typename T >
__global__
void fused_prefix_sum_copy( T *arr, T * copy )
{
    // Expected grid size: 1 x  1 x 1
    // Expected CTA size: 32 x 32 x 1

    // lazy implementation for now. Not even close to a hot spot.
    // just iterate H with one thread block and revisit if we start
    // using *very* large H, e.g., H > 8096, or it shows up in profile.

    assert( "H is a power of 2."  && __popc( H ) == 1 );
    assert( "H uses a full warp." && H >= 32 );

    int const lane_id = threadIdx.x;
    int const warp_id = threadIdx.y;
    int const th_id   = warp_id * blockDim.x + lane_id;

    if( th_id >= H ) { return; } // guard clause for syncthreads later

    // the first location of smem will contain the sum of all the
    // size-1024 chunks so far. The remaining 32 are a staging site
    // to propagate warp-level results across warps.
    int const shared_memory_size = 32 + 1;
    __shared__ T smem[ shared_memory_size ];
    if( th_id == 0 ) { smem[ 0 ] = 0; }

    // iterate in chunks of 1024 at a time
    for( int i = th_id; i < H ; i = i + blockDim.x * blockDim.y )
    {

        T my_val = arr[ i ];

        prefix_sum_warp( my_val );

        // compute partial sums over warp-level results
        // first, last lane in each warp copies result to smem for sharing
        if( lane_id == ( blockDim.x - 1) )
        {
            smem[ warp_id + 1 ] = my_val;
        }
        __syncthreads(); // safe because H is a power of 2 & guard clause earlier

        T sum_of_chunk_sofar = 0;

        // first warp computes prefix scan over 32 warp-level sums
        if( warp_id == 0 )
        {
            // fetch other warps' data from smem
            T warp_level_sum = smem[ lane_id + 1 ]
				            + smem[ 0 ] * ( lane_id == 0 );
            prefix_sum_warp( warp_level_sum );

            // write results back out to smem to broadcast to other warps
            // also update smem[ 0 ] to be first sum for next chunk
            smem[ lane_id + 1 ] = warp_level_sum;
            if( lane_id == ( blockDim.x - 1 ) )
            {
                sum_of_chunk_sofar = warp_level_sum;
            }
        }

        // propagate partial results across all threads
        // each thread only needs the partial sum for its warp
        __syncthreads(); // safe for same reasons as previous sync

        my_val += smem[ warp_id ];

        arr [ i ] = my_val;
        copy[ i ] = my_val;

        if(warp_id == 0 && lane_id == ( blockDim.x - 1 )) { smem[0] = sum_of_chunk_sofar; }
    }
}

/**
 * Physically resorts an array with a small domain of V unique values using O(nV) work using an out-of-place
 * struct-of-arrays decomposition.
 */
template <class R>
__global__
void BucketSort(idx_t n, R * arr_x, R *arr_y, R *arr_z, idx_t * arr_idx, R const* points, idx_t const* dH_assignments, idx_t * dH_psum )
{
    
    idx_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < n )
    {
        idx_t const hub_idx = dH_assignments[idx];
        idx_t const loc = atomicAdd(&dH_psum[hub_idx], 1);

        arr_x[loc] = points[idx*dim+0];
        arr_y[loc] = points[idx*dim+1];
        arr_z[loc] = points[idx*dim+2];
        arr_idx[loc] = idx ; 
    }
}

__global__ void set_max_float(float *D, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        D[idx] = __FLT_MAX__;
    }
}

/**
 * Builds the HxH distance matrix, D, in which the asymmetric distance from hub H_i to hub H_j
 * is the distance from H_i to the closest point in H_j.
 */
__global__
void Construct_D( float const * distances, idx_t const * assignments, idx_t b_id, idx_t b_size, idx_t n, float * D )
{

    int constexpr shared_memory_size = 1024;

    assert( "Array fits in shared memory" && H <= shared_memory_size );

    // Each thread block will work on one row of the HxH matrix
    // unless the hub is empty in which case this thread block will just return.
    int const hub_id = blockIdx.x;

    float const * this_hubs_distances = &distances[ hub_id * b_size ];

    int const block_level_lane_id = threadIdx.x;
    assert( "Expect to have one __shared__ lane per thread" && block_level_lane_id < shared_memory_size );

    __shared__ int s_dists[shared_memory_size];

    int R = int (( H + blockDim.x  -1 ) / blockDim.x );

    for ( int r = 0; r < R; r ++)
    {
        if( r * blockDim.x + block_level_lane_id < H )
        {
            s_dists[ r * blockDim.x + block_level_lane_id ] = 0x7f000000;
        }
    }

    __syncwarp(); __syncthreads();

    for( idx_t p = block_level_lane_id; p < b_size; p += blockDim.x )
    {
        idx_t idx = b_id * b_size + p;
        if( idx < n)
        {
            idx_t const this_H = assignments[ idx ];
            assert( "Retrieved a valid hub id" && this_H < H );
            atomicMin( &s_dists[this_H], __float_as_int( this_hubs_distances[ p ] ) );
        }
    }

    __syncwarp(); __syncthreads();

    for ( int r = 0; r < R; r ++)
    {
        if( r * blockDim.x + block_level_lane_id < H )
        {
            atomicMin(&s_dists[ r * blockDim.x + block_level_lane_id ],  __float_as_int( D[ H * hub_id + r * blockDim.x + block_level_lane_id ] ));
            D[ H * hub_id + r * blockDim.x + block_level_lane_id ] = __int_as_float( s_dists[ r * blockDim.x + block_level_lane_id ] );
        } 
    }
}
template < typename R, int ROUNDS >
__global__
void fused_transform_sort_D( R     const * D // square matrix with lower dist bound from hub i to j
                           , idx_t       * sorted_hub_ids  // square matrix where (i,j) is id of j'th closest hub to i
                           , R           * sorted_hub_dist // square matrix where (i,j) is dist of j'th closest hub to i
                           )
{
    // NOTE: this currently uses a lot of smem, reducing warp occupancy by 2x at H=1024.
    __shared__ R smem[ 2 * H ];

    // each block will sort one row of size H.
    // each thread is responsible for determining the final contents of one cell
    auto  const     warp_size = 32u;
    auto  const     block_size = 1024u;
    idx_t const     lane_id   = threadIdx.x;
    idx_t const     warp_id   = threadIdx.y;
    idx_t const     sort_id   = warp_id * warp_size + lane_id;
    idx_t const     hub_id    = blockIdx.x;

    if(sort_id >= H || hub_id >=H) {return;}

    // each thread grabs the contents of its cell in the input distance matrix
    R     dist[ROUNDS] ;
    idx_t hub[ROUNDS] ;

    for (int r=0; r< ROUNDS; r++)
    {
        dist[r] = D[ H * hub_id + sort_id + block_size * r ];
        hub[r]   = sort_id + block_size * r;
    }

    // create num_hubs >> 5 sorted runs in registers
    //bitonic::sort<warp_id % 2, 1>( &hub, &dist );
    //branch divergence here

    for ( int r = 0; r < ROUNDS; r ++)
    {
        if ( warp_id % 2 == 0  ) {
            bitonic::sort<true, 1>( &hub[r], &dist[r] );  
        } else {
            bitonic::sort<false, 1>( &hub[r], &dist[r] ); 
        }
    }

    // perform repeated merges with a given number of cooperating threads
    for( idx_t coop = warp_size << 1; coop <= H; coop = coop << 1 )
    {
        // do first steps of merge in shared memory 
        for( idx_t stride = coop >> 1; stride >= warp_size; stride = stride >> 1 )
        {
            for ( int r = 0; r < ROUNDS; r ++)
            {
                int const global_lane_id = r * block_size + sort_id;
                smem[ global_lane_id ] = dist[r];
                smem[ global_lane_id + H ] = float(hub[r]);
            }

            __syncthreads();

            for ( int r = 0; r < ROUNDS; r ++)
            {
                int const global_lane_id = r * block_size + sort_id;
                // TODO: optimise this part to reduce trips to smem somehow
                // something more tiled? each thread only reads two vals per sync
                // TODO: this is a guaranteed bank conflict (BC) followed immediately
                // by a sync to force *all* threads to wait for the BC

                // TODO: this is a guaranteed bank conflict, too.
                // but maybe these are inevitable anyway due
                // to pigeon hole principle?
                idx_t paired_thread = (global_lane_id)  ^ stride;
                R     const paired_dist = smem[ paired_thread ];
                idx_t const paired_hub  = int(smem[ paired_thread + H ]);
        
                if( ( paired_thread > global_lane_id && ( global_lane_id & coop ) == 0 && ( paired_dist < dist[r] ) )
                || ( paired_thread < global_lane_id && ( global_lane_id & coop ) != 0 && ( paired_dist < dist[r] ) )
                || ( paired_thread > global_lane_id && ( global_lane_id & coop ) != 0 && ( paired_dist > dist[r] ) )
                || ( paired_thread < global_lane_id && ( global_lane_id & coop ) == 0 && ( paired_dist > dist[r] ) ) )
                {
                    dist[r] = paired_dist;
                    hub[r]  = paired_hub;
                }
                __syncthreads();

            }
        }

        for ( int r = 0; r < ROUNDS; r ++)
        {
            int const global_lane_id = r * block_size + sort_id;
            if ( ( global_lane_id & coop ) == 0 ){
                bitonic::sort<true, 1>( &hub[r], &dist[r] );  
            } else {
                bitonic::sort<false, 1>( &hub[r], &dist[r] ); 
            }
        }

    }

    __syncthreads();

    for (int r = 0 ; r < ROUNDS ; r ++)
    {
        sorted_hub_ids [ hub_id * H + sort_id + r * block_size ] = hub[r];
        sorted_hub_dist[ hub_id * H + sort_id + r * block_size ] = dist[r];
    }
}

//-----------------Helper functions for grid-related queries
void compute_cell_bounds(
    const float* min_vals,
    const float* max_vals,
    int p,               
    int dim,
    std::vector<float>& cell_mins,
    std::vector<float>& cell_maxs
) {
    int num_cells = std::pow(p, dim);

    std::vector<float> cell_size(dim);
    std::vector<float> half_cell(dim);
    std::vector<std::vector<float>> mids_per_axis(dim);

    for (int d = 0; d < dim; ++d) {
        cell_size[d] = (max_vals[d] - min_vals[d]) / p;
        half_cell[d] = cell_size[d] / 2.0f;

        mids_per_axis[d].resize(p);
        for (int i = 0; i < p; ++i) {
            mids_per_axis[d][i] = min_vals[d] + half_cell[d] + i * cell_size[d];
        }
    }

    cell_mins.resize(num_cells * dim);
    cell_maxs.resize(num_cells * dim);

    int cell_id = 0;
    std::vector<int> cell_idx(dim, 0);

    for (int linear_id = 0; linear_id < num_cells; ++linear_id) {
        int rem = linear_id;
        for (int d = dim - 1; d >= 0; --d) {
            cell_idx[d] = rem % p;
            rem /= p;
        }

        for (int d = 0; d < dim; ++d) {
            float center = mids_per_axis[d][cell_idx[d]];
            cell_mins[cell_id * dim + d] = center - half_cell[d];
            cell_maxs[cell_id * dim + d] = center + half_cell[d];
        }
        ++cell_id;
    }
}

__device__ float distance_to_cell_aabb(float* query, float* min_c, float* max_c, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float v = query[d];
        float lo = min_c[d];
        float hi = max_c[d];
        float clipped = fmaxf(fminf(v, hi), lo);
        float diff = clipped - v;
        sum += diff * diff;
    }
    return sqrtf(sum);
}

__device__ int get_assignment_another_cell(const float q_xyz[3], int cell, idx_t** dH, float** data) {
    constexpr int warp_size = 32;
    int lane_id = threadIdx.x;

    int best_hub_idx = -1;
    float best_dist = FLT_MAX;

    for (int i=0; i < H; i += warp_size) {
        int hub_idx = dH[cell][i]; 
        float hx = data[cell][hub_idx * 3 + 0];
        float hy = data[cell][hub_idx * 3 + 1];
        float hz = data[cell][hub_idx * 3 + 2];

        float dx = q_xyz[0] - hx;
        float dy = q_xyz[1] - hy;
        float dz = q_xyz[2] - hz;

        float dist = dx * dx + dy * dy + dz * dz;

        if (dist < best_dist) {
            best_dist = dist;
            best_hub_idx = i;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xffffffff, best_dist, offset);
        int other_idx  = __shfl_down_sync(0xffffffff, best_hub_idx, offset);

        if (other_dist < best_dist) {
            best_dist = other_dist;
            best_hub_idx = other_idx;
        }
    }
    best_hub_idx = __shfl_sync(0xffffffff, best_hub_idx, 0);

    return best_hub_idx;
}

//-----------------Query kernel for Grid-related version
template <std::size_t ROUNDS>
__device__ void query_own_cell(
    int cell,
    int qp,
    int hub_containing_qp,
    int K,
    idx_t** arr_idx_arr,
    float** arr_x_arr,
    float** arr_y_arr,
    float** arr_z_arr,
    float* q_xyz,
    float** data,               
    idx_t** dH,                 
    idx_t* global_idx_current_cell,
    idx_t** dH_psum,
    idx_t* best_point_id,
    float* best_distance,
    float** dD_arr,
    idx_t** iD_arr,
    int H,
    int n_stream,
    int* hubs_scanned,      
    int* points_scanned 
)
{
    int lane_id = threadIdx.x;

    int current_H = hub_containing_qp;
    
    const idx_t* arr_idx = arr_idx_arr[cell];
    const float* arr_x  = arr_x_arr[cell];
    const float* arr_y  = arr_y_arr[cell];
    const float* arr_z  = arr_z_arr[cell];
    const idx_t* psum         = dH_psum[cell];

    idx_t scan_hub_from = psum[hub_containing_qp];
    idx_t scan_hub_to   = psum[hub_containing_qp + 1];

    const float q_x = q_xyz[0];
    const float q_y = q_xyz[1];
    const float q_z = q_xyz[2];

    int hubs_processed = 0;
    int points_scanned_local = 0;

    idx_t const lane_K  = (K - 1) % warp_size;
    idx_t const round_K = ROUNDS - 1;
    
    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        idx_t idx = lane_id + r * warpSize;
        idx_t const* hub_points = arr_idx + scan_hub_from;
        if (scan_hub_from + idx < scan_hub_to) {
            best_point_id[r] = global_idx_current_cell[hub_points[idx]];

            float next_x = arr_x[scan_hub_from + idx];
            float next_y = arr_y[scan_hub_from + idx];
            float next_z = arr_z[scan_hub_from + idx];

            best_distance[r] = spatial::l2dist(q_x, q_y, q_z, next_x, next_y, next_z);
        } else {
            best_distance[r] = FLT_MAX;
        }
    }
    points_scanned_local += ((ROUNDS * warp_size) > (scan_hub_to - scan_hub_from))?(scan_hub_to - scan_hub_from):(ROUNDS * warp_size);
    scan_hub_from += ROUNDS * warp_size; 

    bitonic::sort<true, ROUNDS>( best_point_id, best_distance );

    float kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );

    float h_x = data[cell][ dH[cell][hub_containing_qp] * 3 ];
    float h_y = data[cell][ dH[cell][hub_containing_qp] * 3 +1];
    float h_z = data[cell][ dH[cell][hub_containing_qp] * 3 +2];

    float * dD = dD_arr[cell];
    idx_t * iD = iD_arr[cell];

    float dist_to_my_hub   = sqrt(spatial::l2dist( q_x, q_y, q_z, h_x, h_y, h_z ));
    float dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];

    if( scan_hub_from >= scan_hub_to )   
    {
        if( ++hubs_processed < H )
        {
            current_H        = iD[ hub_containing_qp * H + hubs_processed ];
            dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];
            scan_hub_from    = psum[ current_H ];
            scan_hub_to      = psum[ current_H + 1 ];
        }
    }
    
    while( hubs_processed < H && sqrt( kth_distance ) > dist_to_this_hub - dist_to_my_hub )
    {
        idx_t next_point_id = n_stream;
        float next_distance = FLT_MAX;
        
        if( scan_hub_from + lane_id < scan_hub_to )
        {
            next_point_id      = global_idx_current_cell[arr_idx[scan_hub_from + lane_id]];
            
            float const next_x =   arr_x[ scan_hub_from + lane_id ];
            float const next_y =   arr_y[ scan_hub_from + lane_id ];
            float const next_z =   arr_z[ scan_hub_from + lane_id ];
            next_distance = spatial::l2dist( q_x, q_y, q_z, next_x, next_y, next_z );
        }
        
        if( __any_sync( 0xFFFFFFFF, next_distance < kth_distance ) )
        {
            
            bitonic::sort<false, 1>( &next_point_id, &next_distance );
            if( next_distance < best_distance[ ROUNDS - 1 ] )
            {
                util::swap( next_distance, best_distance[ ROUNDS - 1 ] );
                util::swap( next_point_id, best_point_id[ ROUNDS - 1 ] );
            }
            bitonic::sort<true, ROUNDS>( best_point_id, best_distance, warp_size * ROUNDS );
            kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );
        }
        
        points_scanned_local += (warp_size > (scan_hub_to - scan_hub_from))?(scan_hub_to - scan_hub_from) : warp_size;
        scan_hub_from += warp_size;

        if( scan_hub_from >= scan_hub_to )
        {
            if( ++hubs_processed < H )
            {
                current_H        = iD[ hub_containing_qp * H + hubs_processed ];
                dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];
                scan_hub_from    = psum[ current_H ];
                scan_hub_to      = psum[ current_H + 1 ];
            }
        }
    }
    
    if( lane_id == 0 )
    {
        hubs_scanned[0] += hubs_processed;
        points_scanned[0] += points_scanned_local;
    }
}
template <std::size_t ROUNDS>
__device__ void query_cell(
    int cell,
    int qp,
    int hub_containing_qp,
    int K,
    idx_t** arr_idx_arr,
    float** arr_x_arr,
    float** arr_y_arr,
    float** arr_z_arr,
    float* q_xyz,
    float** data,               
    idx_t** dH,                 
    idx_t* global_idx_current_cell,
    idx_t** dH_psum,
    idx_t* best_point_id,
    float* best_distance,
    float** dD_arr,
    idx_t** iD_arr,
    int H,
    int n_stream,
    int* hubs_scanned,      
    int* points_scanned 
)
{
    int lane_id = threadIdx.x;

    int current_H = hub_containing_qp;

    const idx_t* arr_idx = arr_idx_arr[cell];
    const float* arr_x  = arr_x_arr[cell];
    const float* arr_y  = arr_y_arr[cell];
    const float* arr_z  = arr_z_arr[cell];
    const idx_t* psum         = dH_psum[cell];

    idx_t scan_hub_from = psum[hub_containing_qp];
    idx_t scan_hub_to   = psum[hub_containing_qp + 1];

    const float q_x = q_xyz[0];
    const float q_y = q_xyz[1];
    const float q_z = q_xyz[2];

    int hubs_processed = 0;
    int points_scanned_local = 0;

    idx_t const lane_K  = (K - 1) % warp_size;
    idx_t const round_K = ROUNDS - 1;

    float kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );

    float h_x = data[cell][ dH[cell][hub_containing_qp] * 3 ];
    float h_y = data[cell][ dH[cell][hub_containing_qp] * 3 +1];
    float h_z = data[cell][ dH[cell][hub_containing_qp] * 3 +2];

    float * dD = dD_arr[cell];
    idx_t * iD = iD_arr[cell];

    float dist_to_my_hub   = sqrt(spatial::l2dist( q_x, q_y, q_z, h_x, h_y, h_z ));
    float dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];
    
    while( hubs_processed < H && sqrt( kth_distance ) > dist_to_this_hub - dist_to_my_hub) 
    {        
        idx_t next_point_id = n_stream;
        float next_distance = FLT_MAX;

        if( scan_hub_from + lane_id < scan_hub_to )
        {
            next_point_id      = global_idx_current_cell[arr_idx[scan_hub_from + lane_id]];
            
            float const next_x =   arr_x[ scan_hub_from + lane_id ];
            float const next_y =   arr_y[ scan_hub_from + lane_id ];
            float const next_z =   arr_z[ scan_hub_from + lane_id ];
            next_distance = spatial::l2dist( q_x, q_y, q_z, next_x, next_y, next_z );
        }
        
        if( __any_sync( 0xFFFFFFFF, next_distance < kth_distance ) )
        {
            
            bitonic::sort<false, 1>( &next_point_id, &next_distance );
            if( next_distance < best_distance[ ROUNDS - 1 ] )
            {
                util::swap( next_distance, best_distance[ ROUNDS - 1 ] );
                util::swap( next_point_id, best_point_id[ ROUNDS - 1 ] );
            }
            bitonic::sort<true, ROUNDS>( best_point_id, best_distance, warp_size * ROUNDS );
            kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );
        }
        
        points_scanned_local +=(warp_size > (scan_hub_to - scan_hub_from))?(scan_hub_to - scan_hub_from) : warp_size;
        scan_hub_from += warp_size;

        if( scan_hub_from >= scan_hub_to )
        {
            if( ++hubs_processed < H )
            {   
                current_H        = iD[ hub_containing_qp * H + hubs_processed ];
                dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];
                scan_hub_from    = psum[ current_H ];
                scan_hub_to      = psum[ current_H + 1 ];

            }
        }
    }
    
    if( lane_id == 0 )
    {
        hubs_scanned[0] += hubs_processed;
        points_scanned[0] += points_scanned_local;
    }
}

template < std::size_t ROUNDS >
__global__
__launch_bounds__(32, 2)
void Query( int stream_id, int K, int* __restrict__ cell_bin_counts,
    float* __restrict__ solutions_distances,
    idx_t* __restrict__ solutions_knn,
    idx_t** __restrict__ arr_idx_arr,
    idx_t* __restrict__ sorted_global_idx,
    float** __restrict__ arr_x_arr,
    float** __restrict__ arr_y_arr,
    float** __restrict__ arr_z_arr,
    float** __restrict__ data,
    idx_t** __restrict__ dH,
    idx_t** __restrict__ assignments,
    idx_t** __restrict__ dH_psum, 
    const int* __restrict__ cell_psum,
    float** __restrict__ D_arr,
    idx_t** __restrict__ iD_arr,
    float** __restrict__ dD_arr, 
    float* __restrict__ cell_maxs,
    float* __restrict__ cell_mins,
    int* __restrict__ hubs_scanned,
    int* __restrict__ pointsScanned,
    int num_cells, int * bin_ids,
    unsigned char* __restrict__ frozen
)
{
    int const lane_id           = threadIdx.x;
    int const query_id_in_block = threadIdx.y;
    int const queries_per_block = blockDim.y;
    int const query_sequence_id = blockIdx.x * queries_per_block + query_id_in_block;

    int n_stream = cell_bin_counts[stream_id];
    
    if( query_sequence_id >= n_stream ) { return; }
    
    int const qp = arr_idx_arr[stream_id][query_sequence_id] ; //Local to cell
    int const global_qp = sorted_global_idx[cell_psum[stream_id] + qp]; 
    if(frozen[global_qp]) { return; }

    idx_t best_point_id[ROUNDS];
    float best_distance[ROUNDS];

    for (int i = 0; i < ROUNDS; i++) {
        best_point_id[i] = -1;       
        best_distance[i] = FLT_MAX;  
    }

    idx_t const lane_K  = (K - 1) % warp_size;
    idx_t const round_K = ROUNDS - 1;
    //---------Query its own cell First
    int cell = stream_id;

    float const q_x = data[cell][ qp * 3 ];
    float const q_y = data[cell][ qp * 3 +1 ];
    float const q_z = data[cell][ qp * 3 +2 ];

    float q_xyz[3] = { q_x, q_y, q_z };

    int hub_containing_qp = assignments[stream_id][qp];

    idx_t * global_idx_current_cell = sorted_global_idx + cell_psum[cell];

    query_own_cell<ROUNDS>(
        cell,
        qp,
        hub_containing_qp,
        K,
        arr_idx_arr,
        arr_x_arr,
        arr_y_arr,
        arr_z_arr,
        q_xyz,
        data,
        dH,
        global_idx_current_cell,
        dH_psum,
        best_point_id,
        best_distance,
        dD_arr,
        iD_arr,
        H,
        n_stream,
        hubs_scanned + sorted_global_idx[cell_psum[stream_id] + qp],  
        pointsScanned + sorted_global_idx[cell_psum[stream_id] + qp]
    );

    for (int i = 0; i < num_cells ; i ++ )
    {
        if(i!=stream_id)
        {
            float kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );
            if( distance_to_cell_aabb(q_xyz, &cell_mins[i * dim], &cell_maxs[i * dim], dim) > sqrt(kth_distance))
            {
                continue;
            }
            hub_containing_qp = get_assignment_another_cell(q_xyz, i, dH, data);
            global_idx_current_cell = sorted_global_idx + cell_psum[i];
            query_cell<ROUNDS>(
                i,
                qp,
                hub_containing_qp,
                K,
                arr_idx_arr,
                arr_x_arr,
                arr_y_arr,
                arr_z_arr,
                q_xyz,
                data,
                dH,
                global_idx_current_cell,
                dH_psum,
                best_point_id,
                best_distance,
                dD_arr,
                iD_arr,
                H,
                cell_bin_counts[i],
                hubs_scanned + sorted_global_idx[cell_psum[stream_id] + qp],  
                pointsScanned + sorted_global_idx[cell_psum[stream_id] + qp]
            );
        }
    }

    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r )
    {
        int const global_lane_id = r * warp_size + lane_id;
        if( global_lane_id < K )
        {
            int const neighbour_location = global_qp * K + global_lane_id;
            solutions_knn[ neighbour_location ] = best_point_id[r];
            solutions_distances[ neighbour_location ] = best_distance[r];
        }
    }
}

//--------------End of query kernels-------------------------------
//--------------Construction and Query-----------------------------
template <class R>
void C_and_Q(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, unsigned char* __restrict__ frozen, idx_t *results_knn, R *results_distances, std::string& mesh_name)
{

    int constexpr block_size = 1024;
    
    //------------------------Start running algorithm-------------
    for ( int p = 1; p <= 1; p++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        std::vector<std::string> section_names;
        std::vector<float> section_times;

        //---------------Grid-related-proprocessing----------------------------------------------------
        cudaEventRecord(start);
        int num_cells = p * p * p;
        
        int dim = 3;
        int* d_min_vals_int;
        int* d_max_vals_int;
        cudaMalloc(&d_min_vals_int, sizeof(int) * dim);
        cudaMalloc(&d_max_vals_int, sizeof(int) * dim);

        int h_init_min[3] = {
            float_to_ordered_int(FLT_MAX),
            float_to_ordered_int(FLT_MAX),
            float_to_ordered_int(FLT_MAX)
        };
        int h_init_max[3] = {
            float_to_ordered_int(-FLT_MAX),
            float_to_ordered_int(-FLT_MAX),
            float_to_ordered_int(-FLT_MAX)
        };

        cudaMemcpy(d_min_vals_int, h_init_min, sizeof(int) * dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_vals_int, h_init_max, sizeof(int) * dim, cudaMemcpyHostToDevice);

        compute_min_max<<<(n + block_size - 1) / block_size, block_size>>>(data, d_min_vals_int, d_max_vals_int, n);

        cudaMemcpy( h_init_min, d_min_vals_int, sizeof(int) * dim, cudaMemcpyDeviceToHost);
        cudaMemcpy( h_init_max, d_max_vals_int, sizeof(int) * dim, cudaMemcpyDeviceToHost);

        float min_vals[dim], max_vals[dim];
        for (int i = 0; i < dim; ++i) {
            min_vals[i] = ordered_int_to_float(h_init_min[i]);
            max_vals[i] = ordered_int_to_float(h_init_max[i]);
        }

        float* d_min_vals;
        float* d_max_vals;
        cudaMalloc(&d_min_vals, sizeof(float) * dim);
        cudaMalloc(&d_max_vals, sizeof(float) * dim);

        cudaMemcpy(d_min_vals, min_vals, sizeof(float) * dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_vals, max_vals, sizeof(float) * dim, cudaMemcpyHostToDevice);

        int* d_bin_ids, *d_bin_counts;
        cudaMalloc(&d_bin_ids, sizeof(int) * n);
        cudaMalloc(&d_bin_counts, sizeof(int) * num_cells);
        cudaMemset(d_bin_counts, 0, sizeof(int) * num_cells);

        assign_bins<<<(n + block_size - 1) / block_size, block_size>>>(data, d_min_vals, d_max_vals, d_bin_ids, d_bin_counts, n, p);

        int* d_cell_psum;
        cudaMalloc(&d_cell_psum, sizeof(int) * num_cells);
        cudaMemset(d_cell_psum, 0, sizeof(int) * num_cells);

        int h_bin_counts[num_cells];
        cudaMemcpy(h_bin_counts, d_bin_counts, sizeof(int) * num_cells, cudaMemcpyDeviceToHost);

        int h_cell_psum[num_cells];
        h_cell_psum[0] = 0;
        for (int i = 1; i < num_cells; ++i) {
            h_cell_psum[i] = h_cell_psum[i - 1] + h_bin_counts[i - 1];
        }

        cudaMemcpy(d_cell_psum, h_cell_psum, sizeof(int) * num_cells, cudaMemcpyHostToDevice);

        float* d_data;
        cudaMalloc(&d_data, sizeof(float) * dim * n);

        idx_t* d_sorted_idx;
        cudaMalloc(&d_sorted_idx, sizeof(idx_t) * n);

        int* d_cell_psum_copy;
        cudaMalloc(&d_cell_psum_copy, sizeof(int) * num_cells);
        cudaMemcpy(d_cell_psum_copy, d_cell_psum, sizeof(int) * num_cells, cudaMemcpyDeviceToDevice);

        BucketSortPoints<<<(n + block_size - 1) / block_size, block_size>>>(n, data, d_data, d_bin_ids, d_cell_psum_copy, d_sorted_idx);
        cudaFree(d_cell_psum_copy);

        std::vector<float> cell_mins, cell_maxs;
        compute_cell_bounds(min_vals, max_vals, p, dim, cell_mins, cell_maxs);

        float* d_cell_mins;
        float* d_cell_maxs;
        size_t bounds_size = cell_mins.size() * sizeof(float);
        cudaMalloc(&d_cell_mins, bounds_size);
        cudaMalloc(&d_cell_maxs, bounds_size);

        cudaMemcpy(d_cell_mins, cell_mins.data(), bounds_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cell_maxs, cell_maxs.data(), bounds_size, cudaMemcpyHostToDevice);
        //----------------End of grid proprocessing ------------------------------------------------------
        //----------------Separating into CudaStrams------------------------------------------------------
        int num_streams = num_cells;
        cudaStream_t* streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; ++i){   cudaStreamCreate(&streams[i]);  }

        R* chunk_ptrs[num_streams], *results_distances_chunks[num_streams];       
        idx_t* idx_chunks[num_streams], *results_knn_chunks[num_streams];    
        for (int i = 0; i < num_streams; ++i) {   
            chunk_ptrs[i] = d_data + dim * h_cell_psum[i]; 
            idx_chunks[i] = d_sorted_idx + h_cell_psum[i];
        }    

        time_section(section_times, "grid-related pre-processing", section_names, start, stop);
        //---------------Clover Construction for streams--------------------------------------------------
        //-----------Memory Allocations
        idx_t* dH[num_streams];
        idx_t* dH_psum[num_streams];
        idx_t* dH_psum_copy[num_streams];
        idx_t* d_psum_placeholder[num_streams];
        idx_t* dH_assignments[num_streams];

        float* distances_arr[num_streams];
        float* arr_x_arr[num_streams], *arr_y_arr[num_streams], *arr_z_arr[num_streams];
        idx_t* arr_idx_arr[num_streams];
        float* D_arr[num_streams];
        idx_t* iD_arr[num_streams];
        float* dD_arr[num_streams];

        std::vector<std::thread> threads;

        for (int i = 0; i < num_streams; ++i) {
            std::size_t n_stream = h_bin_counts[i];  
            CUDA_CALL(cudaMalloc((void**)&dH[i], sizeof(idx_t) * H));
            CUDA_CALL(cudaMalloc((void**)&dH_psum[i], sizeof(idx_t) * (H + 1)));
            CUDA_CALL(cudaMalloc((void**)&dH_psum_copy[i], sizeof(idx_t) * (H + 1)));
            CUDA_CALL(cudaMalloc((void**)&d_psum_placeholder[i], sizeof(idx_t) * (H + 1)));

            CUDA_CALL(cudaMalloc((void**)&dH_assignments[i], sizeof(idx_t) * n_stream));

            CUDA_CALL(cudaMemsetAsync(dH_psum[i], 0, sizeof(idx_t) * (H + 1), streams[i]));
            CUDA_CALL(cudaMemsetAsync(dH_psum_copy[i], 0, sizeof(idx_t) * (H + 1), streams[i]));
            CUDA_CALL(cudaMemsetAsync(d_psum_placeholder[i], 0, sizeof(idx_t) * (H + 1), streams[i]));
            CUDA_CALL(cudaMemsetAsync(dH_assignments[i], 0, sizeof(idx_t) * n_stream, streams[i]));

            idx_t batch_size = 1024 * 4 ;
            CUDA_CALL(cudaMalloc((void**)&distances_arr[i], sizeof(float) * H * batch_size));
            
            CUDA_CALL(cudaMalloc((void**)&arr_x_arr[i], sizeof(float) * n_stream));
            CUDA_CALL(cudaMalloc((void**)&arr_y_arr[i], sizeof(float) * n_stream));
            CUDA_CALL(cudaMalloc((void**)&arr_z_arr[i], sizeof(float) * n_stream));
            CUDA_CALL(cudaMalloc((void**)&arr_idx_arr[i], sizeof(idx_t) * n_stream));

            CUDA_CALL(cudaMalloc((void**)&D_arr[i], sizeof(float) * H * H));
            CUDA_CALL(cudaMalloc((void**)&iD_arr[i], sizeof(idx_t) * H * H));
            CUDA_CALL(cudaMalloc((void**)&dD_arr[i], sizeof(float) * H * H));
        }
        time_section(section_times, "MemAlloc", section_names, start, stop);
        //-----------End of Memory Allocations
        //-----------Kernels in order
        for (int i = 0; i < num_streams; ++i) {
            std::size_t n_stream = h_bin_counts[i];

            idx_t*        dH_i     = dH[i];
            cudaStream_t  stream_i = streams[i];

            bool ok = GenerateUniqueHubsAndUpload(n_stream, H, dH_i, stream_i, i);
            if (!ok) {
                // *best_p  = best_p_val;  
                // *time_ms = best_total; 
                return;  
            }
        }

        time_section(section_times, "HubSelect", section_names, start, stop);
        for (int i = 0; i < num_streams; ++i) {
            std::size_t n_stream = h_bin_counts[i]; 
            idx_t batch_size = 1024 * 4 ;
            std::size_t num_blocks = (batch_size + block_size - 1) / block_size;
            set_max_float<<<( H * H + block_size - 1 ) / block_size, block_size>>>(D_arr[i], H * H);

            int tem_block_size = 128;
            num_blocks = (batch_size + tem_block_size - 1) / tem_block_size;

            idx_t batch_number = (n_stream + batch_size -1) / batch_size;
            idx_t batch_id;
            for (batch_id = 0; batch_id < batch_number; batch_id++)
            {
                Calculate_Distances<<<num_blocks, tem_block_size, 3 * H * sizeof(float), streams[i]>>>(batch_id, batch_size, n_stream, dH[i], distances_arr[i], chunk_ptrs[i], dH_psum[i], dH_assignments[i]);
                Construct_D<<<H, block_size, sizeof(int) * H , streams[i]>>>(distances_arr[i], dH_assignments[i], batch_id, batch_size, n_stream, D_arr[i]);
            }
            cudaFree( distances_arr[i] );
            
        }
         time_section(section_times, "Calc Dist & D", section_names, start, stop);

        for (int i = 0; i < num_streams; ++i) {
            fused_prefix_sum_copy<<<1, dim3( warp_size,  warp_size, 1) , sizeof(int) * (warp_size + 1),streams[i] >>>(dH_psum[i], dH_psum_copy[i]);

            cudaMemcpy(d_psum_placeholder[i], dH_psum_copy[i], (H + 1 )* sizeof(idx_t), cudaMemcpyDeviceToDevice);

            cudaMemcpy(dH_psum_copy[i] + 1, d_psum_placeholder[i], H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dH_psum[i] + 1, d_psum_placeholder[i], H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
            cudaMemset(dH_psum[i], 0, sizeof(idx_t));
            cudaMemset(dH_psum_copy[i], 0, sizeof(idx_t));
            cudaFree(d_psum_placeholder[i]);
            
        }
        time_section(section_times, "prefix_sum", section_names, start, stop);
        for (int i = 0; i < num_streams; ++i) {
            std::size_t n_stream = h_bin_counts[i]; 
            std::size_t num_blocks = (n_stream + block_size - 1) / block_size;

            BucketSort<float><<<num_blocks, block_size, 0, streams[i]>>>(n_stream, arr_x_arr[i], arr_y_arr[i], arr_z_arr[i], arr_idx_arr[i], chunk_ptrs[i], dH_assignments[i], dH_psum_copy[i]);
            //cudaFree(dH_psum_copy);
        }
        time_section(section_times, "Bucket Sort", section_names, start, stop);
        for (int i = 0; i < num_streams; ++i) {
            fused_transform_sort_D<float, (H + block_size - 1) / block_size> <<<H, dim3 { warp_size, block_size/warp_size, 1 }, 2 * H * sizeof(float) ,streams[i]>>> (D_arr[i], iD_arr[i], dD_arr[i]);
            cudaFree(D_arr[i]); 
        }
        time_section(section_times, "Sort D", section_names, start, stop);
        //----------------End of Clover Construction--------------------------
        //----------------Pointer Transfer-----------------------
        idx_t **d_dH, **d_dH_psum, **d_dH_assignments;
        float **d_arr_x_arr, **d_arr_y_arr, **d_arr_z_arr, **d_data_chunks;
        idx_t **d_arr_idx_arr;
        float **d_D_arr;
        idx_t **d_iD_arr;
        float **d_dD_arr;

        cudaMalloc(&d_dH, sizeof(idx_t*) * num_streams);
        cudaMalloc(&d_dH_psum, sizeof(idx_t*) * num_streams);
        cudaMalloc(&d_dH_assignments, sizeof(idx_t*) * num_streams);
        
        cudaMalloc(&d_arr_x_arr, sizeof(float*) * num_streams);
        cudaMalloc(&d_arr_y_arr, sizeof(float*) * num_streams);
        cudaMalloc(&d_arr_z_arr, sizeof(float*) * num_streams);
        cudaMalloc(&d_arr_idx_arr, sizeof(idx_t*) * num_streams);
        cudaMalloc(&d_data_chunks, sizeof(float*) * num_streams);

        cudaMalloc(&d_D_arr, sizeof(float*) * num_streams);
        cudaMalloc(&d_iD_arr, sizeof(idx_t*) * num_streams);
        cudaMalloc(&d_dD_arr, sizeof(float*) * num_streams);

        cudaMemcpy(d_dH, dH, sizeof(idx_t*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dH_psum, dH_psum, sizeof(idx_t*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dH_assignments, dH_assignments, sizeof(idx_t*) * num_streams, cudaMemcpyHostToDevice);

        cudaMemcpy(d_arr_x_arr, arr_x_arr, sizeof(float*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr_y_arr, arr_y_arr, sizeof(float*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr_z_arr, arr_z_arr, sizeof(float*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr_idx_arr, arr_idx_arr, sizeof(idx_t*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data_chunks, chunk_ptrs, sizeof(float*) * num_streams, cudaMemcpyHostToDevice);

        cudaMemcpy(d_D_arr, D_arr, sizeof(float*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_iD_arr, iD_arr, sizeof(idx_t*) * num_streams, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dD_arr, dD_arr, sizeof(float*) * num_streams, cudaMemcpyHostToDevice);
        time_section(section_times, "Pointer Transfer", section_names, start, stop);
        //----------------End of Pointer Transfer-----------------------
        std::vector<int> hubsScanned(n, 0);
        std::vector<int> pointsScanned(n, 0);
        int * d_hubsScanned, * d_pointsScanned;
        CUDA_CALL(cudaMalloc((void **) &d_hubsScanned, sizeof(int)* n));
        CUDA_CALL(cudaMalloc((void **) &d_pointsScanned, sizeof(int)* n));
        cudaMemset(d_hubsScanned, 0, sizeof(int)* n);
        cudaMemset(d_pointsScanned, 0, sizeof(int)* n);

        time_section(section_times, "Log MemAlloc", section_names, start, stop);
        //--------------Query Launch------------------------------
        
        for (int i = 0; i < num_streams; ++i) {
            std::size_t n_stream = h_bin_counts[i];  
            std::size_t constexpr queries_per_block = 32 / warp_size;
            int num_blocks = util::CEIL_DIV(n_stream, queries_per_block);

            switch (util::CEIL_DIV(k, warp_size))
            {   
                case 1: { Query<1> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 2: { Query<2> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 3: { Query<3> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 4: { Query<4> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 5: { Query<5> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 6: { Query<6> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 7: { Query<7> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                case 8: { Query<8> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }, 0, streams[i]>>>(
                    i, k, d_bin_counts,
                    results_distances, results_knn, 
                    d_arr_idx_arr, d_sorted_idx, d_arr_x_arr, d_arr_y_arr, d_arr_z_arr, d_data_chunks, 
                    d_dH, d_dH_assignments, d_dH_psum, d_cell_psum,
                    d_D_arr, d_iD_arr, d_dD_arr,
                    d_cell_maxs, d_cell_mins,
                    d_hubsScanned, d_pointsScanned,
                    num_cells, d_bin_ids, frozen
                ); } break;
                default: assert(false && "Rounds required to fulfill k value will exceed thread register allotment.");
            }
        }
        time_section(section_times, "Query", section_names, start, stop);
        //-----------------Memory Free------------------------
        for (int i = 0; i < num_streams; ++i) {
            cudaFree(dH[i]);
            cudaFree(dH_psum[i]);
            cudaFree(dH_assignments[i]);
            cudaFree(arr_x_arr[i]);
            cudaFree(arr_y_arr[i]);
            cudaFree(arr_z_arr[i]);
            cudaFree(arr_idx_arr[i]);
            cudaFree(iD_arr[i]);
            cudaFree(dD_arr[i]);
            cudaStreamDestroy(streams[i]);
        }

        cudaFree(d_dH);
        cudaFree(d_dH_psum);
        cudaFree(d_dH_assignments);
        cudaFree(d_arr_x_arr);
        cudaFree(d_arr_y_arr);
        cudaFree(d_arr_z_arr);
        cudaFree(d_D_arr);
        cudaFree(d_iD_arr);
        cudaFree(d_dD_arr);
        cudaFree(d_hubsScanned);
        cudaFree(d_pointsScanned);
        cudaFree(d_min_vals_int);
        cudaFree(d_max_vals_int);
        cudaFree(d_min_vals);
        cudaFree(d_max_vals);
        cudaFree(d_bin_ids);
        cudaFree(d_bin_counts);
        cudaFree(d_cell_psum);
        cudaFree(d_data);
        cudaFree(d_sorted_idx);
        cudaFree(d_cell_mins);
        cudaFree(d_cell_maxs);
        time_section(section_times, "Pointer free", section_names, start, stop);
        //-----------------End------------------------
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float construction = 0.0f, query = 0.0f, total = 0.0f;

        for (size_t i = 0; i < section_names.size(); ++i) {
            if (section_names[i] != "Log" && section_names[i] != "Log MemAlloc" && section_names[i] != "Query")
                construction += section_times[i];
            else if (section_names[i] == "Query" )
                query += section_times[i];
            if(section_names[i] != "Log" && section_names[i] != "Log MemAlloc")
                total += section_times[i];
        }
    }
}
}