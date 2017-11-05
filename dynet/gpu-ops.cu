#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#include "dynet/gpu-kernels.h"
#include "dynet/functors.h"

#include <algorithm>				// For std::min and std::max

namespace dynet {
namespace gpu {

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_dense_to_sparse_assign(int n, const unsigned int *idx, float *src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    trg[idx[id]] = src[id];
}

void dense_to_sparse_assign(int n, const unsigned int *idx, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
      ker_dense_to_sparse_assign<<<tb.first, tb.second>>>(
          std::min(total_size, n-curr_pos), idx+curr_pos, src+curr_pos, trg);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_sparse_to_dense_assign(int n, const unsigned int *idx, float *src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    trg[id] = src[idx[id]];
}

void sparse_to_dense_assign(int n, const unsigned int *idx, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
      ker_sparse_to_dense_assign<<<tb.first, tb.second>>>(
          std::min(total_size, n-curr_pos), idx+curr_pos, src, trg+curr_pos);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_dense_to_sparse_subtract(int n, const unsigned int *idx, float *src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    atomicAdd(trg + idx[id], -src[id]);
}

void dense_to_sparse_subtract(int n, const unsigned int *idx, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
      ker_dense_to_sparse_subtract<<<tb.first, tb.second>>>(
          std::min(total_size, n-curr_pos), idx+curr_pos, src+curr_pos, trg);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_sparse_to_dense_block_assign_and_multiply(int n, const unsigned *idx, int bsize, float mult, float* src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n*bsize)
    trg[id] = src[idx[id/bsize]*bsize+id%bsize] * mult;
}

void sparse_to_dense_block_assign_and_multiply(int n, const unsigned *idx, int bsize, float mult, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n*bsize);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size/bsize)
      ker_sparse_to_dense_block_assign_and_multiply<<<tb.first, tb.second>>>(
          std::min(total_size/bsize, n-curr_pos),
          idx+curr_pos, bsize, mult, src, trg+curr_pos*bsize);
  }
}

// CUDA kernel. Each thread takes care of one row copy.
__global__ void ker_parallel_memcpy(int num_seqs, float **src, float **trg,
                                    std::size_t *len) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  int seq_id = id % num_seqs;
  int i = id / num_seqs;
  if (i < len[seq_id])
    trg[seq_id][i] = src[seq_id][i];

  __syncthreads();
}

void parallel_memcpy(int num_seqs, int max_len, float **src, float **trg,
                     std::size_t *len) {
  if(num_seqs > 0) {
    auto tb = SizeToBlockThreadPair(num_seqs * max_len);
    for (int pos = 0; pos < num_seqs; pos += tb.second) {
      ker_parallel_memcpy<<<tb.first, tb.second>>>(
		      min(tb.second, num_seqs - pos),
		      src + pos,
		      trg + pos,
		      len + pos);
    }
  }
}

// CUDA kernel. Each thread takes care of one row copy.
__global__ void ker_parallel_accumulate(int num_seqs,
                                        float **src,
                                        float **trg,
                                        std::size_t *len) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  int seq_id = id % num_seqs;
  int i = id / num_seqs;
  if (i < len[seq_id])
    atomicAdd(&trg[seq_id][i], src[seq_id][i]);

  __syncthreads();
}

void parallel_accumulate(int num_seqs,
                         int max_len,
                         float **src,
                         float **trg,
                         std::size_t *len) {
  if(num_seqs > 0) {
    auto tb = SizeToBlockThreadPair(num_seqs*max_len);
    ker_parallel_accumulate<<<tb.first, tb.second>>>(num_seqs, src, trg, len);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_dense_to_sparse_block_add(int n, const unsigned *idx, int bsize, float* src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n*bsize)
    atomicAdd(trg + idx[id/bsize]*bsize+id%bsize, src[id]);
}

void dense_to_sparse_block_add(int n, const unsigned *idx, int bsize, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n*bsize);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size/bsize)
      ker_dense_to_sparse_block_add<<<tb.first, tb.second>>>(
          std::min(total_size/bsize, n-curr_pos),
          idx+curr_pos, bsize, src+curr_pos*bsize, trg);
  }
}

} // namespace gpu
} // namespace dynet
