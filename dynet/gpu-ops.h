#ifndef DYNET_GPU_OPS_H
#define DYNET_GPU_OPS_H

#include <cstddef>

namespace dynet {
namespace gpu {

void dense_to_sparse_assign(int n, const unsigned int* ids, float* src, float* trg);
void sparse_to_dense_assign(int n, const unsigned int* ids, float* src, float* trg);
void dense_to_sparse_subtract(int n, const unsigned int* ids, float* src, float* trg);
void sparse_to_dense_block_assign_and_multiply(int n, const unsigned *idx, int bsize, float mult, float *src, float *trg);
void dense_to_sparse_block_add(int n, const unsigned* ids, int bsize, float* src, float* trg);
void parallel_memcpy(int num_seqs, int max_len, float **src, float **trg, std::size_t *len);
void parallel_accumulate(int num_seqs, int max_len, float **src, float **trg, std::size_t *len);
} // namespace gpu
} // namespace dynet

#endif
