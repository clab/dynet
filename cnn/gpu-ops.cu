#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#include "cnn/gpu-kernels.h"
#include "cnn/functors.h"

namespace cnn {
namespace gpu {

// CUDA kernel. Each thread takes care of one element of c
__global__ void sparse_assign(int n, unsigned int *idx, float *src, float *trg)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        trg[idx[id]] = src[id];
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void const_init(int n, float val, float *trg)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        trg[id] = val;
}

} // namespace gpu
} // namespace cnn
