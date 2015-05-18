#ifndef CNN_GPU_OPS_H
#define CNN_GPU_OPS_H

namespace cnn {
namespace gpu {

void vtanh(int n, float* x, float* y);
void vtanh_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void sqeucdist(int n, float* x, float *y, float* res);
void sqeucdist_backward(int n, const float* dEdy, const float* x0, const float* x1, float* dEdx, int i);

void sgd_update(int n, const float* g, float* x, float scale, float lambda);

} // namespace gpu
} // namespace cnn

#endif
