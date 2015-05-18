#ifndef CNN_GPU_OPS_H
#define CNN_GPU_OPS_H

namespace cnn {
namespace gpu {

void vnegate(int n, float* x, float* y);
void vnegate_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void vrelu(int n, float* x, float* y);
void vrelu_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void vtanh(int n, float* x, float* y);
void vtanh_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void vlogistic(int n, float* x, float* y);
void vlogistic_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void sqeucdist(int n, float* x, float *y, float* res);
void sqeucdist_backward(int n, const float* dEdy, const float* x0, const float* x1, float* dEdx, int i);

void sgd_update(int n, const float* g, float* x, float scale, float lambda);

} // namespace gpu
} // namespace cnn

#endif
