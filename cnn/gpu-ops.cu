#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#include "cnn/gpu-kernels.h"
#include "cnn/functors.h"

namespace cnn {
namespace gpu {

// this wraps kernel dispatches for various operations (preventing us from
// having to compile a version of nodes.cc with NVCC)

void vpairwise_rank_loss(int n, float margin, const float* xgood, const float* xbad, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprKernel<<<tb.first, tb.second>>>(n, xgood, xbad, y, FPairwiseRankLoss(margin));
}

void vpairwise_rank_loss_backward(int n, bool d_wrt_correct, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  if (d_wrt_correct) {
    accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyNegateBackward());
  } else {
    accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyBackward());
  }
}

void vcwise_product(int n, const float* x0, const float* x1, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprKernel<<<tb.first, tb.second>>>(n, x0, x1, y, FProduct());
}

void vcwise_product_backward(int n, const float* dEdy, const float* x_other, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, dEdy, x_other, dEdx, FProduct());
}

void vconstant_minusx(int n, float c, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FConstantMinus(c));
}

void vnegate(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FNegate());
}

void vnegate_backward(int n, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accUnaryExprKernel<<<tb.first, tb.second>>>(n, dEdf, dEdx, FNegate());
}

void vrelu(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FRectify());
}

void vrelu_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyBackward());
}

void vtanh(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FTanh());
}

void vtanh_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FTanhBackward());
}

void vlogistic(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FLogisticSigmoid());
}

void vlogistic_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FLogisticSigmoidBackward());
}

void sqeucdist_backward(int n, const float* dEdy, const float* x0, const float* x1, float* dEdx, int i) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, x0, x1, dEdx, FEuclideanBackward(i, dEdy));
}

void sgd_update(int n, const float* g, float* x, float scale, float lambda) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, x, g, x, FL2SGDUpdate(lambda, scale));
}

//// BROKEN STUFF UNDER HERE ///////////////////////////////////

void sqeucdist(int n, const float* x0, const float *x1, float* y) {
  slowReduceKernel<<<1,1>>>(n, x0, x1, y, FSqDist());
}

} // namespace gpu
} // namespace cnn
