#include "dynet/nodes-contract.h"

#include <limits>
#include <cmath>
#include <stdexcept>

#include "dynet/nodes-macros.h"
#include "dynet/nodes.h"

// This file takes a long time to compile on GPU. Uncomment this line to skip it.
// #define DYNET_SKIP_CUDA_CONTRACTIONS


#if defined(__CUDACC__) && !defined(DYNET_SKIP_CUDA_CONTRACTIONS)
#include "dynet/nodes.cc"
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif


using namespace std;

namespace dynet {

#ifndef __CUDACC__

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dot(" << arg_names[0] << "," << arg_names[1] << ')';
  if (arg_names.size() == 3) s << " + " << arg_names[2];
  return s.str();
}

Dim InnerProduct3D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 && xs.size() != 3)
    throw std::invalid_argument("Expected two or three arguments in InnerProduct3D_1D");
  if (xs[0].ndims() != 3 ||
      !LooksLikeVector(xs[1]) ||
      xs[0].size(2) != xs[1].size(0)) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  Dim d({xs[0].size(0), xs[0].size(1)}, max(xs[0].bd, xs[1].bd));
  if (xs.size() == 3) d.bd = max(d.bd, xs[2].bd);
  if (xs.size() == 3 && xs[2].single_batch() != d.single_batch()) {
    ostringstream s; s << "Bad bias dimensions in InnerProduct3D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  return d;
}

string InnerProduct3D_1D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dotdot(" << arg_names[0] << "," << arg_names[1] << "," << arg_names[2] << ')';
  if (arg_names.size() == 4) s << " + " << arg_names[3];
  return s.str();
}

Dim InnerProduct3D_1D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 3 && xs.size() != 4)
    throw std::invalid_argument("Expected three or four arguments in InnerProduct3D_1D");
  if (xs[0].ndims() != 3 ||
      !LooksLikeVector(xs[1]) ||
      !LooksLikeVector(xs[2])) {
    // TODO fix add check
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  Dim d({xs[0].size(0)}, max(max(xs[0].bd, xs[1].bd), xs[2].bd));
  if (xs.size() == 4) d.bd = max(d.bd, xs[3].bd);
  if (xs.size() == 4 && xs[3] != d) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  return d;
}

#endif

//   Y_ij = A_ijk * B_k (+ C_ij)
template<class MyDevice>
void InnerProduct3D_1D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D::forward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
  if (xs.size() == 3) {
    auto C = xs[2]->tb<2>();
    Eigen::array<int, 3> bcast_C = {1, 1, (int)(xs[2]->d.bd == 1 ? fx.d.bd : 1)};
    fx.tb<2>().device(*dev.edevice) = C.broadcast(bcast_C);
  }
#if defined(__CUDACC__) && !defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  // Reshape xs[0] to a matrix
  Dim new_xs0_d({xs[0]->d[0] * xs[0]->d[1], xs[0]->d[2]}, xs[0]->d.bd);
  const Tensor new_xs0(new_xs0_d, xs[0]->v, xs[0]->device, xs[0]->mem_pool);
  // Reshape fx to a vector
  Dim new_fx_d({fx.d[0] * fx.d[1]}, fx.d.bd);
  Tensor new_fx(new_fx_d, fx.v, fx.device, fx.mem_pool);
  // CUDA matrix multiply ftw
  CUDAMatrixMultiply(dev, new_xs0, *xs[1], new_fx, kSCALAR_ONE);
#else
  if (xs[0]->d.bd == 1) {  // A is a 3 tensor
    Eigen::array<int, 2> bcast_b = {1, (int)(xs[1]->d.bd == 1 ? fx.d.bd : 1)};
    auto b = xs[1]->tb<1>();
    auto A = xs[0]->t<3>();
    fx.tb<2>().device(*dev.edevice) += A.contract(b.broadcast(bcast_b), dims);
  } else { // A is a 4 tensor
    auto A = xs[0]->tb<3>();
    if (xs[1]->d.bd == 1) { // b is a 1 tensor
      auto b = xs[1]->t<1>();
      fx.tb<2>().device(*dev.edevice) += A.contract(b, dims);
    } else {
      // If both A and b are batched loop over batches
      auto b = xs[1]->tb<1>();
      for (unsigned i = 0; i < fx.d.bd; ++i) {
        auto b_ = b.chip<1>(i);
        fx.tb<2>().chip<2>(i).device(*dev.edevice) += A.chip<3>(i).contract(b_, dims);
      }
    }
  }
#endif
#endif
}

template<class MyDevice>
void InnerProduct3D_1D::backward_dev_impl(const MyDevice & dev,
    const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D::backward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto tdEdf = dEdf.tb<2>();  // 2 tensor
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  if (i == 0) {
    if (xs[0]->d.bd == 1) { // A is a 3 tensor
      // tensor product
#if defined(__CUDACC__) && !defined(DYNET_SKIP_CUDA_CONTRACTIONS)
      if (dEdf.d.bd == xs[1]->d.bd) {
        // Basically here dEdxi[i,j,k] = \sum_b dEdf[i,j,b] * B[k,b]
        // Which we do as matrix multiplication dEdxi[i*j, k] = \sum_b dEdf[i*j,b] * B^T[b,k]
        // CUDA matrix multiply ftw
        CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 dEdxi.d[0] * dEdxi.d[1], dEdxi.d[2] , dEdf.d.bd,
                                 kSCALAR_ONE,
                                 dEdf.v, dEdf.d.batch_size(),
                                 xs[1]->v, dEdxi.d[2],
                                 kSCALAR_ONE, dEdxi.v, dEdxi.d[0] * dEdxi.d[1]));
      } else {
        // This is the tricky case where dEdf is batched but not xs[1] (ie xs[2] is batched) is not supported
        // Iterate over the batches of dEdf and then do an outer product beween flattened dEdf and xs[1]
        // and accumulate the result in dEdxi
        for (unsigned b = 0; b < dEdf.d.bd; b++) {
          CUBLAS_CHECK(cublasSger(dev.cublas_handle,
                                  dEdxi.d[0] * dEdxi.d[1], dEdxi.d[2] ,
                                  kSCALAR_ONE,
                                  dEdf.batch_ptr(b), 1,
                                  xs[1]->v, 1,
                                  dEdxi.v, dEdxi.d[0] * dEdxi.d[1]));
        }
      }
#else
      auto b = xs[1]->tb<1>();
      Eigen::array<int, 2> bcast_b = {1, (int)(xs[1]->d.bd == 1 ? fx.d.bd : 1)};
      Eigen::array<DimPair, 1> dims({{DimPair(2, 1)}});
      dEdxi.t<3>().device(*dev.edevice) += tdEdf.contract(b.broadcast(bcast_b), dims);
#endif
    } else {
      if (xs[1]->d.bd == 1) {
        // auto b = xs[1]->t<1>();
        // Eigen::array<int, 4> morph {dEdf.d[0], dEdf.d[1], xs[1]->d[0], dEdf.d.bd};
        // dEdxi.tb<3>().device(*dev.edevice) += tdEdf.contract(b, Eigen::array<DimPair, 0> {{}}).reshape(morph);
        auto b = xs[1]->t<1>();
        for (unsigned i = 0; i < fx.d.bd; ++i) {
          dEdxi.tb<3>().chip<3>(i).device(*dev.edevice) += tdEdf.chip<2>(i).contract(b, Eigen::array<DimPair, 0> {{}});
        }
      } else {
        auto b = xs[1]->tb<1>();
        for (unsigned i = 0; i < fx.d.bd; ++i) {
          dEdxi.tb<3>().chip<3>(i).device(*dev.edevice) += tdEdf.chip<2>(i).contract(b.chip<1>(i), Eigen::array<DimPair, 0> {{}});
        }

      }
    }
  } else if (i == 1) {
#if defined(__CUDACC__) && !defined(DYNET_SKIP_CUDA_CONTRACTIONS)
    if (xs[0]->d.bd == 1) {

      CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               dEdxi.d.rows(), dEdxi.d.batch_elems(), dEdf.d.batch_size(),
                               kSCALAR_ONE,
                               xs[0]->v, xs[0]->d[0] * xs[0]->d[1],
                               dEdf.v, dEdf.d.batch_size(),
                               kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      const float* dEdfv = dEdf.v;
      for (unsigned b = 0; b < xs[0]->d.bd; ++b) {
        if (dEdf.d.bd > 1) {
          dEdfv = dEdf.batch_ptr(b);
        }
        CUBLAS_CHECK(cublasSgemv(dev.cublas_handle, CUBLAS_OP_T,
                                 dEdf.d.batch_size(), dEdxi.d.rows(),
                                 kSCALAR_ONE,
                                 xs[0]->batch_ptr(b), dEdf.d.batch_size(),
                                 dEdfv, 1,
                                 kSCALAR_ONE, dEdxi.batch_ptr(b), 1));
      }
    }
#else
    if (xs[1]->d.bd == 1) { // b is a 1 tensor
      if (xs[0]->d.bd == 1) {
        auto A = xs[0]->t<3>();  // A is 3 tensor
        Eigen::array<int, 1> red_axis; red_axis[0] = 0;
        Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});
        dEdxi.t<1>().device(*dev.edevice) += tdEdf.contract(A, dims).sum(red_axis);
      } else {
        auto A = xs[0]->tb<3>();  // A is 4 tensor
        Eigen::array<DimPair, 3> dims({{DimPair(0, 0), DimPair(1, 1), DimPair(2, 3)}});
        dEdxi.t<1>().device(*dev.edevice) += tdEdf.contract(A, dims);
      }
    } else { // b is a 2 tensor
      if (xs[0]->d.bd == 1) {
        auto A = xs[0]->t<3>();  // A is 3 tensor
        Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});
        dEdxi.tb<1>().device(*dev.edevice) += A.contract(tdEdf, dims);
      } else {
        auto A = xs[0]->tb<3>();  // A is 4 tensor
        Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});
        for (unsigned i = 0; i < fx.d.bd; ++i) {
          dEdxi.tb<1>().chip<1>(i).device(*dev.edevice) += tdEdf.chip<2>(i).contract(A.chip<3>(i), dims);
        }
      }
    }
#endif
  } else if (i == 2) {
    if (xs[2]->d.bd == 1) {
      Eigen::array<int, 1> red_axis; red_axis[0] = 2;
      dEdxi.t<2>().device(*dev.edevice) += tdEdf.sum(red_axis);
    } else {
      dEdxi.tb<2>().device(*dev.edevice) += tdEdf;
    }
  } else {
    throw std::runtime_error("Illegal configuration in InnerProduct3D");
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(InnerProduct3D_1D)

//   Y_ij = A_ijk * B_k * C_j (+ D_i)
template<class MyDevice>
void InnerProduct3D_1D_1D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D_1D::forward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto A = xs[0]->t<3>();
  auto b = xs[1]->t<1>();
  auto c = xs[2]->t<1>();
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
  Eigen::array<DimPair, 1> dims2({{DimPair(1, 0)}});
  if (xs.size() == 3) {
    fx.t<1>().device(*dev.edevice) = A.contract(b, dims).contract(c, dims2);
  } else {
    auto d = xs[3]->t<1>();
    fx.t<1>().device(*dev.edevice) = A.contract(b, dims).contract(c, dims2) + d;
  }
#endif
}

template<class MyDevice>
void InnerProduct3D_1D_1D::backward_dev_impl(const MyDevice & dev,
    const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D_1D::backward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto tdEdf = dEdf.t<1>();  // vector
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  if (i == 0) { // 3 tensor
    // tensor product
    auto b = xs[1]->t<1>();
    auto c = xs[2]->t<1>();
    dEdxi.t<3>().device(*dev.edevice) += tdEdf.contract(c, Eigen::array<DimPair, 0> {{}}).contract(b, Eigen::array<DimPair, 0> {{}});
  } else if (i == 1) { // vector 1
    // TODO these should be reorganized so the contraction is first with tdEdf and then with c or b.
    // in theory, that intermediate result could be cached (although DYNET doesn't support this). the fact that it
    // this part of the product is redone when i=1 and again when i=2 is probably why this is slower
    // (or maybe it's the contract implementation?)
    Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});
    Eigen::array<DimPair, 1> dims2({{DimPair(0, 0)}});
    auto A = xs[0]->t<3>();
    auto c = xs[2]->t<1>();
    dEdxi.t<1>().device(*dev.edevice) += A.contract(c, dims).contract(tdEdf, dims2);
  } else if (i == 2) { // vector 2
    Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
    Eigen::array<DimPair, 1> dims2({{DimPair(0, 0)}});
    auto A = xs[0]->t<3>();
    auto b = xs[1]->t<1>();
    dEdxi.t<1>().device(*dev.edevice) += A.contract(b, dims).contract(tdEdf, dims2);
  } else if (i == 3) { // vector bias
    dEdxi.t<1>().device(*dev.edevice) += tdEdf;
  } else {
    throw std::runtime_error("Illegal configuration in InnerProduct3D");
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(InnerProduct3D_1D_1D)


} // namespace dynet
