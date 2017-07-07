#include "dynet/nodes.h"

#include <limits>
#include <cmath>
#include <stdexcept>

#include "dynet/simd-functors.h"
#include "dynet/functors.h"
#include "dynet/nodes-macros.h"
#include "dynet/globals.h"

#ifdef __CUDACC__
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

inline string print_vec(const std::vector<float> & vec) {
  string sep = "[";
  ostringstream oss;
  for(auto f : vec) {
    oss << sep << f; sep = ",";
  }
  oss << "]";
  return oss.str();
}

// notes on implementing differentiable components
// 1) fx can be understood as a pointer to the (preallocated) location for the result
//    of forward to be stored
// 2) fx is not initialized, so after calling forward fx must point to the correct answer
// 3) fx can be repointed to an input, if forward(x) evaluates to x (e.g., in reshaping)
// 4) dEdxi MUST **ACCUMULATE** a result since multiple calls to forward may depend on
//    the same x_i. Even, e.g., Identity must be implemented as
//    dEdx1 += dEdf. THIS IS EXTREMELY IMPORTANT
// 5) scalars results of forward are placed in fx.v[0]
// 6) DYNET manages its own memory, not Eigen, and it is configured with the
//    EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate
//    memory, it is (probably) because of an implicit creation of a temporary variable.
//    To tell Eigen this is not necessary, the noalias() method is available. If you really
//    do need a temporary variable, its capacity must be requested by Node::aux_storage_size
//
// notes on debugging problems with differentiable components
// 1) fx is uninitialized when forward is called- are you relying on it being 0?
// 2) dEdxi must accummulate (see point 4 above!)
//

namespace dynet {

// ======= Shared definitions
#define MAX_LOG_SUM_EXP 65536
#define MAX_SPARSEMAX_LOSS_ROWS 65536

// ======= Functions to be compiled on only CPU
#ifndef __CUDACC__

// set use_cholesky if M is symmetric - it's faster and more stable
// for dep paring it won't be
template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;
  if (use_cholesky) {
    LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
    auto& U = chol.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
  } else {
    PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
    auto& LU = lu.matrixLU();
    Scalar c = lu.permutationP().determinant(); // -1 or 1
    for (unsigned i = 0; i < LU.rows(); ++i) {
      const auto& lii = LU(i,i);
      if (lii < Scalar(0)) c *= -1;
      ld += log(abs(lii));
    }
    ld += log(c);
  }
  return ld;
}

template <class T>
EIGEN_STRONG_INLINE real logsumexp(const T& x, const vector<unsigned>& denom) {
  real m = x(denom[0],0);
  for (auto i : denom) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (auto i : denom)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

// ===== Auxiliary functions

size_t BlockDropout::aux_storage_size() const {
  // we just need to remember whether this entire block is turned on (1.0) or off (0.0)
  return 1 * sizeof(float);
}

size_t Dropout::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

size_t DropoutDim::aux_storage_size() const {
  return (dim.size() / dim[dimension]) * sizeof(float);
}

size_t DropoutBatch::aux_storage_size() const {
  return dim.batch_elems() * sizeof(float);
}

size_t GaussianNoise::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

size_t LogSoftmax::aux_storage_size() const {
  return 2 * dim.size() / dim.rows() * sizeof(float);
}

// this i need to do something better, but this is a work-around
// if this is too small, just make it bigger
size_t LogSumExp::aux_storage_size() const {
  return (MAX_LOG_SUM_EXP + 1) * sizeof(float);
}

size_t Max::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

size_t Min::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

size_t Softmax::aux_storage_size() const {
  return 2 * dim.size() / dim.rows() * sizeof(float);
}

size_t Sparsemax::aux_storage_size() const {
  return (dim.size() + 1) * sizeof(float);
}

size_t SparsemaxLoss::aux_storage_size() const {
  // first dim.size dimensions is the sparsemax
  const unsigned rows = MAX_SPARSEMAX_LOSS_ROWS;  // this should be xs[0]->d.rows()
  return rows * sizeof(float);
}

size_t MaxDimension::aux_storage_size() const {
  return sizeof(Eigen::DenseIndex) * dim.size();
}

size_t MinDimension::aux_storage_size() const {
  return sizeof(Eigen::DenseIndex) * dim.size();
}

#endif // Finish CPU only functions

// ===== Functions to be compiled on both CPU and GPU

template<class MyDevice>
void AddVectorToAllColumns::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  // Broadcasting is slow on CPU, so split codepaths
#ifdef __CUDACC__
  if(xs[0]->d.bd >= xs[1]->d.bd) {
    Eigen::array<int, 3> bcasts = {1, (int)xs[0]->d[1], (int)(xs[0]->d.bd/xs[1]->d.bd)};
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<2>() + xs[1]->tb<2>().broadcast(bcasts);
  } else {
    DYNET_ASSERT(xs[0]->d.bd == 1,
                 "Bad dimensions in AddVectorToAllColumns::forward: " << xs[0]->d << ", " << xs[1]->d);
    Eigen::array<int, 3> bcasts0 = {1, 1, (int)xs[1]->d.bd};
    Eigen::array<int, 3> bcasts1 = {1, (int)xs[0]->d[1], 1};
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<2>().broadcast(bcasts0) + xs[1]->tb<2>().broadcast(bcasts1);
  }
#else
  // First, add the matrix
  if(xs[0]->d.bd == fx.d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  else
    for(size_t b = 0; b < fx.d.bd; ++b)
      fx.tbvec().chip<1>(b).device(*dev.edevice) = xs[0]->tvec();
  // Second, add the columns
  if(xs[1]->d.bd == fx.d.bd) {
    for(size_t i = 0; i < xs[0]->d[1]; ++i) 
      fx.tb<2>().chip<1>(i).device(*dev.edevice) += xs[1]->tb<1>();
  } else {
    for(size_t b = 0; b < fx.d.bd; ++b)
      for(size_t i = 0; i < fx.d[1]; ++i) 
        fx.tb<2>().chip<2>(b).chip<1>(i).device(*dev.edevice) += xs[1]->t<1>();
  }
#endif
}

template<class MyDevice>
void AddVectorToAllColumns::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in AddVetorToAllColumns::backward");
  // TODO: profile on CPU and see whether the chip version is better
  if (i == 0) { // x
    if(dEdf.d.bd == dEdxi.d.bd) {
      dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
    } else {
      Eigen::array<int, 1> red_axis = {2};
      dEdxi.t<2>().device(*dev.edevice) += dEdf.tb<2>().sum(red_axis);
    }
  } else { // bias
    if(dEdf.d.bd == dEdxi.d.bd) {
      Eigen::array<int, 1> red_axis = {1};
      dEdxi.tb<1>().device(*dev.edevice) += dEdf.tb<2>().sum(red_axis);
    } else {
      DYNET_ASSERT(dEdxi.d.bd == 1,
                   "Bad dimensions in AddVectorToAllColumns::backward: " << xs[0]->d << ", " << xs[1]->d);
      Eigen::array<int, 2> red_axis = {1,2};
      dEdxi.t<1>().device(*dev.edevice) += dEdf.tb<2>().sum(red_axis);
    }
  }
}  
DYNET_NODE_INST_DEV_IMPL(AddVectorToAllColumns)

template<class MyDevice>
void Average::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
    return;
  }
  if (num_args == 2 && xs[0]->d.bd == xs[1]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec();
  else if (num_args == 3 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec();
  else if (num_args == 4 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd && xs[2]->d.bd == xs[3]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();
  else {
    bool allSameBatchSize = std::all_of(xs.begin(), xs.end(), [&](const Tensor* x) { return x->d.bd == xs[0]->d.bd;});
    if (allSameBatchSize) {
      // Since they are all the same batch size, we can easily unroll the addition (results in lower GPU latency by merging multiple adds together in one CUDA call):
      DYNET_ASSERT(num_args > 4, "Bad loop unrolling in Sum::forward");        // If it was <=4, we would have handled it in the special cases above
      fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();

      const unsigned remainder = (num_args - 4 ) % 4;
      switch (remainder) {
        case 0: break;
        case 1: fx.tvec().device(*dev.edevice) += xs[4]->tvec(); break;
        case 2: fx.tvec().device(*dev.edevice) += xs[4]->tvec() + xs[5]->tvec(); break;
        case 3: fx.tvec().device(*dev.edevice) += xs[4]->tvec() + xs[5]->tvec() + xs[6]->tvec(); break;
      }
      for (unsigned i = 4 + remainder; i < num_args; i += 4)
        fx.tvec().device(*dev.edevice) += xs[i]->tvec() + xs[i + 1]->tvec() + xs[i + 2]->tvec() + xs[i + 3]->tvec();
    }
    else {
      // Not all the same batch size, so need to broadcast in the cases where they differ
      TensorTools::zero(fx);
#ifdef __CUDACC__
      Eigen::array<int, 2> bcast({ 1, (int)fx.d.bd });
#endif
      for (unsigned i = 0; i < num_args; ++i) {
        if (xs[i]->d.bd == fx.d.bd) {
          fx.tvec().device(*dev.edevice) += xs[i]->tvec();
        }
        else {
#ifdef __CUDACC__
          fx.tbvec().device(*dev.edevice) += xs[i]->tbvec().broadcast(bcast);
#else
          for (unsigned b = 0; b < fx.d.bd; ++b)
            fx.tbvec().chip<1>(b).device(*dev.edevice) += xs[i]->tvec();
#endif
        }
      }
    }
  }
  fx.tvec().device(*dev.edevice) = fx.tvec() / (float)xs.size();
}

template<class MyDevice>
void Average::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += (dEdf.tvec() / (float)xs.size());
}
DYNET_NODE_INST_DEV_IMPL(Average)

template<class MyDevice>
void Concatenate::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned curr_row = 0;
  src_indices.resize(xs.size());
  Eigen::DSizes<ptrdiff_t, 5> indices(0,0,0,0,0);
  Eigen::DSizes<ptrdiff_t, 5> sizes(fx.d[0], fx.d[1], fx.d[2], fx.d[3],static_cast<ptrdiff_t>(fx.d.bd));
  for (unsigned i = 0; i < xs.size(); ++i) {
    indices[dimension] = src_indices[i] = curr_row;
    const unsigned row_size = xs[i]->d[dimension];
    sizes[dimension] = row_size;
    if(fx.d.bd == xs[i]->d.bd) {
      fx.tb<4>().slice(indices, sizes).device(*dev.edevice) = xs[i]->tb<4>();
    } else {
      Eigen::array<ptrdiff_t, 5> bcast; bcast[0] = bcast[1] = bcast[2] = bcast[3] = 1; bcast[4] = fx.d.bd;
      fx.tb<4>().slice(indices, sizes).device(*dev.edevice) = xs[i]->tb<4>().broadcast(bcast);
    }
    curr_row += row_size;
  }
}

template<class MyDevice>
void Concatenate::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < src_indices.size(), "Failed boundary check in Concatenate::backward: " << i << " >= " << src_indices.size());
  Eigen::DSizes<ptrdiff_t, 5> indices(0,0,0,0,0); indices[dimension] = src_indices[i];
  Eigen::DSizes<ptrdiff_t, 5> sizes(static_cast<ptrdiff_t>(dEdxi.d[0]),
                                    static_cast<ptrdiff_t>(dEdxi.d[1]),
                                    static_cast<ptrdiff_t>(dEdxi.d[2]),
                                    static_cast<ptrdiff_t>(dEdxi.d[3]),
                                    static_cast<ptrdiff_t>(fx.d.bd));
  if(dEdxi.d.bd == dEdf.d.bd) {
    dEdxi.tb<4>().device(*dev.edevice) += dEdf.tb<4>().slice(indices, sizes);
  } else {
    Eigen::array<int, 1> red_axis; red_axis[0] = 4;
    dEdxi.t<4>().device(*dev.edevice) += dEdf.tb<4>().slice(indices, sizes).sum(red_axis);
  }
}
DYNET_NODE_INST_DEV_IMPL(Concatenate)

template<class MyDevice>
void ConcatenateToBatch::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const { 
  unsigned curr_e = 0;
  src_element_indices.resize(xs.size());
  Eigen::DSizes<ptrdiff_t, 2> indices(0,0);
  Eigen::DSizes<ptrdiff_t, 2> sizes(static_cast<ptrdiff_t>(fx.d.batch_size()), 0);
  for (unsigned i = 0; i < xs.size(); ++i) {
    indices[1] = src_element_indices[i] = curr_e;
    sizes[1] = xs[i]->d.bd;
    fx.tbvec().slice(indices, sizes).device(*dev.edevice) = xs[i]->tbvec();
    curr_e += xs[i]->d.bd;
  }
  
}

template<class MyDevice>
void ConcatenateToBatch::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < src_element_indices.size(), "Failed boundary check in ConcatenateToBatch::backward: " << i << " >= " << src_element_indices.size());
  Eigen::DSizes<ptrdiff_t, 2> indices(0, static_cast<ptrdiff_t>(src_element_indices[i]));
  Eigen::DSizes<ptrdiff_t, 2> sizes(static_cast<ptrdiff_t>(fx.d.batch_size()), static_cast<ptrdiff_t>(xs[i]->d.bd));
  dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().slice(indices, sizes);
}
DYNET_NODE_INST_DEV_IMPL(ConcatenateToBatch)

template<class MyDevice>
void BinaryLogLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.t<0>().device(*dev.edevice) = xs[0]->tvec().binaryExpr(xs[1]->tvec(), FBinaryLogLoss()).sum();
}

template<class MyDevice>
void BinaryLogLoss::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[i]->tvec().binaryExpr(xs[1-i]->tvec(), FBinaryLogLossBackward(as_scalar(dEdf)));
}
DYNET_NODE_INST_DEV_IMPL(BinaryLogLoss)

template<class MyDevice>
void BlockDropout::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  bernoulli_distribution distribution(1.0 - dropout_probability);
  float block_multiplier = distribution(*rndeng)? 1.0 : 0.0;
  block_multiplier = 
    dropout_probability == 1.0? 0.0 : block_multiplier / (1.0 - dropout_probability);
  if (dropout_probability > 1.0 || dropout_probability < 0.0)
    DYNET_INVALID_ARG("Dropout probability must be in the range [0, 1]");
  *(static_cast<float*>(aux_mem)) = block_multiplier;
  fx.tvec().device(*dev.edevice) = xs[0]->tvec() * block_multiplier;
}

template<class MyDevice>
void BlockDropout::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  float block_multiplier = *(static_cast<float*>(aux_mem));
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * block_multiplier;
}
DYNET_NODE_INST_DEV_IMPL(BlockDropout)

template<class MyDevice>
void ConstantMinusX::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(const_minus_op<float>(c));
}

template<class MyDevice>
void ConstantMinusX::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) -= dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(ConstantMinusX)

template<class MyDevice>
void ConstantPlusX::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(const_add_op<float>(c));
}

template<class MyDevice>
void ConstantPlusX::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(ConstantPlusX)

template<class MyDevice>
void ConstScalarMultiply::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec() * alpha;
}

template<class MyDevice>
void ConstScalarMultiply::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in ConstScalarMultiply");
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * alpha;
}
DYNET_NODE_INST_DEV_IMPL(ConstScalarMultiply)

template<class MyDevice>
void CwiseQuotient::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in CwiseQuotient::forward (cdiv)");
  if(xs[0]->d.bd == xs[1]->d.bd) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() / xs[1]->tvec();
  } else if(xs[0]->d.bd == 1) {
    Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
    fx.tb<1>().device(*dev.edevice) = xs[0]->tb<1>().broadcast(bcast) / xs[1]->tb<1>();
  } else {
    Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
    fx.tb<1>().device(*dev.edevice) = xs[0]->tb<1>() / xs[1]->tb<1>().broadcast(bcast);
  }
}

template<class MyDevice>
void CwiseQuotient::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in CwiseQuotient::backward (cdiv)");
  if (i == 0) {
    if(xs[0]->d.bd == xs[1]->d.bd) {
      dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() / xs[1]->tvec();
    } else if(xs[1]->d.bd == 1) {
      Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
      dEdxi.tb<1>().device(*dev.edevice) += dEdf.tb<1>() / xs[1]->tb<1>().broadcast(bcast);
    } else {
      Eigen::array<int, 1> red_axis; red_axis[0] = 1;
      dEdxi.t<1>().device(*dev.edevice) += (dEdf.tb<1>() / xs[1]->tb<1>()).sum(red_axis);
    }
  } else { // i = 1
    if(xs[0]->d.bd == xs[1]->d.bd) {
      dEdxi.tvec().device(*dev.edevice) -= dEdf.tvec() / xs[1]->tvec().square() * xs[0]->tvec();
    } else if(xs[1]->d.bd == 1) {
      Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
      Eigen::array<int, 1> red_axis; red_axis[0] = 1;
      dEdxi.t<1>().device(*dev.edevice) -= (dEdf.tb<1>() / xs[1]->tb<1>().square().broadcast(bcast) * xs[0]->tb<1>()).sum(red_axis);
    } else {
      Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
      dEdxi.tb<1>().device(*dev.edevice) -= dEdf.tb<1>() / xs[1]->tb<1>().square() * xs[0]->tb<1>().broadcast(bcast);
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(CwiseQuotient)

template<class MyDevice>
void CwiseMultiply::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in CwiseMultiply::forward (cmult)");
  if(xs[0]->d.bd == xs[1]->d.bd) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() * xs[1]->tvec();
  } else {
    Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
    if(xs[0]->d.bd == 1)
      fx.tbvec().device(*dev.edevice) = xs[0]->tbvec().broadcast(bcast) * xs[1]->tbvec();
    else
      fx.tbvec().device(*dev.edevice) = xs[0]->tbvec() * xs[1]->tbvec().broadcast(bcast);
  }
}

template<class MyDevice>
void CwiseMultiply::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in CwiseMultiply::backward (cmult)");
  if(xs[0]->d.bd == xs[1]->d.bd) {
    dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * xs[1-i]->tvec();
  } else if(xs[1-i]->d.bd == 1) {
    Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
    dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec() * xs[1-i]->tbvec().broadcast(bcast);
  } else {
    Eigen::array<int, 1> red_axis; red_axis[0] = 1;
    dEdxi.tvec().device(*dev.edevice) += (dEdf.tbvec() * xs[1-i]->tbvec()).sum(red_axis);
  }
}
DYNET_NODE_INST_DEV_IMPL(CwiseMultiply)

template<class MyDevice>
void ScalarAdd::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in ScalarAdd::forward (+)");
  Eigen::array<int, 2> bcast_0 = {1, (int) (fx.d.bd == xs[0]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 2> bcast_1 = {(int) fx.d.batch_size(), (int) (fx.d.bd == xs[1]->d.bd ? 1 : fx.d.bd)};
  fx.tbvec().device(*dev.edevice) = xs[0]->tbvec().broadcast(bcast_0) + xs[1]->tbvec().broadcast(bcast_1);
}

template<class MyDevice>
void ScalarAdd::backward_dev_impl(const MyDevice & dev,
                                  const vector<const Tensor*>& xs,
                                  const Tensor& fx,
                                  const Tensor& dEdf,
                                  unsigned i,
                                  Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in ScalarAdd::backward (+)");
  Eigen::array<int, 1> red_axis_0 = {0}, red_axis_1 = {1};
  Eigen::array<int, 2> red_axes_01 = {0, 1};
  if (i == 0) {
    if (xs[0]->d.bd == 1)
      dEdxi.tvec().device(*dev.edevice) += dEdf.tbvec().sum(red_axis_1);
    else
      dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec();
  } else {
    if (xs[1]->d.bd == 1)
      dEdxi.t<0>().device(*dev.edevice) += dEdf.tbvec().sum(red_axes_01);
    else
      dEdxi.tb<0>().device(*dev.edevice) += dEdf.tbvec().sum(red_axis_0);
  }
}
DYNET_NODE_INST_DEV_IMPL(ScalarAdd)

template<class MyDevice>
void ScalarMultiply::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in ScalarMultiply::forward (cmult)");

  Eigen::array<int, 2> bcast_0 = {(int) fx.d.batch_size(), (int) (fx.d.bd == xs[0]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 2> bcast_1 = {1, (int) (fx.d.bd == xs[1]->d.bd ? 1 : fx.d.bd)};
  fx.tbvec().device(*dev.edevice) = xs[0]->tbvec().broadcast(bcast_0) * xs[1]->tbvec().broadcast(bcast_1);
}

template<class MyDevice>
void ScalarMultiply::backward_dev_impl(const MyDevice & dev,
                                       const vector<const Tensor*>& xs,
                                       const Tensor& fx,
                                       const Tensor& dEdf,
                                       unsigned i,
                                       Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in ScalarMultiply::backward (cmult)");
  Eigen::array<int, 2> bcast_0 = {(int) fx.d.batch_size(), (int)( fx.d.bd == xs[0]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 2> bcast_1 = {1, (int)(fx.d.bd == xs[1]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 1> red_axis_0 = {0}, red_axis_1 = {1};
  Eigen::array<int, 2> red_axes_01 = {0, 1};
  if (i == 0) {
    if (xs[0]->d.bd == 1)
      dEdxi.t<0>().device(*dev.edevice) += (dEdf.tbvec() * xs[1]->tbvec().broadcast(bcast_1)).sum(red_axes_01);
    else
      dEdxi.tb<0>().device(*dev.edevice) += (dEdf.tbvec() * xs[1]->tbvec().broadcast(bcast_1)).sum(red_axis_0);
  } else {
    if (xs[1]->d.bd == 1)
      dEdxi.tvec().device(*dev.edevice) += (dEdf.tbvec() * xs[0]->tbvec().broadcast(bcast_0)).sum(red_axis_1);
    else
      dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec() * xs[0]->tbvec().broadcast(bcast_0);
  }
}
DYNET_NODE_INST_DEV_IMPL(ScalarMultiply)

template<class MyDevice>
void ScalarQuotient::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in ScalarQuotient::forward (cdiv)");
  Eigen::array<int, 2> bcast_0 = {1, (int) (fx.d.bd == xs[0]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 2> bcast_1 = {(int) fx.d.batch_size(), (int) (fx.d.bd == xs[1]->d.bd ? 1 : fx.d.bd)};
  fx.tbvec().device(*dev.edevice) = xs[0]->tbvec().broadcast(bcast_0) / xs[1]->tbvec().broadcast(bcast_1);
}

template<class MyDevice>
void ScalarQuotient::backward_dev_impl(const MyDevice & dev,
                                       const vector<const Tensor*>& xs,
                                       const Tensor& fx,
                                       const Tensor& dEdf,
                                       unsigned i,
                                       Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in ScalarQuotient::backward (cdiv)");
  Eigen::array<int, 2> bcast = {(int)fx.d.batch_size(), (int)(fx.d.bd == xs[1]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 2> bcast2 = {1, (int)(fx.d.bd == xs[0]->d.bd ? 1 : fx.d.bd)};
  Eigen::array<int, 1> red_axis_0 = {0}, red_axis_1 = {1};
  Eigen::array<int, 2> red_axes_01 = {0, 1};
  if (i == 0) {
    if (xs[0]->d.bd == 1)
      dEdxi.tvec().device(*dev.edevice) += (dEdf.tbvec() / xs[1]->tbvec().broadcast(bcast)).sum(red_axis_1);
    else
      dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec() / xs[1]->tbvec().broadcast(bcast);
  } else {
    if (xs[1]->d.bd == 1)
      dEdxi.t<0>().device(*dev.edevice) += - (dEdf.tbvec() * xs[0]->tbvec().broadcast(bcast2)).sum(red_axes_01) / xs[1]->t<0>().square();
    else
      dEdxi.tb<0>().device(*dev.edevice) += - (dEdf.tbvec() * xs[0]->tbvec().broadcast(bcast2)).sum(red_axis_0) / xs[1]->tb<0>().square();
  }
}
DYNET_NODE_INST_DEV_IMPL(ScalarQuotient)

template<class MyDevice>
void Dropout::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor m(dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_bernoulli(m, (1.f-p), 1.f / (1.f-p));
  fx.tvec().device(*dev.edevice) = xs[0]->tvec() * m.tvec();
}

template<class MyDevice>
void Dropout::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Tensor m(dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * m.tvec();
}
DYNET_NODE_INST_DEV_IMPL(Dropout)


template<class MyDevice>
void DropoutBatch::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Dim mask_dim({1},xs[0]->d.batch_elems());
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_bernoulli(m, (1.f-p), 1.f / (1.f-p));
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  fx.tbvec().device(*dev.edevice) = xs[0]->tbvec() * m.tbvec().broadcast(bcast);
}

template<class MyDevice>
void DropoutBatch::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Dim mask_dim({1},xs[0]->d.batch_elems());
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec() * m.tbvec().broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(DropoutBatch)


template<class MyDevice>
void DropoutDim::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Dim mask_dim(dim);
  mask_dim.d[dimension]=1;
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_bernoulli(m, (1.f-p), 1.f / (1.f-p));
  Eigen::array<ptrdiff_t, 4> bcast = {1, 1, 1, 1}; bcast[dimension] = xs[0]->d[dimension];
  fx.tb<3>().device(*dev.edevice) = xs[0]->tb<3>() * m.tb<3>().broadcast(bcast);
}

template<class MyDevice>
void DropoutDim::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Dim mask_dim(dim);
  mask_dim.d[dimension]=1;
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> bcast = {1, 1, 1, 1}; bcast[dimension] = dEdf.d[dimension];
  dEdxi.tb<3>().device(*dev.edevice) += dEdf.tb<3>() * m.tb<3>().broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(DropoutDim)

template<class MyDevice>
void Erf::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().erf();
}

template<class MyDevice>
void Erf::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().binaryExpr(dEdf.tvec(), scalar_erf_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Erf)

template<class MyDevice>
void GaussianNoise::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor m(dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_normal(m, 0, stddev);
  fx.tvec().device(*dev.edevice) = xs[0]->tvec() + m.tvec();
}

template<class MyDevice>
void GaussianNoise::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(GaussianNoise)

template<class MyDevice>
void Identity::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void Identity::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(Identity)

template<class MyDevice>
void KMHNGram::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("KMHNGram not implemented for CUDA");
#else
  auto x = **xs[0];
  const int new_cols = x.cols() - n + 1;
  DYNET_ASSERT(new_cols > 0, "Failed dimension check in KMHNGram");
  auto res = *fx;
  res.setZero();
  for (int j = 0; j < new_cols; ++j) {
    auto c_j = res.col(j);
    for (unsigned k = 0; k < n; ++k)
      c_j += x.col(j + k);
  }
#endif
}

template<class MyDevice>
void KMHNGram::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("KMHNGram not implemented for CUDA");
#else
  const int c = dEdf.d.cols();
  for (int j = 0; j < c; ++j)
    for (unsigned k = 0; k < n; ++k)
      (*dEdxi).col(j+k) += (*dEdf).col(j);
#endif
}
DYNET_NODE_INST_DEV_IMPL(KMHNGram)

template<class MyDevice>
void LogDet::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("LogDet not implemented for CUDA");
#else
  fx.v[0] = logdet(**xs[0], false);
#endif
}

template<class MyDevice>
void LogDet::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("KMHNGram not implemented for CUDA");
#else
  auto trans = (**xs[0]).transpose();
  (*dEdxi) += (dEdf.v[0]) * trans.inverse();
#endif
}
DYNET_NODE_INST_DEV_IMPL(LogDet)

template<class MyDevice>
void LogGamma::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().lgamma();
}

template<class MyDevice>
void LogGamma::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().digamma() * dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(LogGamma)

template<class MyDevice>
void LogisticSigmoid::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in LogisticSigmoid::forward");
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(scalar_logistic_sigmoid_op<float>());
}

template<class MyDevice>
void LogisticSigmoid::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), scalar_logistic_sigmoid_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(LogisticSigmoid)

template<class MyDevice>
void LogSoftmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in LogSoftmax::forward");
  Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Tensor m(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem + z.d.size(), fx.device, DeviceMempool::FXS);
  TensorTools::logsumexp_dev(dev, *xs[0], m, z);
  if(fx.d.size() == fx.d.rows()) {
#ifdef __CUDACC__
    Eigen::array<int, 1> bcast;
    bcast[0] = xs[0]->d[0];
    fx.t<1>().device(*dev.edevice) = xs[0]->t<1>() - z.t<1>().broadcast(bcast);
#else
    fx.t<1>().device(*dev.edevice) = xs[0]->t<1>() - as_scalar(z);
#endif
  } else {
    // TODO? Is this broadcast efficient on CPU?
    Eigen::array<int, 3> bcasts = {(int)xs[0]->d.rows(), 1, 1};
    Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<2>() - z.tvec().reshape(morph).broadcast(bcasts);
  }
}

template<class MyDevice>
void LogSoftmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  // TODO? Is this broadcast efficient on CPU?
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  z.tb<1>().device(*dev.edevice) = dEdf.tb<2>().sum(red_axis);
  Eigen::array<int, 3> bcast = {(int)fx.d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  dEdxi.tb<2>().device(*dev.edevice) += fx.tb<2>().exp() * -z.tvec().reshape(morph).broadcast(bcast) + dEdf.tb<2>();
}
DYNET_NODE_INST_DEV_IMPL(LogSoftmax)

template<class MyDevice>
void LogSumExp::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs.size() == 1) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  } else {
    // TODO: Ideally we wouldn't need to allocate this memory permanently.
    //       We need a good method for allocating "scratch" memory that is only used temporarily.
    Tensor ms(fx.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
    Eigen::array<ptrdiff_t, 2> bcast = {1,fx.d.bd};
    // Calculate the max
    if(ms.d.bd == xs[0]->d.bd)
      ms.tvec().device(*dev.edevice) = xs[0]->tvec();
    else
      ms.tbvec().device(*dev.edevice) = xs[0]->tbvec().broadcast(bcast); 
    for (size_t i = 1; i < xs.size(); ++i) {
      if(ms.d.bd == xs[i]->d.bd)
        ms.tvec().device(*dev.edevice) = ms.tvec().cwiseMax(xs[i]->tvec());
      else
        ms.tbvec().device(*dev.edevice) = ms.tbvec().cwiseMax(xs[i]->tbvec().broadcast(bcast)); 
    }
    // sumexp
    if(ms.d.bd == xs[0]->d.bd)
      fx.tvec().device(*dev.edevice) = (xs[0]->tvec() - ms.tvec()).exp();
    else
      fx.tbvec().device(*dev.edevice) = (xs[0]->tbvec().broadcast(bcast) - ms.tbvec()).exp();
    for (size_t i = 1; i < xs.size(); ++i) {
      if(ms.d.bd == xs[i]->d.bd)
        fx.tvec().device(*dev.edevice) += (xs[i]->tvec() - ms.tvec()).exp();
      else
        fx.tbvec().device(*dev.edevice) += (xs[i]->tbvec().broadcast(bcast) - ms.tbvec()).exp();
    }
    // log and add max
    fx.tvec().device(*dev.edevice) = fx.tvec().log() + ms.tvec();
  }
}

template<class MyDevice>
void LogSumExp::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if (xs.size() == 1) {
    dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
  } else {
    // df/dx_i = 1/{sum_j exp(x_j)} * exp(x_i)}
    //         = 1/{exp f(x)} * exp(x_i)
    //         = exp(x_i - f(x))
    if(fx.d.bd == xs[i]->d.bd) {
      dEdxi.tvec().device(*dev.edevice) += (xs[i]->tvec() - fx.tvec()).exp() * dEdf.tvec();
    } else {
      Eigen::array<ptrdiff_t, 2> bcast = {1,fx.d.bd};
      Eigen::array<int, 1> red_axis = {1};
      dEdxi.tvec().device(*dev.edevice) += ((xs[i]->tbvec().broadcast(bcast) - fx.tbvec()).exp() * dEdf.tbvec()).sum(red_axis);
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(LogSumExp)

template<class MyDevice>
void MatrixInverse::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MatrixInverse::forward");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("MatrixInverse not yet implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = *fx;
  y = x.inverse();
#endif
  // TODO: Change into tensors after resolving test errors
  // fx.t<2>().device(*dev.edevice) = xs[0]->t<2>().inverse();
}

template<class MyDevice>
void MatrixInverse::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MatrixInverse::backward");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("MatrixInverse not yet implemented for CUDA");
#else
  auto d = *dEdf;
  auto y = *fx;
  (*dEdxi) -= y * d * y;
#endif
}
DYNET_NODE_INST_DEV_IMPL(MatrixInverse)

template<class MyDevice>
void Max::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor t(fx.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  t.tvec().device(*dev.edevice) = (xs[0]->tvec() > xs[1]->tvec()).cast<float>();
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().cwiseMax(xs[1]->tvec());
}

template<class MyDevice>
void Max::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in Max::backward");
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  if (i == 0) {
    dEdxi.tvec().device(*dev.edevice) += t.tvec() * dEdf.tvec();
  } else {
    dEdxi.tvec().device(*dev.edevice) += t.tvec().binaryExpr(dEdf.tvec(), FMaxBackwardInv());
  }
}
DYNET_NODE_INST_DEV_IMPL(Max)

template<class MyDevice>
void NoBackprop::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void NoBackprop::backward_dev_impl(const MyDevice & dev,
                                   const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  // no op
}
DYNET_NODE_INST_DEV_IMPL(NoBackprop)

template<class MyDevice>
void FlipGradient::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void FlipGradient::backward_dev_impl(const MyDevice & dev,
                                   const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  // takes negative on backprop
  dEdxi.tvec().device(*dev.edevice) -= dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(FlipGradient)  
  
template<class MyDevice>
void MaxPooling1D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_RUNTIME_ERR("MaxPooling1D::forward_dev_impl not implemented yet");
#if 0
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MaxPooling1D::forward");
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  DYNET_ASSERT(x.cols() == 1, "Failed dimension check in MaxPooling1D::forward");
  const unsigned fx_rows = x_rows / width;
  ind.resize(fx_rows);
  Tensor fx = Zero(Dim(fx_rows, 1));
  for (unsigned i = 0; i < fx_rows; ++i) {
    unsigned from = i * width;
    unsigned to = from + width;
    if (to > x_rows) to = x_rows;
    real best = x(from, 0);
    unsigned bestr = from;
    for (unsigned r = from + 1; r < to; ++r) {
      if (x(r, 0) > best) {
        best = x(r,0);
        bestr = r;
      }
    }
    ind[i] = bestr;
    fx(i, 0) = best;
  }
  return fx;
#endif
}

template<class MyDevice>
void MaxPooling1D::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("MaxPooling1D::backward_dev_impl not implemented yet");
#if 0
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  Tensor dEdx = Zero(Dim(x_rows, 1));
  const unsigned fx_rows = x_rows / width;
  DYNET_ASSERT(fx_rows == ind.size(), "Failed dimension check in MaxPooling1D::backward");
  DYNET_ASSERT(fx_rows == dEdf.rows(), "Failed dimension check in MaxPooling1D::backward");
  for (unsigned i = 0; i < fx_rows; ++i)
    dEdx(ind[i], 0) = dEdf(i, 0);
  return dEdx;
#endif
}
DYNET_NODE_INST_DEV_IMPL(MaxPooling1D)

template<class MyDevice>
void Min::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor t(fx.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  t.tvec().device(*dev.edevice) = (xs[0]->tvec() < xs[1]->tvec()).cast<float>();
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().cwiseMin(xs[1]->tvec());
}

template<class MyDevice>
void Min::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in Min::backward");
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  if (i == 0) {
    dEdxi.tvec().device(*dev.edevice) += t.tvec() * dEdf.tvec();
  } else {
    dEdxi.tvec().device(*dev.edevice) += t.tvec().binaryExpr(dEdf.tvec(), FMaxBackwardInv());
  }
}
DYNET_NODE_INST_DEV_IMPL(Min)

template<class MyDevice>
void PairwiseRankLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().binaryExpr(xs[1]->tvec(), FPairwiseRankLoss(margin));
}

template<class MyDevice>
void PairwiseRankLoss::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if (i == 0) {
    dEdxi.tvec().device(*dev.edevice) -= fx.tvec().binaryExpr(dEdf.tvec(), FRectifyBackward());
  } else {
    dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FRectifyBackward());
  }
}
DYNET_NODE_INST_DEV_IMPL(PairwiseRankLoss)

// x_1 is a vector
// y = (x_1)_{*pval}
template<class MyDevice>
void PickElement::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if(pval) {
    DYNET_ARG_CHECK(*pval < xs[0]->d[dimension], 
                            "PickElement::forward_impl requested element " << *pval << " from a dimension of length " << xs[0]->d[dimension]);
    // TODO: This limit of up to 4 is somewhat arbitrary. We need to decide how to handle
    //       things with "maximum tensor size".
    fx.tb<3>().device(*dev.edevice) = xs[0]->tb<4>().chip(*pval, dimension); 
  } else {
    DYNET_ASSERT(pvals != nullptr, "Neither single nor vector of elements available in PickElement::forward");
    DYNET_ARG_CHECK(pvals->size() == fx.d.batch_elems(),
                            "In PickElement::forward, number of elements in the passed-in index vector (" <<  pvals->size() << ")"
                            " did not match number of elements in mini-batch elements in expression (of dimension" << fx.d << ")");
    for(unsigned b = 0; b < pvals->size(); ++b) {
      DYNET_ARG_CHECK((*pvals)[b] < xs[0]->d[dimension], 
                              "PickElement::forward_impl requested element " << (*pvals)[b] << " from a dimension of length " << xs[0]->d[dimension]);
      if(xs[0]->d.bd == 1){
        fx.tb<2>().chip<2>(b).device(*dev.edevice) = xs[0]->t<3>().chip((*pvals)[b], dimension); 
      }else{
        fx.tb<2>().chip<2>(b).device(*dev.edevice) = xs[0]->tb<3>().chip<3>(b).chip((*pvals)[b], dimension); 
      }
    }
  }
}

// derivative is 0 in all dimensions except 1 for the selected element
template<class MyDevice>
void PickElement::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in PickElement::backward");
  if(pval) {
    dEdxi.tb<3>().chip(*pval, dimension).device(*dev.edevice) += dEdf.tb<2>();
  } else {
    DYNET_ASSERT(pvals, "Neither single nor vector of elements available in PickElement::forward");
    for(unsigned b = 0; b < pvals->size(); ++b){
      if(xs[0]->d.bd == 1){
        dEdxi.t<3>().chip((*pvals)[b], dimension).device(*dev.edevice) += dEdf.tb<2>().chip<2>(b);
      }else{
        dEdxi.tb<3>().chip<3>(b).chip((*pvals)[b], dimension).device(*dev.edevice) += dEdf.tb<2>().chip<2>(b);
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(PickElement)

// x_1 is a matrix
// y = (x_1)[start:end]
// slice of matrix from index start (inclusive) to index end (exclusive)
template<class MyDevice>
void PickRange::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DSizes<ptrdiff_t, 5> indices(0,0,0,0,0);
  indices[dim] = start;
  Eigen::DSizes<ptrdiff_t, 5> sizes(static_cast<ptrdiff_t>(fx.d[0]), 
                                    static_cast<ptrdiff_t>(fx.d[1]),
                                    static_cast<ptrdiff_t>(fx.d[2]),
                                    static_cast<ptrdiff_t>(fx.d[3]),
                                    static_cast<ptrdiff_t>(fx.d.bd));
  sizes[dim] = end-start;
  fx.tb<4>().device(*dev.edevice) = xs[0]->tb<4>().slice(indices, sizes);
}

// derivative is 0 in all dimensions except the slice range
template<class MyDevice>
void PickRange::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::DSizes<ptrdiff_t, 5> indices(0,0,0,0,0);
  indices[dim] = start;
  Eigen::DSizes<ptrdiff_t, 5> sizes(static_cast<ptrdiff_t>(fx.d[0]), 
                                    static_cast<ptrdiff_t>(fx.d[1]),
                                    static_cast<ptrdiff_t>(fx.d[2]),
                                    static_cast<ptrdiff_t>(fx.d[3]),
                                    static_cast<ptrdiff_t>(fx.d.bd));
  sizes[dim] = end-start;
  dEdxi.tb<4>().slice(indices, sizes).device(*dev.edevice) += dEdf.tb<4>();
}
DYNET_NODE_INST_DEV_IMPL(PickRange)

template<class MyDevice>
void PickBatchElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (pval) {
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().chip<1>(*pval);
  } else {
    DYNET_ASSERT(pvals != nullptr, "Neither single nor vector of elements available in PickBatchElements::forward");
    DYNET_ARG_CHECK(pvals->size() == fx.d.batch_elems(), 
                            "In PickBatchElements::forward, number of elements in the passed-in index vector (" << pvals->size() << ") "
                            "did not match number of elements in mini-batch elements in expression (of dimension" << fx.d << ")");
    for (unsigned b = 0; b < pvals->size(); ++b) {
      DYNET_ARG_CHECK((*pvals)[b] < xs[0]->d.bd,
                              "PickBatchElements::forward_impl requested element " << (*pvals)[b] << " from a batch size of " << xs[0]->d.bd);
      fx.tbvec().chip<1>(b).device(*dev.edevice) = xs[0]->tbvec().chip<1>((*pvals)[b]);
    }
  }
}

template<class MyDevice>
void PickBatchElements::backward_dev_impl(const MyDevice & dev,
                                  const vector<const Tensor*>& xs,
                                  const Tensor& fx,
                                  const Tensor& dEdf,
                                  unsigned i,
                                  Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in PickBatchElements::backward");
  if (pval) {
    dEdxi.tbvec().chip<1>(*pval).device(*dev.edevice) += dEdf.tvec();
  } else {
    DYNET_ASSERT(pvals, "Neither single nor vector of elements available in PickBatchElements::backward");
    for (unsigned b = 0; b < pvals->size(); ++b)
      dEdxi.tbvec().chip<1>((*pvals)[b]).device(*dev.edevice) += dEdf.tbvec().chip<1>(b);
  }
}
DYNET_NODE_INST_DEV_IMPL(PickBatchElements)

template<class MyDevice>
void PoissonRegressionLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const real y = *pty;
  const auto z = std::lgamma(y + 1);
  // const auto x = as_scalar(*xs[0]);
  fx.t<0>().device(*dev.edevice) = xs[0]->t<0>().exp() + z - xs[0]->t<0>() * y;
}

template<class MyDevice>
void PoissonRegressionLoss::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  const real y = *pty;
  dEdxi.t<0>().device(*dev.edevice) += xs[0]->t<0>().exp() - y;
}
DYNET_NODE_INST_DEV_IMPL(PoissonRegressionLoss)

template<class MyDevice>
void Pow::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed dimension check in Pow::forward");
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().pow(as_scalar(*xs[1]));
}

template<class MyDevice>
void Pow::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed dimension check in Pow::backward");
  real x2 = as_scalar(*xs[1]);
  if (i == 0) {
    dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().pow(x2 - 1) * dEdf.tvec() * x2;
  } else {
#if defined(__CUDACC__) && defined(EIGEN_NO_MALLOC)
    DYNET_RUNTIME_ERR("CUDA memory allocation in Pow");
#endif
    // y = a^x
    // dy/dx = a^x * log(a)
    dEdxi.t<0>().device(*dev.edevice) += (fx.tvec() * xs[0]->tvec().log() * dEdf.tvec()).sum();
  }
}
DYNET_NODE_INST_DEV_IMPL(Pow)

template<class MyDevice>
void Rectify::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in Rectify::forward");
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().cwiseMax(0.f);
}

template<class MyDevice>
void Rectify::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FRectifyBackward());
}
DYNET_NODE_INST_DEV_IMPL(Rectify)

template<class MyDevice>
void ExponentialLinearUnit::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in ExponentialLinearUnit::forward");
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(FELUForward(alpha, lambda));;
}

template<class MyDevice>
void ExponentialLinearUnit::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().binaryExpr(dEdf.tvec(), FELUBackward(alpha, lambda));
}
DYNET_NODE_INST_DEV_IMPL(ExponentialLinearUnit)

template<class MyDevice>
void Reshape::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  // just point to the input memory and change dimensions
  // dimensions are handled by forward_dim
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void Reshape::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Tensor reshaped(dEdxi.d, dEdf.v, dEdxi.device, dEdf.mem_pool);
  dEdxi.tvec().device(*dev.edevice) += reshaped.tvec();
}
DYNET_NODE_INST_DEV_IMPL(Reshape)

template<class MyDevice>
void RestrictedLogSoftmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in RestrictedLogSoftmax");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("RestrictedLogSoftmax not yet implemented for CUDA (contributions welcome!)");
#else
  // TODO create auxiliary mask with -infty's
  // and do usual LogSoftmax stuff
  if(denom.size() == 0)
    DYNET_INVALID_ARG("Number of elements in denominator of RestrictedLogSoftmax::forward must be zero");
  auto x = **xs[0];
  if(denom.size() == 0)
    DYNET_RUNTIME_ERR("RestrictedLogSoftmax currently only supports single column expressions (contributions expanding support to multiple columns welcome!)");
  const real logz = logsumexp(x, denom);
  TensorTools::constant(fx, -numeric_limits<real>::infinity());
  for (auto i : denom)
    (*fx)(i,0) = x(i,0) - logz;
  if (denom.size() == 1) (*fx)(denom.front(), 0) = 0;
#endif
}

template<class MyDevice>
void RestrictedLogSoftmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in RestrictedLogSoftmax");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("RestrictedLogSoftmax not yet implemented for CUDA (contributions welcome!)");
#else
  float z = 0;
  for (auto ind : denom)
    z += (*dEdf)(ind, 0);
  for (auto ind : denom)
    (*dEdxi)(ind, 0) += (*dEdf)(ind, 0) - expf((*fx)(ind, 0)) * z;
#endif
}
DYNET_NODE_INST_DEV_IMPL(RestrictedLogSoftmax)

template<class MyDevice>
void SelectCols::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectCols::forward");
  auto& rm = *pcols;
  for (unsigned i = 0; i < rm.size(); ++i) {
    DYNET_ARG_CHECK(rm[i] < xs[0]->d.cols(),
                            "Out-of-bounds index " << rm[i] << " in SelectCols over expression of dimensions " << xs[0]->d);
    fx.t<2>().chip<1>(i).device(*dev.edevice) = xs[0]->t<2>().chip<1>(rm[i]);
  }
}

template<class MyDevice>
void SelectCols::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectCols::backward");
  auto& rm = *pcols;
  for (unsigned i = 0; i < rm.size(); ++i)
    dEdxi.t<2>().chip<1>(rm[i]).device(*dev.edevice) += dEdf.t<2>().chip<1>(i);
}
DYNET_NODE_INST_DEV_IMPL(SelectCols)

template<class MyDevice>
void SelectRows::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectRows::forward");
  auto& rm = *prows;
  for (unsigned i = 0; i < rm.size(); ++i) {
    DYNET_ARG_CHECK(rm[i] < xs[0]->d.rows(),
                            "Out-of-bounds index " << rm[i] << " in SelectRows over expression of dimensions " << xs[0]->d);
    fx.t<4>().chip<0>(i).device(*dev.edevice) = xs[0]->t<4>().chip<0>(rm[i]);
  }
}

template<class MyDevice>
void SelectRows::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectRows::backward");
  auto& rm = *prows;
  for (unsigned i = 0; i < rm.size(); ++i)
    dEdxi.t<4>().chip<0>(rm[i]).device(*dev.edevice) += dEdf.t<4>().chip<0>(i);
}
DYNET_NODE_INST_DEV_IMPL(SelectRows)

template<class MyDevice>
void Softmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in Softmax::forward");
  Tensor z(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Tensor m(Dim({xs[0]->d.cols()},fx.d.bd), (float*)aux_mem + z.d.size(), fx.device, DeviceMempool::FXS);
  TensorTools::logsumexp_dev(dev, *xs[0], m, z);
  // TODO? Is this broadcast efficient on CPU?
  Eigen::array<int, 3> bcasts = {(int)xs[0]->d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  fx.tb<2>().device(*dev.edevice) = (xs[0]->tb<2>() - z.tvec().reshape(morph).broadcast(bcasts)).exp();
}

template<class MyDevice>
void Softmax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Tensor z(Dim({fx.d.cols()},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
  // TODO? Is this broadcast efficient on CPU?
  Eigen::array<int, 1> red_axis = {0};
  z.tb<1>().device(*dev.edevice) = (fx.tb<2>() * dEdf.tb<2>()).sum(red_axis);
  Eigen::array<int, 3> bcast = {(int)xs[0]->d.rows(), 1, 1};
  Eigen::array<int, 3> morph = {1, (int)z.d[0], (int)z.d.bd};
  dEdxi.tb<2>().device(*dev.edevice) += (dEdf.tb<2>() - z.tvec().reshape(morph).broadcast(bcast)) * fx.tb<2>();
}
DYNET_NODE_INST_DEV_IMPL(Softmax)

template<class MyDevice>
void SoftSign::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SoftSign::forward");
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(FSoftSign());
}

template<class MyDevice>
void SoftSign::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FSoftSignBackward());
}
DYNET_NODE_INST_DEV_IMPL(SoftSign)

template<class MyDevice>
void Sparsemax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#ifdef __CUDACC__
    DYNET_RUNTIME_ERR("Sparsemax not implemented for CUDA");
#else
    const unsigned rows = xs[0]->d.rows();
    float *zs = static_cast<float*>(aux_mem);
    std::partial_sort_copy(xs[0]->v, xs[0]->v+rows, zs, zs + rows, std::greater<float>());
    float sum = 0, maxsum = 0;
    unsigned k = 0;
    for (k = 0; k < rows; ++k) {
      sum += zs[k];
      float t = 1 + (k + 1) * zs[k];
      if (t <= sum) break;
      maxsum = sum;
    }
    float tau = (maxsum - 1) / k;
    auto y = *fx;
    fx.tvec() = (xs[0]->tvec() - tau).cwiseMax(0.f);
    int c = 1;
    int *cc = static_cast<int*>(aux_mem);
    for (unsigned i = 0; i < rows; ++i)
      if (y(i,0) > 0.f) cc[c++] = i;
    cc[0] = c - 1;
#endif
  } else {
    DYNET_RUNTIME_ERR("Sparsemax not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void Sparsemax::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("Sparsemax not implemented for CUDA");
#else
  const int ssize = static_cast<int*>(aux_mem)[0];
  int *support = static_cast<int*>(aux_mem) + 1;
  float dhat = 0;
  auto& d = *dEdf;
  for (int i = 0; i < ssize; ++i)
    dhat += d(support[i], 0);
  dhat /= ssize;
  for (int i = 0; i < ssize; ++i)
    (*dEdxi)(support[i], 0) += d(support[i], 0) - dhat;
#endif
}
DYNET_NODE_INST_DEV_IMPL(Sparsemax)

template<class MyDevice>
void SparsemaxLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
#ifdef __CUDACC__
    DYNET_RUNTIME_ERR("SparsemaxLoss not implemented for CUDA");
#else
    const int rows = xs[0]->d.rows();
    if (rows > MAX_SPARSEMAX_LOSS_ROWS)
      DYNET_RUNTIME_ERR("MAX_SPARSEMAX_LOSS_ROWS is not sufficient. Recompile with larger value.");
    const unsigned qsupport_size = pq->size();
    const float qprop = 1.f / qsupport_size;

    float *zs = static_cast<float*>(aux_mem);
    std::partial_sort_copy(xs[0]->v, xs[0]->v+rows, zs, zs + rows, std::greater<float>());
    float sum = 0, maxsum = 0;
    int k = 0;
    for (k = 0; k < rows; ++k) {
      sum += zs[k];
      float t = 1 + (k + 1) * zs[k];
      if (t <= sum) break;
      maxsum = sum;
    }
    float tau = (maxsum - 1) / k;
    Tensor tsm(xs[0]->d, (float*)aux_mem, xs[0]->device, DeviceMempool::FXS);
    tsm.t<1>() = (xs[0]->t<1>() - tau).cwiseMax(0.f);
    fx.t<0>() = ( (tsm.t<1>() != 0.f).cast<float>() * (xs[0]->t<1>().square() - (tau * tau)) ).sum();
    fx.t<0>() = ( fx.t<0>() + qprop * qprop * qsupport_size ) / 2.f;
    for (unsigned i = 0; i < qsupport_size; ++i)
      fx.t<0>() = fx.t<0>() - xs[0]->t<1>().chip<0>((*pq)[i]) * qprop;
    fx.t<0>() = fx.t<0>().cwiseMax(0.f);
#endif
  } else {
    DYNET_RUNTIME_ERR("SparsemaxLoss not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void SparsemaxLoss::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("SparsemaxLoss not implemented for CUDA");
#else
  const float d = dEdf.v[0];
  float* psm = static_cast<float*>(aux_mem);
  float dqprop = d / pq->size();
  Tensor tsm(xs[0]->d, psm, xs[0]->device, DeviceMempool::FXS);
  auto sm = *tsm;  // sparsemax(z)
  *dEdxi += sm * d;
  for (unsigned i = 0; i < pq->size(); ++i)
    (*dEdxi)((*pq)[i], 0) -= dqprop;
#endif
}
DYNET_NODE_INST_DEV_IMPL(SparsemaxLoss)

template<class MyDevice>
void Sum::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) 
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  else if (num_args == 2 && xs[0]->d.bd == xs[1]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec();
  else if (num_args == 3 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec();
  else if (num_args == 4 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd && xs[2]->d.bd == xs[3]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();
  else {
    bool allSameBatchSize = std::all_of(xs.begin(), xs.end(), [&](const Tensor* x) { return x->d.bd == xs[0]->d.bd;});
    if (allSameBatchSize) {
      // Since they are all the same batch size, we can easily unroll the addition (results in lower GPU latency by merging multiple adds together in one CUDA call):
      DYNET_ASSERT(num_args > 4, "Bad loop unrolling in Sum::forward");        // If it was <=4, we would have handled it in the special cases above
      fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();

      const unsigned remainder = (num_args - 4 ) % 4;
      switch (remainder) {
        case 0: break;
        case 1: fx.tvec().device(*dev.edevice) += xs[4]->tvec(); break;
        case 2: fx.tvec().device(*dev.edevice) += xs[4]->tvec() + xs[5]->tvec(); break;
        case 3: fx.tvec().device(*dev.edevice) += xs[4]->tvec() + xs[5]->tvec() + xs[6]->tvec(); break;
      }
      for (unsigned i = 4 + remainder; i < num_args; i += 4)
        fx.tvec().device(*dev.edevice) += xs[i]->tvec() + xs[i + 1]->tvec() + xs[i + 2]->tvec() + xs[i + 3]->tvec();
    }
    else {
      // Not all the same batch size, so need to broadcast in the cases where they differ
      TensorTools::zero(fx);
#ifdef __CUDACC__
      Eigen::array<int, 2> bcast({ 1, (int)fx.d.bd });
#endif
      for (unsigned i = 0; i < num_args; ++i) {
        if (xs[i]->d.bd == fx.d.bd) {
          fx.tvec().device(*dev.edevice) += xs[i]->tvec();
        }
        else {
#ifdef __CUDACC__
          fx.tbvec().device(*dev.edevice) += xs[i]->tbvec().broadcast(bcast);
#else
          for (unsigned b = 0; b < fx.d.bd; ++b)
            fx.tbvec().chip<1>(b).device(*dev.edevice) += xs[i]->tvec();
#endif
        }
      }
    }
  }
}

template<class MyDevice>
void Sum::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if(dEdxi.d.bd == fx.d.bd) {
    dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
  } else {
    Eigen::array<int, 1> red_axis = {1};
    dEdxi.tvec().device(*dev.edevice) += dEdf.tbvec().sum(red_axis);
  }
}
DYNET_NODE_INST_DEV_IMPL(Sum)

template<class MyDevice>
void SumElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SumElements::forward");
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().sum(red_axis);
}

template<class MyDevice>
void SumElements::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in SumElements::backward");
  Eigen::array<int, 2> bcast = {(int)xs[0]->d.batch_size(), 1};
  dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(SumElements)

template<class MyDevice>
void MomentElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in MomentElements::forward");
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  if(order == 1)
    fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().sum(red_axis) / (float) xs[0]->d.batch_size();
  else if (order == 2)
    fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().square().sum(red_axis) / (float) xs[0]->d.batch_size();
  else
    fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().pow(order).sum(red_axis) / (float) xs[0]->d.batch_size();
}

template<class MyDevice>
void MomentElements::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentElements::backward");
  Eigen::array<int, 2> bcast = {(int)xs[0]->d.batch_size(), 1};
  if (order == 1)
    dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().broadcast(bcast) / (float) xs[0]->d.batch_size();
  else if (order == 2)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec()) * ( 2.f / (float) xs[0]->d.batch_size());
  else if (order == 3)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().square()) * ( 3.f / (float) xs[0]->d.batch_size());
  else
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().pow(order - 1)) * ( (float) order / (float) xs[0]->d.batch_size());
}
DYNET_NODE_INST_DEV_IMPL(MomentElements)


template<class MyDevice>
void StdElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in StdElements::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  Eigen::array<ptrdiff_t, 2> newaxis = {1, xs[0]->d.bd};
  float n = (float) xs[0]->d.batch_size();
  fx.tb<0>().device(*dev.edevice) = ((xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
}

template<class MyDevice>
void StdElements::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 1, "Failed dimension check in StdElements::backward");
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  Eigen::array<ptrdiff_t, 2> newaxis = {1, xs[0]->d.bd};
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  float n = (float) xs[0]->d.batch_size();
  dEdxi.tbvec().device(*dev.edevice) +=  (2 / n) * (xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)) * (fx.tbvec().binaryExpr(dEdf.tbvec(), FSqrtBackward())).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdElements)

template<class MyDevice>
void MomentBatches::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in MomentBatches::forward");
  Eigen::array<int, 1> red_axis; red_axis[0] = 1;
  if(order == 1)
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().sum(red_axis) / (float) xs[0]->d.bd;
  else if (order == 2)
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().square().sum(red_axis) / (float) xs[0]->d.bd;
  else
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().pow(order).sum(red_axis) / (float) xs[0]->d.bd;
}

template<class MyDevice>
void MomentBatches::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentBatches::backward");
  Eigen::array<int, 2> bcast = {1, (int)xs[0]->d.bd};
  if (order == 1)
    dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().broadcast(bcast) / (float) xs[0]->d.bd;
  else if (order == 2)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec()) * ( 2.f / (float) xs[0]->d.bd);
  else if (order == 3)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().square()) * ( 3.f / (float) xs[0]->d.bd);
  else
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().pow(order - 1)) * ( (float) order / (float) xs[0]->d.bd);
}
DYNET_NODE_INST_DEV_IMPL(MomentBatches)

template<class MyDevice>
void MomentDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  Eigen::array<int, 1> reduction_axis = {(int)dimension};
  float n = (float) xs[0]->d[dimension];
  if(order == 1)
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().sum(reduction_axis) / n;
  else if (order == 2)
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().square().sum(reduction_axis) / n;
  else
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().pow(order).sum(reduction_axis) / n;
}

template<class MyDevice>
void MomentDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentDimension::backward");
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = xs[0]->d[dimension];
  Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dimension] = 1;
  float n = (float) xs[0]->d[dimension];
  if (order == 1)
    dEdxi.tb<3>().device(*dev.edevice) += dEdf.tb<2>().reshape(morph).broadcast(bcast) / n;
  else if (order == 2)
    dEdxi.tb<3>().device(*dev.edevice) += (dEdf.tb<2>().reshape(morph).broadcast(bcast) * xs[0]->tb<3>()) * ( 2.f / n);
  else if (order == 3)
    dEdxi.tb<3>().device(*dev.edevice) += (dEdf.tb<2>().reshape(morph).broadcast(bcast) * xs[0]->tb<3>().square()) * ( 3.f / n);
  else
    dEdxi.tb<3>().device(*dev.edevice) += (dEdf.tb<2>().reshape(morph).broadcast(bcast) * xs[0]->tb<3>().pow(order - 1)) * ( (float) order / n);
}
DYNET_NODE_INST_DEV_IMPL(MomentDimension)

template<class MyDevice>
void StdDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  Eigen::array<int, 1> red_axis = {(int)dimension};
  Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dimension] = 1;
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = xs[0]->d[dimension];
  float n = (float) xs[0]->d[dimension];
  fx.tb<2>().device(*dev.edevice) = ((xs[0]->tb<3>() - (xs[0]->tb<3>().sum(red_axis).reshape(morph) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
}

template<class MyDevice>
void StdDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in StdDimension::backward");
  Eigen::array<int, 1> red_axis = {(int)dimension};
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = xs[0]->d[dimension];
  Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dimension] = 1;
  float n = (float) xs[0]->d[dimension];
  dEdxi.tb<3>().device(*dev.edevice) +=  (2 / n) * (xs[0]->tb<3>() - (xs[0]->tb<3>().sum(red_axis).reshape(morph) / n).broadcast(bcast)) * (fx.tb<2>().binaryExpr(dEdf.tb<2>(), FSqrtBackward())).reshape(morph).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdDimension)


template<class MyDevice>
void StdBatches::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in StdBatches::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {1};
  Eigen::array<ptrdiff_t, 2> newaxis = {xs[0]->d.batch_size(), 1};
  Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
  float n = (float)xs[0]->d.bd;
  fx.t<1>().device(*dev.edevice) = ((xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
}

template<class MyDevice>
void StdBatches::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 1, "Failed dimension check in StdBatches::backward");
  Eigen::array<ptrdiff_t, 1> red_axis = {1};
  Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
  Eigen::array<ptrdiff_t, 2> newaxis = {xs[0]->d.batch_size(), 1};
  float n = (float)xs[0]->d.bd;
  dEdxi.tbvec().device(*dev.edevice) +=  (2 / n) * (xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)) * (fx.tbvec().binaryExpr(dEdf.tbvec(), FSqrtBackward())).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdBatches)


template<class MyDevice>
void SumBatches::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SumBatches::forward");
  unsigned num_args = xs[0]->d.bd;
#ifdef __CUDACC__
  Eigen::array<int, 1> red_axis; red_axis[0] = 2;
  fx.t<2>().device(*dev.edevice) = xs[0]->tb<2>().sum(red_axis);
#else
  // TODO: Is this CPU version really good? Overhead can probably be reduced.
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = xs[0]->batch_matrix(0); break;
    case 2: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1); break;
    case 3: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1) + xs[0]->batch_matrix(2); break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += xs[0]->batch_matrix(i) + xs[0]->batch_matrix(i+1) + xs[0]->batch_matrix(i+2) + xs[0]->batch_matrix(i+3);
#endif
}

template<class MyDevice>
void SumBatches::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in SumBatches::backward");
#ifdef __CUDACC__
  Eigen::array<int, 3> bcast({1, 1, (int)fx.d.bd});
  dEdxi.tb<2>().device(*dev.edevice) += dEdf.tb<2>().broadcast(bcast);
#else
  for (unsigned i = 0; i < dEdxi.d.bd; ++i)
    dEdxi.batch_matrix(i) += *dEdf;
#endif
}
DYNET_NODE_INST_DEV_IMPL(SumBatches)

template<class MyDevice>
void TraceOfProduct::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("TraceOfProduct not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 * x2.transpose()).trace();
#endif
}

template<class MyDevice>
void TraceOfProduct::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i < 2, "Failed dimension check in TraceOfProduce::backward");
#ifdef __CUDACC__
  DYNET_RUNTIME_ERR("TraceOfProduct not yet implemented for CUDA");
#else
  const float d = dEdf.v[0];
  auto xother = **xs[1 - i];
  *dEdxi += d * xother;
#endif
}
DYNET_NODE_INST_DEV_IMPL(TraceOfProduct)

template<class MyDevice>
void Tanh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().tanh();
}

template<class MyDevice>
void Tanh::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), scalar_tanh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Tanh)

template<class MyDevice>
void Transpose::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (dim.num_nonone_dims() <= 1) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  } else {
    Eigen::array<ptrdiff_t, 5> order;
    for(size_t i = 0; i < 5; ++i)
      order[i] = (i >= dims.size() ? i : dims[i]);
    fx.tb<4>().device(*dev.edevice) = xs[0]->tb<4>().shuffle(order);
  }
}

template<class MyDevice>
void Transpose::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::array<ptrdiff_t, 5> order;
  for(size_t i = 0; i < 5; ++i)
    order[(i >= dims.size() ? i : dims[i])] = i;
  dEdxi.tb<4>().device(*dev.edevice) += dEdf.tb<4>().shuffle(order);
}
DYNET_NODE_INST_DEV_IMPL(Transpose)

template<class MyDevice>
void Zeroes::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in Zeroes::forward");
  TensorTools::zero(fx);
}

template<class MyDevice>
void Zeroes::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(Zeroes)

template<class MyDevice>
void RandomNormal::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomNormal::forward");
  TensorTools::randomize_normal(fx);
}

template<class MyDevice>
void RandomNormal::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomNormal)

template<class MyDevice>
void RandomBernoulli::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomBernoulli::forward");
  TensorTools::randomize_bernoulli(fx, p, scale);
}

template<class MyDevice>
void RandomBernoulli::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomBernoulli)

template<class MyDevice>
void RandomUniform::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomUniform::forward");
  TensorTools::randomize_uniform(fx, left, right);
}

template<class MyDevice>
void RandomUniform::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomUniform)

template<class MyDevice>
void RandomGumbel::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomGumbel::forward");
  DYNET_ARG_CHECK(mu == 0.0 && beta == 1.0, "RandomGumbel only supports Gumbel(0,1) at the moment (pull requests welcome)");
  TensorTools::randomize_uniform(fx, 0, 1);
  float eps = 1e-20;
  fx.tvec().device(*dev.edevice) = -(-fx.tvec().cwiseMax(eps).log()).cwiseMax(eps).log();
}

template<class MyDevice>
void RandomGumbel::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomGumbel)

template<class MyDevice>
void MaxDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(maxmap, first_dim_size, second_dim_size, batch_size);
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {reduced_dim};
  locs.device(*dev.edevice) = xs[0]->tb<3>().argmax(reduced_dim);
  fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().maximum(reduction_axis);
}

template<class MyDevice>
void MaxDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MaxDimension::backward");
#ifdef __CUDACC__
  vector<Eigen::DenseIndex> indices(dim.size());
  Eigen::DenseIndex* maxmap = &indices[0];
  CUDA_CHECK(cudaMemcpy((void*)maxmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
#else
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
#endif
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(maxmap, first_dim_size, second_dim_size, batch_size);
  for(unsigned b = 0; b < batch_size; ++b){
    for(unsigned j = 0; j < second_dim_size; ++j){
      for(unsigned i = 0; i < first_dim_size; ++i){
        if (reduced_dim > second_dim)
          dEdxi.tb<3>().chip<3>(b).chip(locs(i, j, b), reduced_dim).chip(j, second_dim).chip(i, first_dim).device(*dev.edevice) 
            += dEdf.tb<2>().chip<2>(b).chip<1>(j).chip<0>(i);
        else if (reduced_dim > first_dim)
          dEdxi.tb<3>().chip<3>(b).chip(j, second_dim).chip(locs(i, j, b), reduced_dim).chip(i, first_dim).device(*dev.edevice) 
            += dEdf.tb<2>().chip<2>(b).chip<1>(j).chip<0>(i);
        else
          dEdxi.tb<3>().chip<3>(b).chip(j, second_dim).chip(i, first_dim).chip(locs(i, j, b), reduced_dim).device(*dev.edevice) 
            += dEdf.tb<2>().chip<2>(b).chip<1>(j).chip<0>(i);
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(MaxDimension)

template<class MyDevice>
void MinDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DenseIndex* minmap = static_cast<Eigen::DenseIndex*>(aux_mem);
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(minmap, first_dim_size, second_dim_size, batch_size);
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {reduced_dim};
  locs.device(*dev.edevice) = xs[0]->tb<3>().argmin(reduced_dim);
  fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().minimum(reduction_axis);
}

template<class MyDevice>
void MinDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MinDimension::backward");
#ifdef __CUDACC__
  vector<Eigen::DenseIndex> indices(dim.size());
  Eigen::DenseIndex* minmap = &indices[0];
  CUDA_CHECK(cudaMemcpy((void*)minmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
#else
  Eigen::DenseIndex* minmap = static_cast<Eigen::DenseIndex*>(aux_mem);
#endif
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(minmap, first_dim_size, second_dim_size, batch_size);
  for(unsigned b = 0; b < batch_size; ++b){
    for(unsigned j = 0; j < second_dim_size; ++j){
      for(unsigned i = 0; i < first_dim_size; ++i){
        if (reduced_dim > second_dim)
          dEdxi.tb<3>().chip<3>(b).chip(locs(i, j, b), reduced_dim).chip(j, second_dim).chip(i, first_dim).device(*dev.edevice) 
            += dEdf.tb<2>().chip<2>(b).chip<1>(j).chip<0>(i);
        else if (reduced_dim > first_dim)
          dEdxi.tb<3>().chip<3>(b).chip(j, second_dim).chip(locs(i, j, b), reduced_dim).chip(i, first_dim).device(*dev.edevice) 
            += dEdf.tb<2>().chip<2>(b).chip<1>(j).chip<0>(i);
        else
          dEdxi.tb<3>().chip<3>(b).chip(j, second_dim).chip(i, first_dim).chip(locs(i, j, b), reduced_dim).device(*dev.edevice) 
            += dEdf.tb<2>().chip<2>(b).chip<1>(j).chip<0>(i);
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(MinDimension)

template<class MyDevice>
void WeightNormalization::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in WeightNormalization::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  Eigen::array<ptrdiff_t, 1> bcast = {xs[0]->d.size()};
  Eigen::array<ptrdiff_t, 1> morph = {1};
  fx.tvec().device(*dev.edevice) = (xs[0]->tvec() / xs[0]->tvec().square().sum(red_axis).sqrt().reshape(morph).broadcast(bcast)) * as_scalar(*xs[1]);
}

template<class MyDevice>
void WeightNormalization::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  Eigen::array<ptrdiff_t, 1> bcast = {xs[0]->d.size()};
  Eigen::array<ptrdiff_t, 1> morph = {1};
  if (i==0){
    dEdxi.tvec().device(*dev.edevice) += (dEdf.tvec() / xs[0]->tvec().square().sum(red_axis).sqrt().reshape(morph).broadcast(bcast)) * as_scalar(*xs[1]) - fx.tvec() * (((dEdf.tvec() * xs[0]->tvec()).sum(red_axis)) / xs[0]->tvec().square().sum(red_axis)).reshape(morph).broadcast(bcast);
  }else{
    dEdxi.t<0>().device(*dev.edevice) += ((dEdf.tvec() * xs[0]->tvec()).sum(red_axis)) /  xs[0]->tvec().square().sum(red_axis).sqrt();
  }
}
DYNET_NODE_INST_DEV_IMPL(WeightNormalization)

} // namespace dynet
