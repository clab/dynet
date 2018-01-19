#include "dynet/tensor-eigen.h"
#include "dynet/nodes-norms.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"
#include "dynet/simd-functors.h"

using namespace std;

namespace dynet {

// ************* SquaredNorm *************

#ifndef __CUDACC__

string SquaredNorm::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " ||^2";
  return s.str();
}

Dim SquaredNorm::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in SquaredNorm")
  return Dim({1}, xs[0].bd);
}

#endif

template<class MyDevice>
void SquaredNorm::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in SquaredNorm::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  tb<0>(fx).device(*dev.edevice) = tbvec(*xs[0]).square().sum(red_axis);
}

template<class MyDevice>
void SquaredNorm::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 1, "Failed dimension check in SquaredNorm::backward");
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  tbvec(dEdxi).device(*dev.edevice) += tbvec(*xs[0]) * tbvec(dEdf).broadcast(bcast) * 2.0f;
}
DYNET_NODE_INST_DEV_IMPL(SquaredNorm)

// ************* L2Norm *************

#ifndef __CUDACC__

string L2Norm::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " ||";
  return s.str();
}

Dim L2Norm::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in L2Norm")
  return Dim({1}, xs[0].bd);
}

#endif

template<class MyDevice>
void L2Norm::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in L2Norm::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  tb<0>(fx).device(*dev.edevice) =
      (tbvec(*xs[0]).square().sum(red_axis)).sqrt();
}

template<class MyDevice>
void L2Norm::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 1, "Failed dimension check in L2Norm::backward");
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  tbvec(dEdxi).device(*dev.edevice) +=
      tbvec(*xs[0]) *
      (2 * tbvec(fx).binaryExpr(tbvec(dEdf),
                                 FSqrtBackward())).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(L2Norm)

}
