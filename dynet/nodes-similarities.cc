#include "dynet/nodes-similarities.h"

#include "dynet/nodes-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

// ************* DotProduct *************

#ifndef __CUDACC__

string DotProduct::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << "^T . " << arg_names[1];
  return s.str();
}

Dim DotProduct::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 &&
                          xs[0].single_batch() == xs[1].single_batch(),
                          "Bad arguments to DotProduct: " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

#endif

template<class MyDevice>
void DotProduct::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  Eigen::array<int, 2> bcast; bcast[0] = 1; bcast[1] = fx.d.bd;
  if(fx.d.bd == 1) {
    fx.t<0>().device(*dev.edevice) = (xs[0]->tvec() * xs[1]->tvec()).sum();
  } else if(xs[0]->d.bd == xs[1]->d.bd) {
    fx.tb<0>().device(*dev.edevice) = (xs[0]->tbvec() * xs[1]->tbvec()).sum(red_axis);
  } else if(xs[0]->d.bd == 1) {
    fx.tb<0>().device(*dev.edevice) = (xs[0]->tbvec().broadcast(bcast) * xs[1]->tbvec()).sum(red_axis);
  } else {
    fx.tb<0>().device(*dev.edevice) = (xs[0]->tbvec() * xs[1]->tbvec().broadcast(bcast)).sum(red_axis);
  }
}

template<class MyDevice>
void DotProduct::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if(fx.d.bd == 1) {
    Eigen::array<int, 1> bcast; bcast[0] = xs[i]->d.batch_size();
    dEdxi.tvec().device(*dev.edevice) += xs[1-i]->tvec() * dEdf.tvec().broadcast(bcast);
  } else {
    Eigen::array<int, 2> bcast; bcast[0] =xs[i]->d.batch_size(); bcast[1] = 1;
    if(xs[0]->d.bd == xs[1]->d.bd) {
      dEdxi.tbvec().device(*dev.edevice) += xs[1-i]->tbvec() * dEdf.tbvec().broadcast(bcast);
    } else if(dEdxi.d.bd == 1) {
      Eigen::array<int, 1> red_axis; red_axis[0] = 1;
      dEdxi.tvec().device(*dev.edevice) += (xs[1-i]->tbvec() * dEdf.tbvec().broadcast(bcast)).sum(red_axis);
    } else {
      Eigen::array<int, 2> batchcast; batchcast[0] = 1; batchcast[1] = fx.d.bd;
      dEdxi.tbvec().device(*dev.edevice) += (xs[1-i]->tbvec().broadcast(batchcast) * dEdf.tbvec().broadcast(bcast));
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(DotProduct)

// ************* HuberDistance *************

#ifndef __CUDACC__

string HuberDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||_H(" << d << ')';
  return s.str();
}

Dim HuberDistance::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in HuberDistance");
  DYNET_ARG_CHECK(xs[0].single_batch() == xs[1].single_batch() ||
                          (LooksLikeVector(xs[0]) && LooksLikeVector(xs[1]) && xs[0].batch_size() == xs[1].batch_size()),
                          "Mismatched input dimensions in HuberDistance: " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

#endif

template<class MyDevice>
void HuberDistance::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "HuberDistance::forward dimension check failed");
  fx.t<0>().device(*dev.edevice) = (xs[0]->tvec() - xs[1]->tvec()).unaryExpr(FHuberForward(d)).sum();
}

template<class MyDevice>
void HuberDistance::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "HuberDistance::backward dimension check failed");
  dEdxi.tvec().device(*dev.edevice) += (xs[i]->tvec() - xs[1-i]->tvec()).unaryExpr(FHuberBackward(d, as_scalar(dEdf)));
}
DYNET_NODE_INST_DEV_IMPL(HuberDistance)

// ************* L1Distance *************

#ifndef __CUDACC__

string L1Distance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||_1";
  return s.str();
}

Dim L1Distance::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in L1Distance")
  DYNET_ARG_CHECK(xs[0].single_batch() == xs[1].single_batch() ||
                          (LooksLikeVector(xs[0]) && LooksLikeVector(xs[1]) && xs[0].batch_size() == xs[1].batch_size()),
                          "Mismatched input dimensions in L1Distance: " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

#endif

template<class MyDevice>
void L1Distance::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in L1Distance::forward");
  fx.t<0>().device(*dev.edevice) = (xs[0]->tvec() - xs[1]->tvec()).abs().sum();
}

template<class MyDevice>
void L1Distance::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in L1Distance::backward");
  dEdxi.tvec().device(*dev.edevice) += (xs[i]->tvec() - xs[1-i]->tvec()).unaryExpr(FL1Backward(as_scalar(dEdf)));
}
DYNET_NODE_INST_DEV_IMPL(L1Distance)

// ************* SquaredEuclideanDistance *************

#ifndef __CUDACC__

string SquaredEuclideanDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||^2";
  return s.str();
}

Dim SquaredEuclideanDistance::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in SquaredEuclideanDistance")
  DYNET_ARG_CHECK(xs[0].single_batch() == xs[1].single_batch() ||
                          (LooksLikeVector(xs[0]) && LooksLikeVector(xs[1]) && xs[0].batch_size() == xs[1].batch_size()),
                          "Bad input dimensions in SquaredEuclideanDistance: " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

int SquaredEuclideanDistance::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::squared_distance);
  const Dim &dleft = cg.nodes[args[0]]->dim, &dright = cg.nodes[args[1]]->dim;
  if(dleft.bd == dright.bd) {
    s.add_node(1);
    s.add_dim(dleft);
  } else if(dleft.bd == 1) {
    s.add_node(2);
    s.add_node(args[0]);
    s.add_dim(dright);
  } else {
    s.add_node(3);
    s.add_node(args[1]);
    s.add_dim(dleft);
  }
  return sm.get_idx(s);
}

std::vector<int> SquaredEuclideanDistance::autobatch_concat(const ComputationGraph & cg) const {
  const Dim &dleft = cg.nodes[args[0]]->dim, &dright = cg.nodes[args[1]]->dim;
  vector<int> ret(2, 1);
  if(dleft.bd != dright.bd) {
    if(dleft.bd == 1)
      ret[0] = 0;
    else
      ret[1] = 0;
  }
  return ret;
}

#endif

template<class MyDevice>
void SquaredEuclideanDistance::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in SquaredEuclideanDistance::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  if(xs[0]->d.bd == xs[1]->d.bd) {
    fx.tb<0>().device(*dev.edevice) = (xs[0]->tbvec() - xs[1]->tbvec()).square().sum(red_axis);
  } else if(xs[0]->d.bd == 1) {
    Eigen::array<ptrdiff_t, 2> bcast = {1, xs[1]->d.bd};
    fx.tb<0>().device(*dev.edevice) = (xs[0]->tbvec().broadcast(bcast) - xs[1]->tbvec()).square().sum(red_axis);
  } else {
    Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
    fx.tb<0>().device(*dev.edevice) = (xs[0]->tbvec() - xs[1]->tbvec().broadcast(bcast)).square().sum(red_axis);
  }
}

template<class MyDevice>
void SquaredEuclideanDistance::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in SquaredEuclideanDistance::backward");
  float multiplier = (i == 1 ? -2.0f : 2.0f);
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  if(xs[0]->d.bd == xs[1]->d.bd) {
    dEdxi.tbvec().device(*dev.edevice) += (xs[0]->tbvec() - xs[1]->tbvec()) * dEdf.tbvec().broadcast(bcast) * multiplier;
  } else if(xs[0]->d.bd == 1) {
    Eigen::array<ptrdiff_t, 2> batchcast = {1, xs[1]->d.bd};
    if(i == 1) {
      dEdxi.tbvec().device(*dev.edevice) += (xs[0]->tbvec().broadcast(batchcast) - xs[1]->tbvec()) * dEdf.tbvec().broadcast(bcast) * multiplier;
    } else {
      Eigen::array<ptrdiff_t, 1> red_axis = {1};
      dEdxi.tvec().device(*dev.edevice) += ((xs[0]->tbvec().broadcast(batchcast) - xs[1]->tbvec()) * dEdf.tbvec().broadcast(bcast) * multiplier).sum(red_axis);
    }
  } else {
    Eigen::array<ptrdiff_t, 2> batchcast = {1, xs[0]->d.bd};
    if(i == 0) {
      dEdxi.tbvec().device(*dev.edevice) += (xs[0]->tbvec() - xs[1]->tbvec().broadcast(batchcast)) * dEdf.tbvec().broadcast(bcast) * multiplier;
    } else {
      Eigen::array<ptrdiff_t, 1> red_axis = {1};
      dEdxi.tvec().device(*dev.edevice) += ((xs[0]->tbvec() - xs[1]->tbvec().broadcast(batchcast)) * dEdf.tbvec().broadcast(bcast) * multiplier).sum(red_axis);
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(SquaredEuclideanDistance)

}
