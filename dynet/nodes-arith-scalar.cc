#include "dynet/nodes-arith-scalar.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* ScalarAdd *************

#ifndef __CUDACC__

string ScalarAdd::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " + " << arg_names[1];
  return s.str();
}

Dim ScalarAdd::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in ScalarAdd")
  Dim d = xs[0].truncate();
  DYNET_ARG_CHECK(xs[1].batch_size() == 1,
                          "Mismatched input dimensions in ScalarAdd: " << xs);
  d.bd = max(xs[1].bd, d.bd);
  return d;
}

#endif

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

// ************* ScalarMultiply *************

#ifndef __CUDACC__

string ScalarMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " \\cdot " << arg_names[1];
  return s.str();
}

Dim ScalarMultiply::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in ScalarMultiply")
  Dim d = xs[1];
  DYNET_ARG_CHECK(xs[0].batch_size() == 1,
                          "Mismatched input dimensions in ScalarMultiply: " << xs);
  d.bd = max(xs[0].bd, d.bd);
  return d;
}

#endif

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

// ************* ScalarQuotient *************

#ifndef __CUDACC__

string ScalarQuotient::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " / " << arg_names[1];
  return s.str();
}

Dim ScalarQuotient::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in ScalarQuotient")
  Dim d = xs[0].truncate();
  DYNET_ARG_CHECK(xs[1].batch_size() == 1,
                          "Mismatched input dimensions in ScalarQuotient: " << xs);
  d.bd = max(xs[1].bd, d.bd);
  return d;
}

#endif

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

}
