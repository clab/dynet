#include "dynet/nodes-logsumexp.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* LogSumExp *************

#define MAX_LOG_SUM_EXP 65536

#ifndef __CUDACC__

string LogSumExp::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log(exp " << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + exp " << arg_names[i];
  s << ")";
  return s.str();
}

Dim LogSumExp::dim_forward(const vector<Dim>& xs) const {
  Dim d = xs[0].truncate();
  for (unsigned i = 1; i < xs.size(); ++i) {
    DYNET_ARG_CHECK(d.single_batch() == xs[i].truncate().single_batch(),
                            "Mismatched input dimensions in LogSumExp: " << xs);
    d.bd = max(xs[i].bd, d.bd);
  }
  return d;
}

#endif

template<class MyDevice>
void LogSumExp::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs.size() == 1) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  } else {
    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
    Tensor ms(fx.d, nullptr, fx.device, fx.mem_pool);
    ms.v = static_cast<float*>(scratch_allocator->allocate(ms.d.size() * sizeof(float)));

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

    scratch_allocator->free();
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

// ************* LogSumExpDimension *************

#define MAX_LOG_SUM_EXP 65536

#ifndef __CUDACC__

string LogSumExpDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "logsumexp_dim(" << arg_names[0] << ", " << dimension << ")";
  return s.str();
}

Dim LogSumExpDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "LogSumExpDimension takes only one argument" << xs);
  DYNET_ARG_CHECK(xs[0].nd <= 2, "LogSumExpDimension, expects 2 or fewer dimensions" << xs);
  DYNET_ARG_CHECK(xs[0].nd > dimension, "LogSumExpDimension, expects its dimension argument (" << 
                    dimension << ") to be smaller than the number of elements in the input " << xs);
  Dim d = xs[0];
  if(dimension < d.nd)
    d.delete_dim(dimension);
  return d;
}

#endif

template<class MyDevice>
void LogSumExpDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor ms(fx.d, nullptr, fx.device, fx.mem_pool), zs(fx.d, nullptr, fx.device, fx.mem_pool);
  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
  ms.v = static_cast<float*>(scratch_allocator->allocate(ms.d.size() * sizeof(float)));
  TensorTools::logsumexp_dev(dev, *xs[0], ms, fx, dimension);
  scratch_allocator->free();
}

template<class MyDevice>
void LogSumExpDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  unsigned other_dim = dimension ^ 1;
  Eigen::array<int, 3> bcast = {1, 1, 1}; bcast[dimension] = xs[0]->d[dimension];
  Eigen::array<int, 3> morph = {1, 1, (int)fx.d.bd}; morph[other_dim] = fx.d[0];
  dEdxi.tb<2>().device(*dev.edevice) += (xs[0]->tb<2>() - fx.tb<1>().reshape(morph).broadcast(bcast)).exp() * dEdf.tb<1>().reshape(morph).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(LogSumExpDimension)

}
