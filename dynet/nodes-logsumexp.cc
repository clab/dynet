#include "dynet/nodes-logsumexp.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* LogSumExp *************

#define MAX_LOG_SUM_EXP 65536

#ifndef __CUDACC__

// template <class T>
// EIGEN_STRONG_INLINE real logsumexp(const T& x, const vector<unsigned>& denom) {
//   real m = x(denom[0],0);
//   for (auto i : denom) {
//     real r = x(i,0);
//     if (r > m) m = r;
//   }
//   real z = 0;
//   for (auto i : denom)
//     z += expf(x(i,0) - m);
//   return m + logf(z);
// }

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

}
