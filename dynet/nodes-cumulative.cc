#include "dynet/tensor-eigen.h"
#include "dynet/nodes-cumulative.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* CumulativeSum *************

#ifndef __CUDACC__

string CumulativeSum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "cumsum(expression=" << arg_names[0] << ',' << d << ')';
  return s.str();
}

Dim CumulativeSum::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in CumulativeSum");
  DYNET_ARG_CHECK(xs[0].nd <= 3, "CumulativeSum implemented up to tensors of order 3 for now")
  DYNET_ARG_CHECK(d <= xs[0].nd, "dimension " << d << " is out of bounds of tensor of order " << xs[0].nd << " in CumulativeSum" )
  Dim ret(xs[0]);
  return ret;
}

size_t CumulativeSum::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

#endif

template<class MyDevice>
void CumulativeSum::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in CumulativeSum");
  tb<3>(fx).device(*dev.edevice) = tb<3>(*(xs[0])).cumsum(d);
}

template<class MyDevice>
void CumulativeSum::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in CumulativeSum::backward");
  Eigen::array<bool, 4> reverse_dim = {false, false, false, false};
  reverse_dim[d] = true;
  // First reverse the gradient
  Tensor dEdf_reversed(dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  tb<3>(dEdf_reversed).device(*dev.edevice) = tb<3>(dEdf).reverse(reverse_dim);
  // Then accumulate and reverse
  tb<3>(dEdxi).device(*dev.edevice) += tb<3>(dEdf_reversed).cumsum(d).reverse(reverse_dim);
}
DYNET_NODE_INST_DEV_IMPL(CumulativeSum)

}
