#include "dynet/nodes-cumulative.h"

#include "dynet/nodes-macros.h"
#include "dynet/functors.h"

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

#endif

template<class MyDevice>
void CumulativeSum::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in CumulativeSum");
  fx.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().cumsum(d);
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
  // Check whether the issue stems from reverse
  dEdxi.tb<3>().device(*dev.edevice) += dEdf.tb<3>().cumsum(d);//.reverse(reverse_dim).cumsum(d).reverse(reverse_dim);
}
DYNET_NODE_INST_DEV_IMPL(CumulativeSum)

}
