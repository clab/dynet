#include "dynet/nodes-const.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* Zeroes *************

#ifndef __CUDACC__

string Zeroes::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "zeroes(" << dim << ')';
  return s.str();
}

Dim Zeroes::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

#endif

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

}
