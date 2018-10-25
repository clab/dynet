#include "dynet/tensor-eigen.h"
#include "dynet/nodes-matrixmultiply.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/matrix-multiply.h"

using namespace std;

namespace dynet {

// ************* MatrixMultiply *************

#ifndef __CUDACC__

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Dim MatrixMultiply::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in MatrixMultiply")
  DYNET_ARG_CHECK(xs[0].cols() == xs[1].rows(), "Mismatched input dimensions in MatrixMultiply: " << xs);
  DYNET_ARG_CHECK(xs[0].nd <= 2 && xs[1].nd <= 2, "Cannot multiply tensors of dimension higher than 2: " << xs);
  if (xs[1].ndims() == 1) return Dim({xs[0].rows()}, max(xs[0].bd, xs[1].bd));
  return Dim({xs[0].rows(), xs[1].cols()}, max(xs[0].bd, xs[1].bd));
}

int MatrixMultiply::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  // Currently assumes there are two args, and batches with a shared first arg.
  // TODO do we want to treat different dimensions of first/second arg differently?
  if(dim.bd == 1) {
    Sig s(nt::matmul);
    s.add_node(args[0]);
    s.add_dim(cg.nodes[args[1]]->dim);
    return sm.get_idx(s);
  } else {
    return 0; // TODO handle the batched case as well? should it differ at all?
  }
}

std::vector<int> MatrixMultiply::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 0);
  if (dim.bd == 1) { ret[1] = 1; }
  return ret;
}

#endif

template<class MyDevice>
void MatrixMultiply::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in MatrixMultiply::forward");
  DYNET_ARG_CHECK(fx.d.bd == max(xs[0]->d.bd, xs[1]->d.bd), "Failed dimension check in MatrixMultiply::forward");
  // fx = mat(fx0) + xs[0] * xs[1]
  dynet::MatrixMultiply(dev, *xs[0], *xs[1], fx, dev.kSCALAR_ZERO);
}

template<class MyDevice>
void MatrixMultiply::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in MatrixMultiply::backward");
  // y = A * B
  if (i == 0) {
    // dA = dy * B^T
    MatrixMultiplyTranspAcc(dev, dEdf, *xs[1], dEdxi);
  } else {
    // dB = A^T * dy
    MatrixTranspMultiplyAcc(dev, *xs[0], dEdf, dEdxi);
  }
}
DYNET_NODE_INST_DEV_IMPL(MatrixMultiply)

}
