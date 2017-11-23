#include "dynet/tensor-eigen.h"
#include "dynet/nodes-concat.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

// ************* Concatenate *************

#ifndef __CUDACC__

string Concatenate::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat({" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << "}, " << dimension << ')';
  return os.str();
}

Dim Concatenate::dim_forward(const vector<Dim>& xs) const {
  unsigned new_rows = 0;
  Dim dr = xs[0];
  for (auto c : xs) {
    if(dr.nd < c.nd) dr.resize(c.nd);
    if(c.nd < dr.nd) c.resize(dr.nd);
    new_rows += c[dimension];
    dr.set(dimension, c[dimension]);
    DYNET_ARG_CHECK(dr.single_batch() == c.single_batch(),
                            "Bad input dimensions in Concatenate: " << xs);
    dr.bd = max(dr.bd, c.bd);
  }
  dr.nd = max(xs[0].nd, dimension+1);
  dr.set(dimension, new_rows);
  return dr;
}

int Concatenate::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const {
  Sig s(nt::concat);
  for (auto arg:args) s.add_dim(cg.nodes[arg]->dim);
  return sm.get_idx(s);
}

#endif

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
      tb<4>(fx).slice(indices, sizes).device(*dev.edevice) = tb<4>(*xs[i]);
    } else {
      Eigen::array<ptrdiff_t, 5> bcast; bcast[0] = bcast[1] = bcast[2] = bcast[3] = 1; bcast[4] = fx.d.bd;
      tb<4>(fx).slice(indices, sizes).device(*dev.edevice) = tb<4>(*xs[i]).broadcast(bcast);
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
    tb<4>(dEdxi).device(*dev.edevice) += tb<4>(dEdf).slice(indices, sizes);
  } else {
    Eigen::array<int, 1> red_axis; red_axis[0] = 4;
    t<4>(dEdxi).device(*dev.edevice) += tb<4>(dEdf).slice(indices, sizes).sum(red_axis);
  }
}
DYNET_NODE_INST_DEV_IMPL(Concatenate)

// ************* ConcatenateToBatch *************

#ifndef __CUDACC__

string ConcatenateToBatch::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat_batch_elems(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Dim ConcatenateToBatch::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() > 0, "Failed input count check in ConcatenateToBatch")
  Dim d(xs[0]);
  for (unsigned i = 1; i < xs.size(); ++i) {
    DYNET_ARG_CHECK(xs[0].single_batch() == xs[i].single_batch(),
                            "Mismatched input dimensions in ConcatenateToBatch: " << xs);
    d.bd += xs[i].bd;
  }
  return d;
}

#endif

template<class MyDevice>
void ConcatenateToBatch::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned curr_e = 0;
  src_element_indices.resize(xs.size());
  Eigen::DSizes<ptrdiff_t, 2> indices(0,0);
  Eigen::DSizes<ptrdiff_t, 2> sizes(static_cast<ptrdiff_t>(fx.d.batch_size()), 0);
  for (unsigned i = 0; i < xs.size(); ++i) {
    indices[1] = src_element_indices[i] = curr_e;
    sizes[1] = xs[i]->d.bd;
    tbvec(fx).slice(indices, sizes).device(*dev.edevice) = tbvec(*xs[i]);
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
  tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).slice(indices, sizes);
}
DYNET_NODE_INST_DEV_IMPL(ConcatenateToBatch)

}
