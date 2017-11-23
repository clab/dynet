#include "dynet/tensor-eigen.h"
#include "dynet/nodes-select.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* SelectRows *************

#ifndef __CUDACC__

string SelectRows::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "select_rows(" << arg_names[0] << ", {rsize=" << prows->size() << "})";
  return s.str();
}

Dim SelectRows::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Bad arguments in SelectRows: " << xs);
  unsigned nrows = prows->size();
  Dim ret(xs[0]);
  ret.d[0] = nrows;
  return ret;
}

#endif

template<class MyDevice>
void SelectRows::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectRows::forward");
  auto& rm = *prows;
  for (unsigned i = 0; i < rm.size(); ++i) {
    DYNET_ARG_CHECK(rm[i] < xs[0]->d.rows(),
                            "Out-of-bounds index " << rm[i] << " in SelectRows over expression of dimensions " << xs[0]->d);
    t<4>(fx).chip<0>(i).device(*dev.edevice) = t<4>(*xs[0]).chip<0>(rm[i]);
  }
}

template<class MyDevice>
void SelectRows::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectRows::backward");
  auto& rm = *prows;
  for (unsigned i = 0; i < rm.size(); ++i)
    t<4>(dEdxi).chip<0>(rm[i]).device(*dev.edevice) += t<4>(dEdf).chip<0>(i);
}
DYNET_NODE_INST_DEV_IMPL(SelectRows)

// ************* SelectCols *************

#ifndef __CUDACC__

string SelectCols::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "select_cols(" << arg_names[0] << ", {csize=" << pcols->size() << "})";
  return s.str();
}

Dim SelectCols::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && xs[0].ndims() == 2, "Bad arguments in SelectCols: " << xs);
  unsigned ncols = pcols->size();
  Dim ret(xs[0]);
  ret.d[1] = ncols;
  return ret;
}

#endif

template<class MyDevice>
void SelectCols::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectCols::forward");
  auto& rm = *pcols;
  for (unsigned i = 0; i < rm.size(); ++i) {
    DYNET_ARG_CHECK(rm[i] < xs[0]->d.cols(),
                            "Out-of-bounds index " << rm[i] << " in SelectCols over expression of dimensions " << xs[0]->d);
    t<2>(fx).chip<1>(i).device(*dev.edevice) = t<2>(*xs[0]).chip<1>(rm[i]);
  }
}

template<class MyDevice>
void SelectCols::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SelectCols::backward");
  auto& rm = *pcols;
  for (unsigned i = 0; i < rm.size(); ++i)
    t<2>(dEdxi).chip<1>(rm[i]).device(*dev.edevice) += t<2>(dEdf).chip<1>(i);
}
DYNET_NODE_INST_DEV_IMPL(SelectCols)

// ************* PickElement *************

#ifndef __CUDACC__

string PickElement::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick(" << arg_names[0] << ',';
  if(pval) {
    s << *pval;
  } else {
    DYNET_ASSERT(pvals, "Have neither index nor index vector in PickElement");
    s << '[';
    if(pvals->size()) {
      s << (*pvals)[0];
      for(size_t i = 1; i < pvals->size(); ++i)
        s << ',' << (*pvals)[i];
    }
    s << "]";
  }
  s << ", " << dimension << ")";
  return s.str();
}

Dim PickElement::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in PickElement");
  DYNET_ARG_CHECK(dimension < xs[0].nd,
                          "Tried to PickElement on dimension " << dimension << " bigger than input " << xs[0]);
  DYNET_ARG_CHECK(xs[0].nd < 4,
                          "PickElement not currently supported for tensors of 4 or more dimensions.");

  Dim ret(xs[0]);
  if (pvals){
    DYNET_ARG_CHECK(xs[0].bd == 1 || xs[0].bd == pvals->size(),
                          "Number of elements in the passed-in index vector (" <<  pvals->size() << ")"
                            " did not match number of elements in mini-batch elements in expression (of dimension " << xs[0].bd << ") in PickElement");
    ret.bd = pvals->size();
  }

  ret.delete_dim(dimension);
  return ret;
}

#endif

// x_1 is a vector
// y = (x_1)_{*pval}
template<class MyDevice>
void PickElement::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if(pval) {
    DYNET_ARG_CHECK(*pval < xs[0]->d[dimension],
                            "PickElement::forward_impl requested element " << *pval << " from a dimension of length " << xs[0]->d[dimension]);
    // TODO: This limit of up to 4 is somewhat arbitrary. We need to decide how to handle
    //       things with "maximum tensor size".
    tb<3>(fx).device(*dev.edevice) = tb<4>(*xs[0]).chip(*pval, dimension);
  } else {
    DYNET_ASSERT(pvals != nullptr, "Neither single nor vector of elements available in PickElement::forward");
    DYNET_ARG_CHECK(pvals->size() == fx.d.batch_elems(),
                            "In PickElement::forward, number of elements in the passed-in index vector (" <<  pvals->size() << ")"
                            " did not match number of elements in mini-batch elements in expression (of dimension" << fx.d << ")");
    for(unsigned b = 0; b < pvals->size(); ++b) {
      DYNET_ARG_CHECK((*pvals)[b] < xs[0]->d[dimension],
                              "PickElement::forward_impl requested element " << (*pvals)[b] << " from a dimension of length " << xs[0]->d[dimension]);
      if(xs[0]->d.bd == 1){
        tb<2>(fx).chip<2>(b).device(*dev.edevice) = t<3>(*xs[0]).chip((*pvals)[b], dimension);
      }else{
        tb<2>(fx).chip<2>(b).device(*dev.edevice) = tb<3>(*xs[0]).chip<3>(b).chip((*pvals)[b], dimension);
      }
    }
  }
}

// derivative is 0 in all dimensions except 1 for the selected element
template<class MyDevice>
void PickElement::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in PickElement::backward");
  if(pval) {
    tb<3>(dEdxi).chip(*pval, dimension).device(*dev.edevice) += tb<2>(dEdf);
  } else {
    DYNET_ASSERT(pvals, "Neither single nor vector of elements available in PickElement::forward");
    for(unsigned b = 0; b < pvals->size(); ++b){
      if(xs[0]->d.bd == 1){
        t<3>(dEdxi).chip((*pvals)[b], dimension).device(*dev.edevice) += tb<2>(dEdf).chip<2>(b);
      }else{
        tb<3>(dEdxi).chip<3>(b).chip((*pvals)[b], dimension).device(*dev.edevice) += tb<2>(dEdf).chip<2>(b);
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(PickElement)

// ************* PickRange *************

#ifndef __CUDACC__

// x_1 is a vector
// y = (x_1)[start:end]
string PickRange::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "slice(" << arg_names[0] << ',' << start << ':' << end << ", dim=" << dim << ')';
  return s.str();
}

Dim PickRange::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in PickRange");
  DYNET_ARG_CHECK(dim < xs[0].nd && start < end && xs[0][dim] >= end,
                          "Bad input dimensions or range in PickRange: " << xs << " range(" << start << ", " << end << ") with dim=" << dim);
  Dim ret = xs[0]; ret.d[dim] = end-start;
  return ret;
}

int PickRange::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::pickrange);
  const Dim &in_dim = cg.nodes[args[0]]->dim;
  s.add_dim(in_dim);
  s.add_node(start);
  s.add_node(end);
  return sm.get_idx(s);
}

#endif

// x_1 is a matrix
// y = (x_1)[start:end]
// slice of matrix from index start (inclusive) to index end (exclusive)
template<class MyDevice>
void PickRange::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DSizes<ptrdiff_t, 5> indices(0,0,0,0,0);
  indices[dim] = start;
  Eigen::DSizes<ptrdiff_t, 5> sizes(static_cast<ptrdiff_t>(fx.d[0]),
                                    static_cast<ptrdiff_t>(fx.d[1]),
                                    static_cast<ptrdiff_t>(fx.d[2]),
                                    static_cast<ptrdiff_t>(fx.d[3]),
                                    static_cast<ptrdiff_t>(fx.d.bd));
  sizes[dim] = end-start;
  tb<4>(fx).device(*dev.edevice) = tb<4>(*xs[0]).slice(indices, sizes);
}

// derivative is 0 in all dimensions except the slice range
template<class MyDevice>
void PickRange::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::DSizes<ptrdiff_t, 5> indices(0,0,0,0,0);
  indices[dim] = start;
  Eigen::DSizes<ptrdiff_t, 5> sizes(static_cast<ptrdiff_t>(fx.d[0]),
                                    static_cast<ptrdiff_t>(fx.d[1]),
                                    static_cast<ptrdiff_t>(fx.d[2]),
                                    static_cast<ptrdiff_t>(fx.d[3]),
                                    static_cast<ptrdiff_t>(fx.d.bd));
  sizes[dim] = end-start;
  tb<4>(dEdxi).slice(indices, sizes).device(*dev.edevice) += tb<4>(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(PickRange)

// ************* PickBatchElements *************

#ifndef __CUDACC__

string PickBatchElements::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick_batch_elems(" << arg_names[0] << ',';
  if (pval) {
    s << *pval;
  } else {
    DYNET_ASSERT(pvals, "Have neither index nor index vector in PickBatchElements");
    s << '[';
    if (pvals->size()) {
      s << (*pvals)[0];
      for (size_t i = 1; i < pvals->size(); ++i)
        s << ',' << (*pvals)[i];
    }
    s << "]";
  }
  s << ")";
  return s.str();
}

Dim PickBatchElements::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in PickBatchElements")
  DYNET_ARG_CHECK(xs[0].nd < 4, "PickElement not currently supported for tensors of 4 or more dimensions.");
  Dim ret(xs[0]);
  if (pval) {
    // set batch size to one.
    ret.bd = 1;
  } else {
    DYNET_ASSERT(pvals, "Have neither index nor index vector in PickBatchElements");
    ret.bd = pvals->size();
  }
  return ret;
}

#endif

template<class MyDevice>
void PickBatchElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (pval) {
    DYNET_ARG_CHECK(*pval < xs[0]->d.bd,
                    "PickBatchElements::forward_impl requested element " << *pval << " from a batch size of " << xs[0]->d.bd);
    tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).chip<1>(*pval);
  } else {
    DYNET_ASSERT(pvals != nullptr, "Neither single nor vector of elements available in PickBatchElements::forward");
    DYNET_ARG_CHECK(pvals->size() == fx.d.batch_elems(),
                            "In PickBatchElements::forward, number of elements in the passed-in index vector (" << pvals->size() << ") "
                            "did not match number of elements in mini-batch elements in expression (of dimension" << fx.d << ")");
    for (unsigned b = 0; b < pvals->size(); ++b) {
      DYNET_ARG_CHECK((*pvals)[b] < xs[0]->d.bd,
                              "PickBatchElements::forward_impl requested element " << (*pvals)[b] << " from a batch size of " << xs[0]->d.bd);
      tbvec(fx).chip<1>(b).device(*dev.edevice) = tbvec(*xs[0]).chip<1>((*pvals)[b]);
    }
  }
}

template<class MyDevice>
void PickBatchElements::backward_dev_impl(const MyDevice & dev,
                                  const vector<const Tensor*>& xs,
                                  const Tensor& fx,
                                  const Tensor& dEdf,
                                  unsigned i,
                                  Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in PickBatchElements::backward");
  if (pval) {
    tbvec(dEdxi).chip<1>(*pval).device(*dev.edevice) += tvec(dEdf);
  } else {
    DYNET_ASSERT(pvals, "Neither single nor vector of elements available in PickBatchElements::backward");
    for (unsigned b = 0; b < pvals->size(); ++b)
      tbvec(dEdxi).chip<1>((*pvals)[b]).device(*dev.edevice) += tbvec(dEdf).chip<1>(b);
  }
}
DYNET_NODE_INST_DEV_IMPL(PickBatchElements)

}
