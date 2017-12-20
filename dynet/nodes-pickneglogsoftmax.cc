#include "dynet/tensor-eigen.h"
#include "dynet/nodes-pickneglogsoftmax.h"

#include "dynet/nodes-impl-macros.h"

#ifdef __CUDACC__
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

// ************* PickNegLogSoftmax *************

#ifndef __CUDACC__

string PickNegLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  if(pval) {
    s << "pickneglogsoftmax(" << arg_names[0] << ")_{" << *pval << '}';
  } else {
    s << "pickneglogsoftmax(" << arg_names[0] << ")_{";
    string sep = "";
    for(auto v : *pvals) { s << sep << v; sep = ","; }
    s << '}';
  }
  return s.str();
}

Dim PickNegLogSoftmax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in PickNegLogSoftmax");
  DYNET_ARG_CHECK(LooksLikeVector(xs[0]), "Bad input dimensions in PickNegLogSoftmax: " << xs);
  DYNET_ARG_CHECK((pval == nullptr || xs[0].bd == 1),
                          "PickNegLogSoftmax was called with a single ID (" << *pval <<
                          "), but the expression under consideration had multiple mini-batch elements (" <<
                          xs[0].bd << "). A vector of IDs of size " << xs[0].bd << " must be passed instead.");
  DYNET_ARG_CHECK((pvals == nullptr || xs[0].bd == pvals->size()),
                          "The number of IDs passed to PickNegLogSoftmax (" << pvals->size() <<
                          "), did not match the number of mini-batch elements in the expression under consideration (" <<
                          xs[0].bd << "). These numbers must match.");
  return Dim({1}, xs[0].bd);
}

int PickNegLogSoftmax::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::pnls);
  const Dim &in_dim = cg.nodes[args[0]]->dim;
  s.add_dim(in_dim);
  return sm.get_idx(s);
}

std::vector<int> PickNegLogSoftmax::autobatch_concat(const ComputationGraph & cg) const {
  return vector<int>(1, 1);
}

Node* PickNegLogSoftmax::autobatch_pseudo_node(const ComputationGraph & cg,
                                        const std::vector<VariableIndex> & batch_ids) const {
  vector<unsigned> ids;
  PickNegLogSoftmax* ln;
  for(auto batch_id : batch_ids) {
    ln = static_cast<PickNegLogSoftmax*>(cg.nodes[batch_id]);
    if(ln->pval != nullptr)
      ids.push_back(*ln->pval);
    else
      for(auto word_id : *ln->pvals)
        ids.push_back(word_id);
  }
  return new PickNegLogSoftmax({(VariableIndex)1}, ids);
}

size_t PickNegLogSoftmax::aux_storage_size() const {
  return 2 * dim.batch_elems() * sizeof(float) + dim.batch_elems() * sizeof(unsigned int);
}

#endif

template<class MyDevice>
void PickNegLogSoftmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
    Tensor z(Dim({1},fx.d.bd), (float*)aux_mem, fx.device, DeviceMempool::FXS);
    Tensor m(Dim({1},fx.d.bd), (float*)aux_mem + fx.d.bd, fx.device, DeviceMempool::FXS);
    unsigned int *ids_dev = (unsigned int*)((float*)aux_mem + 2*fx.d.bd), *ids_host;
#ifdef __CUDACC__
    ids_host = (unsigned int*)malloc(fx.d.bd * sizeof(unsigned int));
#else
    ids_host = ids_dev;
#endif
    if(pval) {
      *ids_host = *pval;
      DYNET_ARG_CHECK(*pval < xs[0]->d.rows(),
                      "Index error in PickNegLogSoftmax: Index " << *pval << " out of bounds for input tensor " << xs[0]->d);
    } else {
      DYNET_ASSERT(pvals, "Neither single nor vector of elements available in PickNegLogSoftmax::forward");
      DYNET_ARG_CHECK(pvals->size() == fx.d.batch_elems(),
                              "In PickNegLogSoftmax::forward, number of elements in the passed-in index vector (" << pvals->size() << ")"
                              " did not match number of elements in mini-batch elements in expression (of dimension" << fx.d << ")");
      size_t batch_size = xs[0]->d.batch_size();
      for(unsigned b = 0; b < fx.d.bd; ++b) {
        DYNET_ARG_CHECK((*pvals)[b] < xs[0]->d.rows(),
                        "Index error in PickNegLogSoftmax: Index " << (*pvals)[b] << " out of bounds for input tensor " << xs[0]->d);
        ids_host[b] = batch_size * b + (*pvals)[b];
      }
    }
#ifdef __CUDACC__
    CUDA_CHECK(cudaMemcpyAsync(ids_dev, ids_host, fx.d.bd * sizeof(unsigned int), cudaMemcpyHostToDevice));
    TensorTools::logsumexp_dev(dev, *xs[0], m, z);
    dynet::gpu::sparse_to_dense_assign(fx.d.bd, ids_dev, xs[0]->v, fx.v);
    free(ids_host);
#else
    TensorTools::logsumexp_dev(dev, *xs[0], m, z);
    for(unsigned b = 0; b < fx.d.bd; ++b)
      fx.v[b] = xs[0]->v[ids_dev[b]];
#endif
    tvec(fx).device(*dev.edevice) = tvec(z) - tvec(fx);
  } else {
    DYNET_RUNTIME_ERR("PickNegLogSoftmax::forward not yet implemented for multiple columns");
  }
}

template<class MyDevice>
void PickNegLogSoftmax::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
    Tensor z(Dim({1},fx.d.batch_elems()), (float*)aux_mem, fx.device, DeviceMempool::FXS);
    unsigned int *ids_dev = (unsigned int*)((float*)aux_mem + 2*fx.d.bd);
#ifdef __CUDACC__
    Eigen::array<int, 2> bcast({(int)xs[0]->d[0],1});
    tb<1>(dEdxi).device(*dev.edevice) += (tb<1>(*xs[0]) - tb<1>(z).broadcast(bcast)).exp() * tb<1>(dEdf).broadcast(bcast);
    dynet::gpu::dense_to_sparse_subtract(fx.d.bd, ids_dev, dEdf.v, dEdxi.v);
#else
    // TODO: We want to do broadcasting here too, but it's slow
    for(unsigned b = 0; b < fx.d.bd; ++b) {
      tb<1>(dEdxi).chip<1>(b).device(*dev.edevice) += (tb<1>(*xs[0]).chip<1>(b) - z.v[b]).exp() * dEdf.v[b];
      dEdxi.v[ids_dev[b]] -= dEdf.v[b];
    }
#endif
  } else {
    DYNET_RUNTIME_ERR("PickNegLogSoftmax::backward not yet implemented for multiple columns");
  }
}
DYNET_NODE_INST_DEV_IMPL(PickNegLogSoftmax)

}
