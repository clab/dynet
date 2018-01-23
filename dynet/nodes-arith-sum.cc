#include "dynet/tensor-eigen.h"
#include "dynet/nodes-arith-sum.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* Sum *************

#ifndef __CUDACC__

string Sum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + " << arg_names[i];
  return s.str();
}

Dim Sum::dim_forward(const vector<Dim>& xs) const {
  Dim d = xs[0].truncate();
  unsigned int batch = d.bd;
  for (unsigned i = 1; i < xs.size(); ++i) {
    DYNET_ARG_CHECK(d.single_batch() == xs[i].truncate().single_batch(),
                            "Mismatched input dimensions in Sum: " << xs);
    batch = max(xs[i].bd, batch);
  }
  d = xs[0]; d.bd = batch;
  return d;
}

int Sum::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const {
  Sig s(nt::sum);
  s.add_node(args.size());
  // Two cases:
  // If unbatched, it's just an elementwise addition
  // TODO: This will be more efficient if we identify arguments that are used
  //       multiple times (e.g. bias vectors)
  if(dim.bd == 1) {
    s.add_int(-2);
  // Otherwise, make sure the dimensions match and that batched nodes don't intersect
  } else {
    s.add_dim(dim);
    for(auto ai : args) {
      s.add_int(cg.nodes[ai]->dim.bd == 1 ? ai : -1);
    }
  }
  return sm.get_idx(s);
}

std::vector<int> Sum::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 1);
  // If batched, true if multiple batched input as well
  if(dim.bd != 1)
    for(size_t i = 0; i < args.size(); ++i)
      ret[i] = cg.nodes[args[i]]->dim.bd == 1 ? 0 : 1;
  return ret;
}

#endif

template<class MyDevice>
void Sum::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
  else if (num_args == 2 && xs[0]->d.bd == xs[1]->d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]);
  else if (num_args == 3 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]) + tvec(*xs[2]);
  else if (num_args == 4 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd && xs[2]->d.bd == xs[3]->d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]) + tvec(*xs[2]) + tvec(*xs[3]);
  else {
    bool allSameBatchSize = std::all_of(xs.begin(), xs.end(), [&](const Tensor* x) { return x->d.bd == xs[0]->d.bd;});
    if (allSameBatchSize) {
      // Since they are all the same batch size, we can easily unroll the addition (results in lower GPU latency by merging multiple adds together in one CUDA call):
      DYNET_ASSERT(num_args > 4, "Bad loop unrolling in Sum::forward");        // If it was <=4, we would have handled it in the special cases above
      tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]) + tvec(*xs[2]) + tvec(*xs[3]);

      const unsigned remainder = (num_args - 4 ) % 4;
      switch (remainder) {
        case 0: break;
        case 1: tvec(fx).device(*dev.edevice) += tvec(*xs[4]); break;
        case 2: tvec(fx).device(*dev.edevice) += tvec(*xs[4]) + tvec(*xs[5]); break;
        case 3: tvec(fx).device(*dev.edevice) += tvec(*xs[4]) + tvec(*xs[5]) + tvec(*xs[6]); break;
      }
      for (unsigned i = 4 + remainder; i < num_args; i += 4)
        tvec(fx).device(*dev.edevice) += tvec(*xs[i]) + tvec(*xs[i + 1]) + tvec(*xs[i + 2]) + tvec(*xs[i + 3]);
    }
    else {
      // Not all the same batch size, so need to broadcast in the cases where they differ
      TensorTools::zero(fx);
#ifdef __CUDACC__
      Eigen::array<int, 2> bcast({ 1, (int)fx.d.bd });
#endif
      for (unsigned i = 0; i < num_args; ++i) {
        if (xs[i]->d.bd == fx.d.bd) {
          tvec(fx).device(*dev.edevice) += tvec(*xs[i]);
        }
        else {
#ifdef __CUDACC__
          tbvec(fx).device(*dev.edevice) += tbvec(*xs[i]).broadcast(bcast);
#else
          for (unsigned b = 0; b < fx.d.bd; ++b)
            tbvec(fx).chip<1>(b).device(*dev.edevice) += tvec(*xs[i]);
#endif
        }
      }
    }
  }
}

template<class MyDevice>
void Sum::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if(dEdxi.d.bd == fx.d.bd) {
    tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
  } else {
    Eigen::array<int, 1> red_axis = {1};
    tvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).sum(red_axis);
  }
}
DYNET_NODE_INST_DEV_IMPL(Sum)

// ************* SumElements *************

#ifndef __CUDACC__

string SumElements::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_elems( " << arg_names[0] << " )";
  return s.str();
}

Dim SumElements::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in SumElements")
  return Dim({1}, xs[0].bd);
}

#endif

template<class MyDevice>
void SumElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SumElements::forward");
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  tb<0>(fx).device(*dev.edevice) = tbvec(*xs[0]).sum(red_axis);
}

template<class MyDevice>
void SumElements::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in SumElements::backward");
  Eigen::array<int, 2> bcast = {(int)xs[0]->d.batch_size(), 1};
  tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(SumElements)

// ************* SumDimension *************

#ifndef __CUDACC__

string SumDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_dim(expression=" << arg_names[0] << ',';
  for(size_t i = 0; i < dims.size(); ++i)
    s << (i == 0?'{':',') << dims[i];
  s << "})";
  return s.str();
}

Dim SumDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  DYNET_ARG_CHECK(xs[0].nd <= 3, "SumDimension implemented up to tensors of order 3 (with minibatch) for now")
  for (unsigned i = 0; i < dims.size(); ++i)
    DYNET_ARG_CHECK(dims[i] <= xs[0].nd, "dimension " << dims[i]<< " is out of bounds of tensor of order " << xs[0].nd << " in SumDimension" )
  DYNET_ARG_CHECK(dims.size()<=2, "Number of dimensions to reduce (excluding batch dimension) implemented up to 2 in SumDimension (received "<< dims.size() <<")")
  if(dims.size()==0)
    DYNET_ARG_CHECK(include_batch_dim, "At least one dimension has to be reduced (including batch dimension) in SumDimension")
  Dim ret(xs[0]);
  ret.delete_dims(dims, include_batch_dim);
  return ret;
}

#endif

template<class MyDevice>
void SumDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");

  if(dims.size()==0 && include_batch_dim){
    Eigen::array<int, 1> reduction_axis = {1};
    tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).sum(reduction_axis);
  } else if(dims.size()==1 && !include_batch_dim){
    Eigen::array<int, 1> reduction_axis = {(int)dims[0]};
    tb<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis);
  } else if(dims.size()==1 && include_batch_dim){
    Eigen::array<int, 2> reduction_axis = {(int)dims[0], 3};
    t<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis);
  } else if(dims.size()==2 && !include_batch_dim){
    Eigen::array<int, 2> reduction_axis = {(int)dims[0], (int)dims[1]};
    tb<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis);
  } else if(dims.size()==2 && include_batch_dim){
    Eigen::array<int, 3> reduction_axis = {(int)dims[0], (int)dims[1], 3};
    t<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis);
  }
}

template<class MyDevice>
void SumDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in SumDimension::backward");

  if(dims.size()==0 && include_batch_dim){
    Eigen::array<int, 2> bcast = {1, (int)xs[0]->d.bd};
    tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).broadcast(bcast);
  } else if(dims.size()==1 && !include_batch_dim){
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]];
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1;
    tb<3>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).reshape(morph).broadcast(bcast);
  } else if(dims.size()==1 && include_batch_dim){
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[3] = xs[0]->d.bd;
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1;
    tb<3>(dEdxi).device(*dev.edevice) += t<2>(dEdf).reshape(morph).broadcast(bcast);
  } else if(dims.size()==2 && !include_batch_dim){
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]];
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1; morph[dims[1]] = 1;
    tb<3>(dEdxi).device(*dev.edevice) += tb<1>(dEdf).reshape(morph).broadcast(bcast);
  } else if(dims.size()==2 && include_batch_dim){
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]]; bcast[3] = xs[0]->d.bd;
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1; morph[dims[1]] = 1;
    tb<3>(dEdxi).device(*dev.edevice) += t<1>(dEdf).reshape(morph).broadcast(bcast);
  }
}
DYNET_NODE_INST_DEV_IMPL(SumDimension)

// ************* AddVectorToAllColumns *************

#ifndef __CUDACC__

string AddVectorToAllColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "colwise_add(" << arg_names[0] << ", " << arg_names[1] << ')';
  return os.str();
}

Dim AddVectorToAllColumns::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 &&
                          xs[0].rows() == xs[1].rows() &&
                          xs[0].ndims() == 2 &&
                          (xs[1].ndims() == 1 || (xs[1].ndims() == 2 && xs[1].cols() == 1)),
                          "Bad input dimensions in AddVectorToAllColumns: " << xs);
  return Dim({xs[0][0], xs[0][1]}, max(xs[0].bd,xs[1].bd));
}

#endif

template<class MyDevice>
void AddVectorToAllColumns::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  // Broadcasting is slow on CPU, so split codepaths
#ifdef __CUDACC__
  if(xs[0]->d.bd >= xs[1]->d.bd) {
    Eigen::array<int, 3> bcasts = {1, (int)xs[0]->d[1], (int)(xs[0]->d.bd/xs[1]->d.bd)};
    tb<2>(fx).device(*dev.edevice) = tb<2>(*xs[0]) + tb<2>(*xs[1]).broadcast(bcasts);
  } else {
    DYNET_ASSERT(xs[0]->d.bd == 1,
                 "Bad dimensions in AddVectorToAllColumns::forward: " << xs[0]->d << ", " << xs[1]->d);
    Eigen::array<int, 3> bcasts0 = {1, 1, (int)xs[1]->d.bd};
    Eigen::array<int, 3> bcasts1 = {1, (int)xs[0]->d[1], 1};
    tb<2>(fx).device(*dev.edevice) = tb<2>(*xs[0]).broadcast(bcasts0) + tb<2>(*xs[1]).broadcast(bcasts1);
  }
#else
  // First, add the matrix
  if(xs[0]->d.bd == fx.d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
  else
    for(size_t b = 0; b < fx.d.bd; ++b)
      tbvec(fx).chip<1>(b).device(*dev.edevice) = tvec(*xs[0]);
  // Second, add the columns
  if(xs[1]->d.bd == fx.d.bd) {
    for(size_t i = 0; i < xs[0]->d[1]; ++i)
      tb<2>(fx).chip<1>(i).device(*dev.edevice) += tb<1>(*xs[1]);
  } else {
    for(size_t b = 0; b < fx.d.bd; ++b)
      for(size_t i = 0; i < fx.d[1]; ++i)
        tb<2>(fx).chip<2>(b).chip<1>(i).device(*dev.edevice) += t<1>(*xs[1]);
  }
#endif
}

template<class MyDevice>
void AddVectorToAllColumns::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in AddVetorToAllColumns::backward");
  // TODO: profile on CPU and see whether the chip version is better
  if (i == 0) { // x
    if(dEdf.d.bd == dEdxi.d.bd) {
      tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
    } else {
      Eigen::array<int, 1> red_axis = {2};
      t<2>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).sum(red_axis);
    }
  } else { // bias
    if(dEdf.d.bd == dEdxi.d.bd) {
      Eigen::array<int, 1> red_axis = {1};
      tb<1>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).sum(red_axis);
    } else {
      DYNET_ASSERT(dEdxi.d.bd == 1,
                   "Bad dimensions in AddVectorToAllColumns::backward: " << xs[0]->d << ", " << xs[1]->d);
      Eigen::array<int, 2> red_axis = {1,2};
      t<1>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).sum(red_axis);
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(AddVectorToAllColumns)

}
