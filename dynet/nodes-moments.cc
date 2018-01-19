#include "dynet/tensor-eigen.h"
#include "dynet/nodes-moments.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"
#include "dynet/simd-functors.h"

using namespace std;

namespace dynet {

// ************* Average *************

#ifndef __CUDACC__

string Average::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "average(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << ", " << arg_names[i];
  s << ")";
  return s.str();
}

Dim Average::dim_forward(const vector<Dim>& xs) const {
  Dim d(xs[0]);
  for (unsigned i = 1; i < xs.size(); ++i) {
    DYNET_ARG_CHECK(xs[0].single_batch() == xs[i].single_batch(),
                            "Mismatched input dimensions in Average: " << xs);
    d.bd = max(xs[i].bd, d.bd);
  }
  return d;
}

#endif

template<class MyDevice>
void Average::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
    return;
  }
  if (num_args == 2 && xs[0]->d.bd == xs[1]->d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]);
  else if (num_args == 3 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]) + tvec(*xs[2]);
  else if (num_args == 4 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd && xs[2]->d.bd == xs[3]->d.bd)
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(*xs[1]) + tvec(*xs[2]) + tvec(*xs[3]);
  else {
    bool allSameBatchSize = std::all_of(xs.begin(), xs.end(), [&](const Tensor* x) { return x->d.bd == xs[0]->d.bd;});
    if (allSameBatchSize) {
      // Since they are all the same batch size, we can easily unroll the addition (results in lower GPU latency by merging multiple adds together in one CUDA call):
      DYNET_ASSERT(num_args > 4, "Bad loop unrolling in Average::forward");        // If it was <=4, we would have handled it in the special cases above
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
  tvec(fx).device(*dev.edevice) = tvec(fx) / (float)xs.size();
}

template<class MyDevice>
void Average::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += (tvec(dEdf) / (float)xs.size());
}
DYNET_NODE_INST_DEV_IMPL(Average)

// ************* AverageColumns *************

#ifndef __CUDACC__

string AverageColumns::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "average_cols(matrix=" << arg_names[0] << ')';
  return s.str();
}

Dim AverageColumns::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1 || xs.size() == 2, "Failed input count check in AverageColumns");
  int bd = (xs.size() == 1 ? xs[0].bd : max(xs[0].bd, xs[1].bd));
  return Dim({xs[0].rows()}, bd);
}

#endif

template<class MyDevice>
void AverageColumns::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in AverageColumns");
  unsigned cols = xs[0]->d.cols();
#ifdef __CUDACC__
  // The reduction used on CPU is better, but not implemented in GPU
  t<1>(fx).device(*dev.edevice) = t<2>(*xs[0]).chip<1>(0);
  for(unsigned i = 1; i < cols; ++i)
    t<1>(fx).device(*dev.edevice) += t<2>(*xs[0]).chip<1>(i);
  t<1>(fx).device(*dev.edevice) = t<1>(fx) / (float)cols;
#else
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
  t<1>(fx).device(*dev.edevice) = t<2>(*xs[0]).sum(reduction_axis) / (float)cols;
#endif
}

template<class MyDevice>
void AverageColumns::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Eigen::array<Eigen::DenseIndex, 2> broadcasts = {1, xs[0]->d[1]};
  t<2>(dEdxi).device(*dev.edevice) += (t<2>(dEdf) / (float)xs[0]->d[1]).broadcast(broadcasts);
}
DYNET_NODE_INST_DEV_IMPL(AverageColumns)

// ************* MomentElements *************

#ifndef __CUDACC__

string MomentElements::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "moment_elems( expression=" << arg_names[0] << ", order=" << order << " )";
  return s.str();
}

Dim MomentElements::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in MomentElements")
  DYNET_ARG_CHECK(order>= 1, "Order of moment should be >=1 in MomentElements (recieved "<<order<<")")
  return Dim({1}, xs[0].bd);
}

#endif

template<class MyDevice>
void MomentElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in MomentElements::forward");
  Eigen::array<int, 1> red_axis; red_axis[0] = 0;
  if(order == 1)
    tb<0>(fx).device(*dev.edevice) = tbvec(*xs[0]).sum(red_axis) / (float) xs[0]->d.batch_size();
  else if (order == 2)
    tb<0>(fx).device(*dev.edevice) = tbvec(*xs[0]).square().sum(red_axis) / (float) xs[0]->d.batch_size();
  else
    tb<0>(fx).device(*dev.edevice) = tbvec(*xs[0]).pow(order).sum(red_axis) / (float) xs[0]->d.batch_size();
}

template<class MyDevice>
void MomentElements::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentElements::backward");
  Eigen::array<int, 2> bcast = {(int)xs[0]->d.batch_size(), 1};
  if (order == 1)
    tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).broadcast(bcast) / (float) xs[0]->d.batch_size();
  else if (order == 2)
    tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0])) * ( 2.f / (float) xs[0]->d.batch_size());
  else if (order == 3)
    tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0]).square()) * ( 3.f / (float) xs[0]->d.batch_size());
  else
    tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0]).pow(order - 1)) * ( (float) order / (float) xs[0]->d.batch_size());
}
DYNET_NODE_INST_DEV_IMPL(MomentElements)

// ************* MomentDimension *************

#ifndef __CUDACC__

string MomentDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "moment_dim(expression=" << arg_names[0] << ',';
  for(size_t i = 0; i < dims.size(); ++i)
    s << (i == 0?'{':',') << dims[i];
  s << "}), order="<<order;
  return s.str();
}

Dim MomentDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in MomentDimension");
  DYNET_ARG_CHECK(xs[0].nd <= 3, "MomentDimension implemented up to tensors of order 3 (with minibatch) for now")
  for (unsigned i = 0; i < dims.size(); ++i)
    DYNET_ARG_CHECK(dims[i] <= xs[0].nd, "dimension " << dims[i]<< " is out of bounds of tensor of order " << xs[0].nd << " in MomentDimension" )
  DYNET_ARG_CHECK(order>= 1, "Order of moment should be >=1 in MomentDimension (received "<<order<<")")
  DYNET_ARG_CHECK(dims.size()<=2, "Number of dimensions to reduce (excluding batch dimension) implemented up to 2 in MomentDimension (received "<< dims.size() <<")")
  if(dims.size()==0)
    DYNET_ARG_CHECK(include_batch_dim, "At least one dimension has to be reduced (including batch dimension) in MomentDimension")
  Dim ret(xs[0]);
  ret.delete_dims(dims, include_batch_dim);
  return ret;
}

#endif

template<class MyDevice>
void MomentDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");

  float n = 1.0;
  if(overwrite_n==0){
    for(unsigned i=0; i<dims.size(); i++) n *= (float) xs[0]->d[dims[i]];
    if(include_batch_dim) n *= xs[0]->d.bd;
  } else {
    n = overwrite_n;
  }

  if(dims.size()==0 && include_batch_dim){
    Eigen::array<int, 1> reduction_axis = {1};
    if(order == 1)
      tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).sum(reduction_axis) / n;
    else if (order == 2)
      tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).square().sum(reduction_axis) / n;
    else
      tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).pow(order).sum(reduction_axis) / n;
  } else if(dims.size()==1 && !include_batch_dim){
    // original code:
    Eigen::array<int, 1> reduction_axis = {(int)dims[0]};
    if(order == 1)
      tb<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis) / n;
    else if (order == 2)
      tb<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).square().sum(reduction_axis) / n;
    else
      tb<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).pow(order).sum(reduction_axis) / n;
  } else if(dims.size()==1 && include_batch_dim){
    Eigen::array<int, 2> reduction_axis = {(int)dims[0], 3};
    if(order == 1)
      t<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis) / n;
    else if (order == 2)
      t<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).square().sum(reduction_axis) / n;
    else
      t<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).pow(order).sum(reduction_axis) / n;
  } else if(dims.size()==2 && !include_batch_dim){
      Eigen::array<int, 2> reduction_axis = {(int)dims[0], (int)dims[1]};
      if(order == 1)
        tb<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis) / n;
      else if (order == 2)
        tb<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).square().sum(reduction_axis) / n;
      else
        tb<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).pow(order).sum(reduction_axis) / n;
  } else if(dims.size()==2 && include_batch_dim){
      Eigen::array<int, 3> reduction_axis = {(int)dims[0], (int)dims[1], 3};
      if(order == 1)
        t<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).sum(reduction_axis) / n;
      else if (order == 2)
        t<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).square().sum(reduction_axis) / n;
      else
        t<1>(fx).device(*dev.edevice) = tb<3>(*xs[0]).pow(order).sum(reduction_axis) / n;
  }
}

template<class MyDevice>
void MomentDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentDimension::backward");

  float n = 1.0;
  if(overwrite_n==0){
    for(unsigned i=0; i<dims.size(); i++) n *= (float) xs[0]->d[dims[i]];
    if(include_batch_dim) n *= xs[0]->d.bd;
  } else {
    n = overwrite_n;
  }

  if(dims.size()==0 && include_batch_dim){
    Eigen::array<int, 2> bcast = {1, (int)xs[0]->d.bd};
    if (order == 1)
      tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).broadcast(bcast) / n;
    else if (order == 2)
      tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0])) * ( 2.f / n);
    else if (order == 3)
      tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0]).square()) * ( 3.f / n);
    else
      tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0]).pow(order - 1)) * ( (float) order / n);
  } else if(dims.size()==1 && !include_batch_dim){
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]];
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1;
    if (order == 1)
      tb<3>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).reshape(morph).broadcast(bcast) / n;
    else if (order == 2)
      tb<3>(dEdxi).device(*dev.edevice) += (tb<2>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0])) * ( 2.f / n);
    else if (order == 3)
      tb<3>(dEdxi).device(*dev.edevice) += (tb<2>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).square()) * ( 3.f / n);
    else
      tb<3>(dEdxi).device(*dev.edevice) += (tb<2>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).pow(order - 1)) * ( (float) order / n);
  } else if(dims.size()==1 && include_batch_dim){
      Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[3] = xs[0]->d.bd;
      Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1;
      if (order == 1)
        tb<3>(dEdxi).device(*dev.edevice) += t<2>(dEdf).reshape(morph).broadcast(bcast) / n;
      else if (order == 2)
        tb<3>(dEdxi).device(*dev.edevice) += (t<2>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0])) * ( 2.f / n);
      else if (order == 3)
        tb<3>(dEdxi).device(*dev.edevice) += (t<2>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).square()) * ( 3.f / n);
      else
        tb<3>(dEdxi).device(*dev.edevice) += (t<2>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).pow(order - 1)) * ( (float) order / n);
  } else if(dims.size()==2 && !include_batch_dim){
      Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]];
      Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1; morph[dims[1]] = 1;
      if (order == 1)
        tb<3>(dEdxi).device(*dev.edevice) += tb<1>(dEdf).reshape(morph).broadcast(bcast) / n;
      else if (order == 2)
        tb<3>(dEdxi).device(*dev.edevice) += (tb<1>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0])) * ( 2.f / n);
      else if (order == 3)
        tb<3>(dEdxi).device(*dev.edevice) += (tb<1>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).square()) * ( 3.f / n);
      else
        tb<3>(dEdxi).device(*dev.edevice) += (tb<1>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).pow(order - 1)) * ( (float) order / n);
  } else if(dims.size()==2 && include_batch_dim){
      Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]]; bcast[3] = xs[0]->d.bd;
      Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1; morph[dims[1]] = 1;
      if (order == 1)
        tb<3>(dEdxi).device(*dev.edevice) += t<1>(dEdf).reshape(morph).broadcast(bcast) / n;
      else if (order == 2)
        tb<3>(dEdxi).device(*dev.edevice) += (t<1>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0])) * ( 2.f / n);
      else if (order == 3)
        tb<3>(dEdxi).device(*dev.edevice) += (t<1>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).square()) * ( 3.f / n);
      else
        tb<3>(dEdxi).device(*dev.edevice) += (t<1>(dEdf).reshape(morph).broadcast(bcast) * tb<3>(*xs[0]).pow(order - 1)) * ( (float) order / n);
  }
}
DYNET_NODE_INST_DEV_IMPL(MomentDimension)

// ************* MomentBatches *************

#ifndef __CUDACC__

string MomentBatches::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "moment_batches( expression=" << arg_names[0] << ", order=" << order << " )";
  return s.str();
}

Dim MomentBatches::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in MomentBatches")
  DYNET_ARG_CHECK(order>= 1, "Order of moment should be >=1 in MomentBatches (recieved "<<order<<")")
  return xs[0].single_batch();
}

#endif

template<class MyDevice>
void MomentBatches::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in MomentBatches::forward");
  Eigen::array<int, 1> red_axis; red_axis[0] = 1;
  if(order == 1)
    tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).sum(red_axis) / (float) xs[0]->d.bd;
  else if (order == 2)
    tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).square().sum(red_axis) / (float) xs[0]->d.bd;
  else
    tvec(fx).device(*dev.edevice) = tbvec(*xs[0]).pow(order).sum(red_axis) / (float) xs[0]->d.bd;
}

template<class MyDevice>
void MomentBatches::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentBatches::backward");
  Eigen::array<int, 2> bcast = {1, (int)xs[0]->d.bd};
  if (order == 1)
    tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf).broadcast(bcast) / (float) xs[0]->d.bd;
  else if (order == 2)
    tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0])) * ( 2.f / (float) xs[0]->d.bd);
  else if (order == 3)
    tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0]).square()) * ( 3.f / (float) xs[0]->d.bd);
  else
    tbvec(dEdxi).device(*dev.edevice) += (tbvec(dEdf).broadcast(bcast) * tbvec(*xs[0]).pow(order - 1)) * ( (float) order / (float) xs[0]->d.bd);
}
DYNET_NODE_INST_DEV_IMPL(MomentBatches)

// ************* StdElements *************

#ifndef __CUDACC__

string StdElements::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "std_elems( expression=" << arg_names[0] << " )";
  return s.str();
}

Dim StdElements::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in StdElements")
  return Dim({1}, xs[0].bd);
}

#endif

template<class MyDevice>
void StdElements::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in StdElements::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  Eigen::array<ptrdiff_t, 2> newaxis = {1, xs[0]->d.bd};
  float n = (float) xs[0]->d.batch_size();
  tb<0>(fx).device(*dev.edevice) = ((tbvec(*xs[0]) - (tbvec(*xs[0]).sum(red_axis).reshape(newaxis) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
}

template<class MyDevice>
void StdElements::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 1, "Failed dimension check in StdElements::backward");
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  Eigen::array<ptrdiff_t, 2> newaxis = {1, xs[0]->d.bd};
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  float n = (float) xs[0]->d.batch_size();
  tbvec(dEdxi).device(*dev.edevice) +=  (2 / n) * (tbvec(*xs[0]) - (tbvec(*xs[0]).sum(red_axis).reshape(newaxis) / n).broadcast(bcast)) * (tbvec(fx).binaryExpr(tbvec(dEdf), FSqrtBackward())).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdElements)

// ************* StdDimension *************

#ifndef __CUDACC__

string StdDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "std_dim(expression=" << arg_names[0] << ',';
  for(size_t i = 0; i < dims.size(); ++i)
    s << (i == 0?'{':',') << dims[i];
  s << "})";
  return s.str();
}

Dim StdDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in StdDimension");
  DYNET_ARG_CHECK(xs[0].nd <= 3, "StdDimension implemented up to tensors of order 3 (with minibatch) for now")
  for (unsigned i = 0; i < dims.size(); ++i)
    DYNET_ARG_CHECK(dims[i] <= xs[0].nd, "dimension " << dims[i]<< " is out of bounds of tensor of order " << xs[0].nd << " in StdDimension" )
  DYNET_ARG_CHECK(dims.size()<=2, "Number of dimensions to reduce (excluding batch dimension) implemented up to 2 in StdDimension (received "<< dims.size() <<")")
  if(dims.size()==0)
    DYNET_ARG_CHECK(include_batch_dim, "At least one dimension has to be reduced (including batch dimension) in StdDimension")
  Dim ret(xs[0]);
  ret.delete_dims(dims, include_batch_dim);
  return ret;
}

#endif

template<class MyDevice>
void StdDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");

  float n = 1.0;
  if(overwrite_n==0){
    for(unsigned i=0; i<dims.size(); i++) n *= (float) xs[0]->d[dims[i]];
    if(include_batch_dim) n *= xs[0]->d.bd;
  } else {
    n = overwrite_n;
  }

  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];

  if(dims.size()==0 && include_batch_dim){
    Eigen::array<ptrdiff_t, 1> red_axis = {1};
    Eigen::array<ptrdiff_t, 2> morph = {xs[0]->d.batch_size(), 1};
    Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
    Tensor mean(Dim({xs[0]->d.batch_size()}, 1), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tbvec(mean).device(*dev.edevice) = (tbvec(*xs[0]).sum(red_axis).reshape(morph) / n);
    tvec(fx).device(*dev.edevice) = ((tbvec(*xs[0]) - tbvec(mean).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
  } else if(dims.size()==1 && !include_batch_dim){
    Eigen::array<int, 1> red_axis = {(int)dims[0]};
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1;
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]];
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    tb<2>(fx).device(*dev.edevice) = ((tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
  } else if(dims.size()==1 && include_batch_dim){
    Eigen::array<int, 2> red_axis = {(int)dims[0], (int)3};
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1;
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[3] = xs[0]->d.bd;
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    t<2>(fx).device(*dev.edevice) = ((tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
  } else if(dims.size()==2 && !include_batch_dim){
    Eigen::array<int, 2> red_axis = {(int)dims[0], (int)dims[1]};
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1; morph[dims[1]] = 1;
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]];
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    tb<1>(fx).device(*dev.edevice) = ((tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
  } else if(dims.size()==2 && include_batch_dim){
    Eigen::array<int, 3> red_axis = {(int)dims[0], (int)dims[1], (int)3};
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1; morph[dims[1]] = 1;
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]];  bcast[dims[1]] = xs[0]->d[dims[1]]; bcast[3] = xs[0]->d.bd;
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    t<1>(fx).device(*dev.edevice) = ((tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
  }
  scratch_allocator->free();

}

template<class MyDevice>
void StdDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in StdDimension::backward");

  float n = 1.0;
  if(overwrite_n==0){
    for(unsigned i=0; i<dims.size(); i++) n *= (float) xs[0]->d[dims[i]];
    if(include_batch_dim) n *= xs[0]->d.bd;
  } else {
    n = overwrite_n;
  }

  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];

  if(dims.size()==0 && include_batch_dim){
    Eigen::array<ptrdiff_t, 1> red_axis = {1};
    Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
    Eigen::array<ptrdiff_t, 2> morph = {xs[0]->d.batch_size(), 1};
    Tensor mean(Dim({xs[0]->d.batch_size()}, 1), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tbvec(mean).device(*dev.edevice) = (tbvec(*xs[0]).sum(red_axis).reshape(morph) / n);
    tbvec(dEdxi).device(*dev.edevice) +=  (2 / n) * (tbvec(*xs[0]) - tbvec(mean).broadcast(bcast)) * (tbvec(fx).binaryExpr(tbvec(dEdf), FSqrtBackward())).broadcast(bcast);
  } else if(dims.size()==1 && !include_batch_dim){
    Eigen::array<int, 1> red_axis = {(int)dims[0]};
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]];
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1;
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    tb<3>(dEdxi).device(*dev.edevice) +=  (2 / n) * (tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)) * (tb<2>(fx).binaryExpr(tb<2>(dEdf), FSqrtBackward())).reshape(morph).broadcast(bcast);
  } else if(dims.size()==1 && include_batch_dim){
    Eigen::array<int, 2> red_axis = {(int)dims[0], 3};
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[3] = xs[0]->d.bd;
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1;
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    tb<3>(dEdxi).device(*dev.edevice) +=  (2 / n) * (tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)) * (t<2>(fx).binaryExpr(t<2>(dEdf), FSqrtBackward())).reshape(morph).broadcast(bcast);
  } else if(dims.size()==2 && !include_batch_dim){
    Eigen::array<int, 2> red_axis = {(int)dims[0], (int)dims[1]};
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]];
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dims[0]] = 1; morph[dims[1]] = 1;
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    tb<3>(dEdxi).device(*dev.edevice) +=  (2 / n) * (tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)) * (tb<1>(fx).binaryExpr(tb<1>(dEdf), FSqrtBackward())).reshape(morph).broadcast(bcast);
  } else if(dims.size()==2 && include_batch_dim){
    Eigen::array<int, 3> red_axis = {(int)dims[0], (int)dims[1], 3};
    Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dims[0]] = xs[0]->d[dims[0]]; bcast[dims[1]] = xs[0]->d[dims[1]]; bcast[3] = xs[0]->d.bd;
    Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)1}; morph[dims[0]] = 1; morph[dims[1]] = 1;
    Tensor mean(Dim({(unsigned)morph[0], (unsigned)morph[1], (unsigned)morph[2]}, (unsigned)morph[3]), nullptr, fx.device, fx.mem_pool);
    mean.v = static_cast<float*>(scratch_allocator->allocate(mean.d.size() * sizeof(float)));
    tb<3>(mean).device(*dev.edevice) = (tb<3>(*xs[0]).sum(red_axis).reshape(morph) / n);
    tb<3>(dEdxi).device(*dev.edevice) +=  (2 / n) * (tb<3>(*xs[0]) - tb<3>(mean).broadcast(bcast)) * (t<1>(fx).binaryExpr(t<1>(dEdf), FSqrtBackward())).reshape(morph).broadcast(bcast);
  }
  scratch_allocator->free();

}
DYNET_NODE_INST_DEV_IMPL(StdDimension)


// ************* StdBatches *************

#ifndef __CUDACC__

string StdBatches::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "std_batches( expression=" << arg_names[0] << " )";
  return s.str();
}

Dim StdBatches::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in StdBatches")

  return xs[0].single_batch();
}

#endif

template<class MyDevice>
void StdBatches::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in StdBatches::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {1};
  Eigen::array<ptrdiff_t, 2> newaxis = {xs[0]->d.batch_size(), 1};
  Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
  float n = (float)xs[0]->d.bd;
  t<1>(fx).device(*dev.edevice) = ((tbvec(*xs[0]) - (tbvec(*xs[0]).sum(red_axis).reshape(newaxis) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
}

template<class MyDevice>
void StdBatches::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 1, "Failed dimension check in StdBatches::backward");
  Eigen::array<ptrdiff_t, 1> red_axis = {1};
  Eigen::array<ptrdiff_t, 2> bcast = {1, xs[0]->d.bd};
  Eigen::array<ptrdiff_t, 2> newaxis = {xs[0]->d.batch_size(), 1};
  float n = (float)xs[0]->d.bd;
  tbvec(dEdxi).device(*dev.edevice) +=  (2 / n) * (tbvec(*xs[0]) - (tbvec(*xs[0]).sum(red_axis).reshape(newaxis) / n).broadcast(bcast)) * (tbvec(fx).binaryExpr(tbvec(dEdf), FSqrtBackward())).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdBatches)

}
