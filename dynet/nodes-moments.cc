#include "dynet/nodes-moments.h"

#include "dynet/nodes-macros.h"
#include "dynet/functors.h"

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
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
    return;
  }
  if (num_args == 2 && xs[0]->d.bd == xs[1]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec();
  else if (num_args == 3 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec();
  else if (num_args == 4 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd && xs[2]->d.bd == xs[3]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();
  else {
    bool allSameBatchSize = std::all_of(xs.begin(), xs.end(), [&](const Tensor* x) { return x->d.bd == xs[0]->d.bd;});
    if (allSameBatchSize) {
      // Since they are all the same batch size, we can easily unroll the addition (results in lower GPU latency by merging multiple adds together in one CUDA call):
      DYNET_ASSERT(num_args > 4, "Bad loop unrolling in Average::forward");        // If it was <=4, we would have handled it in the special cases above
      fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();

      const unsigned remainder = (num_args - 4 ) % 4;
      switch (remainder) {
        case 0: break;
        case 1: fx.tvec().device(*dev.edevice) += xs[4]->tvec(); break;
        case 2: fx.tvec().device(*dev.edevice) += xs[4]->tvec() + xs[5]->tvec(); break;
        case 3: fx.tvec().device(*dev.edevice) += xs[4]->tvec() + xs[5]->tvec() + xs[6]->tvec(); break;
      }
      for (unsigned i = 4 + remainder; i < num_args; i += 4)
        fx.tvec().device(*dev.edevice) += xs[i]->tvec() + xs[i + 1]->tvec() + xs[i + 2]->tvec() + xs[i + 3]->tvec();
    }
    else {
      // Not all the same batch size, so need to broadcast in the cases where they differ
      TensorTools::zero(fx);
#ifdef __CUDACC__
      Eigen::array<int, 2> bcast({ 1, (int)fx.d.bd });
#endif
      for (unsigned i = 0; i < num_args; ++i) {
        if (xs[i]->d.bd == fx.d.bd) {
          fx.tvec().device(*dev.edevice) += xs[i]->tvec();
        }
        else {
#ifdef __CUDACC__
          fx.tbvec().device(*dev.edevice) += xs[i]->tbvec().broadcast(bcast);
#else
          for (unsigned b = 0; b < fx.d.bd; ++b)
            fx.tbvec().chip<1>(b).device(*dev.edevice) += xs[i]->tvec();
#endif
        }
      }
    }
  }
  fx.tvec().device(*dev.edevice) = fx.tvec() / (float)xs.size();
}

template<class MyDevice>
void Average::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += (dEdf.tvec() / (float)xs.size());
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
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().chip<1>(0);
  for(unsigned i = 1; i < cols; ++i)
    fx.t<1>().device(*dev.edevice) += xs[0]->t<2>().chip<1>(i);
  fx.t<1>().device(*dev.edevice) = fx.t<1>() / (float)cols;
#else
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().sum(reduction_axis) / (float)cols;
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
  dEdxi.t<2>().device(*dev.edevice) += (dEdf.t<2>() / (float)xs[0]->d[1]).broadcast(broadcasts);
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
    fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().sum(red_axis) / (float) xs[0]->d.batch_size();
  else if (order == 2)
    fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().square().sum(red_axis) / (float) xs[0]->d.batch_size();
  else
    fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().pow(order).sum(red_axis) / (float) xs[0]->d.batch_size();
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
    dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().broadcast(bcast) / (float) xs[0]->d.batch_size();
  else if (order == 2)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec()) * ( 2.f / (float) xs[0]->d.batch_size());
  else if (order == 3)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().square()) * ( 3.f / (float) xs[0]->d.batch_size());
  else
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().pow(order - 1)) * ( (float) order / (float) xs[0]->d.batch_size());
}
DYNET_NODE_INST_DEV_IMPL(MomentElements)

// ************* MomentDimension *************

#ifndef __CUDACC__

string MomentDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "moment_dim(expression=" << arg_names[0] << ',' << dimension << ", order="<<order<<'}';
  return s.str();
}

Dim MomentDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in MomentDimension");
  DYNET_ARG_CHECK(xs[0].nd <= 3, "MomentDimension implemented up to tensors of order 3 (with minibatch) for now")
  DYNET_ARG_CHECK(dimension < xs[0].nd, "dimension " << dimension << " is out of bounds of tensor of order " << xs[0].nd << " in MomentDimension" )
  DYNET_ARG_CHECK(order>= 1, "Order of moment should be >=1 in MomentDimension (recieved "<<order<<")")
  Dim ret(xs[0]);
  ret.delete_dim(dimension);
  return ret;
}

#endif

template<class MyDevice>
void MomentDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in MomentDimension");
  Eigen::array<int, 1> reduction_axis = {(int)dimension};
  float n = (float) xs[0]->d[dimension];
  if(order == 1)
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().sum(reduction_axis) / n;
  else if (order == 2)
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().square().sum(reduction_axis) / n;
  else
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<3>().pow(order).sum(reduction_axis) / n;
}

template<class MyDevice>
void MomentDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MomentDimension::backward");
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = xs[0]->d[dimension];
  Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dimension] = 1;
  float n = (float) xs[0]->d[dimension];
  if (order == 1)
    dEdxi.tb<3>().device(*dev.edevice) += dEdf.tb<2>().reshape(morph).broadcast(bcast) / n;
  else if (order == 2)
    dEdxi.tb<3>().device(*dev.edevice) += (dEdf.tb<2>().reshape(morph).broadcast(bcast) * xs[0]->tb<3>()) * ( 2.f / n);
  else if (order == 3)
    dEdxi.tb<3>().device(*dev.edevice) += (dEdf.tb<2>().reshape(morph).broadcast(bcast) * xs[0]->tb<3>().square()) * ( 3.f / n);
  else
    dEdxi.tb<3>().device(*dev.edevice) += (dEdf.tb<2>().reshape(morph).broadcast(bcast) * xs[0]->tb<3>().pow(order - 1)) * ( (float) order / n);
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
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().sum(red_axis) / (float) xs[0]->d.bd;
  else if (order == 2)
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().square().sum(red_axis) / (float) xs[0]->d.bd;
  else
    fx.tvec().device(*dev.edevice) = xs[0]->tbvec().pow(order).sum(red_axis) / (float) xs[0]->d.bd;
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
    dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().broadcast(bcast) / (float) xs[0]->d.bd;
  else if (order == 2)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec()) * ( 2.f / (float) xs[0]->d.bd);
  else if (order == 3)
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().square()) * ( 3.f / (float) xs[0]->d.bd);
  else
    dEdxi.tbvec().device(*dev.edevice) += (dEdf.tbvec().broadcast(bcast) * xs[0]->tbvec().pow(order - 1)) * ( (float) order / (float) xs[0]->d.bd);
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
  fx.tb<0>().device(*dev.edevice) = ((xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
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
  dEdxi.tbvec().device(*dev.edevice) +=  (2 / n) * (xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)) * (fx.tbvec().binaryExpr(dEdf.tbvec(), FSqrtBackward())).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdElements)

// ************* StdDimension *************

#ifndef __CUDACC__

string StdDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "moment_dim(expression=" << arg_names[0] << ',' << dimension <<'}';
  return s.str();
}

Dim StdDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in StdDimension");
  DYNET_ARG_CHECK(xs[0].nd <= 3, "StdDimension implemented up to tensors of order 3 (with minibatch) for now")
  DYNET_ARG_CHECK(dimension < xs[0].nd, "dimension " << dimension << " is out of bounds of tensor of order " << xs[0].nd << " in StdDimension" )
  Dim ret(xs[0]);
  ret.delete_dim(dimension);
  return ret;
}

#endif

template<class MyDevice>
void StdDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in StdDimension");
  Eigen::array<int, 1> red_axis = {(int)dimension};
  Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dimension] = 1;
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = xs[0]->d[dimension];
  float n = (float) xs[0]->d[dimension];
  fx.tb<2>().device(*dev.edevice) = ((xs[0]->tb<3>() - (xs[0]->tb<3>().sum(red_axis).reshape(morph) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
}

template<class MyDevice>
void StdDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in StdDimension::backward");
  Eigen::array<int, 1> red_axis = {(int)dimension};
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = xs[0]->d[dimension];
  Eigen::array<int, 4> morph = {(int)xs[0]->d[0],(int)xs[0]->d[1],(int)xs[0]->d[2],(int)xs[0]->d.bd}; morph[dimension] = 1;
  float n = (float) xs[0]->d[dimension];
  dEdxi.tb<3>().device(*dev.edevice) +=  (2 / n) * (xs[0]->tb<3>() - (xs[0]->tb<3>().sum(red_axis).reshape(morph) / n).broadcast(bcast)) * (fx.tb<2>().binaryExpr(dEdf.tb<2>(), FSqrtBackward())).reshape(morph).broadcast(bcast);

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
  fx.t<1>().device(*dev.edevice) = ((xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)).square().sum(red_axis) / n).sqrt();
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
  dEdxi.tbvec().device(*dev.edevice) +=  (2 / n) * (xs[0]->tbvec() - (xs[0]->tbvec().sum(red_axis).reshape(newaxis) / n).broadcast(bcast)) * (fx.tbvec().binaryExpr(dEdf.tbvec(), FSqrtBackward())).broadcast(bcast);

}
DYNET_NODE_INST_DEV_IMPL(StdBatches)

}
