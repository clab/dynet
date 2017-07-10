#include "dynet/nodes-arith-sum.h"

#include "dynet/nodes-macros.h"

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
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  else if (num_args == 2 && xs[0]->d.bd == xs[1]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec();
  else if (num_args == 3 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec();
  else if (num_args == 4 && xs[0]->d.bd == xs[1]->d.bd && xs[1]->d.bd == xs[2]->d.bd && xs[2]->d.bd == xs[3]->d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec() + xs[1]->tvec() + xs[2]->tvec() + xs[3]->tvec();
  else {
    bool allSameBatchSize = std::all_of(xs.begin(), xs.end(), [&](const Tensor* x) { return x->d.bd == xs[0]->d.bd;});
    if (allSameBatchSize) {
      // Since they are all the same batch size, we can easily unroll the addition (results in lower GPU latency by merging multiple adds together in one CUDA call):
      DYNET_ASSERT(num_args > 4, "Bad loop unrolling in Sum::forward");        // If it was <=4, we would have handled it in the special cases above
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
}

template<class MyDevice>
void Sum::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if(dEdxi.d.bd == fx.d.bd) {
    dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
  } else {
    Eigen::array<int, 1> red_axis = {1};
    dEdxi.tvec().device(*dev.edevice) += dEdf.tbvec().sum(red_axis);
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
  fx.tb<0>().device(*dev.edevice) = xs[0]->tbvec().sum(red_axis);
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
  dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec().broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(SumElements)

// ************* SumDimension *************

#ifndef __CUDACC__

string SumDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_dim(matrix=" << arg_names[0] << ',' << dimension << '}';
  return s.str();
}

Dim SumDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  Dim ret(xs[0]);
  ret.delete_dim(dimension);
  return ret;
}

#endif

template<class MyDevice>
void SumDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  Eigen::array<int, 1> reduction_axis = {(int)dimension};
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().sum(reduction_axis);
}

template<class MyDevice>
void SumDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  // TODO: limit to 3-dimensional tensor is arbitrary
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = dEdxi.d[dimension];
  Eigen::array<int, 4> morph = {(int)dEdxi.d[0],(int)dEdxi.d[1],(int)dEdxi.d[2],(int)dEdxi.d.bd}; morph[dimension] = 1;
  dEdxi.tb<3>().device(*dev.edevice) += dEdf.tb<3>().reshape(morph).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(SumDimension)

// ************* SumBatches *************

#ifndef __CUDACC__

string SumBatches::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_batches( " << arg_names[0] << " )";
  return s.str();
}

Dim SumBatches::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in SumBatches")
  return xs[0].single_batch();
}

#endif

template<class MyDevice>
void SumBatches::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SumBatches::forward");
  unsigned num_args = xs[0]->d.bd;
#ifdef __CUDACC__
  Eigen::array<int, 1> red_axis; red_axis[0] = 2;
  fx.t<2>().device(*dev.edevice) = xs[0]->tb<2>().sum(red_axis);
#else
  // TODO: Is this CPU version really good? Overhead can probably be reduced.
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = xs[0]->batch_matrix(0); break;
    case 2: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1); break;
    case 3: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1) + xs[0]->batch_matrix(2); break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += xs[0]->batch_matrix(i) + xs[0]->batch_matrix(i+1) + xs[0]->batch_matrix(i+2) + xs[0]->batch_matrix(i+3);
#endif
}

template<class MyDevice>
void SumBatches::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in SumBatches::backward");
#ifdef __CUDACC__
  Eigen::array<int, 3> bcast({1, 1, (int)fx.d.bd});
  dEdxi.tb<2>().device(*dev.edevice) += dEdf.tb<2>().broadcast(bcast);
#else
  for (unsigned i = 0; i < dEdxi.d.bd; ++i)
    dEdxi.batch_matrix(i) += *dEdf;
#endif
}
DYNET_NODE_INST_DEV_IMPL(SumBatches)

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
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<2>() + xs[1]->tb<2>().broadcast(bcasts);
  } else {
    DYNET_ASSERT(xs[0]->d.bd == 1,
                 "Bad dimensions in AddVectorToAllColumns::forward: " << xs[0]->d << ", " << xs[1]->d);
    Eigen::array<int, 3> bcasts0 = {1, 1, (int)xs[1]->d.bd};
    Eigen::array<int, 3> bcasts1 = {1, (int)xs[0]->d[1], 1};
    fx.tb<2>().device(*dev.edevice) = xs[0]->tb<2>().broadcast(bcasts0) + xs[1]->tb<2>().broadcast(bcasts1);
  }
#else
  // First, add the matrix
  if(xs[0]->d.bd == fx.d.bd)
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
  else
    for(size_t b = 0; b < fx.d.bd; ++b)
      fx.tbvec().chip<1>(b).device(*dev.edevice) = xs[0]->tvec();
  // Second, add the columns
  if(xs[1]->d.bd == fx.d.bd) {
    for(size_t i = 0; i < xs[0]->d[1]; ++i) 
      fx.tb<2>().chip<1>(i).device(*dev.edevice) += xs[1]->tb<1>();
  } else {
    for(size_t b = 0; b < fx.d.bd; ++b)
      for(size_t i = 0; i < fx.d[1]; ++i) 
        fx.tb<2>().chip<2>(b).chip<1>(i).device(*dev.edevice) += xs[1]->t<1>();
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
      dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
    } else {
      Eigen::array<int, 1> red_axis = {2};
      dEdxi.t<2>().device(*dev.edevice) += dEdf.tb<2>().sum(red_axis);
    }
  } else { // bias
    if(dEdf.d.bd == dEdxi.d.bd) {
      Eigen::array<int, 1> red_axis = {1};
      dEdxi.tb<1>().device(*dev.edevice) += dEdf.tb<2>().sum(red_axis);
    } else {
      DYNET_ASSERT(dEdxi.d.bd == 1,
                   "Bad dimensions in AddVectorToAllColumns::backward: " << xs[0]->d << ", " << xs[1]->d);
      Eigen::array<int, 2> red_axis = {1,2};
      dEdxi.t<1>().device(*dev.edevice) += dEdf.tb<2>().sum(red_axis);
    }
  }
}  
DYNET_NODE_INST_DEV_IMPL(AddVectorToAllColumns)

}
