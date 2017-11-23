#include "dynet/tensor-eigen.h"
#include "dynet/nodes-minmax.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

// ************* Min *************

#ifndef __CUDACC__

string Min::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "min{" << arg_names[0] << ", " << arg_names[1] << "}";
  return s.str();
}

Dim Min::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 && xs[0] == xs[1], "Bad arguments in Min: " << xs);
  return xs[0].bd >= xs[1].bd ? xs[0] : xs[1];
}

size_t Min::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

#endif

template<class MyDevice>
void Min::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor t(fx.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  tvec(t).device(*dev.edevice) = (tvec(*xs[0]) < tvec(*xs[1])).cast<float>();
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).cwiseMin(tvec(*xs[1]));
}

template<class MyDevice>
void Min::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in Min::backward");
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  if (i == 0) {
    tvec(dEdxi).device(*dev.edevice) += tvec(t) * tvec(dEdf);
  } else {
    tvec(dEdxi).device(*dev.edevice) += tvec(t).binaryExpr(tvec(dEdf), FMaxBackwardInv());
  }
}
DYNET_NODE_INST_DEV_IMPL(Min)

// ************* Max *************

#ifndef __CUDACC__

string Max::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "max{" << arg_names[0] << ", " << arg_names[1] << "}";
  return s.str();
}

Dim Max::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 && xs[0] == xs[1], "Bad arguments in Max: " << xs);
  return xs[0].bd >= xs[1].bd ? xs[0] : xs[1];
}

size_t Max::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

#endif

template<class MyDevice>
void Max::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor t(fx.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  tvec(t).device(*dev.edevice) = (tvec(*xs[0]) > tvec(*xs[1])).cast<float>();
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).cwiseMax(tvec(*xs[1]));
}

template<class MyDevice>
void Max::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in Max::backward");
  const Tensor t(dEdxi.d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  if (i == 0) {
    tvec(dEdxi).device(*dev.edevice) += tvec(t) * tvec(dEdf);
  } else {
    tvec(dEdxi).device(*dev.edevice) += tvec(t).binaryExpr(tvec(dEdf), FMaxBackwardInv());
  }
}
DYNET_NODE_INST_DEV_IMPL(Max)

// ************* MinDimension *************

#ifndef __CUDACC__

string MinDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "min_dim(" << arg_names[0] << ", reduced_dim=" << reduced_dim << ')';
  return s.str();
}

Dim MinDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in MinDimension");
  DYNET_ARG_CHECK(reduced_dim < xs[0].nd,
                          "Tried to MinDimension on dimension " << reduced_dim << " bigger than input " << xs[0]);
  DYNET_ARG_CHECK(xs[0].nd < 4,
                          "MinDimension not currently supported for tensors of 4 or more dimensions.");
  Dim ret(xs[0]);
  ret.delete_dim(reduced_dim);
  return ret;
}

size_t MinDimension::aux_storage_size() const {
  return sizeof(Eigen::DenseIndex) * dim.size();
}

#endif

template<class MyDevice>
void MinDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DenseIndex* minmap = static_cast<Eigen::DenseIndex*>(aux_mem);
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(minmap, first_dim_size, second_dim_size, batch_size);
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {reduced_dim};
  locs.device(*dev.edevice) = tb<3>(*xs[0]).argmin(reduced_dim);
  tb<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).minimum(reduction_axis);
}

template<class MyDevice>
void MinDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MinDimension::backward");
#ifdef __CUDACC__
  vector<Eigen::DenseIndex> indices(dim.size());
  Eigen::DenseIndex* minmap = &indices[0];
  CUDA_CHECK(cudaMemcpy((void*)minmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
#else
  Eigen::DenseIndex* minmap = static_cast<Eigen::DenseIndex*>(aux_mem);
#endif
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(minmap, first_dim_size, second_dim_size, batch_size);
  for(unsigned b = 0; b < batch_size; ++b){
    for(unsigned j = 0; j < second_dim_size; ++j){
      for(unsigned i = 0; i < first_dim_size; ++i){
        if (reduced_dim > second_dim)
          tb<3>(dEdxi).chip<3>(b).chip(locs(i, j, b), reduced_dim).chip(j, second_dim).chip(i, first_dim).device(*dev.edevice)
            += tb<2>(dEdf).chip<2>(b).chip<1>(j).chip<0>(i);
        else if (reduced_dim > first_dim)
          tb<3>(dEdxi).chip<3>(b).chip(j, second_dim).chip(locs(i, j, b), reduced_dim).chip(i, first_dim).device(*dev.edevice)
            += tb<2>(dEdf).chip<2>(b).chip<1>(j).chip<0>(i);
        else
          tb<3>(dEdxi).chip<3>(b).chip(j, second_dim).chip(i, first_dim).chip(locs(i, j, b), reduced_dim).device(*dev.edevice)
            += tb<2>(dEdf).chip<2>(b).chip<1>(j).chip<0>(i);
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(MinDimension)

// ************* MaxDimension *************

#ifndef __CUDACC__

string MaxDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "max_dim(" << arg_names[0] << ", reduced_dim=" << reduced_dim << ')';
  return s.str();
}

Dim MaxDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in MaxDimension");
  DYNET_ARG_CHECK(reduced_dim < xs[0].nd,
                          "Tried to MaxDimension on dimension " << reduced_dim << " bigger than input " << xs[0]);
  DYNET_ARG_CHECK(xs[0].nd < 4,
                          "MaxDimension not currently supported for tensors of 4 or more dimensions.");
  Dim ret(xs[0]);
  ret.delete_dim(reduced_dim);
  return ret;
}

size_t MaxDimension::aux_storage_size() const {
  return sizeof(Eigen::DenseIndex) * dim.size();
}

#endif

template<class MyDevice>
void MaxDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(maxmap, first_dim_size, second_dim_size, batch_size);
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {reduced_dim};
  locs.device(*dev.edevice) = tb<3>(*xs[0]).argmax(reduced_dim);
  tb<2>(fx).device(*dev.edevice) = tb<3>(*xs[0]).maximum(reduction_axis);
}

template<class MyDevice>
void MaxDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in MaxDimension::backward");
#ifdef __CUDACC__
  vector<Eigen::DenseIndex> indices(dim.size());
  Eigen::DenseIndex* maxmap = &indices[0];
  CUDA_CHECK(cudaMemcpy((void*)maxmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
#else
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
#endif
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[0];
  const unsigned second_dim_size = dim[1];
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> locs(maxmap, first_dim_size, second_dim_size, batch_size);
  for(unsigned b = 0; b < batch_size; ++b){
    for(unsigned j = 0; j < second_dim_size; ++j){
      for(unsigned i = 0; i < first_dim_size; ++i){
        if (reduced_dim > second_dim)
          tb<3>(dEdxi).chip<3>(b).chip(locs(i, j, b), reduced_dim).chip(j, second_dim).chip(i, first_dim).device(*dev.edevice)
            += tb<2>(dEdf).chip<2>(b).chip<1>(j).chip<0>(i);
        else if (reduced_dim > first_dim)
          tb<3>(dEdxi).chip<3>(b).chip(j, second_dim).chip(locs(i, j, b), reduced_dim).chip(i, first_dim).device(*dev.edevice)
            += tb<2>(dEdf).chip<2>(b).chip<1>(j).chip<0>(i);
        else
          tb<3>(dEdxi).chip<3>(b).chip(j, second_dim).chip(i, first_dim).chip(locs(i, j, b), reduced_dim).device(*dev.edevice)
            += tb<2>(dEdf).chip<2>(b).chip<1>(j).chip<0>(i);
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(MaxDimension)

}
