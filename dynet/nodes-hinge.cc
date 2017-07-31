#include "dynet/nodes-hinge.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* Hinge *************

#ifndef __CUDACC__

string Hinge::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  if(pelement != nullptr)
    os << "hinge(" << arg_names[0] << ", pe=" << *pelement << ", m=" << margin << ')';
  else
    os << "hinge(" << arg_names[0] << ", pe=" << print_vec(*pelements) << ", m=" << margin << ')';
  return os.str();
}

Dim Hinge::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && LooksLikeVector(xs[0]), "Bad input dimensions in Hinge: " << xs);
  // TODO: This const_cast is ugly, but necessary(?). Perhaps we should refactor this.
  const_cast<size_t&>(input_size) = xs[0].size();
  return Dim({1}, xs[0].bd);
}

size_t Hinge::aux_storage_size() const {
  DYNET_ASSERT(input_size != 0, "We should not have an input size of zero in Hinge::aux_storage_size()");
  return input_size * sizeof(float);
}

#endif

template<class MyDevice>
void Hinge::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in Hinge::forward");
  DYNET_ARG_CHECK(margin >= 0, "Hinge loss does not support negative margins (got " << margin << ")");
  Tensor eloss(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  Eigen::array<int, 1> bcasts = {(int)xs[0]->d.rows()};
  if(pelement != nullptr) {
    DYNET_ARG_CHECK(fx.d.bd == 1, 
                            "Hinge was passed a single index but the corresponding expression has multiple mini-batch elements (" << fx.d.bd << ")");
    DYNET_ARG_CHECK(*pelement < xs[0]->d[0], "Index " << *pelement << " is out of bounds for hinge loss over tensor of size " << xs[0]->d);
    eloss.tvec().device(*dev.edevice) = (xs[0]->tvec() - xs[0]->t<2>().chip<0>(*pelement).broadcast(bcasts) + margin).cwiseMax(0.f);
    TensorTools::set_element(eloss, *pelement, 0.f);
  } else {
    DYNET_ASSERT(pelement != nullptr || pelements != nullptr, "Hinge::forward has neither pointer to single element nor vector");
    DYNET_ARG_CHECK(xs[0]->d.bd == pelements->size(),
                            "The list of indexes passed to Hinge has a length (" << pelements->size() <<
                            ") that doesn't match the number of mini-batch elements in the corresponding expression (" << xs[0]->d << ")");
    size_t batch_size = xs[0]->d.batch_size();
    for(size_t b = 0; b < fx.d.bd; b++) {
      DYNET_ARG_CHECK((*pelements)[b] < xs[0]->d[0], "Index for batch " << b << " is " << (*pelements)[b] << ", which is out of bounds for hinge loss over tensor of size " << xs[0]->d);
      eloss.tb<1>().chip<1>(b).device(*dev.edevice) = (xs[0]->tb<1>().chip<1>(b) - xs[0]->tb<2>().chip<2>(b).chip<0>((*pelements)[b]).broadcast(bcasts) + margin).cwiseMax(0.f);
      TensorTools::set_element(eloss, b*batch_size + (*pelements)[b], 0.f);
    }
  }
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  fx.tb<0>().device(*dev.edevice) = eloss.tb<1>().sum(red_axis);
}

template<class MyDevice>
void Hinge::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in Hinge::backward");
  // TODO: Can we do this on device?
  if(pelement != nullptr) {
    if(as_scalar(fx)) { // there was some loss
      const float d = as_scalar(dEdf);
      Tensor eloss(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
      // TODO: The > comparison should not be calculated twice. Keep it in auxiliary memory?
      dEdxi.tvec().device(*dev.edevice) += (eloss.tvec() > 0.f).cast<float>() * d;
#if defined(__CUDACC__) && defined(EIGEN_NO_MALLOC)
      DYNET_RUNTIME_ERR("CUDA memory allocation in hinge");
#endif
      dEdxi.tvec().chip<0>(*pelement).device(*dev.edevice) -= (eloss.tvec() > 0.f).template cast<float>().sum() * d;
    }
  } else {
    DYNET_ASSERT(pelements != nullptr, "Hinge::backward has neither pointer to single element nor vector");
    vector<float> fx_vec = as_vector(fx);
    vector<float> d_vec = as_vector(dEdf);
    Tensor eloss(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
    for(size_t b = 0; b < fx.d.bd; b++) {
      if(fx_vec[b]) { // there was some loss
        dEdxi.tb<1>().chip<1>(b).device(*dev.edevice) += (eloss.tb<1>().chip<1>(b) > 0.f).cast<float>() * d_vec[b];
#if defined(__CUDACC__) && defined(EIGEN_NO_MALLOC)
        DYNET_RUNTIME_ERR("CUDA memory allocation in hinge");
#endif
        dEdxi.tb<1>().chip<1>(b).chip<0>((*pelements)[b]).device(*dev.edevice) -= (eloss.tb<1>().chip<1>(b) > 0.f).template cast<float>().sum() * d_vec[b];
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(Hinge)

// ************* HingeDim *************

#ifndef __CUDACC__

string HingeDim::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  if(pelement != nullptr)
    os << "hinge_dim(" << arg_names[0] << ", pe=" << print_vec(*pelement) << ", d=" << d << ", m=" << margin << ')';
  else
    os << "hinge_dim(" << arg_names[0] << ", pe=" << print_vecs(*pelements) << ", d=" << d << ", m=" << margin << ')';
  return os.str();
}

Dim HingeDim::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && xs[0].nd == 2, "Bad input dimensions in HingeDim, expecting matrix: " << xs);
  const_cast<size_t&>(input_size) = xs[0].size();
  return Dim({xs[0][d ^ 1]}, xs[0].bd);
}

size_t HingeDim::aux_storage_size() const {
  return input_size * sizeof(float);
}

#endif

template<class MyDevice>
void HingeDim::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in HingeDim::forward");
  DYNET_ARG_CHECK(margin >= 0, "HingeDim loss does not support negative margins (got " << margin << ")");
  Tensor eloss(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  Eigen::array<int, 1> bcasts = {(int)xs[0]->d[d]};
  Eigen::array<int, 1> morph = {1};
  DYNET_ASSERT(pelement != nullptr || pelements != nullptr, "HingeDim::forward has neither pointer to single element nor vector");
  DYNET_ARG_CHECK(pelements == nullptr || xs[0]->d.bd == pelements->size(),
                          "The list of indexes passed to HingeDim has a length (" << pelements->size() <<
                          ") that doesn't match the number of mini-batch elements in the corresponding expression (" << xs[0]->d << ")");
  size_t batch_size = xs[0]->d.batch_size(), col_size = xs[0]->d.rows(), scan_size = xs[0]->d[d^1], scan_id = 0;
  // TODO: This will be very slow on GPU due to the one-by-one operations, and not super-fast on CPU either due to the broadcast.
  //       We should write a CUDA kernel and do alternative code paths.
  for(size_t b = 0; b < fx.d.bd; ++b, ++scan_id) {
    size_t my_scan = (pelement != nullptr ? (*pelement) : (*pelements)[b]).size();
    DYNET_ARG_CHECK(my_scan == scan_size, "IDs passed to HingeDim must be same size as # of " << 
                    (d == 0 ? "columns" : "rows") << ", but they didn't match (" <<
                    my_scan << " != " << scan_size << " @ batch " << b << ")");
    for(size_t i = 0; i < scan_size; ++i) {
      size_t id = (pelement != nullptr ? (*pelement)[i] : (*pelements)[b][i]);
      DYNET_ARG_CHECK(id < xs[0]->d[0], "Index for " << (d == 0 ? "column" : "row") <<
                      " " << i << "of batch" << b << " is " << id << " out of bounds for " << xs[0]->d);
      if(d == 0) {
        eloss.tb<2>().chip<2>(b).chip<1>(i).device(*dev.edevice) = 
          (xs[0]->tb<2>().chip<2>(b).chip<1>(i) + margin -
           xs[0]->tb<2>().chip<2>(b).chip<1>(i).chip<0>(id).reshape(morph).broadcast(bcasts)).cwiseMax(0.f);
        TensorTools::set_element(eloss, batch_size * b + col_size * i + id, 0.f);
      } else {
        eloss.tb<2>().chip<2>(b).chip<0>(i).device(*dev.edevice) = 
          (xs[0]->tb<2>().chip<2>(b).chip<0>(i) + margin -
           xs[0]->tb<2>().chip<2>(b).chip<1>(id).chip<0>(i).reshape(morph).broadcast(bcasts)).cwiseMax(0.f);
        TensorTools::set_element(eloss, batch_size * b + col_size * id + i, 0.f);
      }
    }
  }
  Eigen::array<ptrdiff_t, 1> red_axis = {d};
  fx.tb<1>().device(*dev.edevice) = eloss.tb<2>().sum(red_axis);
}

template<class MyDevice>
void HingeDim::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(pelement != nullptr || pelements != nullptr, "HingeDim::backward has neither pointer to single element nor vector");
  size_t scan_size = xs[0]->d[d^1], pos=0;
  vector<float> fx_vec = as_vector(fx);
  vector<float> d_vec = as_vector(dEdf);
  Tensor eloss(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  for(size_t b = 0; b < fx.d.bd; b++) {
    size_t my_scan = (pelement != nullptr ? (*pelement) : (*pelements)[b]).size();
    DYNET_ARG_CHECK(my_scan == scan_size, "IDs passed to HingeDim must be same size as # of " << 
                    (d == 0 ? "columns" : "rows") << ", but they didn't match (" <<
                    my_scan << " != " << scan_size << " @ batch " << b << ")");
    for(size_t i = 0; i < scan_size; ++i, ++pos) {
      if(fx_vec[pos]) { // there was some loss
        size_t id = (pelement != nullptr ? (*pelement)[i] : (*pelements)[b][i]);
        DYNET_ARG_CHECK(id < xs[0]->d[0], "Index for " << (d == 0 ? "column" : "row") <<
                        " " << i << "of batch" << b << " is " << id << " out of bounds for " << xs[0]->d);
        if(d == 0) {
          dEdxi.tb<2>().chip<2>(b).chip<1>(i).device(*dev.edevice) += (eloss.tb<2>().chip<2>(b).chip<1>(i) > 0.f).cast<float>() * d_vec[pos];
          dEdxi.tb<2>().chip<2>(b).chip<1>(i).chip<0>(id).device(*dev.edevice) -= (eloss.tb<2>().chip<2>(b).chip<1>(i) > 0.f).template cast<float>().sum() * d_vec[pos];
        } else {
          dEdxi.tb<2>().chip<2>(b).chip<0>(i).device(*dev.edevice) += (eloss.tb<2>().chip<2>(b).chip<0>(i) > 0.f).cast<float>() * d_vec[pos];
          dEdxi.tb<2>().chip<2>(b).chip<1>(id).chip<0>(i).device(*dev.edevice) -= (eloss.tb<2>().chip<2>(b).chip<0>(i) > 0.f).template cast<float>().sum() * d_vec[pos];
        }
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(HingeDim)

// // ************* BidirectionalHinge *************
// 
// #ifndef __CUDACC__
// 
// string BidirectionalHinge::as_string(const vector<string>& arg_names) const {
//   ostringstream os;
//   os << "bidirectional_hinge(" << arg_names[0] << ", m=" << margin << ')';
//   return os.str();
// }
// 
// Dim BidirectionalHinge::dim_forward(const vector<Dim>& xs) const {
//   DYNET_ARG_CHECK(xs.size() == 1, "Got too many inputs in BidirectionalHinge: " << xs);
//   Dim d = xs[0].truncate();
//   DYNET_ARG_CHECK((d.nd == 2 && d[0] == d[1]) ||
//                   (d.nd == 2 && d[0] == d.nd),
//                   "BidirectionalHinge expects a square matrix or a batch of vectors where the batch size is equal to the vector size, but got: " << xs[0]);
//   DYNET_ARG_CHECK(d[0] > 1,
//                   "BidirectionalHinge expects more than one row in its input, but got: " << xs[0]);
//   // TODO: This const_cast is ugly, but necessary(?). Perhaps we should refactor this.
//   const_cast<size_t&>(input_size) = d.size();
//   return Dim({1});
// }
// 
// size_t BidirectionalHinge::aux_storage_size() const {
//   DYNET_ASSERT(input_size != 0, "We should not have an input size of zero in BidirectionalHinge::aux_storage_size()");
//   return 2 * input_size * sizeof(float);
// }
// 
// #endif
// 
// template<class MyDevice>
// void BidirectionalHinge::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
//   DYNET_ASSERT(xs.size() == 1, "Failed dimension check in BidirectionalHinge::forward");
//   DYNET_ARG_CHECK(margin >= 0, "BidirectionalHinge does not support negative margins (got " << margin << ")");
//   Tensor elossf(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
//   Tensor elossb(xs[0]->d, static_cast<float*>(aux_mem) + input_size, fx.device, DeviceMempool::FXS);
//   const Dim & ind = xs[0]->d;
//   // Calculate forward loss
//   Eigen::array<int, 2> morph = {1, (int)ind.cols()};
//   Eigen::array<int, 2> bcasts = {(int)ind.rows(), 1};
//   if(ind.bd == 1)
//     elossf.t<2>().device(*dev.edevice) = 
//       (xs[0]->t<2>() - xs[0]->t<2>().diag().reshape(morph).broadcast(bcasts) + margin).cwiseMax(0.f);
//   else
//     elossf.tb<1>().device(*dev.edevice) = 
//       (xs[0]->tb<1>() - xs[0]->tb<1>().diag().reshape(morph).broadcast(bcasts) + margin).cwiseMax(0.f);
//   // Calculate backward loss
//   morph = {(int)ind.rows(), 1};
//   bcasts = {1, (int)ind.cols()};
//   if(ind.bd == 1)
//     elossb.t<2>().device(*dev.edevice) = 
//       (xs[0]->t<2>() - xs[0]->t<2>().diag().reshape(morph).broadcast(bcasts) + margin).cwiseMax(0.f);
//   else
//     elossb.tb<1>().device(*dev.edevice) = 
//       (xs[0]->tb<1>() - xs[0]->tb<1>().diag().reshape(morph).broadcast(bcasts) + margin).cwiseMax(0.f);
//   // Return the sum of both
//   fx.t<0>().device(*dev.edevice) = elossf.tvec().sum() + elossb.tvec().sum();
// }
// 
// template<class MyDevice>
// void BidirectionalHinge::backward_dev_impl(const MyDevice & dev,
//                              const vector<const Tensor*>& xs,
//                              const Tensor& fx,
//                              const Tensor& dEdf,
//                              unsigned i,
//                              Tensor& dEdxi) const {
//   Tensor elossf(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
//   Tensor elossb(xs[0]->d, static_cast<float*>(aux_mem) + input_size, fx.device, DeviceMempool::FXS);
//   float d = as_scalar(dEdf);
//   if(xs[0]->d.bd == 1) {
//     dEdxi.t<2>().device(*dev.edevice) += ((elossf.t<2>() > 0.f).template cast<float>() + (elossb.t<2>() > 0.f).template cast<float>()) * d;
//     dEdxi.t<2>().diag().device(*dev.edevice) -= ((elossf.t<2>().diag() > 0.f).template cast<float>() + (elossb.t<2>().diag() > 0.f).template cast<float>()) * d;
//   } else {
//     dEdxi.tb<1>().device(*dev.edevice) += ((elossf.tb<1>() > 0.f).template cast<float>() + (elossb.tb<1>() > 0.f).template cast<float>()) * d;
//     dEdxi.tb<1>().diag().device(*dev.edevice) -= ((elossf.tb<1>().diag() > 0.f).template cast<float>() + (elossb.tb<1>().diag() > 0.f).template cast<float>()) * d;
//   }
// }
// DYNET_NODE_INST_DEV_IMPL(BidirectionalHinge)

}
