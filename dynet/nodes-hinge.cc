#include "dynet/nodes-hinge.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* Hinge *************

#ifndef __CUDACC__

string Hinge::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "hinge(" << arg_names[0] << ", pe=" << pelement << ", m=" << margin << ')';
  return os.str();
}

Dim Hinge::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && LooksLikeVector(xs[0]), "Bad input dimensions in Hinge: " << xs);
  return Dim({1}, xs[0].bd);
}

size_t Hinge::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

#endif

template<class MyDevice>
void Hinge::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in Hinge::forward");
  Tensor eloss(xs[0]->d, static_cast<float*>(aux_mem), fx.device, DeviceMempool::FXS);
  // TODO: Can we do this on device?
  if(pelement != nullptr) {
    DYNET_ARG_CHECK(fx.d.bd == 1, 
                            "Hinge was passed a single index but the corresponding expression has multiple mini-batch elements (" << fx.d.bd << ")");
    DYNET_ARG_CHECK(*pelement < xs[0]->d[0], "Index " << *pelement << " is out of bounds for hinge loss over tensor of size " << xs[0]->d);
    const real mlystar = margin - TensorTools::access_element(*xs[0], *pelement);
    eloss.tvec().device(*dev.edevice) = (xs[0]->tvec() + mlystar).cwiseMax(0.f);
    TensorTools::set_element(eloss, *pelement, 0.f);
    fx.t<0>().device(*dev.edevice) = eloss.tvec().sum();
  } else {
    DYNET_ASSERT(pelements != nullptr, "Hinge::forward has neither pointer to single element nor vector");
    DYNET_ARG_CHECK(xs[0]->d.bd == pelements->size(),
                            "The list of indexes passed to Hinge has a length (" << pelements->size() <<
                            ") that doesn't match the number of mini-batch elements in the corresponding expression (" << xs[0]->d << ")");
    size_t batch_size = xs[0]->d.batch_size();
    for(size_t b = 0; b < fx.d.bd; b++) {
      DYNET_ARG_CHECK((*pelements)[b] < xs[0]->d[0], "Index for batch " << b << " is " << (*pelements)[b] << ", which is out of bounds for hinge loss over tensor of size " << xs[0]->d);
      const real mlystar = margin - TensorTools::access_element(*xs[0], b*batch_size + (*pelements)[b]);
      eloss.tb<1>().chip<1>(b).device(*dev.edevice) = (xs[0]->tb<1>().chip<1>(b) + mlystar).cwiseMax(0.f);
      TensorTools::set_element(eloss, b*batch_size + (*pelements)[b], 0.f);
      fx.tb<0>().chip<0>(b).device(*dev.edevice) = eloss.tb<1>().chip<1>(b).sum();
    }
  }
}

template<class MyDevice>
void Hinge::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in Hinge::backward");
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
        // TODO: The > comparison should not be calculated twice. Keep it in auxiliary memory?
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

}
