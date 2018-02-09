#include "dynet/tensor-eigen.h"
#include "dynet/training.h"
#include "dynet/devices.h"

// #include "dynet/gpu-ops.h"
#include "dynet/param-nodes.h"
#include "dynet/weight-decay.h"

// Macros for defining parameter update functions
#ifdef __CUDACC__
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real gscale, const std::vector<Tensor*> & values);
#elif defined(HAVE_CUDA)
// This is correct, but dying when models are read and written.
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  extern template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real gscale, const std::vector<Tensor*> & values); \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real gscale, const std::vector<Tensor*> & values) { \
    if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,gscale,values); } \
    else if(values[0]->device->type == DeviceType::GPU) { \
      cudaSetDevice(((Device_GPU*)values[0]->device)->cuda_device_id); \
      update_rule_dev(*(Device_GPU*)values[0]->device,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#else
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real gscale, const std::vector<Tensor*> & values) { \
    if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#endif

namespace dynet {

using namespace std;

template <class Derived>
bool is_valid(const Eigen::MatrixBase<Derived>& x) {
  return ((x - x).array() == (x - x).array()).all();
}

// --- The actual update code for each operation, implemented on various devices

// Trainer base class is run on CPUs
#ifndef __CUDACC__

Trainer::~Trainer() {}

void Trainer::rescale_and_reset_weight_decay() {
  const float weight_decay = model->get_weight_decay().current_weight_decay();
  for (auto p : model->parameters_list())
    if (p->is_updated())
      p->scale_parameters(weight_decay);
  for (auto p : model->lookup_parameters_list())
    if (p->is_updated())
      p->scale_parameters(weight_decay);
  model->get_weight_decay().reset_weight_decay();
}

float Trainer::clip_gradients() {
  float gscale = 1;
  if (clipping_enabled) {
    float gg = model->gradient_l2_norm();
    if (isnan(gg) || isinf(gg)) {
      ostringstream oss; oss << "Magnitude of gradient is bad: " << gg;
      throw std::runtime_error(oss.str());
    }
    if (gg > clip_threshold) {
      ++clips;
      ++clips_since_status;
      gscale = clip_threshold / gg;
    }
  }
  return gscale;
}

void Trainer::update_epoch(real r) {
  cerr << "Trainer::update_epoch has been deprecated and doesn't do anything. Please remove it from your code, and control the learning rate of the trainer directly, for example by: 'trainer.learning_rate /= (1 - rate_decay)', see https://github.com/clab/dynet/pull/695 for details." << endl;
}

// this calls the rule-specific updates over all updated parameters
void Trainer::update() {
  const auto & params = model->parameters_list();
  const auto & lparams = model->lookup_parameters_list();

  // Allocate if necessary
  if(aux_allocated < params.size()) {
    aux_allocated = alloc_impl();
  }
  if(aux_allocated_lookup < lparams.size()) {
    aux_allocated_lookup = alloc_lookup_impl();
  }

  // Perform gradient clipping and cycle through parameters
  const float gscale = clip_gradients();
  for(size_t i = 0; i < params.size(); ++i) {
    if(params[i]->updated) {
      update_params(gscale, i);
      params[i]->clear();
    }
  }
  for(size_t i = 0; i < lparams.size(); ++i) {
    auto &p = lparams[i];
    if (p->updated) {
      if(sparse_updates_enabled && !p->all_updated) {
        for (auto j : p->non_zero_grads)
          update_lookup_params(gscale, i, j);
      } else {
        update_lookup_params(gscale, i);
      }
      p->clear();
    }
  }
  ++updates;
  ++updates_since_status;

  L2WeightDecay & wd = model->get_weight_decay();
  wd.update_weight_decay(); // update global weight scale
  if (wd.parameters_need_rescaled())
    rescale_and_reset_weight_decay();  // if wdscale is getting to small multiply all weights by wdscale, and set wdscale to 1
}

void Trainer::restart(real lr) {
    this->learning_rate = lr;
    this->restart();
}


#endif

// --- SimpleSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void SimpleSGDTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[1]) * (learning_rate * gscale / model->get_weight_decay().current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(SimpleSGDTrainer)

#ifndef __CUDACC__
void SimpleSGDTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g});
}
void SimpleSGDTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx]});
}
void SimpleSGDTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads});
}
#endif

// --- CyclicalSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void CyclicalSGDTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[1]) * (learning_rate * gscale / model->get_weight_decay().current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(CyclicalSGDTrainer)

#ifndef __CUDACC__
void CyclicalSGDTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g});
}
void CyclicalSGDTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx]});
}
void CyclicalSGDTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads});
}
#endif

// --- MomentumSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=momentum
template <class MyDevice>
void MomentumSGDTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * momentum - tvec(*ts[1]) * (learning_rate * gscale);
  tvec(*ts[0]).device(*dev.edevice) += tvec(*ts[2]) / model->get_weight_decay().current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(MomentumSGDTrainer)

#ifndef __CUDACC__
void MomentumSGDTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &vp[idx].h});
}
void MomentumSGDTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void MomentumSGDTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &vlp[idx].all_h});
}
unsigned MomentumSGDTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, vp);
  return vp.size();
}
unsigned MomentumSGDTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, vlp);
  return vlp.size();
}

void MomentumSGDTrainer::restart() {
  for (auto sp : vp)
    TensorTools::zero(sp.h);
  for (auto slp : vlp)
    TensorTools::zero(slp.all_h);
}

#endif

// --- AdagradTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=stddev
template <class MyDevice>
void AdagradTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) += tvec(*ts[1]).square();
  tvec(*ts[0]).device(*dev.edevice) += tvec(*ts[1]) / (tvec(*ts[2]) + epsilon).sqrt() * (-learning_rate / model->get_weight_decay().current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(AdagradTrainer)

#ifndef __CUDACC__
void AdagradTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &vp[idx].h});
}
void AdagradTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void AdagradTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &vlp[idx].all_h});
}
unsigned AdagradTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, vp);
  return vp.size();
}
unsigned AdagradTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, vlp);
  return vlp.size();
}

void AdagradTrainer::restart() {
  for (auto sp : vp)
    TensorTools::zero(sp.h);
  for (auto slp : vlp)
    TensorTools::zero(slp.all_h);
}

#endif

// --- AdadeltaTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=hg, ts[3]=hd
template <class MyDevice>
void AdadeltaTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * rho + tvec(*ts[1]).square() * (1.f - rho);
  tvec(*ts[1]).device(*dev.edevice) = - tvec(*ts[1]) * (tvec(*ts[3]) + epsilon).sqrt() / (tvec(*ts[2]) + epsilon).sqrt();
  tvec(*ts[3]).device(*dev.edevice) = tvec(*ts[3]) * rho + tvec(*ts[1]).square() * (1.f - rho);
  tvec(*ts[0]).device(*dev.edevice) += tvec(*ts[1]) / model->get_weight_decay().current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(AdadeltaTrainer)

#ifndef __CUDACC__
void AdadeltaTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &hg[idx].h, &hd[idx].h});
}
void AdadeltaTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &hlg[idx].h[lidx], &hld[idx].h[lidx]});
}
void AdadeltaTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &hlg[idx].all_h, &hld[idx].all_h});
}
unsigned AdadeltaTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, hg);
  allocate_shadow_parameters(*model, aux_allocated, hd);
  return hd.size();
}
unsigned AdadeltaTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hlg);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hld);
  return hld.size();
}

void AdadeltaTrainer::restart() {
  for (auto sp : hg)
    TensorTools::zero(sp.h);
  for (auto sp : hd)
    TensorTools::zero(sp.h);
  for (auto slp : hlg)
    TensorTools::zero(slp.all_h);
  for (auto slp : hld)
    TensorTools::zero(slp.all_h);
}

#endif

// --- RMSPropTrainer
// TODO: This is not finished yet, because it memorizes a scalar for each set of parameters, not each parameter itself.
//       We could implement this with one tensor for each scalar, but this is pretty wasteful

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void RMSPropTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale; // Scale gradient
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * rho + tvec(*ts[1]).square() * (1.f - rho); // Update square gradient exponential average
  tvec(*ts[1]).device(*dev.edevice) = - tvec(*ts[1]) / (tvec(*ts[2]) + epsilon).sqrt(); // Divide by the RMS
  tvec(*ts[0]).device(*dev.edevice) += learning_rate * tvec(*ts[1]) / model->get_weight_decay().current_weight_decay(); // Apply weight decay (should we do this?)
  // real& d2 = hg[pi++];
  // real g2 = p->vec(g).squaredNorm();
  // d2 = rho * d2 + (1.f - rho) * g2;
  // p->vec(values) -= ((learning_rate * gscale / sqrt(d2 + epsilon)) * p->vec(g)) / model->get_weight_decay().current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(RMSPropTrainer)

#ifndef __CUDACC__
void RMSPropTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &hmsg[idx].h});
}
void RMSPropTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &hlmsg[idx].h[lidx]});
}
void RMSPropTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &hlmsg[idx].all_h});
}
unsigned RMSPropTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, hmsg);
  return hmsg.size();
}
unsigned RMSPropTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hlmsg);
  return hlmsg.size();
}

void RMSPropTrainer::restart() {
  for (auto sp : hmsg)
    TensorTools::zero(sp.h);
  for (auto slp : hlmsg)
    TensorTools::zero(slp.all_h);
}

#endif

// --- AdamTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=mean, ts[3]=variance
template <class MyDevice>
void AdamTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * beta_1 + tvec(*ts[1]) * (1.f - beta_1);
  tvec(*ts[3]).device(*dev.edevice) = tvec(*ts[3]) * beta_2 + tvec(*ts[1]).square() * (1.f - beta_2);
  float lr_t = learning_rate * sqrt(1-pow(beta_2, updates+1))/(1-pow(beta_1, updates+1))/ model->get_weight_decay().current_weight_decay();
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[2]) / (tvec(*ts[3]).sqrt() + epsilon) * lr_t;
}
DYNET_TRAINER_INST_DEV_IMPL(AdamTrainer)

#ifndef __CUDACC__
void AdamTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h});
}
void AdamTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx]});
}
void AdamTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &lm[idx].all_h, &lv[idx].all_h});
}
unsigned AdamTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, m);
  allocate_shadow_parameters(*model, aux_allocated, v);
  return v.size();
}
unsigned AdamTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lm);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lv);
  return lv.size();
}

void AdamTrainer::restart() {
  for (auto sp : m)
    TensorTools::zero(sp.h);
  for (auto sp : v)
    TensorTools::zero(sp.h);
  for (auto slp : lm)
    TensorTools::zero(slp.all_h);
  for (auto slp : lv)
    TensorTools::zero(slp.all_h);
}

#endif

// --- AMSGradTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=mean, ts[3]=variance, t[4]=max
template <class MyDevice>
void AmsgradTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * beta_1 + tvec(*ts[1]) * (1.f - beta_1);
  tvec(*ts[3]).device(*dev.edevice) = tvec(*ts[3]) * beta_2 + tvec(*ts[1]).square() * (1.f - beta_2);
  tvec(*ts[4]).device(*dev.edevice) = tvec(*ts[4]).cwiseMax(tvec(*ts[3]));
  float lr_t = learning_rate * sqrt(1-pow(beta_2, updates+1))/(1-pow(beta_1, updates+1))/ model->get_weight_decay().current_weight_decay();
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[2]) / (tvec(*ts[4]).sqrt() + epsilon) * lr_t;
}
DYNET_TRAINER_INST_DEV_IMPL(AmsgradTrainer)

#ifndef __CUDACC__
void AmsgradTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h, &vhat[idx].h});
}
void AmsgradTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx], &lvhat[idx].h[lidx]});
}
void AmsgradTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &lm[idx].all_h, &lv[idx].all_h, &lvhat[idx].all_h});
}
unsigned AmsgradTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, m);
  allocate_shadow_parameters(*model, aux_allocated, v);
  allocate_shadow_parameters(*model, aux_allocated, vhat);
  return vhat.size();
}
unsigned AmsgradTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lm);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lv);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lvhat);
  return lvhat.size();
}

void AmsgradTrainer::restart() {
  for (auto sp : m)
    TensorTools::zero(sp.h);
  for (auto sp : v)
    TensorTools::zero(sp.h);
  for (auto sp : vhat)
    TensorTools::zero(sp.h);
  for (auto slp : lm)
    TensorTools::zero(slp.all_h);
  for (auto slp : lv)
    TensorTools::zero(slp.all_h);
  for (auto slp : lvhat)
    TensorTools::zero(slp.all_h);
}

#endif

template <class MyDevice>
void EGTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  // Add momentum
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * momentum - tvec(*ts[1]) * (learning_rate * gscale);
  tvec(*ts[0]).device(*dev.edevice) = tvec(*ts[0]).log() + tvec(*ts[2]) / model->get_weight_decay().current_weight_decay();// with momentum only
  TensorTools::logsumexp_dev(dev, *ts[0], *ts[3], *ts[4]);// z refers to logZ
  tvec(*ts[0]).device(*dev.edevice) = (tvec(*ts[0]) - as_scalar(*ts[4])).exp();// FIXME: other way(s) of not using as_scalar(z)?
}
DYNET_TRAINER_INST_DEV_IMPL(EGTrainer)

#ifndef __CUDACC__
// --- EGTrainer
EGTrainer::EGTrainer(ParameterCollection& mod, real learning_rate, real mom, real ne)
  : Trainer(mod, learning_rate), momentum(mom), isCyclical(false) {
  zeg.d = meg.d = {1};
  zeg.device = meg.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, zeg);
  default_device->allocate_tensor(DeviceMempool::PS, meg);
}
void EGTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &hp[idx].h, &meg, &zeg});
}
void EGTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &hlp[idx].h[lidx], &meg, &zeg});
}
void EGTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_grads, &p->all_grads, &hlp[idx].all_h, &meg, &zeg});
}
unsigned EGTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, hp);
  return hp.size();
}
unsigned EGTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hlp);
  return hlp.size();
}

void EGTrainer::restart() {
  for (auto sp : hp)
    TensorTools::zero(sp.h);
  for (auto slp : hlp)
    TensorTools::zero(slp.all_h);
}

#endif

} // namespace dynet
