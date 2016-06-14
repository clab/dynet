#include "cnn/training.h"

// #include "cnn/gpu-ops.h"
#include "cnn/param-nodes.h"
#include "cnn/weight-decay.h"

// Macros for defining parameter update functions
#ifdef __CUDACC__
#define CNN_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real scale, real gscale, const std::vector<Tensor*> & values);
#elif defined(HAVE_GPU)
#define CNN_TRAINER_INST_DEV_IMPL(MyTrainer) \
  extern template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real scale, real gscale, const std::vector<Tensor*> & values) { \
    if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,scale,gscale,values); } \
    else if(values[0]->device->type == DeviceType::GPU) { update_rule_dev(*(Device_GPU*)values[0]->device,scale,gscale,values); } \
    else { abort(); } \
  }
#else
#define CNN_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real scale, real gscale, const std::vector<Tensor*> & values) { \
    if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,scale,gscale,values); } \
    else { abort(); } \
  }
#endif

namespace cnn {

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
  const float weight_decay = global_weight_decay.CurrentWeightDecay();
  for (auto p : model->parameters_list())
    p->scale_parameters(weight_decay);
  global_weight_decay.ResetWeightDecay();
}

float Trainer::clip_gradients() {
  float gscale = 1;
  if (clipping_enabled) {
    float gg = model->gradient_l2_norm();
    if (isnan(gg) || isinf(gg)) {
      cerr << "Magnitude of gradient is bad: " << gg << endl;
      abort();
    }
    if (gg > clip_threshold) {
      ++clips;
      gscale = clip_threshold / gg;
    }
  }
  return gscale;
}

// this calls the rule-specific
void Trainer::update(real scale) {
  // Allocate if necessary
  if(!aux_allocated) {
    alloc_impl();
    aux_allocated = true;
  }

  // Perform gradient clipping and cycle through parameters
  const float gscale = clip_gradients();
  const auto & params = model->parameters_list();
  for(size_t i = 0; i < params.size(); ++i) {
    update_params(scale, gscale, i);
    params[i]->clear();
  }
  const auto & lookup_params = model->lookup_parameters_list();
  for(size_t i = 0; i < lookup_params.size(); ++i) {
    for (auto j : lookup_params[i]->non_zero_grads)
      update_lookup_params(scale, gscale, i, j);
    lookup_params[i]->clear();
  }
  ++updates;

  global_weight_decay.UpdateWeightDecay(); // update global weight scale
  if (global_weight_decay.ParametersNeedRescaled())
    rescale_and_reset_weight_decay();  // if wdscale is getting to small multiply all weights by wdscale, and set wdscale to 1
}

#endif

// --- SimpleSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void SimpleSGDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[0]->tvec().device(*dev.edevice) -= ts[1]->tvec() * (eta * scale * gscale / global_weight_decay.CurrentWeightDecay());
}
CNN_TRAINER_INST_DEV_IMPL(SimpleSGDTrainer)

#ifndef __CUDACC__
void SimpleSGDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g});
}
void SimpleSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx]});
}
#endif

// --- MomentumSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=momentum
template <class MyDevice>
void MomentumSGDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * momentum - ts[1]->tvec() * (eta * scale * gscale);
  ts[0]->tvec().device(*dev.edevice) += ts[2]->tvec() / global_weight_decay.CurrentWeightDecay();
}
CNN_TRAINER_INST_DEV_IMPL(MomentumSGDTrainer)

#ifndef __CUDACC__
void MomentumSGDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &vp[idx].h});
}
void MomentumSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void MomentumSGDTrainer::alloc_impl() {
  vp = AllocateShadowParameters(*model);
  vlp = AllocateShadowLookupParameters(*model);
}
#endif

// --- AdagradTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=stddev
template <class MyDevice>
void AdagradTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) += ts[1]->tvec().square();
  ts[0]->tvec().device(*dev.edevice) += ts[1]->tvec() * (ts[2]->tvec() + epsilon).sqrt() * (-eta / global_weight_decay.CurrentWeightDecay());
}
CNN_TRAINER_INST_DEV_IMPL(AdagradTrainer)

#ifndef __CUDACC__
void AdagradTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &vp[idx].h});
}
void AdagradTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void AdagradTrainer::alloc_impl() {
  vp = AllocateShadowParameters(*model);
  vlp = AllocateShadowLookupParameters(*model);
}
#endif

// --- AdadeltaTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=hg, ts[3]=hd
template <class MyDevice>
void AdadeltaTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho);
  ts[1]->tvec().device(*dev.edevice) = - ts[1]->tvec() * (ts[3]->tvec() + epsilon).sqrt() * (ts[2]->tvec() + epsilon).sqrt();
  ts[3]->tvec().device(*dev.edevice) = ts[3]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho);
  ts[0]->tvec().device(*dev.edevice) += ts[1]->tvec() / global_weight_decay.CurrentWeightDecay();
}
CNN_TRAINER_INST_DEV_IMPL(AdadeltaTrainer)

#ifndef __CUDACC__
void AdadeltaTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &hg[idx].h, &hd[idx].h});
}
void AdadeltaTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlg[idx].h[lidx], &hld[idx].h[lidx]});
}
void AdadeltaTrainer::alloc_impl() {
  hg = AllocateShadowParameters(*model);
  hlg = AllocateShadowLookupParameters(*model);
  hd = AllocateShadowParameters(*model);
  hld = AllocateShadowLookupParameters(*model);
}
#endif

// --- RmsPropTrainer
// TODO: This is not finished yet, because it memorizes a scalar for each set of parameters, not each parameter itself.
//       We could implement this with one tensor for each scalar, but this is pretty wasteful

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void RmsPropTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  throw std::runtime_error("RMSProp optimization not implemented yet.");
  // real& d2 = hg[pi++];
  // real g2 = p->g.vec().squaredNorm();
  // d2 = rho * d2 + (1.f - rho) * g2;
  // p->values.vec() -= ((eta * scale * gscale / sqrt(d2 + epsilon)) * p->g.vec()) / global_weight_decay.CurrentWeightDecay();
}
CNN_TRAINER_INST_DEV_IMPL(RmsPropTrainer)

#ifndef __CUDACC__
void RmsPropTrainer::update_params(real scale, real gscale, size_t idx) {
  throw std::runtime_error("RMSProp optimization not implemented yet.");
  // auto & p = model->parameters_list()[idx];
  // update_rule(scale, gscale, {&p->values, &p->g, &hg[idx].h, &hd[idx].h});
}
void RmsPropTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  throw std::runtime_error("RMSProp optimization not implemented yet.");
  // auto & p = model->lookup_parameters_list()[idx];
  // update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlg[idx].h[lidx], &hld[idx].h[lidx]});
}
void RmsPropTrainer::alloc_impl() {
  throw std::runtime_error("RMSProp optimization not implemented yet.");
  // hg.resize(model->parameters_list().size());
  // unsigned pi = 0;
  // hlg.resize(model->lookup_parameters_list().size());
  // for (auto p : model->lookup_parameters_list()) {
  //   hlg[pi++].resize(p->size());
  // }
}
#endif

// --- AdamTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=mean, ts[3]=variance
template <class MyDevice>
void AdamTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * beta_1 + ts[1]->tvec() * (1 - beta_1);
  ts[3]->tvec().device(*dev.edevice) = ts[3]->tvec() * beta_2 + ts[1]->tvec().square() * (1 - beta_2);
  // TODO: Is updates really appropriate here?
  float s1 = 1 - pow(beta_1, updates);
  float s2 = 1 - pow(beta_2, updates);
  ts[0]->tvec() += ts[2]->tvec() * ((ts[3]->tvec() / s2).sqrt() + epsilon) * (-eta / s1 / global_weight_decay.CurrentWeightDecay());
}
CNN_TRAINER_INST_DEV_IMPL(AdamTrainer)

#ifndef __CUDACC__
void AdamTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h});
}
void AdamTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx]});
}
void AdamTrainer::alloc_impl() {
  m = AllocateShadowParameters(*model);
  lm = AllocateShadowLookupParameters(*model);
  v = AllocateShadowParameters(*model);
  lv = AllocateShadowLookupParameters(*model);
}
#endif

} // namespace cnn
