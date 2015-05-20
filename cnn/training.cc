#include "cnn/training.h"

#include "cnn/gpu-ops.h"

namespace cnn {

using namespace std;

Trainer::~Trainer() {}

float Trainer::clip_gradients() {
  float gscale = 1;
  if (clipping_enabled) {
    float gg = model->gradient_l2_norm();
    if (gg > clip_threshold) {
      ++clips;
      gscale = clip_threshold / gg;
    }
  }
  return gscale;
}

void SimpleSGDTrainer::update(real scale) {
  const float gscale = clip_gradients();
  for (auto p : model->parameters_list()) {
#if HAVE_CUDA
    gpu::sgd_update(p->values.d.size(), p->g.v, p->values.v, eta * scale * gscale, lambda);
#else
    auto reg = (*p->values) * lambda;
    *p->values -= ((eta * scale * gscale) * *p->g + reg);
#endif
    p->clear();
  }
  for (auto p : model->lookup_parameters_list()) {
    for (auto i : p->non_zero_grads) {
#if HAVE_CUDA
      gpu::sgd_update(p->values[i].d.size(), p->grads[i].v, p->values[i].v, eta * scale * gscale, lambda);
#else
      auto reg = (*p->values[i]) * lambda;
      *p->values[i] -= (*p->grads[i] * (eta * scale * gscale) + reg);
#endif
    }
    p->clear();
  }
  ++updates;
}

void MomentumSGDTrainer::update(real scale) {
#if 0
  clip_gradients();
  for (auto p : model->parameters_list()) {
    Tensor& v = get_or_init(vp[p], p->values);
    const Tensor reg = p->values * lambda;
    v = momentum * v - (eta * scale) * p->g;
    p->values += v;
    p->values -= reg;
    p->clear();
  }
  for (auto p : model->lookup_parameters_list()) {
    unordered_map<unsigned, Tensor>& vx = vl[p];
    for (auto& it : p->g) {
      Tensor& v = get_or_init(vx[it.first], it.second);
      const Tensor reg = p->values[it.first] * lambda;
      v = momentum * v - (eta * scale) * it.second;
      p->values[it.first] += v;
      p->values[it.first] -= reg;
    }
    p->clear();
  }
  ++updates;
}
#endif

#if 0
void RMSPropTrainer::update(real scale) {
  for (auto p : params) {
    Tensor& x = p->values;
    Tensor& g = p->g;
    Tensor& v = vp[p];
    v *= decay;
    v += g.cwiseProduct(g) * (1.0 - decay);
    const Tensor reg = x * lambda;
    x -= eta * g.cwiseQuotient((v + Tensor::Constant(v.rows(),v.cols(),eps)).cwiseSqrt());
    x -= reg;
    p->clear();
  }
  for (auto p : lookup_params) {
    unordered_map<unsigned, Tensor>& vt = vl[p];
    for (auto it : p->g) {
      Tensor& x = p->values[it.first];
      Tensor& g = it.second;
      Tensor& v = vt[it.first];
      if (v.rows() == 0) v = g * 0;
      v *= decay;
      v += g.cwiseProduct(g) * (1.0 - decay);
      const Tensor reg = x * lambda;
      x -= eta * g.cwiseQuotient((v + Tensor::Constant(v.rows(),v.cols(),eps)).cwiseSqrt());
      x -= reg;
    }
    p->clear();
  }
#endif
}

} // namespace cnn
