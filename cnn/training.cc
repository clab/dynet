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
  // executed on the first iteration to create vectors to
  // store the velocity
  if (!velocity_allocated) {
    vp = AllocateShadowParameters(*model);
    vlp = AllocateShadowLookupParameters(*model);
    velocity_allocated = true;
  }

  const float gscale = clip_gradients();
  unsigned pi = 0;
  for (auto p : model->parameters_list()) {
    Tensor& v = vp[pi++].h;
    auto reg = *p->values * lambda;
    (*v) = momentum * (*v) - (eta * scale * gscale) * (*p->g);
    *p->values += *v - reg;
    p->clear();
  }
  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& vx = vlp[pi++].h;
    for (auto i : p->non_zero_grads) {
      Tensor& v = vx[i];
      auto reg = (*p->values[i]) * lambda;
      (*v) = momentum * (*v) - (eta * scale * gscale) * (*p->grads[i]);
      *p->values[i] += *v - reg;
    }
    p->clear();
  }
  ++updates;
}

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
}
#endif

} // namespace cnn
