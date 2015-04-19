#include "cnn/training.h"

namespace cnn {

using namespace std;

Trainer::~Trainer() {}

void Trainer::clip_gradients() {
  if (clipping_enabled) {
    double gg = 0;
    for (auto p : model->all_parameters_list())
      gg+=p->g_squared_l2norm();
    gg = sqrt(gg);
    if (gg > clip_threshold) {
      ++clips;
      for (auto p : model->all_parameters_list())
        p->rescale_gradient(clip_threshold / gg);
    }
  }
}

void SimpleSGDTrainer::update(real scale) {
  clip_gradients();
  for (auto p : model->parameters_list()) {
    const Tensor reg = p->values * lambda;
    p->values -= (eta * scale) * p->g;
    p->values -= reg;
    p->clear();
  }
  for (auto p : model->lookup_parameters_list()) {
    for (auto& it : p->g) {
      const Tensor reg = p->values[it.first] * lambda;
      p->values[it.first] -= it.second * (eta * scale);
      p->values[it.first] -= reg;
    }
    p->clear();
  }
  ++updates;
}

static inline Tensor& get_or_init(Tensor& x, const Tensor& t) {
#if WITH_THPP_BACKEND
  if (x.ndims() == 0) {
    x = Tensor(t.sizes());
    x.zero();
  }
  return x;
#endif
#ifdef WITH_EIGEN_BACKEND
  if (x.rows() == 0) {
    x = t;
    x.setZero();
  }
  return x;
#endif
#if WITH_MINERVA_BACKEND
#endif
}

void MomentumSGDTrainer::update(real scale) {
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
