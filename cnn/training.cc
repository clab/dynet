#include "cnn/training.h"

#include "cnn/gpu-ops.h"

namespace cnn {

using namespace std;

template <class Derived>
bool is_valid(const Eigen::MatrixBase<Derived>& x) {
  return ((x - x).array() == (x - x).array()).all();
}

Trainer::~Trainer() {}

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

void SimpleSGDTrainer::update(real scale) {
    update(model->lookup_parameters_list(), model->parameters_list(), scale);
}

void SimpleSGDTrainer::update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, real scale) {
  const float gscale = clip_gradients();
  for (auto p : params) {
#if HAVE_CUDA
    gpu::sgd_update(p->values.d.size(), p->g.v, p->values.v, eta * scale * gscale, lambda);
#else
    auto reg = (*p->values) * lambda;
    *p->values -= ((eta * scale * gscale) * *p->g + reg);
#endif
    p->clear();
  }
  for (auto p : lookup_params) {
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

void AdagradTrainer::update(real scale) {
  unsigned pi;
  if (!shadow_params_allocated) {
    vp = AllocateShadowParameters(*model);
    vlp = AllocateShadowLookupParameters(*model);
    shadow_params_allocated = true;
  }

  pi = 0;
  const float gscale = clip_gradients();
  for (auto p : model->parameters_list()) {
    Tensor& v = vp[pi++].h;
    auto reg = (*p->values) * lambda;
    auto g = scale * gscale * (*p->g);
    auto g2 = g.cwiseProduct(g);
    (*v) += g2;
    auto delta = -eta * g.cwiseQuotient(((*v).array() + epsilon).matrix().cwiseSqrt());
    *p->values += delta - reg;
    p->clear();
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& vx = vlp[pi++].h;
    for (auto i : p->non_zero_grads) {
      Tensor& v = vx[i];
      auto reg = (*p->values[i]) * lambda;
      auto g = scale * gscale * (*p->grads[i]);
      auto g2 = g.cwiseProduct(g);
      (*v) += g2;
      auto delta = -eta * g.cwiseQuotient(((*v).array() + epsilon).matrix().cwiseSqrt());
      *p->values[i] += delta - reg;
    }
    p->clear();
  }

  ++updates;
}

void AdadeltaTrainer::update(real scale) {
  unsigned pi;
  if (!shadow_params_allocated) {
    hg = AllocateShadowParameters(*model);
    hlg = AllocateShadowLookupParameters(*model);
    hd = AllocateShadowParameters(*model);
    hld = AllocateShadowLookupParameters(*model);

    /*pi = 0;
    for (auto p : model->parameters_list()) {
      TensorTools::Constant(hg[pi].h, epsilon);
      TensorTools::Constant(hd[pi].h, epsilon);
      ++pi;
    }

    pi = 0;
    for (auto p : model->lookup_parameters_list()) {
      vector<Tensor>& hgx = hlg[pi].h;
      vector<Tensor>& hdx = hld[pi].h;
      for (unsigned i = 0; i < hgx.size(); ++i) {
        TensorTools::Constant(hgx[i], epsilon);
        TensorTools::Constant(hdx[i], epsilon);
      }
      ++pi;
    }*/

    shadow_params_allocated = true;
  }

  const float gscale = clip_gradients();
  pi = 0;
  for (auto p : model->parameters_list()) {
    auto& g = (scale * gscale) * *p->g;
    Tensor& hgv = hg[pi].h;
    Tensor& hdv = hd[pi].h;
    auto reg = (*p->values) * lambda;
    auto g2 = g.cwiseProduct(g);
    *hgv = rho * *hgv + (1.0 - rho) * g2;
    auto num = -g.cwiseProduct(((*hdv).array() + epsilon).matrix().cwiseSqrt());
    auto den = ((*hgv).array() + epsilon).matrix().cwiseSqrt();
    auto delta = num.cwiseQuotient(den);
    auto d2 = delta.cwiseProduct(delta);
    *hdv = rho * *hdv + (1.0 - rho) * d2;
    *p->values += delta - reg;
    p->clear();
    pi++;
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& hgvx = hlg[pi].h;
    vector<Tensor>& hdvx = hld[pi].h;
    for (auto i : p->non_zero_grads) {
      Tensor& hgv = hgvx[i];
      Tensor& hdv = hdvx[i];
      auto& g = scale * gscale * *p->grads[i];
      auto reg = (*p->values[i]) * lambda;
      auto g2 = g.cwiseProduct(g);
      *hgv = rho * *hgv + (1.0 - rho) * g2;
      auto num = -g.cwiseProduct(((*hdv).array() + epsilon).matrix().cwiseSqrt());
      auto den = ((*hgv).array() + epsilon).matrix().cwiseSqrt();
      auto delta = num.cwiseQuotient(den);
      auto d2 = delta.cwiseProduct(delta);
      *hdv = rho * *hdv + (1.0 - rho) * d2;
      *p->values[i] += delta - reg;
    }
    p->clear();
    pi++;
  }
  ++updates;
}

void RmsPropTrainer::update(real scale) {
  unsigned pi = 0;
  if (!shadow_params_allocated) {
    hg.resize(model->parameters_list().size());

    pi = 0;
    hlg.resize(model->lookup_parameters_list().size());
    for (auto p : model->lookup_parameters_list()) {
      hlg[pi++].resize(p->size());
    }

    shadow_params_allocated = true;
  }

  const float gscale = clip_gradients();
  pi = 0;
  for (auto p : model->parameters_list()) {
    real& d2 = hg[pi++];
    auto reg = (*p->values) * lambda;
    real g2 = (*p->g).squaredNorm();
    d2 = rho * d2 + (1.0 - rho) * g2;
    *p->values -= ((eta * scale * gscale / sqrt(d2 + epsilon)) * *p->g + reg);
    p->clear();
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<real>& hlgx = hlg[pi++];
    for (auto i : p->non_zero_grads) {
      real& d2 = hlgx[i];
      auto reg = (*p->values[i]) * lambda;
      real g2 = (*p->grads[i]).squaredNorm();
      d2 = rho * d2 + (1.0 - rho) * g2;
      *p->values[i] -= ((eta * scale * gscale / sqrt(d2 + epsilon)) * *p->grads[i] + reg);
    }
    p->clear();
  }
  ++updates;
}

void AdamTrainer::update(real scale) {
  unsigned pi;
  if (!shadow_params_allocated) {
    m = AllocateShadowParameters(*model);
    lm = AllocateShadowLookupParameters(*model);
    v = AllocateShadowParameters(*model);
    lv = AllocateShadowLookupParameters(*model);
    shadow_params_allocated = true;
  }

  const float gscale = clip_gradients();
  pi = 0;
  static unsigned t = 0;
  for (auto p : model->parameters_list()) {
    ++t;
    auto g_t = (scale * gscale) * *p->g;
    auto m_t = *m[pi].h;
    auto v_t = *v[pi].h;
    auto reg = (*p->values) * lambda;
    m_t = beta_1 * m_t + (1 - beta_1) * g_t;
    auto g2 = g_t.cwiseProduct(g_t);
    v_t = beta_2 * v_t + (1 - beta_2) * g2;
    float s1 = 1 - pow(beta_1, t);
    float s2 = 1 - pow(beta_2, t);
    auto mhat = m_t / s1;
    auto vhat = v_t / s2;
    auto delta = (-eta * mhat).cwiseQuotient((vhat.array().sqrt() + eps).matrix());
    *p->values += delta - reg;
    p->clear();
    pi++;
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& vm = lm[pi].h;
    vector<Tensor>& vv = lv[pi].h;
    for (auto i : p->non_zero_grads) {
      auto m_t = *vm[i];
      auto v_t = *vv[i];
      auto g_t = scale * gscale * *p->grads[i];
      auto g2 = g_t.cwiseProduct(g_t);
      auto reg = (*p->values[i]) * lambda;
      m_t = beta_1 * m_t + (1 - beta_1) * g_t;
      v_t = beta_2 * v_t + (1 - beta_2) * g2;
      float s1 = 1 - pow(beta_1, t);
      float s2 = 1 - pow(beta_2, t);
      auto mhat = m_t / s1;
      auto vhat = v_t / s2;
      auto delta = (-eta * mhat).cwiseQuotient((vhat.array().sqrt() + eps).matrix());
      *p->values[i] += delta - reg;
    }
    p->clear();
    pi++;
  }
  ++updates;
}

} // namespace cnn
