#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <vector>
#include "cnn/model.h"
#include "cnn/shadow-params.h"

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m, real lam, real e0) :
    eta0(e0), eta(e0), eta_decay(), epoch(), lambda(lam), clipping_enabled(true), clip_threshold(5), clips(), updates(), model(m) {}
  virtual ~Trainer();

  virtual void update(real scale = 1.0) = 0;
  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  // if clipping is enabled and the gradient is too big, return the amount to
  // scale the gradient by (otherwise 1)
  float clip_gradients();

  // learning rates
  real eta0;
  real eta;
  real eta_decay;
  real epoch;

  real lambda; // weight regularization (l2)

  // clipping
  real clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;

  void status() {
    std::cerr << "[epoch=" << epoch << " eta=" << eta << " clips=" << clips << " updates=" << updates << "] ";
    updates = clips = 0;
  }

  Model* model;  // parameters and gradients live here
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) : Trainer(m, lam, e0) {}
  void update(real scale) override;
  void update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, real scale = 1);
};

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.01, real mom = 0.9) :
    Trainer(m, lam, e0), momentum(mom), velocity_allocated(false) {}
  void update(real scale) override;

  real momentum;

  bool velocity_allocated;

  // the following represent the current velocity
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
  //std::unordered_map<Parameters*, Tensor> vp;
  //std::unordered_map<LookupParameters*, std::unordered_map<unsigned, Tensor>> vl;
};

struct AdagradTrainer : public Trainer {
  explicit AdagradTrainer(Model* m, real lam = 1e-6, real e0 = 0.1, real eps = 1e-20) :
    Trainer(m, lam, e0), epsilon(eps), shadow_params_allocated(false) {}
  void update(real scale) override;

  real epsilon;
  bool shadow_params_allocated;
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
};

struct AdadeltaTrainer : public Trainer {
  explicit AdadeltaTrainer(Model* m, real lam = 1e-6, real eps = 1e-6, real rho = 0.95) :
    Trainer(m, lam, 1.0), epsilon(eps), rho(rho), shadow_params_allocated(false) {}
  void update(real scale) override;

  real epsilon;
  real rho;
  bool shadow_params_allocated;
  std::vector<ShadowParameters> hg; // History of gradients
  std::vector<ShadowLookupParameters> hlg;
  std::vector<ShadowParameters> hd; // History of deltas
  std::vector<ShadowLookupParameters> hld;
};

struct RmsPropTrainer : public Trainer {
  explicit RmsPropTrainer(Model* m, real lam = 1e-6, real e0 = 0.1, real eps = 1e-20, real rho = 0.95) :
    Trainer(m, lam, e0), epsilon(eps), rho(rho), shadow_params_allocated(false) {}
  void update(real scale) override;

  real epsilon;
  real rho;
  bool shadow_params_allocated;
  std::vector<real> hg; // History of gradients
  std::vector<std::vector<real> > hlg;
};

struct AdamTrainer : public Trainer {
  explicit AdamTrainer(Model* m, float lambda = 1e-6, float alpha = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8) :
    Trainer(m, lambda, alpha), beta_1(beta_1), beta_2(beta_2), eps(eps), shadow_params_allocated(false) {}

  void update(real scale) override;

  float beta_1;
  float beta_2;
  float eps;
  bool shadow_params_allocated;
  std::vector<ShadowParameters> m; // History of gradients
  std::vector<ShadowLookupParameters> lm;
  std::vector<ShadowParameters> v; // History of deltas
  std::vector<ShadowLookupParameters> lv;
};

} // namespace cnn

#endif
