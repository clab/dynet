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

} // namespace cnn

#endif
