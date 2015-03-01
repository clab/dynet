#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <initializer_list>
#include <vector>
#include <unordered_map>
#include "cnn/model.h"

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m, real e0) : eta0(e0), eta(e0), eta_decay(), epoch(), model(m) {}
  virtual ~Trainer();

  virtual void update(real scale) = 0;
  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  real eta0;
  real eta;
  real eta_decay;
  real epoch;
  Model* model;  // parameters and gradients live here
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) : Trainer(m, e0), lambda(lam), clipping_enabled(true), clip_threshold(5), clips(), updates() {}
  void update(real scale) override;
  void status() {
    std::cerr << "[eta=" << eta << " clips=" << clips << " updates=" << updates << "] ";
    updates = clips = 0;
  }
  real lambda;

  real clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;
};

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) :
    Trainer(m, e0), lambda(lam), clipping_enabled(true), clip_threshold(5), clips(), updates() {}
  void update(real scale) override;
  void status() {
    std::cerr << "[eta=" << eta << " clips=" << clips << " updates=" << updates << "] ";
    updates = clips = 0;
  }
  real lambda;
  real decay;
  real eps;

  real clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;

  std::unordered_map<Parameters*, Matrix> vp;
  std::unordered_map<LookupParameters*, std::unordered_map<unsigned, Matrix>> vl;
};

} // namespace cnn

#endif
