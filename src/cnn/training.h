#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <initializer_list>
#include <vector>
#include <unordered_map>
#include "cnn/model.h"

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m) : model(m) {}
  virtual ~Trainer();

  virtual void update(real scale) = 0;

  Model* model;  // parameters and gradients live here
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) : Trainer(m), epoch(), lambda(lam), eta0(e0), eta(e0), eta_decay(0.1) {}
  void update(real scale) override;
  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }
  real epoch;
  real lambda;
  real eta0;
  real eta;
  real eta_decay;
};

  // store the velocity for each parameter
  // std::unordered_map<Parameters*, Matrix> vp;
  // std::unordered_map<LookupParameters*, std::unordered_map<unsigned, Matrix>> vl;

  //real lambda;
  //real eta;
  //real decay;
  //real eps;

} // namespace cnn

#endif
