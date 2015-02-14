#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <initializer_list>
#include <vector>
#include <unordered_map>
#include "cnn/params.h"

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m) : model(m) {}
  virtual ~Trainer();

  virtual void update(real scale) = 0;

  Model* model;  // parameters and gradients live here
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real lambda = 1e-6, real eta = 0.1) : Trainer(m), lambda(lambda), eta(eta) {}
  void update(real scale) override;
  real lambda;
  real eta;
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
