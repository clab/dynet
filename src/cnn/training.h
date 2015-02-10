#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <initializer_list>
#include <vector>
#include <unordered_map>
#include "cnn/params.h"

namespace cnn {

struct Trainer {
  virtual ~Trainer();
  void add_params(Parameters* p) {
    params.push_back(p);
    add_params_impl(p);
  }

  void add_params(LookupParameters* p) {
    lookup_params.push_back(p);
    add_params_impl(p);
  }

  void add_params(const std::initializer_list<Parameters*>& ps) {
    for (auto p : ps) add_params(p);
  }

  void add_params(const std::initializer_list<LookupParameters*>& ps) {
    for (auto p : ps) add_params(p);
  }

  virtual void add_params_impl(Parameters* p);
  virtual void add_params_impl(LookupParameters* p);
  virtual void update(real scale) = 0;

  std::vector<Parameters*> params;
  std::vector<LookupParameters*> lookup_params;
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(real lambda = 1e-6, real eta = 0.1) : lambda(lambda), eta(eta) {}
  void update(real scale) override;
  real lambda;
  real eta;
};

struct RMSPropTrainer : public Trainer {
  explicit RMSPropTrainer(real lambda = 1e-6, real eta = 0.01, real decay = 0.999, real eps = 1e-8) : lambda(lambda), eta(eta), decay(decay), eps(eps) {}
  void add_params_impl(Parameters* p) override;
  void add_params_impl(LookupParameters* p) override;
  void update(real scale) override;

  // store the velocity for each parameter
  std::unordered_map<Parameters*, Matrix> vp;
  std::unordered_map<LookupParameters*, std::unordered_map<unsigned, Matrix>> vl;

  real lambda;
  real eta;
  real decay;
  real eps;
};

} // namespace cnn

#endif
