#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <vector>
#include "cnn/model.h"
#include "cnn/shadow-params.h"

#define CNN_TRAINER_DEFINE_DEV_IMPL() \
  void update_params(real scale, real gscale, size_t idx) override; \
  void update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) override; \
  template <class MyDevice> \
  void update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void update_rule(real scale, real gscale, const std::vector<Tensor*> & values) override;

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m, real e0) :
    eta0(e0), eta(e0), eta_decay(), epoch(), clipping_enabled(true), clip_threshold(5), clips(), updates(), aux_allocated(false), model(m) {}
  virtual ~Trainer();

  void update(real scale = 1.0);

  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  // if clipping is enabled and the gradient is too big, return the amount to
  // scale the gradient by (otherwise 1)
  float clip_gradients();

  // TODO: This is unprotected temporarily until there is a better solution
  //       for serializing the weight decay when saving models
  // Rescale all the parameters handled by this model
  void rescale_and_reset_weight_decay();

  // learning rates
  real eta0;
  real eta;
  real eta_decay;
  real epoch;

  // clipping
  real clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;

  bool aux_allocated;

  void status() {
    std::cerr << "[epoch=" << epoch << " eta=" << eta << " clips=" << clips << " updates=" << updates << "] ";
    updates = clips = 0;
  }

  Model* model;  // parameters and gradients live here

 protected:
  virtual void alloc_impl() { }
  virtual void update_rule(real scale, real gscale, const std::vector<Tensor*> & values) = 0;
  virtual void update_params(real scale, real gscale, size_t idx) = 0;
  virtual void update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) = 0;
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real e0 = 0.1) : Trainer(m, e0) {}
 protected:
  CNN_TRAINER_DEFINE_DEV_IMPL()
};

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(Model* m, real e0 = 0.01, real mom = 0.9) :
    Trainer(m, e0), momentum(mom) {}

 protected:
  CNN_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real momentum;

  // the following represent the current velocity
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
  //std::unordered_map<ParameterStorage*, Tensor> vp;
  //std::unordered_map<LookupParameterStorage*, std::unordered_map<unsigned, Tensor>> vl;
};

struct AdagradTrainer : public Trainer {
  explicit AdagradTrainer(Model* m, real e0 = 0.1, real eps = 1e-20) :
    Trainer(m, e0), epsilon(eps) {}
 protected:
  CNN_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real epsilon;
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
};

struct AdadeltaTrainer : public Trainer {
  explicit AdadeltaTrainer(Model* m, real eps = 1e-6, real rho = 0.95) :
    Trainer(m, 1.0), epsilon(eps), rho(rho) {}
 protected:
  CNN_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real epsilon;
  real rho;
  std::vector<ShadowParameters> hg; // History of gradients
  std::vector<ShadowLookupParameters> hlg;
  std::vector<ShadowParameters> hd; // History of deltas
  std::vector<ShadowLookupParameters> hld;
};

struct RmsPropTrainer : public Trainer {
  explicit RmsPropTrainer(Model* m, real e0 = 0.1, real eps = 1e-20, real rho = 0.95) :
    Trainer(m, e0), epsilon(eps), rho(rho) {}
 protected:
  CNN_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real epsilon;
  real rho;
  std::vector<real> hg; // History of gradients
  std::vector<std::vector<real> > hlg;
};

struct AdamTrainer : public Trainer {
  explicit AdamTrainer(Model* m, float alpha = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8) :
    Trainer(m, alpha), beta_1(beta_1), beta_2(beta_2), epsilon(eps) {}

 protected:
  CNN_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  float beta_1;
  float beta_2;
  float epsilon;
  std::vector<ShadowParameters> m; // History of gradients
  std::vector<ShadowLookupParameters> lm;
  std::vector<ShadowParameters> v; // History of deltas
  std::vector<ShadowLookupParameters> lv;
};

} // namespace cnn

#endif
