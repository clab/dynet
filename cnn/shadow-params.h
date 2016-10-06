#ifndef CNN_SHADOW_PARAMS_H
#define CNN_SHADOW_PARAMS_H

#include <vector>
#include "cnn/tensor.h"

// if your learner needs to keep track of an extra set of values (one per
// parameter), use the Shadow classes. this can be used to implement, e.g.,
// momentum or adagrad

namespace cnn {

class Model;
struct ParameterStorage;
struct LookupParameterStorage;

struct ShadowParameters {
  ShadowParameters() {}
  explicit ShadowParameters(const ParameterStorage& p);
  Tensor h;
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

struct ShadowLookupParameters {
  ShadowLookupParameters() {}
  explicit ShadowLookupParameters(const LookupParameterStorage& lp);
  std::vector<Tensor> h;
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);

};

// one per element in model.parameters_list
std::vector<ShadowParameters> allocate_shadow_parameters(const Model& model);
// one per element in model.lookup_parameters_list
std::vector<ShadowLookupParameters> allocate_shadow_lookup_parameters(const Model& model);

} // namespace cnn

#endif
