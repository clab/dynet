#ifndef CNN_SHADOW_PARAMS_H
#define CNN_SHADOW_PARAMS_H

#include <vector>
#include "cnn/tensor.h"

// if your learner needs to keep track of an extra set of values (one per
// parameter), use the Shadow classes. this can be used to implement, e.g.,
// momentum or adagrad

namespace cnn {

class Model;
struct Parameters;
struct LookupParameters;

struct ShadowParameters {
  explicit ShadowParameters(const Parameters& p);
  Tensor h;
};

struct ShadowLookupParameters {
  explicit ShadowLookupParameters(const LookupParameters& lp);
  std::vector<Tensor> h;
};

// one per element in model.parameters_list
std::vector<ShadowParameters> AllocateShadowParameters(const Model& model);
// one per element in model.lookup_parameters_list
std::vector<ShadowLookupParameters> AllocateShadowLookupParameters(const Model& model);

} // namespace cnn

#endif
