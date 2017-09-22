#ifndef DYNET_SHADOW_PARAMS_H
#define DYNET_SHADOW_PARAMS_H

#include <vector>
#include "dynet/tensor.h"

// if your learner needs to keep track of an extra set of values (one per
// parameter), use the Shadow classes. this can be used to implement, e.g.,
// momentum or adagrad

namespace dynet {

class ParameterCollection;
struct ParameterStorage;
struct LookupParameterStorage;

struct ShadowParameters {
  ShadowParameters() {}
  explicit ShadowParameters(const ParameterStorage& p);
  Tensor h;
};

struct ShadowLookupParameters {
  ShadowLookupParameters() {}
  explicit ShadowLookupParameters(const LookupParameterStorage& lp);
  Tensor all_h;
  std::vector<Tensor> h;
 private:
  void initialize_lookups();
};

// one per element in model.parameters_list
void allocate_shadow_parameters(const ParameterCollection& model, unsigned allocated, std::vector<ShadowParameters>& target);
// one per element in model.lookup_parameters_list
void allocate_shadow_lookup_parameters(const ParameterCollection& model, unsigned allocated, std::vector<ShadowLookupParameters>& target);

} // namespace dynet

#endif
