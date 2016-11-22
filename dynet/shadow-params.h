#ifndef DYNET_SHADOW_PARAMS_H
#define DYNET_SHADOW_PARAMS_H

#include <vector>
#include "dynet/tensor.h"

// if your learner needs to keep track of an extra set of values (one per
// parameter), use the Shadow classes. this can be used to implement, e.g.,
// momentum or adagrad

namespace dynet {

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
  Tensor all_h;
  std::vector<Tensor> h;
 private:
  void initialize_lookups();
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int) const;
  template<class Archive>
  void load(Archive& ar, const unsigned int);
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

// one per element in model.parameters_list
std::vector<ShadowParameters> allocate_shadow_parameters(const Model& model);
// one per element in model.lookup_parameters_list
std::vector<ShadowLookupParameters> allocate_shadow_lookup_parameters(const Model& model);

} // namespace dynet

#endif
