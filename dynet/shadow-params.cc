#include "dynet/dynet.h"

#include <boost/serialization/vector.hpp>

#include "dynet/shadow-params.h"
#include "dynet/tensor.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/model.h"
#include "dynet/io-macros.h"

using namespace std;

namespace dynet {

ShadowParameters::ShadowParameters(const ParameterStorage& p) : h(p.values) {
  default_device->allocate_tensor(DeviceMempool::PS, h);
  TensorTools::Zero(h);
}

ShadowLookupParameters::ShadowLookupParameters(const LookupParameterStorage& lp) : h(lp.values) {
  for (auto& t : h) {
    default_device->allocate_tensor(DeviceMempool::PS, t);
    TensorTools::Zero(t);
  }
}

vector<ShadowParameters> allocate_shadow_parameters(const Model& m) {
  vector<ShadowParameters> v;
  v.reserve(m.parameters_list().size());
  for (auto& p : m.parameters_list())
    v.emplace_back(*p);
  return v;
}

vector<ShadowLookupParameters> allocate_shadow_lookup_parameters(const Model& m) {
  vector<ShadowLookupParameters> v;
  v.reserve(m.lookup_parameters_list().size());
  for (auto& p : m.lookup_parameters_list())
    v.emplace_back(*p);
  return v;
}

template<class Archive>
void ShadowParameters::serialize(Archive& ar, const unsigned int) {
  ar & h;
}
DYNET_SERIALIZE_IMPL(ShadowParameters)

template<class Archive>
void ShadowLookupParameters::serialize(Archive& ar, const unsigned int) {
  ar & h;
}
DYNET_SERIALIZE_IMPL(ShadowLookupParameters)

} // namespace dynet

