#include "cnn/cnn.h"

#include <boost/serialization/vector.hpp>

#include "cnn/shadow-params.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/model.h"
#include "cnn/io-macros.h"

using namespace std;

namespace cnn {

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
CNN_SERIALIZE_IMPL(ShadowParameters)

template<class Archive>
void ShadowLookupParameters::serialize(Archive& ar, const unsigned int) {
  ar & h;
}
CNN_SERIALIZE_IMPL(ShadowLookupParameters)

} // namespace cnn

