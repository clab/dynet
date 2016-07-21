#include "cnn/cnn.h"
#include "cnn/shadow-params.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/model.h"

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

vector<ShadowParameters> AllocateShadowParameters(const Model& m) {
  vector<ShadowParameters> v;
  v.reserve(m.parameters_list().size());
  for (auto& p : m.parameters_list())
    v.emplace_back(*p);
  return v;
}

vector<ShadowLookupParameters> AllocateShadowLookupParameters(const Model& m) {
  vector<ShadowLookupParameters> v;
  v.reserve(m.lookup_parameters_list().size());
  for (auto& p : m.lookup_parameters_list())
    v.emplace_back(*p);
  return v;
}

} // namespace cnn

