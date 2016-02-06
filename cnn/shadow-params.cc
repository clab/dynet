#include "cnn/cnn.h"
#include "cnn/shadow-params.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/model.h"

using namespace std;

namespace cnn {

ShadowParameters::ShadowParameters(const Parameters& p) : h(p.values) {
  h.v = (float*)default_device->mem->malloc(h.d.size() * sizeof(float));
  TensorTools::Zero(h);
}

ShadowLookupParameters::ShadowLookupParameters(const LookupParameters& lp) : h(lp.values) {
  for (auto& t : h) {
    t.v = (float*)default_device->mem->malloc(t.d.size() * sizeof(float));
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

