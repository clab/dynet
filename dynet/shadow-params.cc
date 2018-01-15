#include "dynet/dynet.h"

#include "dynet/shadow-params.h"
#include "dynet/tensor.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/model.h"
#include "dynet/devices.h"

#define LOAD_INIT_FUNC() initialize_lookups()

using namespace std;

namespace dynet {

ShadowParameters::ShadowParameters(const ParameterStorage& p) : h(p.values) {
  p.device->allocate_tensor(DeviceMempool::PS, h);
  TensorTools::zero(h);
}

ShadowLookupParameters::ShadowLookupParameters(const LookupParameterStorage& lp) : all_h(lp.all_values) {
  lp.device->allocate_tensor(DeviceMempool::PS, all_h);
  TensorTools::zero(all_h);
  initialize_lookups();
}

void ShadowLookupParameters::initialize_lookups() {
  int num = all_h.d[all_h.d.nd-1];
  Dim dim = all_h.d; dim.nd--;
  int dim_size = dim.size();
  if(h.size() == 0) {
    h.resize(num);
    for(int i = 0; i < num; ++i)
      h[i] = Tensor(dim, all_h.v + i*dim_size, all_h.device, all_h.mem_pool);
  }
}

void allocate_shadow_parameters(const ParameterCollection& m, unsigned allocated, vector<ShadowParameters>& target) {
  auto& params = m.parameters_list();
  vector<shared_ptr<ParameterStorage>> to_allocate(params.begin() + allocated, params.end());
  vector<ShadowParameters> v;
  target.reserve(params.size());
  for (auto& p : to_allocate)
    target.emplace_back(*p);
}

void allocate_shadow_lookup_parameters(const ParameterCollection& m, unsigned allocated, vector<ShadowLookupParameters>& target) {
  auto& params = m.lookup_parameters_list();
  vector<shared_ptr<LookupParameterStorage>> to_allocate(params.begin() + allocated, params.end());
  target.reserve(params.size());
  for (auto& p : to_allocate)
    target.emplace_back(*p);
}

} // namespace dynet

