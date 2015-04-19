#include "cnn/model.h"
#include "cnn/tensor.h"

#include <iostream>

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return num_params(values); }

void Parameters::rescale_gradient(real scale) { g *= scale; }

real Parameters::g_squared_l2norm() const {
  float n = g.normall(2);
  return n * n;
}

void Parameters::accumulate_grad(const Tensor& d) { g += d; }

void Parameters::clear() {
  g.zero();
}

size_t LookupParameters::size() const {
  return values.size() * num_params(values[0]);
}

real LookupParameters::g_squared_l2norm() const {
  real a = 0;
  for (auto& it : this->g) {
    real n = it.second.normall(2);
    a += n*n;
  }
  return a;
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
  auto it = this->g.find(index);
  if (it == this->g.end()) {
    g[index] = d;
  } else {
    it->second += d;
  }
}

void LookupParameters::rescale_gradient(real scale) {
  for (auto& it : this->g)
    it.second *= scale;
}

void LookupParameters::clear() { g.clear(); }

Model::~Model() {
  for (auto p : all_params) delete p;
}

Parameters* Model::add_parameters(const Dim& d) {
  Parameters* p = new Parameters(d);
  all_params.push_back(p);
  params.push_back(p);
  return p;
}

Parameters* Model::add_parameters(const Tensor& m) {  // initial value is m
  Parameters* p = new Parameters(m);
  all_params.push_back(p);
  params.push_back(p);
  return p;
}

LookupParameters* Model::add_lookup_parameters(unsigned n, const Dim& d) {
  LookupParameters* p = new LookupParameters(n,d);
  all_params.push_back(p);
  lookup_params.push_back(p);
  return p;
}

} // namespace cnn
