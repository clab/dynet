#include "cnn/model.h"
#include "cnn/tensor.h"

#include <iostream>

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return dim.size(); }

void Parameters::rescale_gradient(real scale) { g *= scale; }

real Parameters::g_squared_l2norm() const {
  return g.squaredNorm();
}

void Parameters::accumulate_grad(const Tensor& d) { g += d; }

void Parameters::clear() {
  g.setZero();
}

size_t LookupParameters::size() const {
  return values.size() * dim.size();
}

real LookupParameters::g_squared_l2norm() const {
  real a = 0;
  for (auto& it : this->g)
    a += it.second.squaredNorm();
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

LookupParameters* Model::add_lookup_parameters(unsigned n, const Dim& d) {
  LookupParameters* p = new LookupParameters(n,d);
  all_params.push_back(p);
  lookup_params.push_back(p);
  return p;
}

} // namespace cnn
