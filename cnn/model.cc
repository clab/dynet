#include "cnn/model.h"
#include "cnn/tensor.h"

#include <iostream>

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return cnn::size(values).Prod(); }

void Parameters::rescale_gradient(real scale) { g *= scale; }

real Parameters::g_squared_l2norm() const {
#if MINERVA_BACKEND
  Tensor r = g.Reshape({g.Size().Prod()});
  Tensor sq = Elewise::Mult(r, r);
  return sq.Sum(0).Get().get()[0];
#else
  return g.squaredNorm();
#endif
}

void Parameters::accumulate_grad(const Tensor& d) { g += d; }

void Parameters::clear() {
#if MINERVA_BACKEND
  g = NArray::Zeros(g.Size());
#else
  g.setZero();
#endif
}

size_t LookupParameters::size() const {
  return values.size() * dim.Prod();
}

real LookupParameters::g_squared_l2norm() const {
  real a = 0;
#if MINERVA_BACKEND
  cerr << "No impl yet\n"; abort();
#else
  for (auto& it : this->g)
    a += it.second.squaredNorm();
#endif
  return a;
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
#if MINERVA_BACKEND
  cerr << "No impl yet\n"; abort();
#else
  auto it = this->g.find(index);
  if (it == this->g.end()) {
    g[index] = d;
  } else {
    it->second += d;
  }
#endif
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
