#include "cnn/model.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"

#include <unordered_set>
#include <iostream>

#define CNN_ALIGN 32

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

Parameters::Parameters(const Dim& d) : dim(d) {
  values.d = g.d = d;
  values.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
  TensorTools::Randomize(values);
  g.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
  TensorTools::Zero(g);
}

size_t Parameters::size() const { return dim.size(); }

void Parameters::rescale_gradient(real scale) { *g *= scale; }

real Parameters::g_squared_l2norm() const {
  return (*g).squaredNorm();
}

void Parameters::accumulate_grad(const Tensor& d) { *g += *d; }

void Parameters::clear() {
  TensorTools::Zero(g);
}

LookupParameters::LookupParameters(unsigned n, const Dim& d) : dim(d), values(n), grads(n) {
  for (unsigned i = 0; i < n; ++i) {
    auto& v = values[i];
    v.d = d;
    v.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
    TensorTools::Randomize(v);

    auto& g = grads[i];
    g.d = d;
    g.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
    TensorTools::Zero(g);
  }
}

void LookupParameters::Initialize(unsigned index, const vector<float>& val) {
  assert(val.size() == dim.size());
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
}

size_t LookupParameters::size() const {
  return values.size() * dim.size();
}

real LookupParameters::g_squared_l2norm() const {
  real a = 0;
  for (auto i : non_zero_grads)
    a += (*grads[i]).squaredNorm();
  return a;
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
  *grads[index] += *d;
}

void LookupParameters::rescale_gradient(real scale) {
  for (auto i : non_zero_grads)
    *grads[i] *= scale;
}

void LookupParameters::clear() {
  for (auto i : non_zero_grads)
    (*grads[i]).setZero();
  non_zero_grads.clear();
}

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
