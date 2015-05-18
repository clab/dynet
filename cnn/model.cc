#include "cnn/model.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"

#include <unordered_set>
#include <iostream>

#define CNN_ALIGN 256
#if HAVE_CUDA
#include "cnn/cuda.h"
#endif

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

void Parameters::g_squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
//  TODO compute norm of gradient - should this be done sparsely or should the gradient
//  be allocated in a single block?
//  CUBLAS_CHECK(cublasSnrm2(cublas_handle, xs[0]->d.size(), xs[0]->v, 1, fx.v));
//  cerr << "RES: " << fx << endl;
  *sqnorm = 1;
#else
  *sqnorm = (*g).squaredNorm();
#endif
}

void Parameters::accumulate_grad(const Tensor& d) {
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, g.d.size(), kSCALAR_ONE, d.v, 1, g.v, 1));
#else
  *g += *d;
#endif
}

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
  assert(int(val.size()) == int(dim.size()));
#if HAVE_CUDA
  cerr << "implement LookupParameters::Initialize\n"; abort();
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
#endif
}

size_t LookupParameters::size() const {
  return values.size() * dim.size();
}

void LookupParameters::g_squared_l2norm(float* sqnorm) const {
  real a = 0;
  for (auto i : non_zero_grads)
    a += (*grads[i]).squaredNorm();
  *sqnorm = a;
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
  *grads[index] += *d;
}

void LookupParameters::clear() {
  for (auto i : non_zero_grads)
    (*grads[i]).setZero();
  non_zero_grads.clear();
}

Model::~Model() {
  for (auto p : all_params) delete p;
}

void Model::gradient_l2_norm(float* norm) const {
  double gg = 0;
  for (auto p : all_params) {
    float sqn = 0;
    p->g_squared_l2norm(&sqn);
    gg += sqn;
  }
  *norm = sqrt(gg);
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
