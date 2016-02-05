#include "cnn/model.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"

#include <unordered_set>
#include <iostream>

#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#if HAVE_CUDA
#include "cnn/gpu-ops.h"
#include "cnn/cuda.h"
#endif

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

Parameters::Parameters(const Dim& d, float scale) : dim(d) {
  values.d = g.d = d;
  values.v = static_cast<float*>(ps->allocate(d.size() * sizeof(float)));
  if (scale) {
    TensorTools::Randomize(values, scale);
  }
  else {
    TensorTools::Randomize(values);
  }
  g.v = static_cast<float*>(ps->allocate(d.size() * sizeof(float)));
  TensorTools::Zero(g);
}

size_t Parameters::size() const { return dim.size(); }

void Parameters::scale_parameters(float a) {
  (*g) *= a;
}

void Parameters::squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(values.d.size(), values.v, sqnorm, true, false);
#else
  *sqnorm = (*values).squaredNorm();
#endif
}

void Parameters::g_squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(g.d.size(), g.v, sqnorm, true, false);
#else
  *sqnorm = g.vec().squaredNorm();
#endif
}

void Parameters::copy(const Parameters & param) {
  assert(dim == param.dim);
  TensorTools::CopyElements(values, param.values);
}

void Parameters::accumulate_grad(const Tensor& d) {
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, g.d.size(), kSCALAR_ONE, d.v, 1, g.v, 1));
#else
  g.vec() += d.vec();
#endif
}

void Parameters::clear() {
  TensorTools::Zero(g);
}

LookupParameters::LookupParameters(unsigned n, const Dim& d) : dim(d), values(n), grads(n) {
  for (unsigned i = 0; i < n; ++i) {
    auto& v = values[i];
    v.d = d;
    v.v = static_cast<float*>(ps->allocate(d.size() * sizeof(float)));
    TensorTools::Randomize(v);

    auto& g = grads[i];
    g.d = d;
    g.v = static_cast<float*>(ps->allocate(d.size() * sizeof(float)));
    TensorTools::Zero(g);
  }
}

void LookupParameters::scale_parameters(float a) {
  for (auto& p : values)
    (*p) *= a;
}

void LookupParameters::Initialize(unsigned index, const vector<float>& val) {
  assert(int(val.size()) == int(dim.size()));
#if HAVE_CUDA
  cerr << "implement LookupParameters::Initialize\n";
  throw cuda_not_implemented("LookupParameters::Initialize");
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
#endif
}

size_t LookupParameters::size() const {
  return values.size() * dim.size();
}

void LookupParameters::g_squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  bool acc = false;
  for (auto i : non_zero_grads) {
    gpu::l2_norm_reducer(grads[i].d.size(), grads[i].v, sqnorm, true, acc);
    acc = true;
  }
#else
  real a = 0;
  for (auto i : non_zero_grads)
    a += (*grads[i]).squaredNorm();
  *sqnorm = a;
#endif
}

void LookupParameters::squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  bool acc = false;
  for (unsigned i = 0; i < values.size(); ++i) {
    gpu::l2_norm_reducer(values[i].d.size(), values[i].v, sqnorm, true, acc);
    acc = true;
  }
#else
  float a = 0;
  for (unsigned i = 0; i < values.size(); ++i)
    a += (*values[i]).squaredNorm();
  *sqnorm = a;
#endif
}

void LookupParameters::copy(const LookupParameters & param) {
  assert(dim == param.dim);
  for(size_t i = 0; i < param.values.size(); ++i)
    TensorTools::CopyElements(values[i], param.values[i]);
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, d.d.size(), kSCALAR_ONE, d.v, 1, grads[index].v, 1));
#else
  *grads[index] += *d;
#endif
}

void LookupParameters::clear() {
  for (auto i : non_zero_grads)
    TensorTools::Zero(grads[i]);
  non_zero_grads.clear();
}

Model::~Model() {
  for (auto p : all_params) delete p;
}

void Model::project_weights(float radius) {
  static float* project_scratch = 0;
  if (!project_scratch)
    project_scratch = (float*)default_device->mem->malloc(all_params.size() * sizeof(float));
  int pi = 0;
  for (auto p : all_params) {
    p->squared_l2norm(&project_scratch[pi]);
    ++pi;
  }
  double gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += project_scratch[i];
  cerr << "NORM: " << sqrt(gg) << endl;
}

float Model::gradient_l2_norm() const {
  if (!gradient_norm_scratch)
    gradient_norm_scratch = (float*)default_device->mem->malloc(all_params.size() * sizeof(float));
  int pi = 0;
  for (auto p : all_params) {
    p->g_squared_l2norm(&gradient_norm_scratch[pi]);
    ++pi;
  }
#if HAVE_CUDA
  float res = 0;
  gpu::l2_norm_reducer(all_params.size(), gradient_norm_scratch, gradient_norm_scratch, false, false);
  cudaMemcpy(&res, gradient_norm_scratch, sizeof(float),  cudaMemcpyDeviceToHost);
  return sqrt(res);
#else
  double gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += gradient_norm_scratch[i];
  return sqrt(gg);
#endif
}

Parameters* Model::add_parameters(const Dim& d, float scale) {
  Parameters* p = new Parameters(d, scale);
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

void Model::reset_gradient() {
  for (auto p : params) { p->clear(); }
  for (auto p : lookup_params) { p->clear(); }
}

void save_cnn_model(std::string filename, Model* model) {
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << (*model);
};

void load_cnn_model(std::string filename, Model* model) {
    std::ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> (*model);
};

} // namespace cnn
