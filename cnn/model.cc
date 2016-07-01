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

BOOST_CLASS_EXPORT_IMPLEMENT(cnn::ParameterStorage)
BOOST_CLASS_EXPORT_IMPLEMENT(cnn::LookupParameterStorage)

namespace cnn {

ParameterStorageBase::~ParameterStorageBase() {}

ParameterStorage::ParameterStorage(const Dim& d, float scale) : dim(d) {
  values.d = g.d = d;
  values.v = static_cast<float*>(default_device->ps->allocate(d.size() * sizeof(float)));
  values.device = g.device = default_device;
  if (scale) {
    TensorTools::Randomize(values, scale);
  }
  else {
    TensorTools::Randomize(values);
  }
  g.v = static_cast<float*>(default_device->ps->allocate(d.size() * sizeof(float)));
  TensorTools::Zero(g);
}

size_t ParameterStorage::size() const { return dim.size(); }

void ParameterStorage::scale_parameters(float a) {
  values.vec() *= a;
}

void ParameterStorage::zero() {
  TensorTools::Zero(values);
  clear();
}

void ParameterStorage::squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(values.d.size(), values.v, sqnorm, true, false);
#else
  *sqnorm = (*values).squaredNorm();
#endif
}

void ParameterStorage::g_squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(g.d.size(), g.v, sqnorm, true, false);
#else
  *sqnorm = g.vec().squaredNorm();
#endif
}

void ParameterStorage::copy(const ParameterStorage & param) {
  assert(dim == param.dim);
  TensorTools::CopyElements(values, param.values);
}

void ParameterStorage::accumulate_grad(const Tensor& d) {
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, g.d.size(), kSCALAR_ONE, d.v, 1, g.v, 1));
#else
  g.vec() += d.vec();
#endif
}

void ParameterStorage::clear() {
  TensorTools::Zero(g);
}

LookupParameterStorage::LookupParameterStorage(unsigned n, const Dim& d) : dim(d), values(n), grads(n) {
  for (unsigned i = 0; i < n; ++i) {
    auto& v = values[i];
    v.d = d;
    v.v = static_cast<float*>(default_device->ps->allocate(d.size() * sizeof(float)));
    v.device = default_device;
    TensorTools::Randomize(v);

    auto& g = grads[i];
    g.d = d;
    g.v = static_cast<float*>(default_device->ps->allocate(d.size() * sizeof(float)));
    g.device = default_device;
    TensorTools::Zero(g);
  }
}

void LookupParameterStorage::scale_parameters(float a) {
  for (auto& p : values)
    (*p) *= a;
}

void LookupParameterStorage::zero() {
  for (auto& p : values)
    TensorTools::Zero(p);
  clear();
}

void LookupParameterStorage::Initialize(unsigned index, const vector<float>& val) {
  assert(int(val.size()) == int(dim.size()));
#if HAVE_CUDA
  cerr << "implement LookupParameterStorage::Initialize\n";
  throw cuda_not_implemented("LookupParameterStorage::Initialize");
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
#endif
}

size_t LookupParameterStorage::size() const {
  return values.size() * dim.size();
}

void LookupParameterStorage::g_squared_l2norm(float* sqnorm) const {
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

void LookupParameterStorage::squared_l2norm(float* sqnorm) const {
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

void LookupParameterStorage::copy(const LookupParameterStorage& param) {
  assert(dim == param.dim);
  for(size_t i = 0; i < param.values.size(); ++i)
    TensorTools::CopyElements(values[i], param.values[i]);
}

void LookupParameterStorage::accumulate_grad(unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, d.d.size(), kSCALAR_ONE, d.v, 1, grads[index].v, 1));
#else
  *grads[index] += *d;
#endif
}

void LookupParameterStorage::clear() {
  for (auto i : non_zero_grads)
    TensorTools::Zero(grads[i]);
  non_zero_grads.clear();
}

Parameter::Parameter() {
  mp = nullptr;
  index = 0;
}

Parameter::Parameter(const Model* mp, unsigned long index) : mp(mp), index(index) {}

ParameterStorage* Parameter::get() const {
  return mp->parameters_list()[index];
}

void Parameter::zero() {
  return mp->parameters_list()[index]->zero();
}

LookupParameter::LookupParameter() {
  mp = nullptr;
  index = 0;
}

LookupParameter::LookupParameter(const Model* mp, unsigned long index) : mp(mp), index(index) {}

LookupParameterStorage* LookupParameter::get() const {
  return mp->lookup_parameters_list()[index];
}

void LookupParameter::zero() {
  return mp->lookup_parameters_list()[index]->zero();
}

void LookupParameter::Initialize(unsigned index, const std::vector<float>& val) const {
  get()->Initialize(index, val);
}

Model::Model() : gradient_norm_scratch(nullptr) {
  weight_decay.SetLambda(weight_decay_lambda);
}

Model::~Model() {
  for (auto p : all_params) delete p;
  if(gradient_norm_scratch)
    default_device->mem->free(gradient_norm_scratch);
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

Parameter Model::add_parameters(const Dim& d, float scale) {
  ParameterStorage* p = new ParameterStorage(d, scale);
  Parameter r(this, params.size());
  //cerr << "Adding parameters with dim " << d << endl;
  all_params.push_back(p);
  params.push_back(p);
  return r;
}

LookupParameter Model::add_lookup_parameters(unsigned n, const Dim& d) {
  LookupParameterStorage* p = new LookupParameterStorage(n,d);
  LookupParameter r(this, lookup_params.size());
  //cerr << "Adding lookup parameters with dim " << d << " and size " << n << endl;
  all_params.push_back(p);
  lookup_params.push_back(p);
  return r;
}

void Model::reset_gradient() {
  for (auto p : params) { p->clear(); }
  for (auto p : lookup_params) { p->clear(); }
}

size_t Model::parameter_count() const {
  size_t r = 0;
  for (const ParameterStorageBase* param : all_params) {
    r += param->size();
  }
  return r;
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
