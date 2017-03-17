#include "dynet/param-init.h"
#include "dynet/tensor.h"

using namespace dynet;
using namespace std;

#include <fstream>

void ParameterInitNormal::initialize_params(Tensor & values) const {
  TensorTools::RandomizeNormal(values, mean, sqrt(var));
}

void ParameterInitUniform::initialize_params(Tensor & values) const {
  TensorTools::RandomizeUniform(values, left, right);
}

void ParameterInitConst::initialize_params(Tensor & values) const {
  TensorTools::Constant(values, cnst);
}

void ParameterInitIdentity::initialize_params(Tensor & values) const {
  TensorTools::Identity(values);
}

void ParameterInitGlorot::initialize_params(Tensor & values) const {
  int dims = 0, dim_len = values.d.nd - (lookup ? 1 : 0);
  for (int i = 0; i < dim_len; ++i) dims += values.d[i];
  float my_scale = gain * sqrt(6) / sqrt(dims);
  TensorTools::RandomizeUniform(values, -my_scale, my_scale);
}

void ParameterInitSaxe::initialize_params(Tensor & values) const {
  if (values.device->type == DeviceType::GPU)
    throw std::runtime_error("Saxe initialization not implemented for CUDA (we welcome pull requests)");
  else
    TensorTools::RandomizeOrthonormal(values, gain);
}


void ParameterInitFromVector::initialize_params(Tensor & values) const {
  TensorTools::SetElements(values, vec);
}

void ParameterInitFromFile::initialize_params(Tensor & values) const {
  ifstream is(filename);
  istream_iterator<float> start(is), end;
  vector<float> param_vector(start, end);
  TensorTools::SetElements(values, param_vector);
}
