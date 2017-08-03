#include "dynet/param-init.h"
#include "dynet/tensor.h"
#include "dynet/except.h"

using namespace dynet;
using namespace std;

#include <fstream>

void ParameterInitNormal::initialize_params(Tensor & values) const {
  TensorTools::randomize_normal(values, mean, sqrt(var));
}

void ParameterInitUniform::initialize_params(Tensor & values) const {
  TensorTools::randomize_uniform(values, left, right);
}

void ParameterInitConst::initialize_params(Tensor & values) const {
  TensorTools::constant(values, cnst);
}

void ParameterInitIdentity::initialize_params(Tensor & values) const {
  TensorTools::identity(values);
}

void ParameterInitGlorot::initialize_params(Tensor & values) const {
  int dims = 0, dim_len = values.d.nd - (lookup ? 1 : 0);
  for (int i = 0; i < dim_len; ++i) dims += values.d[i];
  float my_scale = gain * sqrt(3 * dim_len) / sqrt(dims);
  TensorTools::randomize_uniform(values, -my_scale, my_scale);
}

void ParameterInitSaxe::initialize_params(Tensor & values) const {
  TensorTools::randomize_orthonormal(values, gain);
}


void ParameterInitFromVector::initialize_params(Tensor & values) const {
  TensorTools::set_elements(values, vec);
}

void ParameterInitFromFile::initialize_params(Tensor & values) const {
  ifstream is(filename);
  istream_iterator<float> start(is), end;
  vector<float> param_vector(start, end);
  TensorTools::set_elements(values, param_vector);
}
