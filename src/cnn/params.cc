#include "cnn/params.h"

#include <iostream>

using namespace std;

namespace cnn {

double eta = 0.1;

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return values.rows() * values.cols(); }

void Parameters::accumulate_grad(const Matrix& d) { g += d; }

void Parameters::update(real scale) {
  values -= g * (eta * scale);
  g *= 0;
}

// since these aren't optimized, don't count them
size_t ConstParameters::size() const { return 0; }

void ConstParameters::accumulate_grad(const Matrix& g) {
  cerr << "acculuate_grad() was called on ConstParameters!\n";
  abort();
}

void ConstParameters::update(real scale) {
  cerr << "update_grad() was called on ConstParameters!\n";
  abort();
}

size_t LookupParameters::size() const {
  return values.size() * dim.rows * dim.cols;
}

void LookupParameters::accumulate_grad(const Matrix& g) {
  auto it = this->g.find(index);
  if (it == this->g.end()) {
    this->g[index] = g;
  } else {
    this->g[index] += g;
  }
}

void LookupParameters::update(real scale) {
  for (auto it : g)
    values[it.first] -= it.second * (eta * scale);
  g.clear();
}

} // namespace cnn
