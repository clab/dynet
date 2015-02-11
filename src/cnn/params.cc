#include "cnn/params.h"

#include <iostream>

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return values.rows() * values.cols(); }

real Parameters::g_squared_l2norm() const { return g.squaredNorm(); }

void Parameters::accumulate_grad(const Matrix& d) { g += d; }

void Parameters::clear() { g *= 0; }

size_t LookupParameters::size() const {
  return values.size() * dim.rows * dim.cols;
}

real LookupParameters::g_squared_l2norm() const {
  real a = 0;
  for (auto it : this->g)
    a += it.second.squaredNorm();
  return a;
}

void LookupParameters::accumulate_grad(unsigned index, const Matrix& d) {
  auto it = this->g.find(index);
  if (it == this->g.end()) {
    g[index] = d;
  } else {
    g[index] += d;
  }
}

void LookupParameters::clear() { g.clear(); }

real ConstParameters::g_squared_l2norm() const { return 0; }

// since these aren't optimized, don't count them
size_t ConstParameters::size() const { return 0; }

} // namespace cnn
