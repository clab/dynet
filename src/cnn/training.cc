#include "cnn/training.h"

namespace cnn {

using namespace std;

Trainer::~Trainer() {}

void SimpleSGDTrainer::update(real scale) {
  double extra_scale = 1.0;
  if (clipping_enabled) {
    double gg = 0;
    for (auto p : model->parameters_list())
      gg+=p->g_squared_l2norm();
    for (auto p : model->lookup_parameters_list())
      gg+=p->g_squared_l2norm();
    gg = sqrt(gg);
    if (gg > clip_threshold) {
      extra_scale = clip_threshold / gg;
      ++clips;
    }
  }
  for (auto p : model->parameters_list()) {
    const Matrix reg = p->values * lambda;
    p->values -= (eta * scale * extra_scale) * p->g;
    p->values -= reg;
    p->clear();
  }
  for (auto p : model->lookup_parameters_list()) {
    for (auto it : p->g) {
      const Matrix reg = p->values[it.first] * lambda;
      p->values[it.first] -= it.second * (eta * scale * extra_scale);
      p->values[it.first] -= reg;
    }
    p->g.clear();
  }
  ++updates;
}

#if 0
void RMSPropTrainer::update(real scale) {
  for (auto p : params) {
    Matrix& x = p->values;
    Matrix& g = p->g;
    Matrix& v = vp[p];
    v *= decay;
    v += g.cwiseProduct(g) * (1.0 - decay);
    const Matrix reg = x * lambda;
    x -= eta * g.cwiseQuotient((v + Matrix::Constant(v.rows(),v.cols(),eps)).cwiseSqrt());
    x -= reg;
    p->clear();
  }
  for (auto p : lookup_params) {
    unordered_map<unsigned, Matrix>& vt = vl[p];
    for (auto it : p->g) {
      Matrix& x = p->values[it.first];
      Matrix& g = it.second;
      Matrix& v = vt[it.first];
      if (v.rows() == 0) v = g * 0;
      v *= decay;
      v += g.cwiseProduct(g) * (1.0 - decay);
      const Matrix reg = x * lambda;
      x -= eta * g.cwiseQuotient((v + Matrix::Constant(v.rows(),v.cols(),eps)).cwiseSqrt());
      x -= reg;
    }
    p->clear();
  }
}
#endif

} // namespace cnn
