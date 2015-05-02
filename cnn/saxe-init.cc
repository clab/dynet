#include "saxe-init.h"

#include <random>
#include <cstring>

#include <Eigen/SVD>

using namespace std;

namespace cnn {

inline Eigen::MatrixXf EigenRandomNormal(int dim, real mean, real stddev) {
  normal_distribution<real> distribution(mean, stddev);
  auto b = [&] (real) {return distribution(*rndeng);};
  Eigen::MatrixXf r = Eigen::MatrixXf::NullaryExpr(dim, dim, b);
  return r;
}

Tensor OrthonormalRandom(int dim, real g) {
  Eigen::MatrixXf m = EigenRandomNormal(dim, 0.0, 0.01);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullU);
  return FromEigenMatrix(svd.matrixU());
}

}

