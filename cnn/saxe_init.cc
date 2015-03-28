#include "saxe_init.h"

#include <random>
#include <cstring>

#include <Eigen/SVD>

using namespace std;

namespace cnn {

std::mt19937 reng(time(0));

real randn(real) {
  std::normal_distribution<real> distribution(0.0,0.01);
  return distribution(reng);
}

Tensor OrthonormalRandom(unsigned dim, real g) {
  Eigen::MatrixXf m = Eigen::MatrixXf::Zero(dim, dim).unaryExpr(ptr_fun(randn));
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullU);
  return FromEigenMatrix(svd.matrixU());
}

}

