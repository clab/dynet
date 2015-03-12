#include "saxe_init.h"

#include <random>

#include <Eigen/SVD>

using namespace std;

namespace cnn {

std::mt19937 reng(time(0));

real randn(real) {
  std::normal_distribution<real> distribution(0.0,0.01);
  return distribution(reng);
}

Matrix OrthonormalRandom(unsigned dim, real g) {
  Matrix m = Matrix::Zero(dim, dim).unaryExpr(ptr_fun(randn));
  Eigen::JacobiSVD<Matrix> svd(m, Eigen::ComputeFullU);
  return svd.matrixU();
}

}


