#include "saxe_init.h"

#include <random>
#include <cstring>

#include <Eigen/SVD>

using namespace std;

namespace cnn {

Tensor OrthonormalRandom(int dim, real g) {
  Eigen::MatrixXf m = RandomNormal(Dim({dim, dim}), 0.0, 0.01);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullU);
  return FromEigenMatrix(svd.matrixU());
}

}

