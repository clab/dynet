#ifndef CNN_EIGEN_BACKEND_H_
#define CNN_EIGEN_BACKEND_H_

#include "Eigen/Eigen"

namespace cnn {

// This is a class that makes some of the Minerva library calls available
// even with the Eigen backend (it will just be used in porting, and until
// Minerva supports everything it should on the CPU, I hope).
//
// Note about names in the Backward functions:
//   Minerva's bottom = CNN's x (the input to the function)
//   Minvera's top = CNN's fx (the output of the function)
//   Minerva's diff = CNN's dEdf (the derivative of the loss with respect to fx)

class Elewise {
 public:
  static Eigen::MatrixXf Ln(const Eigen::MatrixXf&);
  static Eigen::MatrixXf Exp(const Eigen::MatrixXf&);
  static Eigen::MatrixXf SigmoidForward(const Eigen::MatrixXf&);
  static Eigen::MatrixXf SigmoidBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom);
  static Eigen::MatrixXf ReluForward(const Eigen::MatrixXf&);
  static Eigen::MatrixXf ReluBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom);
  static Eigen::MatrixXf TanhForward(const Eigen::MatrixXf&);
  static Eigen::MatrixXf TanhBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, const Eigen::MatrixXf& bottom);
};

typedef unsigned SoftmaxAlgorithm;

class Convolution {
 public:
  static Eigen::MatrixXf SoftmaxForward(const Eigen::MatrixXf& src, SoftmaxAlgorithm algorithm);
  static Eigen::MatrixXf SoftmaxBackward(const Eigen::MatrixXf& diff, const Eigen::MatrixXf& top, SoftmaxAlgorithm algorithm);
};

} // namespace cnn

#endif
