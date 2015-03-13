#ifndef CNN_EIGEN_BACKEND_H_
#define CNN_EIGEN_BACKEND_H_

#include "cnn/tensor.h"

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
  static Matrix SigmoidForward(const Matrix&);
  static Matrix SigmoidBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom);
  static Matrix ReluForward(const Matrix&);
  static Matrix ReluBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom);
  static Matrix TanhForward(const Matrix&);
  static Matrix TanhBackward(const Matrix& diff, const Matrix& top, const Matrix& bottom);
};

typedef unsigned SoftmaxAlgorithm;

class Convolution {
 public:
  static Matrix SoftmaxForward(const Matrix& src, SoftmaxAlgorithm algorithm);
  static Matrix SoftmaxBackward(const Matrix& diff, const Matrix& top, SoftmaxAlgorithm algorithm);
};

} // namespace cnn

#endif
