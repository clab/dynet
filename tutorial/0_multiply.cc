#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"

#include <iostream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
  cnn::initialize(argc, argv);

  ComputationGraph cg;
  float ia, ib;
  Expression a = input(cg, &ia);
  Expression b = input(cg, &ib);
  Expression y = a * b;

  ia = 1;
  ib = 2;
  cout << as_scalar(cg.forward()) << endl; // 2

  ia = 3;
  ib = 3;
  cout << as_scalar(cg.forward()) << endl; // 9
}
