#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"

#include <iostream>

using namespace std;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  ComputationGraph cg;
  float ia, ib;
  Expression a = input(cg, &ia);
  Expression b = input(cg, &ib);
  Expression y = a * b;

  ia = 1;
  ib = 2;
  cout << as_scalar(cg.forward(y)) << endl; // 2

  ia = 3;
  ib = 3;
  cout << as_scalar(cg.forward(y)) << endl; // 9
}
