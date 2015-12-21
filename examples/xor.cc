#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 30;
  Model m;
  SimpleSGDTrainer sgd(&m);
  //MomentumSGDTrainer sgd(&m);

  ComputationGraph cg;

  Expression W = parameter(cg, m.add_parameters({HIDDEN_SIZE, 2}));
  Expression b = parameter(cg, m.add_parameters({HIDDEN_SIZE}));
  Expression V = parameter(cg, m.add_parameters({1, HIDDEN_SIZE}));
  Expression a = parameter(cg, m.add_parameters({1}));

  vector<cnn::real> x_values(2);  // set x_values to change the inputs to the network
  Expression x = input(cg, {2}, &x_values);
  cnn::real y_value;  // set y_value to change the target output
  Expression y = input(cg, &y_value);

  Expression h = tanh(W*x + b);
  //Expression h = tanh(affine_transform({b, W, x}));
  //Expression h = softsign(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss = squared_distance(y_pred, y);

  cg.PrintGraphviz();
  if (argc == 2) {
    ifstream in(argv[1]);
    boost::archive::text_iarchive ia(in);
    ia >> m;
  }

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      y_value = (x1 != x2) ? 1 : -1;
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd.update(1.0);
    }
    sgd.update_epoch();
    loss /= 4;
    cerr << "E = " << loss << endl;
  }
  boost::archive::text_oarchive oa(cout);
  oa << m;
}

