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
  const unsigned ITERATIONS = 200;
  Model m;
  SimpleSGDTrainer sgd(&m);

  ComputationGraph cg;

  Expression W = parameter(cg, m.add_parameters({HIDDEN_SIZE, 2}));
  Expression b = parameter(cg, m.add_parameters({HIDDEN_SIZE}));
  Expression V = parameter(cg, m.add_parameters({1, HIDDEN_SIZE}));
  Expression a = parameter(cg, m.add_parameters({1}));

  // set x_values to change the inputs to the network
  Dim x_dim({2}, 4), y_dim({1}, 4);
  cerr << "x_dim=" << x_dim << ", y_dim=" << y_dim << endl;
  vector<cnn::real> x_values = {1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0};
  Expression x = input(cg, x_dim, &x_values);
  // set y_values expressing the output
  vector<cnn::real> y_values = {-1.0, 1.0, 1.0, -1.0};
  Expression y = input(cg, y_dim, &y_values);

  Expression h = tanh(W*x + b);
  //Expression h = tanh(affine_transform({b, W, x}));
  //Expression h = softsign(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss = squared_distance(y_pred, y);
  Expression sum_loss = sum_batches(loss);

  cg.PrintGraphviz();
  if (argc == 2) {
    ifstream in(argv[1]);
    boost::archive::text_iarchive ia(in);
    ia >> m;
  }

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    float my_loss = as_scalar(cg.forward()) / 4;
    cg.backward();
    sgd.update(0.25);
    sgd.update_epoch();
    cerr << "E = " << my_loss << endl;
  }
  //boost::archive::text_oarchive oa(cout);
  //oa << m;
}

