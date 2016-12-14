#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  // parameters
  const unsigned ITERATIONS = 30;
  Model m;
  SimpleSGDTrainer sgd(m);
  //MomentumSGDTrainer sgd(m);

  ComputationGraph cg;
  Parameter p_W, p_b, p_V, p_a;
  if (argc == 2) {
    // Load the model and parameters from
    // file if given.
    ifstream in(argv[1]);
    boost::archive::text_iarchive ia(in);
    ia >> m >> p_W >> p_b >> p_V >> p_a;
  }
  else {
    // Otherwise, just create a new model.
    const unsigned HIDDEN_SIZE = 8;
    p_W = m.add_parameters({HIDDEN_SIZE, 2});
    p_b = m.add_parameters({HIDDEN_SIZE});
    p_V = m.add_parameters({1, HIDDEN_SIZE});
    p_a = m.add_parameters({1});
  }

  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  vector<dynet::real> x_values(2);  // set x_values to change the inputs to the network
  Expression x = input(cg, {2}, &x_values);
  dynet::real y_value;  // set y_value to change the target output
  Expression y = input(cg, &y_value);

  Expression h = tanh(W*x + b);
  //Expression h = tanh(affine_transform({b, W, x}));
  //Expression h = softsign(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss_expr = squared_distance(y_pred, y);

  // Show the computation graph, just for fun.
  cg.print_graphviz();

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      y_value = (x1 != x2) ? 1 : -1;
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      sgd.update(1.0);
    }
    sgd.update_epoch();
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  // Output the model and parameter objects
  // to a cout.
  boost::archive::text_oarchive oa(cout);
  oa << m << p_W << p_b << p_V << p_a;
}

