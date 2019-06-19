#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include <iostream>

using namespace std;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  const unsigned ITERATIONS = 30;

  // ParameterCollection (all the model parameters).
  ParameterCollection m;
  SimpleSGDTrainer trainer(m);

  const unsigned HIDDEN_SIZE = 8;
  Parameter p_W = m.add_parameters({HIDDEN_SIZE, 2});
  Parameter p_b = m.add_parameters({HIDDEN_SIZE});
  Parameter p_V = m.add_parameters({1, HIDDEN_SIZE});
  Parameter p_a = m.add_parameters({1});
  if (argc == 2) {
    // Load the model and parameters from file if given.
    TextFileLoader loader(argv[1]);
    loader.populate(m);
  }

  // Static declaration of the computation graph.
  ComputationGraph cg;
  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  // Set x_values to change the inputs to the network.
  vector<dynet::real> x_values(2);
  Expression x = input(cg, {2}, &x_values);
  dynet::real y_value;  // Set y_value to change the target output.
  Expression y = input(cg, &y_value);

  Expression h = tanh(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss_expr = squared_distance(y_pred, y);

  // Show the computation graph, just for fun.
  cg.print_graphviz();

  // Train the parameters.
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
      trainer.update();
    }
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  // Check whether our ComputationGraph learns correctly or not.
  x_values[0] = -1;	// Set input value
  x_values[1] = -1; // Set input value
  cg.forward(y_pred); // Calculate until y_pred node
  std::cout << "[-1,-1] -1 : " << as_scalar(y_pred.value()) << std::endl;
  x_values[0] = -1;
  x_values[1] =  1;
  cg.forward(y_pred);
  std::cout << "[-1, 1]  1 : " << as_scalar(y_pred.value()) << std::endl;
  x_values[0] =  1;
  x_values[1] = -1;
  cg.forward(y_pred);
  std::cout << "[ 1,-1]  1 : " << as_scalar(y_pred.value()) << std::endl;
  x_values[0] =  1;
  x_values[1] =  1;
  cg.forward(y_pred);
  std::cout << "[ 1, 1] -1 : " << as_scalar(y_pred.value()) << std::endl;

  // Output the model and parameter objects to a file.
  TextFileSaver saver("xor.model");
  saver.save(m);
}
