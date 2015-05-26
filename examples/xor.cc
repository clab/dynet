#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 30;
  Model m;
  SimpleSGDTrainer sgd(&m);
  //MomentumSGDTrainer sgd(&m);

  Parameters& p_a = *m.add_parameters({1});
  Parameters& p_b = *m.add_parameters({HIDDEN_SIZE});
  Parameters& p_W = *m.add_parameters({HIDDEN_SIZE, 2});
  Parameters& p_V = *m.add_parameters({1, HIDDEN_SIZE});

  // build the graph
  ComputationGraph cg;

  // get symbolic variables corresponding to parameters
  VariableIndex i_b = cg.add_parameters(&p_b);
  VariableIndex i_a = cg.add_parameters(&p_a);
  VariableIndex i_W = cg.add_parameters(&p_W);
  VariableIndex i_V = cg.add_parameters(&p_V);

  vector<cnn::real> x_values(2);  // set x_values to change the inputs to the network
  VariableIndex i_x = cg.add_input({2}, &x_values);
  cnn::real y_value;  // set y_value to change the target output
  VariableIndex i_y = cg.add_input(&y_value);

  // two options: MatrixMultiply and Sum, or AffineTransform
  // these are identical, but AffineTransform may be slightly more efficient
#if 0
  VariableIndex i_f = cg.add_function<MatrixMultiply>({i_W, i_x});
  VariableIndex i_g = cg.add_function<Sum>({i_f, i_b});
#else
  VariableIndex i_g = cg.add_function<AffineTransform>({i_b, i_W, i_x});
#endif
  VariableIndex i_h = cg.add_function<Tanh>({i_g});

#if 1
  VariableIndex i_p = cg.add_function<MatrixMultiply>({i_V, i_h});
  VariableIndex i_y_pred = cg.add_function<Sum>({i_p, i_a});
#else
  VariableIndex i_y_pred = cg.add_function<AffineTransform>({i_a, i_V, i_h});
#endif
  cg.add_function<SquaredEuclideanDistance>({i_y_pred, i_y});
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

