#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;

int main(int argc, char** argv) {
  srand(time(0));

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 30;
  Model m;
  SimpleSGDTrainer sgd(&m);
  //MomentumSGDTrainer sgd(&m);

  Parameters& p_a = *m.add_parameters(Dim(1,1));
  Parameters& p_b = *m.add_parameters(Dim(HIDDEN_SIZE, 1));
  Parameters& p_W = *m.add_parameters(Dim(HIDDEN_SIZE, 2));
  Parameters& p_V = *m.add_parameters(Dim(1, HIDDEN_SIZE));

  // build the graph
  Hypergraph hg;

  // get symbolic variables corresponding to parameters
  VariableIndex i_b = hg.add_parameter(&p_b);
  VariableIndex i_a = hg.add_parameter(&p_a);
  VariableIndex i_W = hg.add_parameter(&p_W);
  VariableIndex i_V = hg.add_parameter(&p_V);

  Tensor* x_values;  // set *x_values to change the inputs to the network
  VariableIndex i_x = hg.add_input(Dim(2), &x_values);
  cnn::real* y_value;  // set *y_value to change the target output
  VariableIndex i_y = hg.add_input(&y_value);

  // two options: MatrixMultiply and Sum, or Multilinear
  // these are identical, but Multilinear may be slightly more efficient
#if 0
  VariableIndex i_f = hg.add_function<MatrixMultiply>({i_W, i_x});
  VariableIndex i_g = hg.add_function<Sum>({i_f, i_b});
#else
  VariableIndex i_g = hg.add_function<Multilinear>({i_b, i_W, i_x});
#endif
  VariableIndex i_h = hg.add_function<Tanh>({i_g});

#if 0
  VariableIndex i_p = hg.add_function<MatrixMultiply>({i_V, i_h});
  VariableIndex i_y_pred = hg.add_function<Sum>({i_p, i_a});
#else
  VariableIndex i_y_pred = hg.add_function<Multilinear>({i_a, i_V, i_h});
#endif
  hg.add_function<SquaredEuclideanDistance>({i_y_pred, i_y});
  hg.PrintGraphviz();
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
      (*x_values)(0,0) = x1 ? 1 : -1;
      (*x_values)(1,0) = x2 ? 1 : -1;
      *y_value = (x1 != x2) ? 1 : -1;
      loss += hg.forward()(0,0);
      hg.backward();
      sgd.update(1.0);
    }
    sgd.update_epoch();
    loss /= 4;
    cerr << "E = " << loss << endl;
  }
  boost::archive::text_oarchive oa(cout);
  oa << m;
}

