#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;

// This is a sample class which implements the xor model from xor.cc
// Everything in this class is just as you would do the usual except for
// parts with provided comments.
class XORModel {
public:
  unsigned hidden_size;

  Expression W, b, V, a;
  Parameter pW, pb, pV, pa;

  // It is important to have a null default constructor for the class, as
  // we would first need to read the class object from the file, followed by
  // the dynet model which has saved parameters.
  XORModel() {}

  XORModel(unsigned hidden_len, ParameterCollection& m) {
    hidden_size = hidden_len;
    InitParams(m);
  }

  void InitParams(ParameterCollection& m) {
    pW = m.add_parameters({hidden_size, 2});
    pb = m.add_parameters({hidden_size});
    pV = m.add_parameters({1, hidden_size});
    pa = m.add_parameters({1});
  }

  void NewGraph(ComputationGraph& cg) {
    W = parameter(cg, pW);
    b = parameter(cg, pb);
    V = parameter(cg, pV);
    a = parameter(cg, pa);
  }

  float Train(const vector<dynet::real>& input, dynet::real gold_output, SimpleSGDTrainer& trainer) {
    ComputationGraph cg;
    NewGraph(cg);

    Expression x = dynet::input(cg, {(unsigned int)input.size()}, &input);
    Expression y = dynet::input(cg, &gold_output);

    Expression h = tanh(W*x + b);
    Expression y_pred = V*h + a;
    Expression loss = squared_distance(y_pred, y);

    float return_loss = as_scalar(cg.forward(loss));
    cg.backward(loss);
    trainer.update();
    return return_loss;
  }

  float Decode(vector<dynet::real>& input) {
    ComputationGraph cg;
    NewGraph(cg);

    Expression x = dynet::input(cg, {(unsigned int)input.size()}, &input);
    Expression h = tanh(W*x + b);
    Expression y_pred = V*h + a;
    return as_scalar(cg.forward(y_pred));
  }
};

void WriteToFile(string& filename, ParameterCollection& dynet_model) {
  TextFileSaver saver(filename);
  saver.save(dynet_model);
}

void ReadFromFile(string& filename, ParameterCollection& dynet_model) {
  TextFileLoader loader(filename);
  loader.populate(dynet_model);
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  const unsigned HIDDEN = 8;
  const unsigned ITERATIONS = 20;
  ParameterCollection m;
  SimpleSGDTrainer trainer(m);
  XORModel model(HIDDEN, m);

  vector<dynet::real> x_values(2);  // set x_values to change the inputs
  dynet::real y_value;  // set y_value to change the target output

  // Train the model
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      y_value = (x1 != x2) ? 1 : -1;
      loss += model.Train(x_values, y_value, trainer);
    }
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  string outfile = "read-write.model";
  cerr << "Written model to File: " << outfile << endl;
  WriteToFile(outfile, m);  // Writing objects to file

  // New objects in which the written model will be read
  ParameterCollection read_dynet_model;
  XORModel read_model(HIDDEN, read_dynet_model);

  cerr << "Reading model from File: " << outfile << endl;
  ReadFromFile(outfile, read_dynet_model);  // Reading from file
  cerr << "Output for the input: " << x_values[0] << " " << x_values[1] << endl;
  cerr << read_model.Decode(x_values);  // Checking output for sanity
}
