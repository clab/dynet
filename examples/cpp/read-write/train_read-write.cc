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

  XORModel(unsigned hidden_len, Model& m) {
    hidden_size = hidden_len;
    InitParams(m);
  }

  void InitParams(Model& m) {
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

  float Train(const vector<dynet::real>& input, dynet::real gold_output, SimpleSGDTrainer& sgd) {
    ComputationGraph cg;
    NewGraph(cg);

    Expression x = dynet::expr::input(cg, {(unsigned int)input.size()}, &input);
    Expression y = dynet::expr::input(cg, &gold_output);

    Expression h = tanh(W*x + b);
    Expression y_pred = V*h + a;
    Expression loss = squared_distance(y_pred, y);

    float return_loss = as_scalar(cg.forward(loss));
    cg.backward(loss);
    sgd.update(1.0);
    return return_loss;
  }

  float Decode(vector<dynet::real>& input) {
    ComputationGraph cg;
    NewGraph(cg);

    Expression x = dynet::expr::input(cg, {(unsigned int)input.size()}, &input);
    Expression h = tanh(W*x + b);
    Expression y_pred = V*h + a;
    return as_scalar(cg.forward(y_pred));
  }

  // This function should save all those variables in the archive, which
  // determine the size of other members of the class, here: hidden_size
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {

    // This can either save or read the value of hidden_size from ar,
    // depending on whether its the output or input archive.
    ar & hidden_size;

    // We may save class data, such as the hidden size
    // but we must be sure to save all Parameter objects
    // that are members of this class.
    ar & pW;
    ar & pV;
    ar & pa;
    ar & pb;
  }
};

void WriteToFile(string& filename, XORModel& model, Model& dynet_model) {
  ofstream outfile(filename);
  if (!outfile.is_open()) {
    cerr << "File opening failed" << endl;
    exit(1);
  }

  // Write out the DYNET model and the XOR model.
  // It's important to write the DYNET model first.
  // Since the XOR model uses the DYNET model,
  // saving in the opposite order will generate a
  // boost archive "Pointer Conflict" exception.
  boost::archive::text_oarchive oa(outfile);
  oa & dynet_model;  // Write down the dynet::Model object.
  oa & model;  // Write down your class object.
  outfile.close();
}

void ReadFromFile(string& filename, XORModel& model, Model& dynet_model) {
  ifstream infile(filename);
  if (!infile.is_open()) {
    cerr << "File opening failed" << endl;
    exit(1);
  }

  boost::archive::text_iarchive ia(infile);
  ia & dynet_model;  // Read the dynet::Model
  ia & model;  // Read your class object

  infile.close();
}


int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  const unsigned HIDDEN = 8;
  const unsigned ITERATIONS = 20;
  Model m;
  SimpleSGDTrainer sgd(m);
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
      loss += model.Train(x_values, y_value, sgd);
    }
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  string outfile = "out.txt";
  cerr << "Written model to File: " << outfile << endl;
  WriteToFile(outfile, model, m);  // Writing objects to file

  // New objects in which the written archive will be read
  Model read_dynet_model;
  XORModel read_model;

  cerr << "Reading model from File: " << outfile << endl;
  ReadFromFile(outfile, read_model, read_dynet_model);  // Reading from file
  cerr << "Output for the input: " << x_values[0] << " " << x_values[1] << endl;
  cerr << read_model.Decode(x_values);  // Checking output for sanity
}

