#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/mp.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;

struct Datum {
  Datum() {}
  Datum(const vector<dynet::real>& x, const dynet::real y) : x(x), y(y) {}

  vector<dynet::real> x;
  dynet::real y;
};

class XorModel {
public:
  XorModel(const unsigned hidden_size, Model& dynet_model) : pcg(nullptr) {
    p_W = dynet_model.add_parameters({hidden_size, 2});
    p_b = dynet_model.add_parameters({hidden_size});
    p_V = dynet_model.add_parameters({1, hidden_size});
    p_a = dynet_model.add_parameters({1});
  }

  void new_graph(ComputationGraph& cg) {
    W = parameter(cg, p_W);
    b = parameter(cg, p_b);
    V = parameter(cg, p_V);
    a = parameter(cg, p_a);
    pcg = &cg;
  }

  Expression compute_loss(const Datum& datum) {
    Expression x = input(*pcg, {2}, &datum.x);
    Expression y = input(*pcg, &datum.y);

    Expression h = tanh(W*x + b);
    Expression y_pred = V*h + a;
    Expression loss_expr = squared_distance(y_pred, y);
    return loss_expr;
  }

private:
  XorModel() : pcg(nullptr) {}

  Parameter p_W, p_b, p_V, p_a;
  Expression W, b, V, a;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & p_W & p_b & p_V & p_a;
  }
};

void serialize(const XorModel* const xor_model, const Model& dynet_model, const Trainer* const trainer) {
  // Remove existing stdout output
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}

  // Move the cursor to the beginning of the stdout stream
  fseek(stdout, 0, SEEK_SET);

  // Dump the model to stdout
  boost::archive::text_oarchive oa(cout);
  oa & dynet_model;
  oa & xor_model;
  oa & trainer;
}

void deserialize(const string& filename, XorModel* xor_model, Model& dynet_model, Trainer* trainer) {
  ifstream in(filename.c_str());
  boost::archive::text_iarchive ia(in);
  ia & dynet_model;
  ia & xor_model;
  ia & trainer;
  in.close();
}

class SufficientStats {
public:
  dynet::real loss;
  unsigned example_count;

  SufficientStats() : loss(), example_count() {}

  SufficientStats(dynet::real loss, unsigned example_count) : loss(loss), example_count(example_count) {}

  SufficientStats& operator+=(const SufficientStats& rhs) {
    loss += rhs.loss;
    example_count += rhs.example_count;
    return *this;
  }

  friend SufficientStats operator+(SufficientStats lhs, const SufficientStats& rhs) {
    lhs += rhs;
    return lhs;
  }

  bool operator<(const SufficientStats& rhs) {
    return loss < rhs.loss;
  }

  friend std::ostream& operator<< (std::ostream& stream, const SufficientStats& stats) {
    return stream << exp(stats.loss / stats.example_count) << " (" << stats.loss << " over " << stats.example_count << " examples)";
  }
};

class Learner : public ILearner<Datum, SufficientStats> {
public:
  Learner(XorModel* xor_model, Model& dynet_model, const Trainer* const trainer, bool quiet) : xor_model(xor_model), dynet_model(dynet_model), trainer(trainer), quiet(quiet) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const Datum& datum, bool learn) {
    ComputationGraph cg;
    xor_model->new_graph(cg);
    Expression loss_expr = xor_model->compute_loss(datum);
    dynet::real loss = as_scalar(loss_expr.value());

    if (learn) {
      cg.backward(loss_expr);
    }
    return SufficientStats(loss, 1);
  }

  void SaveModel() {
    if (!quiet) {
      serialize(xor_model, dynet_model, trainer);
    }
  }

private:
  XorModel* xor_model;
  Model& dynet_model; 
  const Trainer* const trainer;
  bool quiet;
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  // parameters
  const unsigned num_cores = 4;
  const unsigned ITERATIONS = 1000;
  Model dynet_model;
  XorModel* xor_model = nullptr;
  Trainer* trainer = nullptr;

  if (argc == 2) {
    // Load the model and parameters from file if given.
    deserialize(argv[1], xor_model, dynet_model, trainer);
  }
  else {
    // Otherwise, just create a new model.
    const unsigned HIDDEN_SIZE = 8;
    xor_model = new XorModel(HIDDEN_SIZE, dynet_model);
    trainer = new SimpleSGDTrainer(dynet_model);
  }

  vector<Datum> data(4);
  data[0] = Datum({0, 0}, 0);
  data[1] = Datum({0, 1}, 1);
  data[2] = Datum({1, 0}, 1);
  data[3] = Datum({1, 1}, 0);

  Learner learner(xor_model, dynet_model, trainer, false);
  if (num_cores == 0) {
    run_single_process<Datum>(&learner, trainer, data, data, ITERATIONS, data.size(), data.size(), data.size());
  }
  else {
    run_multi_process<Datum>(num_cores, &learner, trainer, data, data, ITERATIONS, data.size(), data.size());
  }
}
