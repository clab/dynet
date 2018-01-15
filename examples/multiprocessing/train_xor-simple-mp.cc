#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/mp.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::mp;

struct Datum {
  Datum() {}
  Datum(const vector<dynet::real>& x, const dynet::real y) : x(x), y(y) {}

  vector<dynet::real> x;
  dynet::real y;
};

class XorModel {
public:
  XorModel(const unsigned hidden_size, ParameterCollection& dynet_model) : pcg(nullptr) {
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
};

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
  Learner(XorModel* xor_model, ParameterCollection& dynet_model, const Trainer* const trainer, bool quiet) : xor_model(xor_model) {}
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

  void SaveModel() {}

private:
  XorModel* xor_model;
  // ParameterCollection& dynet_model; 
  // const Trainer* const trainer;
  // bool quiet;
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  // parameters
  const unsigned num_cores = 4;
  const unsigned ITERATIONS = 1000;
  ParameterCollection dynet_model;
  XorModel* xor_model = nullptr;
  Trainer* trainer = nullptr;

  // Otherwise, just create a new model.
  const unsigned HIDDEN_SIZE = 8;
  xor_model = new XorModel(HIDDEN_SIZE, dynet_model);
  trainer = new SimpleSGDTrainer(dynet_model);

  vector<Datum> data(4);
  data[0] = Datum({0, 0}, 0);
  data[1] = Datum({0, 1}, 1);
  data[2] = Datum({1, 0}, 1);
  data[3] = Datum({1, 1}, 0);

  Learner learner(xor_model, dynet_model, trainer, false);
  for (unsigned i = 0; i < ITERATIONS; ++i) {
    SufficientStats ss = run_mp_minibatch<Datum>(num_cores, &learner, data);
    trainer->update();
    cout << ss << endl;
  }
}
