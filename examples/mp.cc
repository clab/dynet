#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <sstream>
#include <random>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

typedef pair<cnn::real, cnn::real> Datum;
const unsigned num_children = 2;

cnn::real ReadReal(int pipe) {
  cnn::real v;
  read(pipe, &v, sizeof(cnn::real));
  return v;
}

void WriteReal(int pipe, cnn::real v) {
  write(pipe, &v, sizeof(cnn::real));
}

cnn::real Mean(const vector<cnn::real>& values) {
  return accumulate(values.begin(), values.end(), 0.0) / values.size();
}

struct Workload {
  pid_t pid;
  unsigned start;
  unsigned end;
  int pipe[2];
};

struct ModelParameters {
  Parameters* m;
  Parameters* b;
};

void BuildComputationGraph(ComputationGraph& cg, ModelParameters& model_parameters, cnn::real* x_value, cnn::real* y_value) {
  Expression m = parameter(cg, model_parameters.m);
  Expression b = parameter(cg, model_parameters.b);

  Expression x = input(cg, x_value);
  Expression y_star = input(cg, y_value);
  Expression y = m * x + b;
  Expression loss = squared_distance(y, y_star);
}

vector<Datum> ReadData(string filename) {
  vector<Datum> data;
  ifstream fs(filename);
  string line;
  while (getline(fs, line)) {
    if (line.size() > 0 && line[0] == '#') {
      continue;
    }
    vector<string> parts;
    boost::split(parts, line, boost::is_any_of("\t"));
    data.push_back(make_pair(atof(parts[0].c_str()), atof(parts[1].c_str())));
  }
  return data;
}

unsigned SpawnChildren(vector<Workload>& workloads) {
  assert (workloads.size() == num_children);
  pid_t pid;
  unsigned cid;
  for (cid = 0; cid < num_children; ++cid) {
    pid = fork();
    if (pid == -1) {
      cerr << "Fork failed. Exiting ...";
      return 1;
    }
    else if (pid == 0) {
      // children shouldn't continue looping
      break;
    }
    workloads[cid].pid = pid;
  }
  return cid;
}

int RunChild(unsigned cid, ComputationGraph& cg, Trainer* trainer, vector<Workload>& workloads,
    const vector<Datum>& data, cnn::real& x_value, cnn::real& y_value, ModelParameters& model_params) {
  assert (cid >= 0 && cid < num_children);
  unsigned start = workloads[cid].start;
  unsigned end = workloads[cid].end; 
  assert (start < end);
  assert (end <= data.size());

  cnn::real loss = 0;
  for (auto it = data.begin() + start; it != data.begin() + end; ++it) {
    auto p = *it;
    x_value = get<0>(p);
    y_value = get<1>(p);
    loss += as_scalar(cg.forward());
    cg.backward();
    trainer->update(1.0);
  }
  loss /= (end - start);

  cnn::real m_end = as_scalar(model_params.m->values);
  cnn::real b_end = as_scalar(model_params.b->values);

  write(workloads[cid].pipe[1], (char*)&m_end, sizeof(cnn::real));
  write(workloads[cid].pipe[1], (char*)&b_end, sizeof(cnn::real));
  write(workloads[cid].pipe[1], (char*)&loss, sizeof(cnn::real));
  return 0;
}

void RunParent(unsigned iter, vector<Workload>& workloads, ModelParameters& model_params, Trainer* trainer) {
  vector<cnn::real> m_values;
  vector<cnn::real> b_values;
  vector<cnn::real> loss_values;
  for(unsigned cid = 0; cid < num_children; ++cid) { 
    cnn::real m = ReadReal(workloads[cid].pipe[0]);
    cnn::real b = ReadReal(workloads[cid].pipe[0]);
    cnn::real loss = ReadReal(workloads[cid].pipe[0]);
    m_values.push_back(m);
    b_values.push_back(b);
    loss_values.push_back(loss);
    wait(NULL); 
  }

  cnn::real m = Mean(m_values);
  cnn::real b = 0.0;
  cnn::real loss = 0.0;
  for (unsigned i = 0; i < m_values.size(); ++i) {
    b += b_values[i];
    loss += loss_values[i];
  }

  b /= b_values.size();

  // Update parameters to use the new m and b values
  TensorTools::SetElements(model_params.m->values, {m});
  TensorTools::SetElements(model_params.b->values, {b});
  trainer->update_epoch(); 
  cerr << iter << "\t" << "loss = " << loss << "\tm = " << m << "\tb = " << b << endl;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " data.txt" << endl;
    cerr << "Where data.txt contains tab-delimited pairs of floats." << endl;
    return 1;
  }
  vector<Datum> data = ReadData(argv[1]);
  vector<Workload> workloads(num_children);

  Model model;
  AdamTrainer sgd(&model, 0.0);

  ComputationGraph cg;
  cnn::real x_value, y_value;
  Parameters* m_param = model.add_parameters({1, 1});
  Parameters* b_param = model.add_parameters({1});
  ModelParameters model_params = {m_param, b_param};
  BuildComputationGraph(cg, model_params, &x_value, &y_value);

  for (unsigned cid = 0; cid < num_children; cid++) {
    unsigned start = (unsigned)(1.0 * cid / num_children * data.size() + 0.5);
    unsigned end = (unsigned)(1.0 * (cid + 1) / num_children * data.size() + 0.5);
    workloads[cid].start = start;
    workloads[cid].end = end;
    pipe(workloads[cid].pipe);
  }

  // train the parameters
  for (unsigned iter = 0; true; ++iter) {
    random_shuffle(data.begin(), data.end());
    unsigned cid = SpawnChildren(workloads);

    if (cid < num_children) {
      return RunChild(cid, cg, &sgd, workloads, data, x_value, y_value, model_params);
    }
    else {
      RunParent(iter, workloads, model_params, &sgd);
    }
  }
}

