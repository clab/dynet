#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <sstream>
#include <random>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct SharedObject {
  cnn::real m;
  cnn::real b;
  cnn::real loss;

  cnn::real temp_m;
  cnn::real temp_b;
};

typedef pair<cnn::real, cnn::real> Datum;
const unsigned num_children = 4;
SharedObject* shared_memory = nullptr;

cnn::real ReadReal(int pipe) {
  cnn::real v;
  read(pipe, &v, sizeof(cnn::real));
  return v;
}

void WriteReal(int pipe, cnn::real v) {
  write(pipe, &v, sizeof(cnn::real));
}

template <typename T>
void WriteIntVector(int pipe, const vector<T>& vec) {
  unsigned length = vec.size();
  write(pipe, &length, sizeof(unsigned));
  for (T v : vec) {
    write(pipe, &v, sizeof(T));
  }
}

template<typename T>
vector<T> ReadIntVector(int pipe) {
  unsigned length;
  read(pipe, &length, sizeof(unsigned));
  vector<T> vec(length);
  for (unsigned i = 0; i < length; ++i) {
    read(pipe, &vec[i], sizeof(T));
  }
  return vec;
}

cnn::real Mean(const vector<cnn::real>& values) {
  return accumulate(values.begin(), values.end(), 0.0) / values.size();
}

struct Workload {
  pid_t pid;
  int c2p[2]; // Child to parent pipe
  int p2c[2]; // Parent to child pipe
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
  if (!fs.is_open()) {
    cerr << "ERROR: Unable to open " << filename << endl;
    exit(1);
  }
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
  while (true) {
    // Check if the parent wants us to exit
    bool cont = false;
    read(workloads[cid].p2c[0], &cont, sizeof(bool));
    if (!cont) {
      break;
    }

    // Read in our workload and update our local model
    vector<unsigned> indices = ReadIntVector<unsigned>(workloads[cid].p2c[0]);
 
    TensorTools::SetElements(model_params.m->values, {shared_memory->m});
    TensorTools::SetElements(model_params.b->values, {shared_memory->b});

    cnn::real loss = 0;
    for (unsigned i : indices) {
      assert (i < data.size());
      auto p = data[i];
      x_value = get<0>(p);
      y_value = get<1>(p);
      loss += as_scalar(cg.forward());
      cg.backward();
      trainer->update(1.0);
    }
    loss /= indices.size();

    // Get our final values of each parameter and send them back to the parent,
    // along with the current loss value
    cnn::real m = as_scalar(model_params.m->values);
    cnn::real b = as_scalar(model_params.b->values);
    shared_memory->temp_m += m;
    shared_memory->temp_b += b;
    shared_memory->loss += loss;

    /*write(workloads[cid].c2p[1], (char*)&m, sizeof(cnn::real));
    write(workloads[cid].c2p[1], (char*)&b, sizeof(cnn::real));
    write(workloads[cid].c2p[1], (char*)&loss, sizeof(cnn::real));*/
    WriteReal(workloads[cid].c2p[1], 0.0);
  }
  return 0;
}

void RunParent(vector<Datum>& data, vector<Workload>& workloads, ModelParameters& model_params, Trainer* trainer) {
  shared_memory->m = TensorTools::AccessElement(model_params.m->values, {0, 0});
  shared_memory->b = TensorTools::AccessElement(model_params.b->values, {0, 0});

  for (unsigned iter = 0; iter < 10; ++iter) {
    shared_memory->loss = 0.0;
    shared_memory->temp_m = 0.0;
    shared_memory->temp_b = 0.0;
    /*vector<cnn::real> m_values;
    vector<cnn::real> b_values;
    vector<cnn::real> loss_values;*/
    for(unsigned cid = 0; cid < num_children; ++cid) {
      unsigned start = (unsigned)(1.0 * cid / num_children * data.size() + 0.5);
      unsigned end = (unsigned)(1.0 * (cid + 1) / num_children * data.size() + 0.5);
      vector<unsigned> indices;
      indices.reserve(end - start);
      for (unsigned i = start; i < end; ++i) {
        indices.push_back(i);
      }
      bool cont = true;
      write(workloads[cid].p2c[1], &cont, sizeof(bool)); 
      WriteIntVector(workloads[cid].p2c[1], indices);

      /*cnn::real m = ReadReal(workloads[cid].c2p[0]);
      cnn::real b = ReadReal(workloads[cid].c2p[0]);
      cnn::real loss = ReadReal(workloads[cid].c2p[0]);
      m_values.push_back(m);
      b_values.push_back(b);
      loss_values.push_back(loss);*/
    }

    for(unsigned cid = 0; cid < num_children; ++cid) {
      ReadReal(workloads[cid].c2p[0]);
    }

    /*cnn::real m = Mean(m_values);
    cnn::real b = 0.0;
    cnn::real loss = 0.0;
    for (unsigned i = 0; i < m_values.size(); ++i) {
      b += b_values[i];
      loss += loss_values[i];
    }

    b /= b_values.size();*/

    shared_memory->m = shared_memory->temp_m / num_children;
    shared_memory->b = shared_memory->temp_b / num_children;

    // Update parameters to use the new m and b values
    //TensorTools::SetElements(model_params.m->values, {m});
    //TensorTools::SetElements(model_params.b->values, {b});
    trainer->update_epoch();
    //cerr << shared_memory->m << "\t" << iter << "\t" << "loss = " << loss << "\tm = " << m << "\tb = " << b << endl;
    cerr << iter << "\t" << "loss = " << shared_memory->loss << "\tm = " << shared_memory->m << "\tb = " << shared_memory->b << endl;
  }

  // Kill all children one by one and wait for them to exit
  for (unsigned cid = 0; cid < num_children; ++cid) {
    bool cont = false;
    write(workloads[cid].p2c[1], &cont, sizeof(cont));
    wait(NULL);
  }
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

  unsigned shm_size = 1024;
  assert (sizeof(SharedObject) < shm_size);
  key_t shm_key = ftok("/home/austinma/shared", 'R');
  if (shm_key == -1) {
    cerr << "Unable to get shared memory key" << endl;
    return 1;
  }
  int shm_id = shmget(shm_key, shm_size, 0644 | IPC_CREAT);
  if (shm_id == -1) {
    cerr << "Unable to create shared memory" << endl;
    return 1;
  }
  void* shm_p = shmat(shm_id, nullptr, 0);
  if (shm_p == (void*)-1) {
    cerr << "Unable to get shared memory pointer";
    return 1;
  }
  shared_memory = (SharedObject*)shm_p;

  for (unsigned cid = 0; cid < num_children; cid++) {
    pipe(workloads[cid].p2c);
    pipe(workloads[cid].c2p);
  }

  unsigned cid = SpawnChildren(workloads);
  if (cid < num_children) {
    return RunChild(cid, cg, &sgd, workloads, data, x_value, y_value, model_params);
  }
  else {
    RunParent(data, workloads, model_params, &sgd);
  }
}
