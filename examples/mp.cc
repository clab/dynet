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
  cnn::real fake;
};

typedef pair<cnn::real, cnn::real> Datum;
const unsigned num_children = 1;
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

    cnn::real old_m = as_scalar(model_params.m->values);
    cnn::real old_b = as_scalar(model_params.b->values);
 
    // Run the actual training loop
    cnn::real loss = 0;
    for (unsigned i : indices) {
      assert (i < data.size());
      auto p = data[i];
      x_value = get<0>(p);
      y_value = get<1>(p);
      loss += as_scalar(cg.forward());
      cg.backward();
      cnn::real old_mt = as_scalar(model_params.m->values);
      trainer->update(1.0);
      cnn::real new_mt = as_scalar(model_params.m->values);
    }
    trainer->update_epoch();

    cnn::real m = as_scalar(model_params.m->values);
    cnn::real b = as_scalar(model_params.b->values);

    // Let the parent know that we're done and return the loss value
    WriteReal(workloads[cid].c2p[1], m);
    WriteReal(workloads[cid].c2p[1], b);
    WriteReal(workloads[cid].c2p[1], loss);
  }
  return 0;
}

void RunParent(vector<Datum>& data, unsigned num_iterations, vector<Workload>& workloads, ModelParameters& model_params, Trainer* trainer) {
  for (unsigned iter = 0; iter < num_iterations; ++iter) {
    vector<cnn::real> ms(num_children);
    vector<cnn::real> bs(num_children);
    vector<cnn::real> losses(num_children);
    // TODO: Store the data in shared RAM, shuffle every iteration, let children's assignments live in smem too 
    for(unsigned cid = 0; cid < num_children; ++cid) {
      // work out the indices of the data points we want this child to consider
      unsigned start = (unsigned)(1.0 * cid / num_children * data.size() + 0.5);
      unsigned end = (unsigned)(1.0 * (cid + 1) / num_children * data.size() + 0.5);
      vector<unsigned> indices;
      indices.reserve(end - start);
      for (unsigned i = start; i < end; ++i) {
        indices.push_back(i);
      }
      // Tell the child it's not time to quit yet
      bool cont = true;
      write(workloads[cid].p2c[1], &cont, sizeof(bool)); 
      WriteIntVector(workloads[cid].p2c[1], indices);
    /*}

    // Wait for each child to finish training its load
    for(unsigned cid = 0; cid < num_children; ++cid) {*/
      ms[cid] = ReadReal(workloads[cid].c2p[0]);
      bs[cid] = ReadReal(workloads[cid].c2p[0]);
      losses[cid] = ReadReal(workloads[cid].c2p[0]);
    }

    cerr << "ID\tm\tb\tloss" << endl;
    cerr << "============================" << endl;
    for (unsigned cid = 0; cid < num_children; ++cid) {
      cerr << cid << "\t" << ms[cid] << "\t" << bs[cid] << "\t" << losses[cid] << endl;
    }

    // TODO: This is currently uneffective because it doesn't affect the Trainers on the child processes
    trainer->update_epoch();

    cnn::real loss = accumulate(losses.begin(), losses.end(), 0.0) / data.size();
    cerr << iter << "\t" << "loss = " << loss << endl; 
  }

  // Kill all children one by one and wait for them to exit
  for (unsigned cid = 0; cid < num_children; ++cid) {
    bool cont = false;
    write(workloads[cid].p2c[1], &cont, sizeof(cont));
    wait(NULL);
  }

  cnn::Cleanup();
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " data.txt" << endl;
    cerr << "Where data.txt contains tab-delimited pairs of floats." << endl;
    return 1;
  }
  unsigned num_iterations = (argc >= 3) ? atoi(argv[2]) : 10;
  vector<Datum> data = ReadData(argv[1]);
  vector<Workload> workloads(num_children);

  Model model; 
  SimpleSGDTrainer sgd(&model, 0.0, 0.001);
  //AdamTrainer sgd(&model, 0.0);

  ComputationGraph cg;
  cnn::real x_value, y_value;
  Parameters* m_param = model.add_parameters({1, 1});
  Parameters* b_param = model.add_parameters({1});
  ModelParameters model_params = {m_param, b_param};
  BuildComputationGraph(cg, model_params, &x_value, &y_value);

  unsigned shm_size = 1024;
  assert (sizeof(SharedObject) < shm_size);
  key_t shm_key = ftok("/Users/austinma/shared2", 'R');
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
    RunParent(data, num_iterations, workloads, model_params, &sgd);
  }
}
