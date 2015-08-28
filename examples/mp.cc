#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include "cnn/dict.h"
#include "cnn/lstm.h"
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
/*
TODO:
- Can we move the data vector into shared memory so that we can shuffle it
  between iterations and not have to send huge vectors of integers around?
- Can we use some sort of shared memory queue to allow threads to spread
  work more evenly?
- The shadow params in the trainers need to be shared.
*/

using namespace std;
using namespace cnn;
using namespace cnn::expr;

typedef vector<int> Datum;
unsigned num_children = 1;

// Some simple functions that do IO to/from pipes.
// These are used to send data from child processes
// to the parent process or vice/versa.
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

cnn::real SumValues(const vector<cnn::real>& values) {
  return accumulate(values.begin(), values.end(), 0.0);
}

cnn::real Mean(const vector<cnn::real>& values) {
  return SumValues(values) / values.size();
}

struct Workload {
  pid_t pid;
  int c2p[2]; // Child to parent pipe
  int p2c[2]; // Parent to child pipe
};

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;  //256
unsigned HIDDEN_DIM = 24;  // 1024
unsigned VOCAB_SIZE = 5500;

cnn::Dict d;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_c;
  Parameters* p_R;
  Parameters* p_bias;
  Builder builder;
  explicit RNNLanguageModel(Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  }

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t =  i_bias + i_R * i_y_t;
      
      // LogSoftmax followed by PickElement can be written in one step
      // using PickNegLogSoftmax
      Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  }

  // return Expression for total loss
  void RandomSample(int max_len = 150) {
    cerr << endl;
    ComputationGraph cg;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    vector<Expression> errs;
    int len = 0;
    int cur = kSOS;
    while(len < max_len && cur != kEOS) {
      ++len;
      Expression i_x_t = lookup(cg, p_c, cur);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t = i_bias + i_R * i_y_t;
      
      Expression ydist = softmax(i_r_t);
      
      unsigned w = 0;
      while (w == 0 || (int)w == kSOS) {
        auto dist = as_vector(cg.incremental_forward());
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        if (w == dist.size()) w = kEOS;
      }
      cerr << (len == 1 ? "" : " ") << d.Convert(w);
      cur = w;
    }
    cerr << endl;
  }
};

vector<Datum> ReadData(string filename) {
  vector<Datum> data;
  ifstream fs(filename);
  if (!fs.is_open()) {
    cerr << "ERROR: Unable to open " << filename << endl;
    exit(1);
  }
  string line;
  while (getline(fs, line)) {
    data.push_back(ReadSentence(line, &d));
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

template <class T>
int RunChild(unsigned cid, RNNLanguageModel<T>& rnnlm, Trainer* trainer, vector<Workload>& workloads,
    const vector<Datum>& data) {
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

    // Run the actual training loop
    cnn::real loss = 0;
    for (unsigned i : indices) {
      assert (i < data.size());
      const Datum& datum = data[i];
      ComputationGraph cg;
      rnnlm.BuildLMGraph(datum, cg);
      loss += as_scalar(cg.forward());
      cg.backward();
      trainer->update(1.0);
    }
    trainer->update_epoch();

    // Let the parent know that we're done and return the loss value
    WriteReal(workloads[cid].c2p[1], loss);
  }
  return 0;
}

void RunParent(vector<Datum>& data, vector<Datum>& dev_data, vector<Workload>& workloads) {
  for (unsigned iter = 0; iter < 0; ++iter) {
    vector<unsigned> indices(data.size());
    for (unsigned i = 0; i < data.size(); ++i) {
      indices[i] = i;
    }
    random_shuffle(indices.begin(), indices.end());

    for(unsigned cid = 0; cid < num_children; ++cid) {
      // work out the indices of the data points we want this child to consider
      unsigned start = (unsigned)(1.0 * cid / num_children * data.size() + 0.5);
      unsigned end = (unsigned)(1.0 * (cid + 1) / num_children * data.size() + 0.5);
      vector<unsigned> child_indices;
      child_indices.reserve(end - start);
      for (unsigned i = start; i < end; ++i) {
        child_indices.push_back(indices[i]);
      }
      // Tell the child it's not time to quit yet
      bool cont = true;
      write(workloads[cid].p2c[1], &cont, sizeof(bool)); 
      WriteIntVector(workloads[cid].p2c[1], child_indices);
    }

    // Wait for each child to finish training its load
    vector<cnn::real> losses(num_children);
    for(unsigned cid = 0; cid < num_children; ++cid) {
      losses[cid] = ReadReal(workloads[cid].c2p[0]);
    }

    cnn::real loss = SumValues(losses) / data.size();
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
  cnn::Initialize(argc, argv, 0, true);

  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " cores corpus.txt dev.txt" << endl;
    return 1;
  }
  num_children = atoi(argv[1]);
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  assert (num_children > 0 && num_children <= 64);

  vector<Datum> data = ReadData(argv[2]);
  vector<Datum> dev_data = ReadData(argv[3]);
  vector<Workload> workloads(num_children);

  Model model; 
  SimpleSGDTrainer sgd(&model, 0.0);
  //AdamTrainer sgd(&model, 0.0);

  RNNLanguageModel<LSTMBuilder> rnnlm(model);

  for (unsigned cid = 0; cid < num_children; cid++) {
    pipe(workloads[cid].p2c);
    pipe(workloads[cid].c2p);
  }

  unsigned cid = SpawnChildren(workloads);
  if (cid < num_children) {
    return RunChild(cid, rnnlm, &sgd, workloads, data);
  }
  else {
    RunParent(data, dev_data, workloads);
  }
}
