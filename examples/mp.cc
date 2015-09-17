#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/mp.h"
#include "rnnlm.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>
#include <vector>
/*
TODO:
- The shadow params in the trainers need to be shared.
*/

using namespace std;
using namespace cnn;
using namespace cnn::expr;
using namespace cnn::mp;
using namespace boost::interprocess;

typedef vector<int> Datum;

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

template<class T, class D>
class Learner : public ILearner<D, cnn::real> {
public:
  explicit Learner(RNNLanguageModel<T>& rnnlm, unsigned data_size) : rnnlm(rnnlm) {}
  ~Learner() {}

  cnn::real LearnFromDatum(const D& datum, bool learn) {
    ComputationGraph cg;
    rnnlm.BuildLMGraph(datum, cg);
    cnn::real loss = as_scalar(cg.forward());
    if (learn) {
      cg.backward();
    }
    return loss;
  }

  void SaveModel() {}

private:
  RNNLanguageModel<T>& rnnlm;
};

int main(int argc, char** argv) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " cores corpus.txt dev.txt [iterations]" << endl;
    return 1;
  }
  srand(time(NULL));
  unsigned num_children = atoi(argv[1]);
  assert (num_children <= 64);
  vector<Datum> data = ReadData(argv[2]);
  vector<Datum> dev_data = ReadData(argv[3]);
  unsigned num_iterations = (argc >= 5) ? atoi(argv[4]) : UINT_MAX;
  unsigned dev_frequency = 5000;
  unsigned report_frequency = 10;

  cnn::Initialize(argc, argv, 1, true);

  Model model;
  SimpleSGDTrainer sgd(&model, 0.0, 0.2);
  //AdagradTrainer sgd(&model, 0.0);
  //AdamTrainer sgd(&model, 0.0);

  RNNLanguageModel<LSTMBuilder> rnnlm(model);

  Learner<LSTMBuilder, Datum> learner(rnnlm, data.size());
  RunMultiProcess<Datum>(num_children, &learner, &sgd, data, dev_data, num_iterations, dev_frequency, report_frequency);
}
