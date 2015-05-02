#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;
unsigned HIDDEN_DIM = 24;
unsigned VOCAB_SIZE = 0;

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

  // return VariableIndex of total loss
  VariableIndex BuildLMGraph(const vector<int>& sent, Hypergraph& hg) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(&hg);  // reset RNN builder for new graph
    builder.start_new_sequence(&hg);
    VariableIndex i_R = hg.add_parameter(p_R); // hidden -> word rep parameter
    VariableIndex i_bias = hg.add_parameter(p_bias);  // word bias
    vector<VariableIndex> errs;
    for (unsigned t = 0; t < slen; ++t) {
      // x_t = lookup sent[t] in parameters p_c
      VariableIndex i_x_t = hg.add_lookup(p_c, sent[t]);
      // y_t = RNN(x_t)
      VariableIndex i_y_t = builder.add_input(i_x_t, &hg);
      // r_t = bias + R * y_t
      VariableIndex i_r_t = hg.add_function<Multilinear>({i_bias, i_R, i_y_t});
      // ydist = softmax(r_t)
      // LogSoftmax followed by PickElement can be written in one step
      // using PickNegLogSoftmax
#if 0
      VariableIndex i_ydist = hg.add_function<LogSoftmax>({i_r_t});
      errs.push_back(hg.add_function<PickElement>({i_ydist}, sent[t+1]));
#endif
      VariableIndex i_err = hg.add_function<PickNegLogSoftmax>({i_r_t}, sent[t+1]);
      errs.push_back(i_err);
    }
    VariableIndex i_nerr = hg.add_function<Sum>(errs);
#if 0
    return hg.add_function<Negate>({i_nerr});
#endif
    return i_nerr;
  }

  // return VariableIndex of total loss
  void RandomSample(int max_len = 150) {
    cerr << endl;
    Hypergraph hg;
    builder.new_graph(&hg);  // reset RNN builder for new graph
    builder.start_new_sequence(&hg);
    VariableIndex i_R = hg.add_parameter(p_R); // hidden -> word rep parameter
    VariableIndex i_bias = hg.add_parameter(p_bias);  // word bias
    vector<VariableIndex> errs;
    int len = 0;
    int cur = kSOS;
    while(len < max_len && cur != kEOS) {
      ++len;
      // x_t = lookup sent[t] in parameters p_c
      VariableIndex i_x_t = hg.add_lookup(p_c, cur);
      // y_t = RNN(x_t)
      VariableIndex i_y_t = builder.add_input(i_x_t, &hg);
      // r_t = bias + R * y_t
      VariableIndex i_r_t = hg.add_function<Multilinear>({i_bias, i_R, i_y_t});
      // ydist = softmax(r_t)
      hg.add_function<Softmax>({i_r_t});
      unsigned w = 0;
      while (w == 0 || (int)w == kSOS) {
        auto dist = as_vector(hg.incremental_forward());
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

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<vector<int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      training.push_back(ReadSentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
        cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      dev.push_back(ReadSentence(line, &d));
      dtoks += dev.back().size();
      if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
        cerr << "Dev sentence in " << argv[2] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  ostringstream os;
  os << "lm"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  bool use_momentum = false;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);

  //RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
  RNNLanguageModel<LSTMBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 500;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }

      // build graph for this instance
      Hypergraph hg;
      auto& sent = training[order[si]];
      chars += sent.size() - 1;
      ++si;
      lm.BuildLMGraph(sent, hg);
      loss += as_scalar(hg.forward());
      hg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
    lm.RandomSample();

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (auto& sent : dev) {
        Hypergraph hg;
        lm.BuildLMGraph(sent, hg);
        dloss += as_scalar(hg.forward());
        dchars += sent.size() - 1;
      }
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
  }
  delete sgd;
}

