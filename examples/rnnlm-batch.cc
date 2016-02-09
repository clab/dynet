#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;  //256
unsigned HIDDEN_DIM = 24;  // 1024
unsigned BATCH_SIZE = 4;
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

  // return Expression of total loss
  Expression BuildLMGraphs(const vector<vector<int> >& sents,
                           unsigned id,
                           unsigned bsize,
                           unsigned & chars,
                           ComputationGraph& cg) {
    const unsigned slen = sents[id].size();
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs;
    vector<unsigned> last_arr(bsize, sents[0][0]), next_arr(bsize);
    for (unsigned t = 1; t < slen; ++t) {
      for (unsigned i = 0; i < bsize; ++i) {
        next_arr[i] = sents[id+i][t];
        if(next_arr[i] != *sents[id].rbegin()) chars++; // add non-EOS
      }
      // y_t = RNN(x_t)
      Expression i_x_t = lookup(cg, p_c, last_arr);
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t = i_bias + i_R * i_y_t;
      Expression i_err = pickneglogsoftmax(i_r_t, next_arr);
      errs.push_back(i_err);
      last_arr = next_arr;
    }
    Expression i_nerr = sum_batches(sum(errs));
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

// Sort in descending order of length
struct CompareLen {
  bool operator()(const std::vector<int>& first, const std::vector<int>& second) {
    return first.size() > second.size();
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

  // Sort the training sentences in descending order of length
  CompareLen comp;
  sort(training.begin(), training.end(), comp);
  // Pad the sentences in the same batch with EOS so they are the same length
  // This modifies the training objective a bit by making it necessary to
  // predict EOS multiple times, but it's easy and not so harmful
  for(size_t i = 0; i < training.size(); i += BATCH_SIZE)
      for(size_t j = 1; j < BATCH_SIZE; ++j)
        while(training[i+j].size() < training[i].size())
            training[i+j].push_back(kEOS);

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
  Trainer* sgd = nullptr;
  sgd = new SimpleSGDTrainer(&model);
  sgd->clip_threshold *= BATCH_SIZE;

  RNNLanguageModel<LSTMBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 500;
  vector<unsigned> order((training.size()+BATCH_SIZE-1)/BATCH_SIZE);
  unsigned si = order.size();
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i*BATCH_SIZE;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i, ++si) {
      if (si == order.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      // build graph for this instance
      ComputationGraph cg;
      unsigned bsize = std::min((unsigned)training.size()-order[si], BATCH_SIZE); // Batch size
      lm.BuildLMGraphs(training, order[si], bsize, chars, cg);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      lines += bsize;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
    lm.RandomSample();

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dchars = 0;
      for (unsigned i = 0; i < dev.size(); ++i) {
        ComputationGraph cg;
        lm.BuildLMGraphs(dev, i, 1, dchars, cg);
        dloss += as_scalar(cg.forward());
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

