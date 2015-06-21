#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned INPUT_DIM = 48;
unsigned OUTPUT_DIM = 48;
unsigned VOCAB_SIZE = 0;
unsigned LABEL_SIZE = 0;

bool eval = false;
cnn::Dict d;
cnn::Dict ld;
int kSOS;
int kEOS;

struct NeuralBagOfWords {
  LookupParameters* p_w;
  Parameters* p_c2h;
  Parameters* p_hbias;
  Parameters* p_h2o;
  Parameters* p_obias;

  explicit NeuralBagOfWords(Model& m) :
      p_w(m.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_c2h(m.add_parameters({OUTPUT_DIM, INPUT_DIM})),
      p_hbias(m.add_parameters({OUTPUT_DIM})),
      p_h2o(m.add_parameters({LABEL_SIZE, OUTPUT_DIM})),
      p_obias(m.add_parameters({LABEL_SIZE})) {}

  Expression BuildClassifier(const vector<int>& x, ComputationGraph& cg) {
    Expression c2h = parameter(cg, p_c2h);
    Expression hbias = parameter(cg, p_hbias);
    Expression h2o = parameter(cg, p_h2o);
    Expression obias = parameter(cg, p_obias);

    vector<Expression> vx(x.size());
    for (unsigned i = 0; i < x.size(); ++i)
      vx[i] = lookup(cg, p_w, x[i]);
    Expression h = rectify(c2h * sum(vx) / x.size() + hbias);
    Expression y_pred = obias + h2o * h;
    return y_pred;
  }
};

bool IsCurrentPredictionCorrection(ComputationGraph& cg, int y_true) {
  auto v = as_vector(cg.incremental_forward());
  assert(v.size() > 1);
  int besti = 0;
  float best = v[0];
  for (unsigned i = 1; i < v.size(); ++i)
    if (v[i] > best) { best = v[i]; besti = i; }
  return (besti == y_true);
}

Expression CrossEntropyLoss(const Expression& y_pred, int y_true) {
  Expression lp = log_softmax(y_pred);
  Expression nll = -pick(lp, y_true);
  return nll;
}

Expression HingeLoss(const Expression& y_pred, int y_true) {
  Expression hl = hinge(y_pred, y_true, 10.0f);
  return hl;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<pair<vector<int>,int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> x,y;
      ReadSentencePair(line, &x, &d, &y, &ld);
      if (x.size() == 0 || y.size() != 1) { cerr << line << endl; abort(); }
      training.push_back(make_pair(x,y[0]));
      ttoks += x.size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
    cerr << "Labels: " << ld.size() << endl;
  }
  LABEL_SIZE = ld.size();
  //d.Freeze(); // no new word types allowed
  ld.Freeze(); // no new tag types allowed

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      vector<int> x,y;
      ReadSentencePair(line, &x, &d, &y, &ld);
      assert(y.size() == 1);
      dev.push_back(make_pair(x,y[0]));
      dtoks += x.size();
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  VOCAB_SIZE = d.size();
  ostringstream os;
  os << "textcat"
     << '_' << INPUT_DIM
     << '_' << OUTPUT_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  bool use_momentum = true;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);

  NeuralBagOfWords nbow(model);

  unsigned report_every_i = min(200, int(training.size()));
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned ttags = 0;
    unsigned correct = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sentx_y = training[order[si]];
      const auto& x = sentx_y.first;
      const int y = sentx_y.second;
      ++si;
      Expression y_pred = nbow.BuildClassifier(x, cg);
      //CrossEntropyLoss(y_pred, y);
      HingeLoss(y_pred, y);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update(0.1);
      ++lines;
      ++ttags;
    }
    sgd->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / (double)ttags) << ") ";

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dtags = 0;
      unsigned dcorr = 0;
      eval = true;
      //lm.p_th2t->scale_parameters(pdrop);
      for (auto& sent : dev) {
        const auto& x = sent.first;
        const int y = sent.second;
        ComputationGraph cg;
        Expression y_pred = nbow.BuildClassifier(x, cg);
        if (IsCurrentPredictionCorrection(cg, y)) dcorr++;
        //CrossEntropyLoss(y_pred, y);
        HingeLoss(y_pred, y);
        dloss += as_scalar(cg.incremental_forward());
        dtags++;
      }
      //lm.p_th2t->scale_parameters(1/pdrop);
      eval = false;
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / (double)dtags) << ' ';
    }
  }
  delete sgd;
}

