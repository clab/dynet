#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/cfsm-builder.h"
#include "cnn/hsm-builder.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>

#include <boost/algorithm/string/join.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/regex.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 0;
unsigned INPUT_DIM = 0;
unsigned HIDDEN_DIM = 0;
unsigned VOCAB_SIZE = 0;
float DROPOUT = 0;
bool SAMPLE = false;
FactoredSoftmaxBuilder* cfsm = nullptr;

cnn::Dict d;
int kSOS;
int kEOS;

volatile bool INTERRUPTED = false;

namespace po = boost::program_options;
void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("train,t", po::value<string>(), "training corpus")
        ("dev,d", po::value<string>(), "development/validation corpus")
        ("test,p", po::value<string>(), "test corpus")
        ("nbest", po::value<string>(), "N-best list to score (- for stdin)")
        ("learn,x", "set this to estimate the language model from the training data")
        ("clusters,c", po::value<string>(), "word cluster file for class factored softmax")
        ("paths,b", po::value<string>(), "word paths file for hierarchical softmax")
        ("sample,s", "periodically generate random samples from model as it trains (recommended)")
        ("model,m", po::value<string>(), "load model from this file")
        ("input_dim,i", po::value<unsigned>()->default_value(128), "input embedding dimension")
        ("hidden_dim,H", po::value<unsigned>()->default_value(128), "hidden layer size")
        ("layers,l", po::value<unsigned>()->default_value(2), "number of layers in RNN")
        ("dropout,D", po::value<float>(), "dropout rate (recommended between 0.2 and 0.5)")
        ("eta0,e", po::value<float>()->default_value(0.1f), "initial learning rate")
        ("eta_decay_onset_epoch", po::value<unsigned>(), "start decaying eta every epoch after this epoch (try 8)")
        ("eta_decay_rate", po::value<float>(), "how much to decay eta by (recommended 0.5)")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("train") == 0) {
    cerr << "Training data must always be specified (it determines the vocab mapping) with --train\n";
    exit(1);
  }
}

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
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg, bool apply_dropout) {
    const unsigned slen = sent.size();
    if (apply_dropout) {
      builder.set_dropout(DROPOUT);
    } else {
      builder.disable_dropout();
    }
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    if (cfsm) cfsm->new_graph(cg);
    Expression R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs(slen + 1);
    Expression h_t = builder.add_input(lookup(cg, p_c, kSOS)); // read <s>
    for (unsigned t = 0; t < slen; ++t) { // h_t = RNN(x_0,...,x_t)
      if (cfsm) { // class-factored softmax
        errs[t] = cfsm->neg_log_softmax(h_t, sent[t]);
      } else { // regular softmax
        Expression u_t = affine_transform({bias, R, h_t});
        errs[t] = pickneglogsoftmax(u_t, sent[t]);
      }
      Expression x_t = lookup(cg, p_c, sent[t]);
      h_t = builder.add_input(x_t);
    }
    // it reamins to deal predict </s>
    if (cfsm) {
      errs.back() = cfsm->neg_log_softmax(h_t, kEOS);
    } else {
      Expression u_last = affine_transform({bias, R, h_t});
      errs.back() = pickneglogsoftmax(u_last, kEOS); // predict </s>
    }
    return sum(errs);
  }

  void RandomSample(int max_len = 200) {
    ComputationGraph cg;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    if (cfsm) cfsm->new_graph(cg);
    Expression R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression bias = parameter(cg, p_bias);  // word bias
    Expression h_t = builder.add_input(lookup(cg, p_c, kSOS)); // read <s>
    int cur = kSOS;
    int len = 0;
    while(len < max_len) {
      if (cfsm) { // class-factored softmax
        cur = cfsm->sample(h_t);
      } else { // regular softmax
        Expression u_t = affine_transform({bias, R, h_t});
        softmax(u_t);
        auto dist = as_vector(cg.incremental_forward());
        double p = rand01();
        cur = 0;
        for (; cur < dist.size(); ++cur) {
          p -= dist[cur];
          if (p < 0.0) { break; }
        }
        if (cur == dist.size()) cur = kEOS;
      }
      if (cur == kEOS) break;
      ++len;
      cerr << (len == 1 ? "" : " ") << d.Convert(cur);
      Expression x_t = lookup(cg, p_c, cur);
      h_t = builder.add_input(x_t);
    }
    cerr << endl;
  }
};

int main(int argc, char** argv) {
  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  cnn::Initialize(argc, argv);
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  SAMPLE = conf.count("sample");
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  Model model;
  if (conf.count("clusters"))
    cfsm = new ClassFactoredSoftmaxBuilder(HIDDEN_DIM, conf["clusters"].as<string>(), &d, &model);
  else if (conf.count("paths"))
    cfsm = new HierarchicalSoftmaxBuilder(HIDDEN_DIM, conf["paths"].as<string>(), &d, &model);
  float eta_decay_rate = 1;
  unsigned eta_decay_onset_epoch = 0;
  if (conf.count("eta_decay_onset_epoch"))
    eta_decay_onset_epoch = conf["eta_decay_onset_epoch"].as<unsigned>();
  if (conf.count("eta_decay_rate"))
    eta_decay_rate = conf["eta_decay_rate"].as<float>();
  vector<vector<int>> training, dev, test;
  string line;
  int tlc = 0;
  int ttoks = 0;
  {
    string trainf = conf["train"].as<string>();
    cerr << "Reading training data from " << trainf << " ...\n";
    ifstream in(trainf);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      training.push_back(ReadSentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() == kSOS || training.back().back() == kEOS) {
        cerr << "Training sentence in " << argv[1] << ":" << tlc << " started with <s> or ended with </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  d.SetUnk("<unk>");
  VOCAB_SIZE = d.size();
  if (!cfsm)
    cfsm = new NonFactoredSoftmaxBuilder(HIDDEN_DIM, VOCAB_SIZE, &model);

  if (conf.count("test")) {
    string testf = conf["test"].as<string>();
    cerr << "Reading test data from " << testf << " ...\n";
    ifstream in(testf);
    assert(in);
    while(getline(in, line)) {
      test.push_back(ReadSentence(line, &d));
      if (test.back().front() == kSOS || test.back().back() == kEOS) {
        cerr << "Test sentence in " << argv[2] << ":" << tlc << " started with <s> or ended with </s>\n";
        abort();
      }
    }
  }

  Trainer* sgd = new SimpleSGDTrainer(&model);
  sgd->eta0 = sgd->eta = conf["eta0"].as<float>();
  RNNLanguageModel<LSTMBuilder> lm(model);

  bool has_model_to_load = conf.count("model");
  if (has_model_to_load) {
    string fname = conf["model"].as<string>();
    cerr << "Reading parameters from " << fname << "...\n";
    ifstream in(fname);
    assert(in);
    boost::archive::binary_iarchive ia(in);
    ia >> model;
  }

  bool LEARN = conf.count("learn");

  if (LEARN) {
    int dlc = 0;
    int dtoks = 0;
    if (conf.count("dev") == 0) {
      cerr << "You must specify a development set (--dev file.txt) with --learn" << endl;
      abort();
    } else {
      string devf = conf["dev"].as<string>();
      cerr << "Reading dev data from " << devf << " ...\n";
      ifstream in(devf);
      assert(in);
      while(getline(in, line)) {
        ++dlc;
        dev.push_back(ReadSentence(line, &d));
        dtoks += dev.back().size();
        if (dev.back().front() == kSOS || dev.back().back() == kEOS) {
          cerr << "Dev sentence in " << argv[2] << ":" << tlc << " started with <s> or ended with </s>\n";
          abort();
        }
      }
      cerr << dlc << " lines, " << dtoks << " tokens\n";
    }
    ostringstream os;
    os << "lm"
       << '_' << DROPOUT
       << '_' << LAYERS
       << '_' << INPUT_DIM
       << '_' << HIDDEN_DIM
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;
    unsigned report_every_i = 100;
    unsigned dev_every_i_reports = 25;
    unsigned si = training.size();
    if (report_every_i > si) report_every_i = si;
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    double lines = 0;
    int completed_epoch = -1;
    while(!INTERRUPTED) {
      if (SAMPLE) lm.RandomSample();
      Timer iteration("completed in");
      double loss = 0;
      unsigned chars = 0;
      for (unsigned i = 0; i < report_every_i; ++i) {
        if (si == training.size()) {
          si = 0;
          if (first) { first = false; } else { sgd->update_epoch(); }
          cerr << "**SHUFFLE\n";
          completed_epoch++;
          if (eta_decay_onset_epoch && completed_epoch >= (int)eta_decay_onset_epoch)
            sgd->eta *= eta_decay_rate;
          shuffle(order.begin(), order.end(), *rndeng);
        }

        // build graph for this instance
        ComputationGraph cg;
        auto& sent = training[order[si]];
        chars += sent.size();
        ++si;
        lm.BuildLMGraph(sent, cg, DROPOUT > 0.f);
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd->update();
        ++lines;
      }
      report++;
      cerr << '#' << report << " [epoch=" << (lines / training.size()) << " eta=" << sgd->eta << "] E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

      // show score on dev data?
      if (report % dev_every_i_reports == 0) {
        double dloss = 0;
        int dchars = 0;
        for (auto& sent : dev) {
          ComputationGraph cg;
          lm.BuildLMGraph(sent, cg, false);
          dloss += as_scalar(cg.forward());
          dchars += sent.size();
        }
        if (dloss < best) {
          best = dloss;
          ofstream out(fname);
          boost::archive::binary_oarchive oa(out);
          oa << model;
        }
        cerr << "\n***DEV [epoch=" << (lines / training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
      }
    }
  }  // train?
  if (conf.count("test")) {
    cerr << "Evaluating test data...\n";
    double tloss = 0;
    int tchars = 0;
    for (auto& sent : test) {
      ComputationGraph cg;
      lm.BuildLMGraph(sent, cg, false);
      tloss += as_scalar(cg.forward());
      tchars += sent.size();
    }
    cerr << "TEST                -LLH = " << tloss << endl;
    cerr << "TEST CROSS ENTOPY (NATS) = " << (tloss / tchars) << endl;
    cerr << "TEST                 PPL = " << exp(tloss / tchars) << endl;
  }

  // N-best scoring
  if (conf.count("nbest")) {
    // cdec: index ||| hypothesis ||| feature=val ... ||| ...
    // Moses: index ||| hypothesis ||| feature= val(s) ... ||| ...
    const int HYP_FIELD = 1;
    const int FEAT_FIELD = 2;
    const string FEAT_NAME = "RNNLM";
    // Input
    string nbestf = conf["nbest"].as<string>();
    cerr << "Scoring N-best list " << nbestf << " ..." << endl;
    shared_ptr<istream> in;
    if (nbestf == "-") {
      in.reset(&cin, [](...){});
    } else {
      in.reset(new ifstream(nbestf));
    }
    // Split on |||, consume whitespace
    boost::regex delim("\\s*\\|\\|\\|\\s*");
    boost::sregex_token_iterator end;
    // Match spacing of input file
    string sep = "=";
    bool sep_detected = false;
    // Input lines
    while (getline(*in, line)) {
      vector<string> fields;
      boost::sregex_token_iterator it(line.begin(), line.end(), delim, -1);
      while (it != end) {
        fields.push_back(*it++);
      }
      // Check sep if needed
      if (!sep_detected) {
        sep_detected = true;
        int i = fields[FEAT_FIELD].find("=");
        if (fields[FEAT_FIELD].substr(i + 1, 1) == " ") {
          sep = "= ";
        }
      }
      // Score hypothesis
      ComputationGraph cg;
      lm.BuildLMGraph(ReadSentence(fields[HYP_FIELD], &d), cg, false);
      double loss = as_scalar(cg.forward());
      // Add score
      ostringstream os;
      os << fields[FEAT_FIELD] << " " << FEAT_NAME << sep << loss;
      fields[FEAT_FIELD] = os.str();
      // Write augmented line
      cout << boost::algorithm::join(fields, " ||| ") << endl;
    }
  }

  delete sgd;
}
