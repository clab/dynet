#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/cfsm-builder.h"
#include "dynet/hsm-builder.h"
#include "dynet/globals.h"
#include "dynet/io.h"
#include "getpid.h"
#include "cl-args.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>

using namespace std;
using namespace dynet;

unsigned LAYERS = 0;
unsigned INPUT_DIM = 0;
unsigned HIDDEN_DIM = 0;
unsigned VOCAB_SIZE = 0;
float DROPOUT = 0;
bool SAMPLE = false;
SoftmaxBuilder* cfsm = nullptr;

dynet::Dict d;
int kSOS;
int kEOS;

volatile bool INTERRUPTED = false;


template <class Builder>
struct RNNLanguageModel {
  LookupParameter p_c;
  Parameter p_R;
  Parameter p_bias;
  Builder builder;
  explicit RNNLanguageModel(ParameterCollection& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model) {
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
    while (len < max_len) {
      if (cfsm) { // class-factored softmax
        cur = cfsm->sample(h_t);
      } else { // regular softmax
        Expression u_t = affine_transform({bias, R, h_t});
        Expression dist_expr = softmax(u_t);
        auto dist = as_vector(cg.incremental_forward(dist_expr));
        double p = rand01();
        cur = 0;
        for (; static_cast<unsigned>(cur) < dist.size(); ++cur) {
          p -= dist[cur];
          if (p < 0.0) { break; }
        }
        if (static_cast<unsigned>(cur) == dist.size()) cur = kEOS;
      }
      if (cur == kEOS) break;
      ++len;
      cerr << (len == 1 ? "" : " ") << d.convert(cur);
      Expression x_t = lookup(cg, p_c, cur);
      h_t = builder.add_input(x_t);
    }
    cerr << endl;
  }
};


void inline read_fields(string line, vector<string>& fields, string delimiter = "|||") {
  string field;
  int start = 0, end = 0, delim_size = delimiter.size();
  while (true) {
    end = line.find(delimiter, start);
    fields.push_back(line.substr(start, end - start));
    if (end == (int)std::string::npos) break;
    start = end + delim_size;
  }
}

// Read the dataset, returns the number of tokens
unsigned read_data(const string& filename,
                   vector<vector<int>>& data) {
  unsigned num_tokens = 0;
  ifstream in(filename);
  assert(in);
  size_t lc = 0;
  string line;
  while (getline(in, line)) {
    ++lc;
    data.push_back(read_sentence(line, d));
    num_tokens += data.back().size();
    if (data.back().front() == kSOS || data.back().back() == kEOS) {
      cerr << "sentence in " << filename << ":" << lc << " started with <s> or ended with </s>\n";
      abort();
    }
  }
  return num_tokens;
}

int main(int argc, char** argv) {
  cerr << "COMMAND LINE:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;

  // Fetch dynet params ----------------------------------------------------------------------------
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);

  // Fetch program specific parameters (see ../utils/cl-args.h) ------------------------------------
  Params params;

  get_args(argc, argv, params, TRAIN);

  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  LAYERS = params.LAYERS;
  INPUT_DIM = params.INPUT_DIM;
  HIDDEN_DIM = params.HIDDEN_DIM;
  SAMPLE = params.sample;
  if (params.dropout_rate)
    DROPOUT = params.dropout_rate;
  ParameterCollection model;
  if (params.clusters_file != "")
    cfsm = new ClassFactoredSoftmaxBuilder(HIDDEN_DIM, params.clusters_file, d, model);
  else if (params.paths_file != "")
    cfsm = new HierarchicalSoftmaxBuilder(HIDDEN_DIM, params.paths_file, d, model);
  float eta_decay_rate = params.eta_decay_rate;
  unsigned eta_decay_onset_epoch = params.eta_decay_onset_epoch;
  vector<vector<int>> training, dev, test;
  string line;
  int tlc = 0;
  int ttoks = 0;
  {
    string trainf = params.train_file;
    cerr << "Reading training data from " << trainf << " ...\n";
    ttoks = read_data(trainf, training);
    tlc = training.size();
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.freeze(); // no new word types allowed
  d.set_unk("<unk>");
  VOCAB_SIZE = d.size();

  if (params.test_file != "") {
    string testf = params.test_file;
    cerr << "Reading test data from " << testf << " ...\n";
    read_data(testf, test);
  }

  std::unique_ptr<Trainer> trainer(new SimpleSGDTrainer(model));
  trainer->learning_rate = params.eta0;
  RNNLanguageModel<LSTMBuilder> lm(model);

  bool has_model_to_load = params.model_file != "";
  if (has_model_to_load) {
    string fname = params.model_file;
    cerr << "Reading parameters from " << fname << "...\n";
    TextFileLoader loader(fname);
    loader.populate(model);
  }

  bool TRAIN = (params.train_file != "");

  if (TRAIN) {
    int dlc = 0;
    int dtoks = 0;
    if (params.dev_file == "") {
      cerr << "You must specify a development set (--dev file.txt) with --train" << endl;
      abort();
    } else {
      string devf = params.dev_file;
      cerr << "Reading dev data from " << devf << " ...\n";
      dtoks = read_data(devf, dev);
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
    int report = 0;
    double lines = 0;
    int completed_epoch = -1;
    while (!INTERRUPTED) {
      if (SAMPLE) lm.RandomSample();
      Timer iteration("completed in");
      double loss = 0;
      unsigned chars = 0;
      for (unsigned i = 0; i < report_every_i; ++i) {
        if (si == training.size()) {
          si = 0;
          cerr << "**SHUFFLE\n";
          completed_epoch++;
          if (eta_decay_onset_epoch && completed_epoch >= (int)eta_decay_onset_epoch)
            trainer->learning_rate *= eta_decay_rate;
          shuffle(order.begin(), order.end(), *rndeng);
        }

        // build graph for this instance
        ComputationGraph cg;
        auto& sent = training[order[si]];
        chars += sent.size();
        ++si;
        Expression loss_expr = lm.BuildLMGraph(sent, cg, DROPOUT > 0.f);
        loss += as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        trainer->update();
        ++lines;
      }
      report++;
      cerr << '#' << report << " [epoch=" << (lines / training.size()) << " lr=" << trainer->learning_rate << "] E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

      // show score on dev data?
      if (report % dev_every_i_reports == 0) {
        double dloss = 0;
        int dchars = 0;
        for (auto& sent : dev) {
          ComputationGraph cg;
          Expression loss_expr = lm.BuildLMGraph(sent, cg, false);
          dloss += as_scalar(cg.forward(loss_expr));
          dchars += sent.size();
        }
        if (dloss < best) {
          best = dloss;
          TextFileSaver saver(fname);
          saver.save(model);
        }
        cerr << "\n***DEV [epoch=" << (lines / training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
      }
    }
  }  // train?
  if (params.test_file != "") {
    cerr << "Evaluating test data...\n";
    double tloss = 0;
    int tchars = 0;
    for (auto& sent : test) {
      ComputationGraph cg;
      Expression loss_expr = lm.BuildLMGraph(sent, cg, false);
      tloss += as_scalar(cg.forward(loss_expr));
      tchars += sent.size();
    }
    cerr << "TEST                -LLH = " << tloss << endl;
    cerr << "TEST CROSS ENTOPY (NATS) = " << (tloss / tchars) << endl;
    cerr << "TEST                 PPL = " << exp(tloss / tchars) << endl;
  }

  // N-best scoring
  if (params.nbest_file != "") {
    // cdec: index ||| hypothesis ||| feature=val ... ||| ...
    // Moses: index ||| hypothesis ||| feature= val(s) ... ||| ...
    const int HYP_FIELD = 1;
    const int FEAT_FIELD = 2;
    const string FEAT_NAME = "RNNLM";
    // Input
    string nbestf = params.nbest_file;
    cerr << "Scoring N-best list " << nbestf << " ..." << endl;
    shared_ptr<istream> in;
    if (nbestf == "-") {
      in.reset(&cin, [](...) {});
    } else {
      in.reset(new ifstream(nbestf));
    }
    // Match spacing of input file
    string sep = "=";
    bool sep_detected = false;
    // Input lines
    while (getline(*in, line)) {
      vector<string> fields;
      read_fields(line, fields);
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
      Expression loss_expr = lm.BuildLMGraph(read_sentence(fields[HYP_FIELD], d), cg, false);
      double loss = as_scalar(cg.forward(loss_expr));
      // Add score
      ostringstream os;
      os << fields[FEAT_FIELD] << " " << FEAT_NAME << sep << loss;
      fields[FEAT_FIELD] = os.str();
      // Write augmented line
      for (unsigned f = 0; f < fields.size(); ++f) {
        if (f > 0)
          cout << " ||| ";
        cout << fields[f];
      }
      cout << endl;
    }
  }
}
