#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"
#include "cnn/cnn-helper.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;

long LAYERS = 2;
long INPUT_DIM = 8;  //256
long HIDDEN_DIM = 24;  // 1024
long VOCAB_SIZE = 0;

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
      
      // we can easily look at intermidiate values
//      std::vector<float> r_t = as_vector(i_r_t.value());
  //    for (float f : r_t) cout << f << " "; cout << endl;
    //  cout << "[" << as_scalar(pick(i_r_t, sent[t+1]).value()) << "]" << endl;

      // LogSoftmax followed by PickElement can be written in one step
      // using PickNegLogSoftmax
#if 0
      Expression i_ydist = logsoftmax(i_r_t);
      errs.push_back(pick(i_ydist, sent[t+1]));
#if 0
      Expression i_ydist = softmax(i_r_t);
      i_ydist = log(i_ydist)
      errs.push_back(pick(i_ydist, sent[t+1]));
#endif
#else
      Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
      errs.push_back(i_err);
#endif
    }
    Expression i_nerr = sum(errs);
#if 0
    return -i_nerr;
#else
    return i_nerr;
#endif
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

template <class LM_t>
void train(Model &model, LM_t &lm,
    const vector<vector<int>>& training,
    const vector<vector<int>>& dev,
    Trainer *sgd, const string& fname,
    bool randomSample)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500;
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    while (1) {
        Timer iteration("completed in");
        double loss = 0;
        unsigned chars = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd->update_epoch(); }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto& sent = training[order[si]];
            chars += sent.size() - 1;
            ++si;
            lm.BuildLMGraph(sent, cg);
            loss += as_scalar(cg.forward());
            cg.backward();
            sgd->update();
            ++lines;
        }
        sgd->status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

        if (randomSample)
            lm.RandomSample();

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (auto& sent : dev) {
                ComputationGraph cg;
                lm.BuildLMGraph(sent, cg);
                dloss += as_scalar(cg.forward());
                dchars += sent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            else{
                sgd->eta *= 0.5;
            }
            cerr << "\n***TEST E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
}

template <class LM_t>
void testcorpus(Model &model, LM_t &lm,
    const vector<vector<int>>& dev)
{
    unsigned lines = 0;
    double dloss = 0;
    int dchars = 0;
    for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildLMGraph(sent, cg);
        dloss += as_scalar(cg.forward());
        dchars += sent.size() - 1;
    }

    cerr << "\n***DEV [epoch=" << (lines / (double)dev.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> model;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // command line processing
  using namespace boost::program_options;
  variables_map vm;
  options_description opts("Allowed options");
  opts.add_options()
      ("help", "print help message")
      ("seed,s", value<int>()->default_value(217), "random seed number")
      ("train,t", value<string>(), "file containing training sentences")
      ("devel,d", value<string>(), "file containing development sentences.")
      ("test,T", value<string>(), "file containing testing source sentences")
      ("initialise,i", value<string>(), "load initial parameters from file")
      ("parameters,p", value<string>(), "save best parameters to this file")
      ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
      ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
      ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
      ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
      ("dglstm", "use depth-gated LSTM for recurrent structure; default RNN")
      ("verbose,v", "be extremely chatty")
      ("generate,g", value<bool>()->default_value(false), "generate random samples")
      ;
  store(parse_command_line(argc, argv, opts), vm);

  string flavour;
  if (vm.count("gru"))	flavour = "gru";
  else if (vm.count("lstm"))	flavour = "lstm";
  else if (vm.count("rnnem"))	flavour = "rnnem";
  else if (vm.count("dglstm")) flavour = "dglstm";
  else			flavour = "rnn";


  LAYERS = vm["layers"].as<int>();
  HIDDEN_DIM = vm["hidden"].as<int>();

  bool generateSample = false;
  generateSample = vm["generate"].as<bool>();

  string fname;
  if (vm.count("parameters")) {
      fname = vm["parameters"].as<string>();
  }
  else {
      ostringstream os;
      os << "lm"
          << '_' << LAYERS
          << '_' << HIDDEN_DIM
          << '_' << flavour
          << "-pid" << getpid() << ".params";
      fname = os.str();
  }

  cerr << "Parameters will be written to: " << fname << endl;

  if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && vm.count("test") != 1)) {
      cout << opts << "\n";
      return 1;
  }

  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<vector<int>> training, dev, test;
  string line;
  int tlc = 0;
  int ttoks = 0;

  string infile = vm["train"].as<string>();
  cerr << "Reading training data from " << infile << "...\n";

  {
    ifstream in(infile);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      training.push_back(ReadSentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
        cerr << "Training sentence in " << infile << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();

  if (vm.count("devel") > 0)
  {
      int dlc = 0;
      int dtoks = 0;
      string devfile = vm["devel"].as<string>();
      cerr << "Reading training data from " << devfile << "...\n";
      {
          ifstream in(devfile);
          assert(in);
          while (getline(in, line)) {
              ++dlc;
              dev.push_back(ReadSentence(line, &d));
              dtoks += dev.back().size();
              if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
                  cerr << "Dev sentence in " << devfile << ":" << tlc << " didn't start or end with <s>, </s>\n";
                  abort();
              }
          }
          cerr << dlc << " lines, " << dtoks << " tokens\n";
      }
  }

  Model model;
  bool use_momentum = false;
  Trainer* sgd = nullptr;
  //if (use_momentum)
  //  sgd = new MomentumSGDTrainer(&model);
  //else
  sgd = new SimpleSGDTrainer(&model);

  if (vm.count("test") == 0)
  {
      if (vm.count("lstm")) {
          cerr << "%% Using LSTM recurrent units" << endl;
          RNNLanguageModel<LSTMBuilder> lm(model);
          train(model, lm, training, dev, sgd, fname, generateSample);
      }
      else if (vm.count("dglstm")) {
          cerr << "%% Using DGLSTM recurrent units" << endl;
          RNNLanguageModel<DGLSTMBuilder> lm(model);
          train(model, lm, training, dev, sgd, fname, generateSample);
      }
  }
  else
  {
      string testfile = vm["test"].as<string>();
      int dlc = 0;
      int dtoks = 0;
      cerr << "Reading training data from " << testfile << "...\n";
      {
          ifstream in(testfile);
          assert(in);
          while (getline(in, line)) {
              ++dlc;
              test.push_back(ReadSentence(line, &d));
              dtoks += test.back().size();
              if (test.back().front() != kSOS && test.back().back() != kEOS) {
                  cerr << "Dev sentence in " << testfile << ":" << tlc << " didn't start or end with <s>, </s>\n";
                  abort();
              }
          }
          cerr << dlc << " lines, " << dtoks << " tokens\n";
      }

      if (vm.count("test"))
      {
          if (vm.count("lstm")){
              cerr << "%% using LSTM recurrent units" << endl;
              RNNLanguageModel<LSTMBuilder> lm(model);
              if (vm.count("initialise"))
                  initialise(model, vm["initialise"].as<string>());
              testcorpus(model, lm, test);
          }
          if (vm.count("dglstm")){
              cerr << "%% using LSTM recurrent units" << endl;
              RNNLanguageModel<DGLSTMBuilder> lm(model);
              if (vm.count("initialise"))
                  initialise(model, vm["initialise"].as<string>());
              testcorpus(model, lm, test);
          }
      }
  }

  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  delete sgd;
}

