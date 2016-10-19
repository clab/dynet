#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "getpid.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

//parameters
unsigned LAYERS = 3;
unsigned INPUT_DIM = 500;
unsigned HIDDEN_DIM = 500;
unsigned INPUT_VOCAB_SIZE = 0;
unsigned OUTPUT_VOCAB_SIZE = 0;

dynet::Dict d, devd;
int kSOS;
int kEOS;

template <class Builder>
struct EncoderDecoder {
  LookupParameter p_c;
  LookupParameter p_ec;  // map input to embedding (used in fwd and rev models)
  Parameter p_ie2h;
  Parameter p_bie;
  Parameter p_h2oe;
  Parameter p_boe;
  Parameter p_R;
  Parameter p_bias;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;

  EncoderDecoder() {}

  explicit EncoderDecoder(Model& model) :
      dec_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {


    p_ie2h = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5), unsigned(HIDDEN_DIM * LAYERS * 2)});
    p_bie = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5)});
    p_h2oe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS), unsigned(HIDDEN_DIM * LAYERS * 1.5)});
    p_boe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS)});
    p_c = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
    p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({OUTPUT_VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({OUTPUT_VOCAB_SIZE});
  }

  // build graph and return Expression for total loss
  Expression BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
    // forward encoder
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    for (unsigned t = 0; t < insent.size(); ++t) {
    	Expression i_x_t = lookup(cg,p_ec,insent[t]);
      fwd_enc_builder.add_input(i_x_t);
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int t = insent.size() - 1; t >= 0; --t) {
      Expression i_x_t = lookup(cg, p_ec, insent[t]);
      rev_enc_builder.add_input(i_x_t);
    }
    
    // encoder -> decoder transformation
    vector<Expression> to;
    for (auto h_l : fwd_enc_builder.final_h()) to.push_back(h_l);
    for (auto h_l : rev_enc_builder.final_h()) to.push_back(h_l);
    
    Expression i_combined = concatenate(to);
    Expression i_ie2h = parameter(cg, p_ie2h);
    Expression i_bie = parameter(cg, p_bie);
    Expression i_t = i_bie + i_ie2h * i_combined;

    Expression i_h = rectify(i_t);
    Expression i_h2oe = parameter(cg,p_h2oe);
    Expression i_boe = parameter(cg,p_boe);
    Expression i_nc = i_boe + i_h2oe * i_h;
    
    vector<Expression> oein1, oein2, oein;
    for (unsigned i = 0; i < LAYERS; ++i) {
      oein1.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
      oein2.push_back(tanh(oein1[i]));
    }
    for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein1[i]);
    for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein2[i]);

    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(oein);

    // decoder
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    vector<Expression> errs;

    const unsigned oslen = osent.size() - 1;
    for (unsigned t = 0; t < oslen; ++t) {
    	Expression i_x_t = lookup(cg, p_c, osent[t]);
    	Expression i_y_t = dec_builder.add_input(i_x_t);
    	Expression i_r_t = i_bias + i_R * i_y_t;
    	Expression i_ydist = log_softmax(i_r_t);
    	errs.push_back(pick(i_ydist,osent[t+1]));
    }
    Expression i_nerr = sum(errs);
    return -i_nerr;
  }

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & p_c & p_ec;
    ar & p_ie2h & p_bie & p_h2oe & p_boe & p_R & p_bias;
    ar & dec_builder & rev_enc_builder & fwd_enc_builder;
  }
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
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
      training.push_back(read_sentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
	cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.freeze(); // no new word types allowed
  INPUT_VOCAB_SIZE = d.size();
  OUTPUT_VOCAB_SIZE = d.size();
  
  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      dev.push_back(read_sentence(line, &devd));
      dtoks += dev.back().size();
      if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
	cerr << "Dev sentence in " << argv[2] << ":" << dlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  
  ostringstream os;
  os << "bilm"
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
  //EncoderDecoder<SimpleRNNBuilder> lm(model);
  EncoderDecoder<LSTMBuilder> lm;
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }
  else {
    lm = EncoderDecoder<LSTMBuilder>(model);
  }
  
  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 10;
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
        ComputationGraph cg;
        auto& sent = training[order[si]];
        chars += sent.size() - 1;
        ++si;
        Expression loss_expr = lm.BuildGraph(sent, sent, cg);
        //cg.print_graphviz();
        loss += as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        sgd->update();
        ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
    
#if 0
    lm.RandomSample();
#endif
    
    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (auto& sent : dev) {
	ComputationGraph cg;
	Expression loss_expr = lm.BuildGraph(sent, sent, cg);
	dloss += as_scalar(cg.forward(loss_expr));
	dchars += sent.size() - 1;
      }
      if (dloss < best) {
	best = dloss;
	ofstream out(fname);
	boost::archive::text_oarchive oa(out);
	oa << model << lm;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
  }
  delete sgd;
}
