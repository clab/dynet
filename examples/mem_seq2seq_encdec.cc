#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/rnnem.h"

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef WIN32
#include <process.h>
#endif
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

/** this implements the following model
@inproceedings{Seq2Seq,
title = {Recurrent neural networks with external memory},
author  = {K. Yao, B. Peng and C. Dyer},
year  = 2016,
}
*/

//parameters
unsigned LAYERS = 3;
unsigned INPUT_DIM = 500;
unsigned HIDDEN_DIM = 500;
unsigned VOCAB_SIZE_SRC = 0;
unsigned VOCAB_SIZE_TGT = 0;

cnn::Dict sd, td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;

typedef vector<int> Sentence;     
typedef pair<Sentence, Sentence> SentencePair;  
typedef vector<SentencePair> Corpus;  

Corpus read_corpus(const string &filename)
{
  ifstream in(filename);
  assert(in);
  Corpus corpus;
  string line;
  int lc = 0, stoks = 0, ttoks = 0;
  while(getline(in, line)) {
    ++lc;
    Sentence source, target;
    ReadSentencePair(line, &source, &sd, &target, &td);
    corpus.push_back(SentencePair(source, target));
    stoks += source.size();
    ttoks += target.size();

    if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
	(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
      cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
      abort();
    }
  }
  cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
  return corpus;
}


template <class Builder>
struct EncoderDecoder {
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_R;
  Parameters* p_bias;
  Builder dec_builder;
  Builder enc_builder;

  explicit EncoderDecoder(Model& model,
			  long layers,
			  long vocab_size_src, 
			  long vocab_size_tgt,
			  long input_dim,
			  long hidden_dim) :
      dec_builder(layers, input_dim, hidden_dim, &model),
      enc_builder(layers, input_dim, hidden_dim, &model) {


    p_c = model.add_lookup_parameters(vocab_size_src, {input_dim}); 
    p_ec = model.add_lookup_parameters(vocab_size_src, {input_dim}); 
    p_R = model.add_parameters({vocab_size_tgt, hidden_dim});
    p_bias = model.add_parameters({vocab_size_tgt});
  }

  // build graph and return Expression for total loss
  Expression BuildGraph(const Sentence& insent, const Sentence& osent, ComputationGraph& cg) {
    // forward encoder
    enc_builder.new_graph(cg);
    
    enc_builder.start_new_sequence();
    for (int t = insent.size() - 1; t >= 0; --t) {
      Expression i_x_t = lookup(cg, p_ec, insent[t]);
      enc_builder.add_input(i_x_t);
    }
    
    cg.incremental_forward();

    // encoder -> decoder transformation
    vector<Expression> i_combined;
    for (auto s_l : enc_builder.final_s()) i_combined.push_back(s_l);
    

    //    cerr << "dim= " << i_combined.size() << endl;
    
    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(i_combined);

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
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSRC_SOS = sd.Convert("<s>");
  kSRC_EOS = sd.Convert("</s>");
  kTGT_SOS = td.Convert("<s>");
  kTGT_EOS = td.Convert("</s>");

  vector<SentencePair> training, dev;
  string line;
  cerr << "Reading training data from " << argv[1] << "...\n";

  ifstream in(argv[1]);
  training = read_corpus(argv[1]);
 
  VOCAB_SIZE_SRC = sd.size();
  VOCAB_SIZE_TGT = td.size();

  cerr << "vocab size src " << VOCAB_SIZE_SRC << "vocab size targt " << VOCAB_SIZE_TGT << endl;

  sd.Freeze(); // no new word types allowed
  td.Freeze();



  cerr << "Reading dev data from " << argv[2] << "...\n"; 
  dev = read_corpus(argv[2]); 

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
//  EncoderDecoder<LSTMBuilder> lm(model, LAYERS, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, INPUT_DIM, HIDDEN_DIM);
  EncoderDecoder<RNNEMBuilder> lm(model, LAYERS, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, INPUT_DIM, HIDDEN_DIM);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }
  
  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 2;
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
      Sentence src_sent = sent.first;
      Sentence tgt_sent = sent.second;

      chars += tgt_sent.size() - 1;
      ++si;
      lm.BuildGraph(src_sent, tgt_sent, cg);
      //cg.PrintGraphviz();
      loss += as_scalar(cg.forward());
      cg.backward();
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
	Sentence src_sent = sent.first;
	Sentence tgt_sent = sent.second;

	lm.BuildGraph(src_sent, tgt_sent, cg); 
	dloss += as_scalar(cg.forward());
	dchars += tgt_sent.size() - 1;
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
