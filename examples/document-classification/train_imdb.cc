/**
 * An implementation of [Document Modeling with Gated Recurrent Neural
 * Network for Sentiment Classification](http://aclweb.org/anthology/D15-1167)
 * using `pick_batch_elem`.
 * 
 * The model in use a bidirectional GRU to represents every sentence in the
 * document and feed the sentence representations into the another bi-GRU
 * to get the final representation of the document.
 *
 * This implementation compiles each i-th words in the sentences into a batch
 * (like that in the rnnlm-batch) and use `pick_batch_elem` to get last representation
 * in corresponding batch.
 *
 * On a small proportion of the IMDB data (2500 for training, 500 for dev.), this
 * model achieves 80% accuracy on two-way classification.
 */
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/gru.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"
#include "dynet/io.h"
#include "getpid.h"

#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;
using namespace dynet;

unsigned INPUT_DIM = 100;
unsigned SENTENCE_DIM = 200;
unsigned DOCUMENT_DIM = 400;
unsigned HIDDEN_DIM = 200;
unsigned VOCAB_SIZE = 0;
unsigned LABEL_SIZE = 0;
unsigned N_LAYERS = 1;

dynet::Dict d;
dynet::Dict ld;
int kSOS;
int kEOS;

typedef std::vector<std::vector<int>> Document;
typedef std::pair<Document, int> Instance;

struct DocumentModel {
  LookupParameter p_w;
  GRUBuilder fwd_sent_builder;
  GRUBuilder bwd_sent_builder;
  GRUBuilder fwd_doc_builder;
  GRUBuilder bwd_doc_builder;
  Parameter p_d2h;
  Parameter p_hbias;
  Parameter p_h2o;
  Parameter p_obias;

  DocumentModel(ParameterCollection & m) :
    p_w(m.add_lookup_parameters(VOCAB_SIZE, { INPUT_DIM })),
    fwd_sent_builder(N_LAYERS, INPUT_DIM, SENTENCE_DIM, m),
    bwd_sent_builder(N_LAYERS, INPUT_DIM, SENTENCE_DIM, m),
    fwd_doc_builder(N_LAYERS, SENTENCE_DIM * 2, DOCUMENT_DIM, m),
    bwd_doc_builder(N_LAYERS, SENTENCE_DIM * 2, DOCUMENT_DIM, m),
    p_d2h(m.add_parameters({ HIDDEN_DIM , DOCUMENT_DIM * 2 })),
    p_hbias(m.add_parameters({ HIDDEN_DIM })),
    p_h2o(m.add_parameters({ LABEL_SIZE, HIDDEN_DIM })),
    p_obias(m.add_parameters({ LABEL_SIZE })) {
  }

  Expression classify(ComputationGraph & cg, Instance & inst) {
    unsigned n_sents = inst.first.size();
    std::vector<Expression> sent_repr(n_sents);
    get_sentence_repr(cg, inst, sent_repr);
    Expression doc_repr = get_document_repr(cg, sent_repr);
    Expression logits = get_logits(cg, doc_repr);
    return logits;
  }
  
  Expression objective(ComputationGraph & cg, Instance & inst, Expression & logits) {
    return pickneglogsoftmax(logits, inst.second);
  }

  unsigned predict(ComputationGraph & cg, Instance & inst) {
    Expression logits = classify(cg, inst);
    std::vector<float> pred = dynet::as_vector(cg.get_value(logits));
    return std::max_element(pred.begin(), pred.end()) - pred.begin();
  }

  unsigned get_max_steps(const Instance & inst) {
    unsigned max_steps = 0;
    for (const auto & sentence : inst.first) {
      if (max_steps < sentence.size()) { max_steps = sentence.size(); }
    }
    return max_steps;
  }

  void get_sentence_repr(ComputationGraph & cg,
                         const Instance & inst, 
                         std::vector<Expression> & sent_repr) {
    unsigned max_steps = get_max_steps(inst);
    unsigned n_sents = inst.first.size();
    std::vector<unsigned> fwd_last(n_sents);
    std::vector<unsigned> bwd_last(n_sents);
    sent_repr.resize(n_sents);

    fwd_sent_builder.new_graph(cg);
    bwd_sent_builder.new_graph(cg);
    fwd_sent_builder.start_new_sequence();
    bwd_sent_builder.start_new_sequence();
    for (unsigned i = 0; i < max_steps; ++i) {
      for (unsigned b = 0; b < n_sents; ++b) {
        const std::vector<int> & sent = inst.first[b];
        fwd_last[b] = (i < sent.size() ? sent[i] : kEOS);
        bwd_last[b] = (i < sent.size() ? sent[sent.size() - 1 - i] : kSOS);
      }
      Expression fwd_i = lookup(cg, p_w, fwd_last);
      Expression bwd_i = lookup(cg, p_w, bwd_last);
      Expression fwd_output = fwd_sent_builder.add_input(fwd_i);
      Expression bwd_output = bwd_sent_builder.add_input(bwd_i);
      for (unsigned b = 0; b < n_sents; ++b) {
        const std::vector<int> & sent = inst.first[b];
        if (i + 1 == sent.size()) {
          sent_repr[b] = concatenate({ pick_batch_elem(fwd_output, b), pick_batch_elem(bwd_output, b) });
        }
      }
    }
  }

  Expression get_document_repr(ComputationGraph & cg,
                               std::vector<Expression> & sent_expr) {
    unsigned n_sents = sent_expr.size();
    fwd_doc_builder.new_graph(cg);
    bwd_doc_builder.new_graph(cg);
    fwd_doc_builder.start_new_sequence();
    bwd_doc_builder.start_new_sequence();
    for (unsigned i = 0; i < n_sents; ++i) {
      fwd_doc_builder.add_input(sent_expr[i]);
      bwd_doc_builder.add_input(sent_expr[n_sents - 1 - i]);
    }

    return concatenate({ fwd_doc_builder.back(), bwd_doc_builder.back() });
  }

  Expression get_logits(ComputationGraph & cg, Expression & doc_repr) {
    Expression h = rectify(parameter(cg, p_hbias) + parameter(cg, p_d2h) * doc_repr);
    Expression logits = parameter(cg, p_obias) + parameter(cg, p_h2o) * h;
    return logits;
  }
};

void read_one_line(const string & line, Instance & inst, Dict & wd, Dict & td) {
  istringstream in(line);
  string token;
  string label_doc_sep = "|||";
  string sent_sep = "|";
  std::vector<int> sentence;
  bool reading_label = true;
  while (in) {
    in >> token;
    if (!in) { break; }
    if (token == label_doc_sep) { reading_label = false; continue; }
    if (reading_label) {
      inst.second = td.convert(token);
    } else {
      if (token == sent_sep) { inst.first.push_back(sentence); sentence.clear(); }
      else { sentence.push_back(wd.convert(token)); }
    }
  }
  if (sentence.size() > 0) { inst.first.push_back(sentence); }
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.file]\n";
    return 1;
  }
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  vector<Instance> training, dev;
  string line;
  int tlc = 0;
  int tsents = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while (getline(in, line)) {
      ++tlc;
      Instance inst;
      read_one_line(line, inst, d, ld);
      training.push_back(inst);
      for (auto & sent : inst.first) { ttoks += sent.size(); }
      tsents += inst.first.size();
    }
    cerr << tlc << " lines, " << tsents << " sentences, " << ttoks << " tokens, " << d.size() << " types\n";
    cerr << "Labels: " << ld.size() << endl;
  }
  LABEL_SIZE = ld.size();
  d.freeze(); // no new word types allowed
  d.set_unk("_unk_");
  ld.freeze(); // no new tag types allowed

  int dlc = 0;
  int dsents = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while (getline(in, line)) {
      ++dlc;
      Instance inst;
      read_one_line(line, inst, d, ld);
      dev.push_back(inst);
      for (auto & sent : inst.first) { dtoks += sent.size(); }
      dsents += inst.first.size();
    }
    cerr << dlc << " lines, " << dsents << " sentences, " << dtoks << " tokens\n";
  }
  VOCAB_SIZE = d.size();
  ostringstream os;
  os << "imdb"
    << '_' << INPUT_DIM
    << '_' << SENTENCE_DIM
    << '_' << DOCUMENT_DIM
    << '_' << HIDDEN_DIM
    << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  ParameterCollection model;
  std::unique_ptr<Trainer> trainer(new AdamTrainer(model));

  DocumentModel engine(model);

  if (argc == 4) {
    TextFileLoader loader(argv[3]);
    loader.populate(model);
  }

  unsigned report_every_i = min(100, int(training.size()));
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  int report = 0;
  unsigned lines = 0;
  TextFileSaver saver("imdb.model");
  while (1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned ttags = 0;
    unsigned correct = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& inst = training[order[si]];
      ++si;
      //cerr << "LINE: " << order[si] << endl;
      Expression logits = engine.classify(cg, inst);
      std::vector<float> pred = dynet::as_vector(cg.get_value(logits));
      unsigned y_pred = std::max_element(pred.begin(), pred.end()) - pred.begin();
      if (y_pred == static_cast<unsigned>(inst.second)) { correct ++; }
      Expression loss_expr = engine.objective(cg, inst, logits);
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer->update();
      ++lines;
      ++ttags;
    }
    trainer->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / (double)ttags) << ") ";
    model.project_weights();

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dtags = 0;
      unsigned dcorr = 0;
      for (auto& inst : dev) {
        ComputationGraph cg;
        unsigned y_pred = engine.predict(cg, inst);
        if (y_pred == static_cast<unsigned>(inst.second)) dcorr++;
        dtags++;
      }
      if (dloss < best) {
        best = dloss;
        saver.save(model);
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / (double)dtags) << ' ';
    }
  }
}
