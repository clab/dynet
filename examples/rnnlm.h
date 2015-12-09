#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/dict.h"
#include "cnn/lstm.h"

#include <vector>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;  //256
unsigned HIDDEN_DIM = 24;  // 1024
unsigned VOCAB_SIZE = 5500;

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
    kSOS = d.Convert("<s>");
    kEOS = d.Convert("</s>");
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

      // LogSoftmax followed by PickElement can be written in one step
      // using PickNegLogSoftmax
      Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
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
