/**
 * \file rnnlm-batch.h
 * \defgroup lmbuilders lmbuilders
 * \brief Language models builders
 *
 * An example implementation of a simple neural language model
 * based on RNNs
 *
 */
#ifndef RNNLM_BATCH_H
#define RNNLM_BATCH_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"


#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int kSOS;
int kEOS;


unsigned INPUT_VOCAB_SIZE;
unsigned OUTPUT_VOCAB_SIZE;

/**
 * \ingroup lmbuilders
 *
 * \struct RNNBatchLanguageModel
 * \brief This structure wraps any RNN to train a language model with minibatching
 * \details Recurrent neural network based language modelling maximizes the likelihood
 * of a sentence \f$\textbf s=(w_1,\dots,w_n)\f$ by modelling it as :
 *
 * \f$L(\textbf s)=p(w_1,\dots,w_n)=\prod_{i=1}^n p(w_i\vert w_1,\dots,w_{i-1})\f$
 *
 * Where \f$p(w_i\vert w_1,\dots,w_{i-1})\f$ is given by the output of the RNN at step \f$i\f$
 *
 * In the case of training with minibatching, the sentences must be of the same length in
 * each minibatch. This requires some preprocessing (see `train_rnnlm-batch.cc` for example).
 * 
 * Reference : [Mikolov et al., 2010](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
 *
 * \tparam Builder This can be any RNNBuilder
 */
template <class Builder>
struct RNNBatchLanguageModel {

protected:
  // Hyper-parameters
  unsigned LAYERS = 2;
  unsigned INPUT_DIM = 8;  //256
  unsigned HIDDEN_DIM = 24;  // 1024
  unsigned VOCAB_SIZE = 0;
  bool cust_l2;

  LookupParameter p_c;
  Parameter p_R;
  Parameter p_bias;
  Expression i_c;
  Expression i_R;
  Expression i_bias;
  Builder rnn;

public:
  /**
   * \brief Constructor for the batched RNN language model
   *
   * \param model Model to hold all parameters for training
   * \param LAYERS Number of layers of the RNN
   * \param INPUT_DIM Embedding dimension for the words
   * \param HIDDEN_DIM Dimension of the hidden states
   * \param VOCAB_SIZE Size of the input vocabulary
   */
  explicit RNNBatchLanguageModel(Model& model,
                                 unsigned LAYERS,
                                 unsigned INPUT_DIM,
                                 unsigned HIDDEN_DIM,
                                 unsigned VOCAB_SIZE) :
    LAYERS(LAYERS), INPUT_DIM(INPUT_DIM),
    HIDDEN_DIM(HIDDEN_DIM), VOCAB_SIZE(VOCAB_SIZE),
    rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, model)  {
    // Add embedding parameters to the model
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  }

  /**
   * \brief Computes the negative log probability on a batch
   *
   * \param sents Full training set
   * \param id Start index of the batch
   * \param bsize Batch size (`id` + `bsize` should be smaller than the size of the dataset)
   * \param tokens Number of tokens processed by the model (used for loos per token computation)
   * \param cg Computation graph
   * \return Expression for $\f$\sum_{s\in\mathrm{batch}}\log(p(s))\f$
   */
  Expression getNegLogProb(const vector<vector<int> >& sents,
                           unsigned id,
                           unsigned bsize,
                           unsigned & tokens,
                           ComputationGraph& cg) {
    const unsigned slen = sents[id].size();
    // Initialize the RNN for a new computation graph
    rnn.new_graph(cg);
    // Prepare for new sequence (essentially set hidden states to 0)
    rnn.start_new_sequence();
    // Instantiate embedding parameters in the computation graph
    // output -> word rep parameters (matrix + bias)
    i_R = parameter(cg, p_R);
    i_bias = parameter(cg, p_bias);
    // Initialize variables for batch errors
    vector<Expression> errs;
    // Set all inputs to the SOS symbol
    vector<unsigned> last_arr(bsize, sents[0][0]), next_arr(bsize);
    // Run rnn on batch
    for (unsigned t = 1; t < slen; ++t) {
      // Fill next_arr (tokens to be predicted)
      for (unsigned i = 0; i < bsize; ++i) {
        next_arr[i] = sents[id + i][t];
        // count non-EOS tokens
        if (next_arr[i] != *sents[id].rbegin()) tokens++;
      }
      // Embed the current tokens
      Expression i_x_t = lookup(cg, p_c, last_arr);
      // Run one step of the rnn : y_t = RNN(x_t)
      Expression i_y_t = rnn.add_input(i_x_t);
      // Project to the token space using an affine transform
      Expression i_r_t = i_bias + i_R * i_y_t;
      // Compute error for each member of the batch
      Expression i_err = pickneglogsoftmax(i_r_t, next_arr);
      errs.push_back(i_err);
      // Change input
      last_arr = next_arr;
    }
    // Add all errors
    Expression i_nerr = sum_batches(sum(errs));
    return i_nerr;
  }

  /**
   * \brief Samples a string of words/characters from the model
   * \details This can be used to debug and/or have fun. Try it on
   * new datasets!
   *
   * \param d Dictionary to use (should be same as the one used for training)
   * \param max_len maximu number of tokens to generate
   * \param temp Temperature for sampling (the softmax computed is
   * \f$\frac{e^{\frac{r_t^{(i)}}{T}}}{\sum_{j=1}^{\vert V\vert}e^{\frac{r_t^{(j)}}{T}}}\f$).
   *  Intuitively lower temperature -> less deviation from the distribution (= more "standard" samples)
   */
  void RandomSample(const dynet::Dict& d, int max_len = 150, float temp = 1.0) {
    // Make some space
    cerr << endl;
    // Initialize computation graph
    ComputationGraph cg;
    // Initialize the RNN for the new computation graph
    rnn.new_graph(cg);
    // Initialize for new sequence (set hidden states, etc..)
    rnn.start_new_sequence();
    // Instantiate embedding parameters in the computation graph
    // output -> word rep parameters (matrix + bias)
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);

    // Start generating
    int len = 0;
    int cur = kSOS;
    while (len < max_len) {
      ++len;
      // Embed current token
      Expression i_x_t = lookup(cg, p_c, cur);
      // Run one step of the rnn
      // y_t = RNN(x_t)
      Expression i_y_t = rnn.add_input(i_x_t);
      // Project into token space
      Expression i_r_t = i_bias + i_R * i_y_t;
      // Get distribution over tokens (with temperature)
      Expression ydist = softmax(i_r_t / temp);

      // Sample token
      unsigned w = 0;
      while (w == 0 || (int)w == kSOS) {
        auto dist = as_vector(cg.incremental_forward(ydist));
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        if (w == dist.size()) w = kEOS;
      }

      if (w == kEOS) {
        // If the sampled token is an EOS, reinitialize network and start generating a new sample
        rnn.start_new_sequence();
        cerr << endl;
        cur = kSOS;
      } else {
        // Otherwise print token and continue
        cerr << (cur == kSOS ? "" : " ") << d.convert(w);
        cur = w;
      }

    }
    cerr << endl;
  }

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int) {
    ar & LAYERS & INPUT_DIM & HIDDEN_DIM;
    ar & p_c & p_R & p_bias;
    ar & rnn;
  }
};

#endif
