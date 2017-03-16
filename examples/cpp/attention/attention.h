/*
 * attention.h
 *
 * \brief cpp implementation of attention.py in examples/python, learns to reproduce input sequence
 */

#ifndef EXAMPLES_CPP_ATTENTION_ATTENTION_H_
#define EXAMPLES_CPP_ATTENTION_ATTENTION_H_

#include "attention.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/lstm.h"
using namespace std;
using namespace dynet;

class attention {
public:
  /**
   * \brief Initializes vocabulary, builders and parameters
   *
   * \param model ParameterCollection holding the parameters
   */
  void initialize(ParameterCollection& model);

  /**
   * \brief computes loss for the network for a sample
   *
   * \param enc_fwd_lstm forward lstm
   * \param enc_bwd_lstm backward lstm
   * \param dec_lstm Decoder lstm
   */
  Expression get_loss(string input_sentence, string output_sentence, LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, LSTMBuilder& dec_lstm,
      ComputationGraph& cg);

  /**
   * \brief executes code to train the network
   *
   * \param model ParameterCollection holding the parameters
   * \param sentence Input sentences
   * \param trainer Trainer instance
   */
  void train(ParameterCollection& model, string sentence, SimpleSGDTrainer& trainer);

  /**
    * \brief generates the learnt sequence
    *
    * \param enc_fwd_lstm forward lstm
    * \param enc_bwd_lstm backward lstm
    * \param dec_lstm Decoder lstm
    * \param cg Computation graph
    */
  string generate(string in_seq, LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, LSTMBuilder& dec_lstm, ComputationGraph& cg);

  /**
    * \brief runs the decoder lstm with attention over the input encoded sequence and computes loss
    *
    * \param enc_fwd_lstm forward lstm
    * \param enc_bwd_lstm backward lstm
    * \param dec_lstm Decoder lstm
    * \param cg Computation graph
    */
  Expression decode(LSTMBuilder& dec_lstm, vector<Expression>& encoded, string output_sentence, ComputationGraph& cg);

  /**
    * \brief Computes attention values using input and the lstm states of input
    *
    * \param input_mat encoded input
    * \param state decoder lstm
    * \param w1dt input weighted by w1
    * \param cg Computation graph
    */
  Expression attend(Expression input_mat, LSTMBuilder& state, Expression w1dt, ComputationGraph& cg);

  /**
   * \brief encodes input sentence using bidirectional lstm
   *
   * \param enc_fwd_lstm forward lstm
   * \param enc_bwd_lstm backward lstm
   * \param embedded input character embedding
   */
  vector<Expression> encode_sentence(LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, vector<Expression>& embedded);

  /**
   * \brief runs lstm over the input embeddings
   *
   * \param init_state lstm instance
   * \param input_vecs input character embeddings
   */
  vector<Expression> run_lstm(LSTMBuilder& init_state, const vector<Expression>& input_vecs);

  /**
   * \brief constructs input character embedding
   *
   * \param sentence input sentence
   * \param cg computation graph instance
   */
  vector<Expression> embed_sentence(string sentence, ComputationGraph& cg);

private:
  static const int REP_SIZE = 32;
};

#endif /* EXAMPLES_CPP_ATTENTION_ATTENTION_H_ */
