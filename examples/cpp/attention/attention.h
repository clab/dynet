/*
 * attention.h
 *
 *  cpp implementation of attention.py in examples/python
 */

#ifndef EXAMPLES_CPP_ATTENTION_ATTENTION_H_
#define EXAMPLES_CPP_ATTENTION_ATTENTION_H_

#include "attention.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/lstm.h"
#include <map>
using namespace std;
using namespace dynet;

class attention{
public:
	void initialize(Model& model);
	Expression get_loss(string input_sentence, string output_sentence, LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, LSTMBuilder& dec_lstm, ComputationGraph& cg);
	void train(Model& model, string sentence, SimpleSGDTrainer& trainer);
	string generate( string in_seq, LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, LSTMBuilder& dec_lstm, ComputationGraph& cg);
	Expression decode(LSTMBuilder& dec_lstm, vector<Expression>& encoded, string output_sentence, ComputationGraph& cg);
	Expression attend(Expression input_mat, LSTMBuilder& state, Expression w1dt, ComputationGraph& cg);
	vector<Expression> encode_sentence(LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, vector<Expression>& embedded);
	vector<Expression> run_lstm(LSTMBuilder& init_state, const vector<Expression>&  input_vecs);
	vector<Expression> embed_sentence(string sentence, ComputationGraph& cg);

private:
	static const int REP_SIZE = 32;
};


#endif /* EXAMPLES_CPP_ATTENTION_ATTENTION_H_ */
