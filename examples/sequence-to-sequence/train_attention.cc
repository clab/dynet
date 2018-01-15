/*
 * attention.cpp
 *
 *  cpp implementation of attention.py in examples/python
 */

#include "attention.h"
#include "dynet/training.h"
#include "dynet/lstm.h"
#include <map>
using namespace std;
using namespace dynet;
LookupParameter input_lookup;
LSTMBuilder enc_fwd_lstm;
LSTMBuilder enc_bwd_lstm;
LSTMBuilder dec_lstm;
map<string, int> char2int;
map<int, string> int2char;
string EOS = "<EOS>";
Parameter decoder_w;
Parameter decoder_b;
Parameter attention_w1;
Parameter attention_w2;
Parameter attention_v;
LookupParameter output_lookup;
unsigned STATE_SIZE = 0;

int main(int argc, char **argv) {
  dynet::initialize(argc, argv);
  attention attention_example;
  ParameterCollection model;
  SimpleSGDTrainer trainer(model);
  attention_example.initialize(model);
  attention_example.train(model, "it is working", trainer);
}

void attention::initialize(ParameterCollection& model) {
  string characters = "abcdefghijklmnopqrstuvwxyz ";
  vector<string> alphabets;
  for (auto c : characters) {
    alphabets.push_back(string(1, c));
  }
  alphabets.push_back(EOS);
  int index = 0;
  for (auto elem : alphabets) {
    char2int[elem] = index;
    int2char[index] = elem;
    index++;
  }
  unsigned VOCAB_SIZE = alphabets.size();
  unsigned LSTM_NUM_OF_LAYERS = 2;
  unsigned EMBEDDINGS_SIZE = REP_SIZE;
  STATE_SIZE = REP_SIZE;
  unsigned ATTENTION_SIZE = REP_SIZE;

  enc_fwd_lstm = LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model);
  enc_bwd_lstm = LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model);
  dec_lstm = LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + EMBEDDINGS_SIZE, STATE_SIZE, model);
  input_lookup = model.add_lookup_parameters(VOCAB_SIZE, { EMBEDDINGS_SIZE });
  output_lookup = model.add_lookup_parameters(VOCAB_SIZE, { EMBEDDINGS_SIZE });
  attention_w1 = model.add_parameters( { ATTENTION_SIZE, STATE_SIZE * 2 });
  attention_w2 = model.add_parameters( { ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 2 });
  attention_v = model.add_parameters( { 1, ATTENTION_SIZE });
  decoder_w = model.add_parameters( { VOCAB_SIZE, STATE_SIZE });
  decoder_b = model.add_parameters( { VOCAB_SIZE });
}

vector<Expression> attention::embed_sentence(string sentence, ComputationGraph& cg) {
  vector<Expression> output_exprs(sentence.size() + 2); //character encoding
  int index = 0;
  output_exprs.at(index) = lookup(cg, input_lookup, char2int[EOS]);
  index++;
  for (auto c : sentence) {
    output_exprs.at(index) = lookup(cg, input_lookup, char2int[string(1, c)]);
    index++;
  }
  output_exprs.at(index) = lookup(cg, input_lookup, char2int[EOS]);
  return output_exprs;
}

vector<Expression> attention::run_lstm(LSTMBuilder& init_state, const vector<Expression>& input_vecs) {
  LSTMBuilder& s = init_state;
  vector<Expression> out_vectors;
  vector<Expression>::const_iterator input_vecs_it;
  for (input_vecs_it = input_vecs.begin(); input_vecs_it != input_vecs.end(); input_vecs_it++) {
    s.add_input(*input_vecs_it);        //run lstm through the inputs
    out_vectors.push_back(s.back());
  }
  return out_vectors;
}

vector<Expression> attention::encode_sentence(LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, vector<Expression>& embedded) {
  vector<Expression> fwd_vectors = run_lstm(enc_fwd_lstm, embedded);    //forward lstm encoding
  vector<Expression> embedded_rev;
  for (vector<Expression>::reverse_iterator i = embedded.rbegin(); i != embedded.rend(); ++i) {
    embedded_rev.push_back(*i);
  }
  vector<Expression> bwd_vectors = run_lstm(enc_bwd_lstm, embedded_rev);    //backward lstm encoding
  reverse(bwd_vectors.begin(), bwd_vectors.end());
  vector<Expression> encoded;
  for (auto loop_index = 0U; loop_index < fwd_vectors.size(); loop_index++) {
    encoded.push_back(concatenate( { fwd_vectors.at(loop_index), bwd_vectors.at(loop_index) }));    //bi-lstm encoding
  }
  return encoded;
}

Expression attention::attend(Expression input_mat, LSTMBuilder& state, Expression w1dt, ComputationGraph& cg) {
  //att_weights=v∗tanh(encodedInput*w1+decoderstate*w2)
  Expression w2 = parameter(cg, attention_w2);
  Expression v = parameter(cg, attention_v);
  Expression w2dt = w2 * concatenate(state.final_s());
  Expression unnormalized = transpose(v * tanh(colwise_add(w1dt, w2dt)));
  Expression att_weights = softmax(unnormalized);
  Expression context = input_mat * att_weights;
  return context;
}

Expression attention::decode(LSTMBuilder& dec_lstm, vector<Expression>& encoded, string output_sentence, ComputationGraph& cg) {
  vector<string> output;
  output.push_back(EOS);
  for (auto c : output_sentence) {
    output.push_back(string(1, c));
  }
  output.push_back(EOS);

  vector<int> embeddings;
  for (auto c : output) {
    embeddings.push_back(char2int[c]);
  }

  Expression w = parameter(cg, decoder_w);
  Expression b = parameter(cg, decoder_b);
  Expression w1 = parameter(cg, attention_w1);

  Expression input_mat = concatenate_cols(encoded);

  Expression last_output_embeddings = lookup(cg, output_lookup, char2int[EOS]);
  Expression w1dt = w1 * input_mat;

  vector<dynet::real> x_values(STATE_SIZE * 2);
  dec_lstm.add_input(concatenate( { input(cg, { STATE_SIZE * 2 }, x_values), last_output_embeddings }));
  vector<Expression> loss;

  for (int c : embeddings) {
    Expression vector = concatenate( { attend(input_mat, dec_lstm, w1dt, cg), last_output_embeddings }); //concatenate input weighted by attention and decoder lstm state
    dec_lstm.add_input(vector);
    Expression out_vector = w * dec_lstm.back() + b;
    Expression probs = softmax(out_vector);
    last_output_embeddings = lookup(cg, output_lookup, c);
    loss.push_back(-log(pick(probs, c)));   //compare with gold output c and compute loss
  }
  return sum(loss);
}

string attention::generate(string in_seq, LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, LSTMBuilder& dec_lstm, ComputationGraph& cg) {
  vector<Expression> embedded = embed_sentence(in_seq, cg);
  vector<Expression> encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded);

  Expression w = parameter(cg, decoder_w);
  Expression b = parameter(cg, decoder_b);
  Expression w1 = parameter(cg, attention_w1);
  Expression input_mat = concatenate_cols(encoded);
  Expression w1dt = w1 * input_mat;

  Expression last_output_embeddings = lookup(cg, output_lookup, char2int[EOS]); //initialize with embedding for EOS
  vector<dynet::real> x_values(STATE_SIZE * 2);
  dec_lstm.new_graph(cg);       //initialize decoder lstm
  dec_lstm.start_new_sequence();
  dec_lstm.add_input(concatenate( { input(cg, { STATE_SIZE * 2 }, x_values), last_output_embeddings }));
  string out;

  int count_EOS = 0;
  for (auto loop_index = 0U; loop_index < in_seq.size() * 2; loop_index++) {
    if (count_EOS == 2) {       //if seen begin and end of sentence then exit loop
      break;
    }
    Expression vector = concatenate( { attend(input_mat, dec_lstm, w1dt, cg), last_output_embeddings });    //concatenate input weighted by attention and decoder lstm state
    dec_lstm.add_input(vector);
    Expression out_vector = w * dec_lstm.back() + b;
    std::vector<float> probs = as_vector(softmax(out_vector).value());
    int next_char = distance(probs.begin(), max_element(probs.begin(), probs.end()));   //predicted char
    last_output_embeddings = lookup(cg, output_lookup, next_char);
    if (int2char.at(next_char) == EOS) {
      count_EOS++;
      continue;
    }
    out = out + int2char.at(next_char);
  }
  return out;
}

Expression attention::get_loss(string input_sentence, string output_sentence, LSTMBuilder& enc_fwd_lstm, LSTMBuilder& enc_bwd_lstm, LSTMBuilder& dec_lstm,
    ComputationGraph& cg) {
  vector<Expression> embedded = embed_sentence(input_sentence, cg);
  vector<Expression> encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded);
  return decode(dec_lstm, encoded, output_sentence, cg);
}

void attention::train(ParameterCollection& model, string sentence, SimpleSGDTrainer& trainer) {
  for (int i = 0; i < 600; i++) {
    ComputationGraph cg;
    enc_fwd_lstm.new_graph(cg);
    enc_bwd_lstm.new_graph(cg);
    dec_lstm.new_graph(cg);
    enc_fwd_lstm.start_new_sequence();
    enc_bwd_lstm.start_new_sequence();
    dec_lstm.start_new_sequence();
    Expression loss = get_loss(sentence, sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, cg);
    float loss_value = as_scalar(cg.forward(loss)); //forward propagation
    cg.backward(loss);  //backward propagation
    trainer.update();   //update network weights
    if (i % 20 == 0) {
      cout << loss_value << endl;
      cout << generate(sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, cg) << endl; //generate output every 20 samples
    }
  }
}
