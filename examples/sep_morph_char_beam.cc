#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <limits>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";
unsigned MAX_PRED_LEN = 100;
float NEG_INF = numeric_limits<int>::min();

vector<string> split_line(const string& line, char delim) {
  vector<string> words;
  stringstream ss(line);
  string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty())
      words.push_back(item);
  }
  return words;
}

template <class Builder>
class MorphTrans {
 public:
  Builder input_forward, input_backward, output_forward;
  LookupParameters* char_vecs;

  Expression hidden_to_output, hidden_to_output_bias;
  Parameters *phidden_to_output, *phidden_to_output_bias;

  Expression transform_encoded, transform_encoded_bias;
  Parameters *ptransform_encoded, *ptransform_encoded_bias;
  
  unsigned char_len;
  Expression EPS;
  Parameters *peps_vec;

  MorphTrans(const int& char_length, const int& hidden_length,
             const int& vocab_length, const int& layers, Model *m) :
           input_forward(layers, char_length, hidden_length, m),
           input_backward(layers, char_length, hidden_length, m),
           output_forward(layers, 2 * char_length + hidden_length, hidden_length, m) {
    char_len = char_length;
    char_vecs = m->add_lookup_parameters(vocab_length, {char_length});

    phidden_to_output = m->add_parameters({vocab_length, hidden_length});
    phidden_to_output_bias = m->add_parameters({vocab_length, 1});

    ptransform_encoded = m->add_parameters({hidden_length, 2 * hidden_length});
    ptransform_encoded_bias = m->add_parameters({hidden_length, 1});
    
    peps_vec = m->add_parameters({char_len, 1});
  }

  void AddParamsToCG(ComputationGraph* cg) {
    input_forward.new_graph(*cg);
    input_backward.new_graph(*cg);
    output_forward.new_graph(*cg);

    hidden_to_output = parameter(*cg, phidden_to_output);
    hidden_to_output_bias = parameter(*cg, phidden_to_output_bias);

    transform_encoded = parameter(*cg, ptransform_encoded);
    transform_encoded_bias = parameter(*cg, ptransform_encoded_bias);
    
    EPS = parameter(*cg, peps_vec);
  }

  void RunFwdBwd(const vector<unsigned>& inputs,
                 Expression* hidden, ComputationGraph *cg) {
    vector<Expression> input_vecs;
    for (const unsigned& input_id : inputs) {
      input_vecs.push_back(lookup(*cg, char_vecs, input_id));
    }

    // Run forward LSTM
    Expression forward_unit;
    input_forward.start_new_sequence();
    for (unsigned i = 0; i < input_vecs.size(); ++i) {
      forward_unit = input_forward.add_input(input_vecs[i]);
    }

    // Run backward LSTM
    reverse(input_vecs.begin(), input_vecs.end());
    Expression backward_unit;
    input_backward.start_new_sequence();
    for (unsigned i = 0; i < input_vecs.size(); ++i) {
      backward_unit = input_backward.add_input(input_vecs[i]);
    }

    // Concatenate the forward and back hidden layers
    *hidden = concatenate({forward_unit, backward_unit});
  }

  void TransformEncodedInputForDecoding(Expression* encoded_input) const {
    *encoded_input = affine_transform({transform_encoded_bias,
                                       transform_encoded, *encoded_input});
  }

  void ProjectToVocab(const Expression& hidden, Expression* out) const {
    *out = affine_transform({hidden_to_output_bias, hidden_to_output, hidden});
  }

  Expression ComputeLoss(const vector<Expression>& hidden_units,
                         const vector<unsigned>& targets) const {
    assert(hidden_units.size() == targets.size());
    vector<Expression> losses;
    for (unsigned i = 0; i < hidden_units.size(); ++i) {
      Expression out;
      ProjectToVocab(hidden_units[i], &out);
      losses.push_back(pickneglogsoftmax(out, targets[i]));
    }
    return sum(losses);
  }

  float Train(const vector<unsigned>& inputs, const vector<unsigned>& outputs,
              AdadeltaTrainer* ada_gd) {
    ComputationGraph cg;
    AddParamsToCG(&cg);

    // Encode and Transform to feed into decoder
    Expression encoded_input_vec;
    RunFwdBwd(inputs, &encoded_input_vec, &cg);
    TransformEncodedInputForDecoding(&encoded_input_vec);

    // Use this encoded word vector to predict the transformed word
    vector<Expression> input_vecs_for_dec;
    vector<unsigned> output_ids_for_pred;
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (i < outputs.size() - 1) { 
        // '</s>' will not be fed as input -- it needs to be predicted.
        if (i < inputs.size() - 1) {
          input_vecs_for_dec.push_back(concatenate(
              {encoded_input_vec, lookup(cg, char_vecs, outputs[i]),
               lookup(cg, char_vecs, inputs[i+1])}));
        } else {
          input_vecs_for_dec.push_back(concatenate(
              {encoded_input_vec, lookup(cg, char_vecs, outputs[i]), EPS}));
        }
      }
      if (i > 0) {  // '<s>' will not be predicted in the output -- its fed in.
        output_ids_for_pred.push_back(outputs[i]);
      }
    }

    vector<Expression> decoder_hidden_units;
    output_forward.start_new_sequence();
    for (const auto& vec : input_vecs_for_dec) {
      decoder_hidden_units.push_back(output_forward.add_input(vec));
    }
    Expression loss = ComputeLoss(decoder_hidden_units, output_ids_for_pred);

    float return_loss = as_scalar(cg.forward());
    cg.backward();
    ada_gd->update(1.0f);
    return return_loss;
  }

  void Decode(const Expression& encoded_word_vec,
              unordered_map<string, unsigned>& char_to_id,
              vector<unsigned>* pred_target_ids,
              const vector<unsigned> input_ids, ComputationGraph* cg) {
    Expression input_word_vec = lookup(*cg, char_vecs, char_to_id[BOW]);
    pred_target_ids->push_back(char_to_id[BOW]);
    output_forward.start_new_sequence();
    unsigned out_index = 1;
    while (pred_target_ids->size() < MAX_PRED_LEN) {
      Expression input;
      if (out_index < input_ids.size()) {
        input = concatenate({encoded_word_vec, input_word_vec,
                            lookup(*cg, char_vecs, input_ids[out_index])});
      } else {
        input = concatenate({encoded_word_vec, input_word_vec, EPS});
      }
      Expression hidden = output_forward.add_input(input);
      Expression out;
      ProjectToVocab(hidden, &out);
      vector<float> dist = as_vector(cg->incremental_forward());
      unsigned pred_index = 0;
      float best_score = dist[pred_index];
      for (unsigned index = 1; index < dist.size(); ++index) {
        if (dist[index] > best_score) {
          best_score = dist[index];
          pred_index = index;
        }
      }
      pred_target_ids->push_back(pred_index);
      if (pred_index == char_to_id[EOW]) {
        return;  // If the end is found, break from the loop and return
      }
      input_word_vec = lookup(*cg, char_vecs, pred_index);
      out_index++;
    }
  }

  void BeamDecode(const Expression& encoded_word_vec,
                  unordered_map<string, unsigned>& char_to_id,
                  vector<unsigned>* pred_target_ids,
                  const vector<unsigned>& input_ids, const unsigned beam_size,
                  ComputationGraph* cg) {
    Expression prev_output_vec = lookup(*cg, char_vecs, char_to_id[BOW]);
    output_forward.start_new_sequence();
    unsigned out_index = 1;

    // Enter the start symbol in the decoder.
    Expression input = concatenate({encoded_word_vec, prev_output_vec,
                                    lookup(*cg, char_vecs, input_ids[out_index])});
    Expression hidden = output_forward.add_input(input);
    Expression out;
    ProjectToVocab(hidden, &out);
    out = log_softmax(out);
    vector<float> log_dist = as_vector(cg->incremental_forward());
    priority_queue<pair<float, unsigned> > init_queue;
    for (unsigned i = 0; i < log_dist.size(); ++i) {
      init_queue.push(make_pair(log_dist[i], i));
    }
    unsigned vocab_size = log_dist.size();

    // Initialise the beam_size sequences, scores, hidden states.
    vector<vector<unsigned> > sequences;
    vector<float> log_scores;
    vector<RNNPointer> prev_states;
    for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
      vector<unsigned> seq;
      seq.push_back(char_to_id[BOW]);
      seq.push_back(init_queue.top().second);
      sequences.push_back(seq);

      log_scores.push_back(init_queue.top().first);
      prev_states.push_back(output_forward.state());
      init_queue.pop();
    }

    vector<cnn::real> neg_inf(vocab_size, NEG_INF);
    Expression neg_inf_vec = cnn::expr::input(*cg, {vocab_size}, &neg_inf);
 
    vector<bool> active_beams(beam_size, true);
    while (true) {
      out_index++;
      Expression input_char_vec;
      if (out_index < input_ids.size()) {
        input_char_vec = lookup(*cg, char_vecs, input_ids[out_index]);
      } else {
        input_char_vec = EPS;
      }
      
      priority_queue<pair<float, pair<unsigned, unsigned> > > probs_queue;
      unordered_map<unsigned, RNNPointer> curr_states;
      vector<Expression> out_dist;
      for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
        if (active_beams[beam_id]) {
          unsigned prev_out_char = sequences[beam_id].back();
          Expression prev_out_vec = lookup(*cg, char_vecs, prev_out_char);
          Expression input = concatenate({encoded_word_vec, prev_out_vec,
                                          input_char_vec});
          hidden = output_forward.add_input(prev_states[beam_id], input);
          curr_states[beam_id] = output_forward.state();

          ProjectToVocab(hidden, &out);
          out_dist.push_back(log_softmax(out));
        } else {
          out_dist.push_back(neg_inf_vec);
        }
      }
      Expression all_scores = concatenate(out_dist);
      vector<float> log_dist = as_vector(cg->incremental_forward());

      for (unsigned index = 0; index < log_dist.size(); ++index) {
        unsigned beam_id = index / vocab_size;
        unsigned char_id = index % vocab_size;
        if (active_beams[beam_id]) {
          pair<unsigned, unsigned> location = make_pair(beam_id, char_id);
          probs_queue.push(pair<float, pair<unsigned, unsigned> >(
                           log_scores[beam_id] + log_dist[index], location));
        }
      }
      
      // Find the beam_size best now and update the variables.
      unordered_map<unsigned, vector<unsigned> > new_seq;
      for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
        if (active_beams[beam_id]) {
          float log_prob = probs_queue.top().first;
          pair<unsigned, unsigned> location = probs_queue.top().second;
          unsigned old_beam_id = location.first, char_id = location.second;

          vector<unsigned> seq = sequences[old_beam_id];
          seq.push_back(char_id);
          new_seq[beam_id] = seq;
          log_scores[beam_id] = log_prob;  // Update the score
          prev_states[beam_id] = curr_states[old_beam_id];  // Update hidden state

          probs_queue.pop();
        }
      }
      
      // Update the sequences now.
      for (auto& it : new_seq) {
        sequences[it.first] = it.second;
      }

      // Check if a sequence should be made inactive.
      for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
        if (active_beams[beam_id] && 
            (sequences[beam_id].back() == char_to_id[EOW] ||
            sequences[beam_id].size() > MAX_PRED_LEN)) {
          active_beams[beam_id] = false;
        }
      }

      // Check if all sequences are inactive.
      /*bool all_inactive = true;
      for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
        if (active_beams[beam_id]) {
          all_inactive = false;
          break;
        }
      }*/

      // When the sequence with highest score has become inactive, return.
      unsigned max_beam_index = distance(log_scores.begin(),
                                         max_element(log_scores.begin(),
                                                     log_scores.end()));
      if (!active_beams[max_beam_index]) {
        *pred_target_ids = sequences[max_beam_index];
        return;
      }
    }
  }

  void Predict(const vector<unsigned>& inputs,
               unordered_map<string, unsigned>& char_to_id,
               const unsigned beam_size,
               vector<unsigned>* outputs) {
    ComputationGraph cg;
    AddParamsToCG(&cg);

    // Encode and Transofrm to feed into decoder
    Expression encoded_input_vec;
    RunFwdBwd(inputs, &encoded_input_vec, &cg);
    TransformEncodedInputForDecoding(&encoded_input_vec);

    // Make preditions using the decoder.
    BeamDecode(encoded_input_vec, char_to_id, outputs, inputs, beam_size, &cg);
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  string vocab_filename = argv[1];  // vocabulary of words/characters
  string morph_filename = argv[2];
  string train_filename = argv[3];
  string test_filename = argv[4];
  unsigned char_size = atoi(argv[5]);
  unsigned hidden_size = atoi(argv[6]);
  unsigned num_iter = atoi(argv[7]);
  float reg_strength = atof(argv[8]);
  unsigned layers = atoi(argv[9]);
  unsigned beam_size = atoi(argv[10]);

  ifstream vocab_file(vocab_filename);
  vector<string> chars;
  if (vocab_file.is_open()) {  // Reading the vocab file
    string line;
    getline(vocab_file, line);
    chars = split_line(line, ' ');
  } else {
    cerr << "File opening failed" << endl;
  }
  unordered_map<string, unsigned> char_to_id;
  unordered_map<unsigned, string> id_to_char;
  unsigned char_id = 0;
  for (const string& ch : chars) {
    char_to_id[ch] = char_id;
    id_to_char[char_id] = ch;
    char_id++;
  }
  unsigned vocab_size = char_to_id.size();

  ifstream morph_file(morph_filename);
  vector<string> morph_attrs;
  if (morph_file.is_open()) {  // Reading the vocab file
    string line;
    getline(morph_file, line);
    morph_attrs = split_line(line, ' ');
  } else {
    cerr << "File opening failed" << endl;
  }
  unordered_map<string, unsigned> morph_to_id;
  unordered_map<unsigned, string> id_to_morph;
  unsigned morph_id = 0;
  for (const string& ch : morph_attrs) {
    morph_to_id[ch] = morph_id;
    id_to_morph[morph_id] = ch;
    morph_id++;
  }
  unsigned morph_size = morph_to_id.size();

  vector<Model*> m;
  vector<AdadeltaTrainer> optimizer;
  vector<MorphTrans<LSTMBuilder> > nn;
  for (unsigned i = 0; i < morph_size; ++i) {
    m.push_back(new Model());
    AdadeltaTrainer ada(m[i], reg_strength);
    optimizer.push_back(ada);
    MorphTrans<LSTMBuilder> neural(char_size, hidden_size, vocab_size, layers, m[i]);
    nn.push_back(neural);
  }

  // Read the training file in a vector
  vector<string> train_data;
  ifstream train_file(train_filename);
  if (train_file.is_open()) {
    string line;
    while (getline(train_file, line)) {
      train_data.push_back(line);
    }
  }
  train_file.close();

  // Read the test file in a vector
  vector<string> test_data;
  ifstream test_file(test_filename);
  if (test_file.is_open()) {
    string line;
    while (getline(test_file, line)) {
      test_data.push_back(line);
    }
  }
  test_file.close();

  // Read the training file and train the model
  for (unsigned iter = 0; iter < num_iter; ++iter) {
    unsigned line_id = 0;
    random_shuffle(train_data.begin(), train_data.end());
    vector<float> loss(morph_size, 0.0f);
    for (string& line : train_data) {
      vector<string> items = split_line(line, '|');
      vector<unsigned> input_ids, target_ids;
      input_ids.clear(); target_ids.clear();
      for (const string& ch : split_line(items[0], ' ')) {
        input_ids.push_back(char_to_id[ch]);
      }
      for (const string& ch : split_line(items[1], ' ')) {
        target_ids.push_back(char_to_id[ch]);
      }
      unsigned morph_id = morph_to_id[items[2]];
      loss[morph_id] += nn[morph_id].Train(input_ids, target_ids, &optimizer[morph_id]);
      cerr << ++line_id << "\r";
    }

    cerr << "Iter " << iter + 1 << " ";
    for (unsigned i = 0; i < loss.size(); ++i) {
      cerr << loss[i] << " ";
    }
    cerr << "Sum: " << accumulate(loss.begin(), loss.end(), 0.) << endl;

    // Read the test file and output predictions for the words.
    string line;
    double correct = 0, total = 0;
    for (string& line : test_data) {
      vector<string> items = split_line(line, '|');
      vector<unsigned> input_ids, target_ids, pred_target_ids;
      input_ids.clear(); target_ids.clear(); pred_target_ids.clear();
      for (const string& ch : split_line(items[0], ' ')) {
        input_ids.push_back(char_to_id[ch]);
      }
      for (const string& ch : split_line(items[1], ' ')) {
        target_ids.push_back(char_to_id[ch]);
      }
      unsigned morph_id = morph_to_id[items[2]];
      nn[morph_id].Predict(input_ids, char_to_id, beam_size, &pred_target_ids);

      string prediction = "";
      for (unsigned i = 0; i < pred_target_ids.size(); ++i) {
        prediction += id_to_char[pred_target_ids[i]];
        if (i != pred_target_ids.size() - 1) {
          prediction += " ";
        }
      }
      if (prediction == items[1]) {
        correct += 1;
      } else {  // If wrong, print prediction and correct answer
        if (iter == num_iter - 1) {
          cout << items[0] << '|' << items[1] << '|' << items[2] << endl;
          cout << items[0] << '|' << prediction << '|' << items[2] << endl;
        }
      }
      total += 1;
    }
    cerr << "Prediction Accuracy: " << correct / total << endl;
  }
  return 1;
}
