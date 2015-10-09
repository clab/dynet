#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";

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

class LSTM {

 protected:
  /* The parameters of the model */
  Expression W_ix, W_ih, W_ic, W_cx, W_ch,
             W_ox, W_oh, W_oc;  // Weight matrices
  Expression b_i, b_f, b_c, b_o;  // Bias vectors

  // Project hidden layer for predicting words (only used by decoder)
  Expression hidden_to_output, hidden_to_output_bias;

  Parameters *pW_ix, *pW_ih, *pW_ic, *pW_cx, *pW_ch, *pW_ox, *pW_oh, *pW_oc;
  Parameters *pb_i, *pb_f, *pb_c, *pb_o;
  Parameters *phidden_to_output, *phidden_to_output_bias;

  vector<float> ZERO;
  Expression h_init;

 public:
  int char_len, hidden_len, vocab_len;

  LSTM() {}

  void Init(const int& char_length, const int& hidden_length,
            const int& vocab_length, Model *m) {
    char_len = char_length;
    hidden_len = hidden_length;
    vocab_len = vocab_length;

    pW_ix = m->add_parameters({hidden_len, char_len});
    pW_ih = m->add_parameters({hidden_len, hidden_len});
    pW_ic = m->add_parameters({hidden_len, hidden_len});

    pW_cx = m->add_parameters({hidden_len, char_len});
    pW_ch = m->add_parameters({hidden_len, hidden_len});

    pW_ox = m->add_parameters({hidden_len, char_len});
    pW_oh = m->add_parameters({hidden_len, hidden_len});
    pW_oc = m->add_parameters({hidden_len, hidden_len});

    pb_i = m->add_parameters({hidden_len, 1});
    pb_f = m->add_parameters({hidden_len, 1});
    pb_c = m->add_parameters({hidden_len, 1});
    pb_o = m->add_parameters({hidden_len, 1});

    phidden_to_output = m->add_parameters({vocab_len, hidden_len});
    phidden_to_output_bias = m->add_parameters({vocab_len, 1});

    for (int i = 0; i < hidden_len; ++i) {
      ZERO.push_back(0.);
    }
  }

  void AddParamsToCG(ComputationGraph* cg) {
    W_ix = parameter(*cg, pW_ix);
    W_ih = parameter(*cg, pW_ih);
    W_ic = parameter(*cg, pW_ic);

    W_cx = parameter(*cg, pW_cx);
    W_ch = parameter(*cg, pW_ch);

    W_ox = parameter(*cg, pW_ox);
    W_oh = parameter(*cg, pW_oh);
    W_oc = parameter(*cg, pW_oc);

    b_i = parameter(*cg, pb_i);
    b_f = parameter(*cg, pb_f);
    b_c = parameter(*cg, pb_c);
    b_o = parameter(*cg, pb_o);

    hidden_to_output = parameter(*cg, phidden_to_output);
    hidden_to_output_bias = parameter(*cg, phidden_to_output_bias);

    h_init = input(*cg, {hidden_len}, &ZERO);
  }

  void ComputeHC (const Expression& input, Expression* h, Expression* c) const {
    Expression i = logistic(affine_transform({b_i, W_ix, input, W_ih, *h, W_ic, *c}));
    Expression f = 1.0f - i;

    Expression temp = tanh(affine_transform({b_c, W_cx, input, W_ch, *h}));
    *c = cwise_multiply(f, *c) + cwise_multiply(i, temp);  // Update c

    Expression o = logistic(affine_transform({b_o, W_ox, input, W_oh, *h, W_oc, *c}));
    *h = cwise_multiply(o, tanh(*c));  // Update h
  }

  void GetAllHiddenUnits(const vector<Expression>& cols,
                         vector<Expression>* hidden) const {
    Expression h = h_init, c = h_init;
    for (unsigned t = 0; t < cols.size(); ++t) {
      ComputeHC(cols[t], &h, &c);
      hidden->push_back(h);
    }
  }

  void GetLastHiddenUnit(const vector<Expression>& cols, Expression* hidden)
                         const {
    Expression h = h_init, c = h_init;
    for (unsigned t = 0; t < cols.size(); ++t) {
      ComputeHC(cols[t], &h, &c);
    }
    *hidden = h;
  }
};

class Encoder : public LSTM {
 public:
  void EncodeInputIntoVector(const vector<Expression>& cols,
                             Expression* hidden) const {
    GetLastHiddenUnit(cols, hidden);
  }
};

class Decoder : public LSTM {

 public:
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

  void DecodeUntilTermFound(const Expression& encoded_word_vec,
                            LookupParameters* char_vecs,
                            unordered_map<string, unsigned>& char_to_id,
                            vector<unsigned>* pred_target_ids,
                            ComputationGraph* cg) const {
    Expression input_word_vec = lookup(*cg, char_vecs, char_to_id[BOW]);
    Expression h = h_init, c = h_init;
    while (true) {
      Expression input = concatenate({encoded_word_vec, input_word_vec});
      ComputeHC(input, &h, &c);
      Expression out;
      ProjectToVocab(h, &out);
      vector<float> dist = as_vector(cg->incremental_forward());
      unsigned pred_index = 0;
      float best_score = dist[pred_index];
      for (unsigned index = 1; index < dist.size(); ++index) {
        if (dist[index] > best_score) {
          best_score = dist[index];
          pred_index = index; 
        }
      }
      if (pred_index == char_to_id[EOW]) {
        return;  // If the end is found, break from the loop and return
      }
      pred_target_ids->push_back(pred_index);
      input_word_vec = lookup(*cg, char_vecs, pred_index);
    }
  }
};

class MorphTrans {
 public:
  Encoder encoder;
  Decoder decoder;
  LookupParameters* char_vecs;
  Expression transform_encoded, transform_encoded_bias;
  Parameters *ptransform_encoded, *ptransform_encoded_bias;

  MorphTrans(const int& char_length, const int& hidden_length,
             const int& vocab_length, Model *m) {
    encoder.Init(char_length, hidden_length, vocab_length, m);
    decoder.Init(char_length + hidden_length, hidden_length, vocab_length, m);

    char_vecs = m->add_lookup_parameters(vocab_length, {char_length});
    ptransform_encoded = m->add_parameters({hidden_length, hidden_length});
    ptransform_encoded_bias = m->add_parameters({hidden_length, 1});
  }

  void AddParamsToCG(ComputationGraph* cg) {
    encoder.AddParamsToCG(cg);
    decoder.AddParamsToCG(cg);
    transform_encoded = parameter(*cg, ptransform_encoded);
    transform_encoded_bias = parameter(*cg, ptransform_encoded_bias);
  }

  void TransformEncodedInputForDecoding(Expression* encoded_input) const {
    return;
    *encoded_input = affine_transform(
        {transform_encoded_bias, transform_encoded, *encoded_input});
  }

  void EncodeInputAndTransform(const vector<unsigned>& inputs,
                               Expression* encoded_input_vec,
                               ComputationGraph* cg) const {
    vector<Expression> input_vecs;
    for (const unsigned& input_id : inputs) {
      input_vecs.push_back(lookup(*cg, char_vecs, input_id));
    }

    // Running the encoder now.
    encoder.EncodeInputIntoVector(input_vecs, encoded_input_vec);

    // Transform it to feed it into the decoder
    TransformEncodedInputForDecoding(encoded_input_vec);
  }

  float Train(const vector<unsigned>& inputs, const vector<unsigned>& outputs,
              Model *m, AdadeltaTrainer* ada_gd) {
    ComputationGraph cg;
    AddParamsToCG(&cg);

    // Encode and Transofrm to feed into decoder
    Expression encoded_input_vec;
    EncodeInputAndTransform(inputs, &encoded_input_vec, &cg);

    // Use this encoded word vector to predict the transformed word
    vector<Expression> input_vecs_for_dec;
    vector<unsigned> output_ids_for_pred;
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (i < outputs.size() - 1) { 
        // '</s>' will not be fed as input -- it needs to be predicted.
        input_vecs_for_dec.push_back(concatenate(
            {encoded_input_vec, lookup(cg, char_vecs, outputs[i])}));
      }
      if (i > 0) {  // '<s>' will not be predicted in the output -- its fed in.
        output_ids_for_pred.push_back(outputs[i]);
      }
    }

    vector<Expression> dec_hidden_units;
    decoder.GetAllHiddenUnits(input_vecs_for_dec, &dec_hidden_units);
    Expression loss = decoder.ComputeLoss(dec_hidden_units, output_ids_for_pred);

    float return_loss = as_scalar(cg.forward());
    cg.backward();
    ada_gd->update(1.0f);
    return return_loss;
  }

  void Predict(vector<unsigned>& inputs,
               unordered_map<string, unsigned>& char_to_id,
               vector<unsigned>* outputs) {
    ComputationGraph cg;
    AddParamsToCG(&cg);

    // Encode and Transofrm to feed into decoder
    Expression encoded_input_vec;
    EncodeInputAndTransform(inputs, &encoded_input_vec, &cg);

    // Make preditions using the decoder.
    decoder.DecodeUntilTermFound(encoded_input_vec, char_vecs, char_to_id,
                                 outputs, &cg);
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // Read the vocab file
  ifstream vocab_file(argv[1]);
  vector<string> chars;
  if (vocab_file.is_open()) {
    string line;
    getline(vocab_file, line);
    chars = split_line(line, ' ');
  } else {
    cerr << "File opening failed" << endl;
  }
  unsigned vocab_size = chars.size();
  unordered_map<string, unsigned> char_to_id;
  unordered_map<unsigned, string> id_to_char;
  unsigned num_chars = 0;
  for (const string& ch : chars) {
    char_to_id[ch] = num_chars;
    id_to_char[num_chars] = ch;
    num_chars++;
  }

  string train_filename = argv[2];  // train file
  string test_filename = argv[3];
  unsigned char_size = atoi(argv[4]);
  unsigned hidden_size = atoi(argv[5]);
  unsigned num_iter = atoi(argv[6]);

  Model m;
  float regularization_strength = 1e-6f;
  AdadeltaTrainer ada_gd(&m, regularization_strength);
  MorphTrans nn(char_size, hidden_size, vocab_size, &m);

  // Read the training file and train the model
  for (unsigned iter = 0; iter < num_iter; ++iter) {
    ifstream train_file(train_filename);
    if (train_file.is_open()) {
      string line;
      float loss = 0;
      while (getline(train_file, line)) {
        chars = split_line(line, ' ');
        vector<unsigned> input_ids, target_ids;
        input_ids.clear(); target_ids.clear();
        bool reading_target = false;
        for (const string& ch : chars) {
          input_ids.push_back(char_to_id[ch]);
          if (reading_target) {
            target_ids.push_back(char_to_id[ch]);
          }
          if (ch == EOW) {
            reading_target = true;
          }
        }
        loss += nn.Train(input_ids, target_ids, &m, &ada_gd);
      }
      cerr << "Iter " << iter + 1 << " nllh: " << loss << endl;
      train_file.close();
    } else {
      cerr << "Failed to open the file" << endl;
    }
  }

  // Read the test file and output predictions for the words.
  ifstream test_file(test_filename);
  if (test_file.is_open()) {
    string line;
    while (getline(test_file, line)) {
      chars = split_line(line, ' ');
      vector<unsigned> input_ids, target_ids;
      vector<unsigned> pred_target_ids;
      input_ids.clear(); pred_target_ids.clear();
      cerr << "Input: " << line;
      for (const string& ch : chars) {
        input_ids.push_back(char_to_id[ch]);
      }
      // Read the input char vectors and obtain word representation
      nn.Predict(input_ids, char_to_id, &pred_target_ids);
      cerr << "Output: ";
      for (unsigned i = 0; i < pred_target_ids.size(); ++i) {
        cerr << id_to_char[pred_target_ids[i]];
      }
      cerr << endl;
    }
  } else {
    cerr << "Test file opening failed" << endl;
  }
  return 1;
}
