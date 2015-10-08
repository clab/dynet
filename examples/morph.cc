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
  Parameters *pW_ix, *pW_ih, *pW_ic, *pW_cx, *pW_ch, *pW_ox, *pW_oh, *pW_oc;
  Parameters *pb_i, *pb_f, *pb_c, *pb_o;
  int char_len, hidden_len;

 public:
  LSTM() {}

  void Init(const int& char_length, const int& hidden_length, Model *m) {
    char_len = char_length;
    hidden_len = hidden_length;

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
  }

  void InitNewCG(ComputationGraph* cg) {
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
  }

  void ComputeHC (const Expression& input, Expression* h, Expression* c) {
    Expression i = logistic(affine_transform({b_i, W_ix, input, W_ih, *h, W_ic, *c}));
    Expression f = 1.f - i;

    Expression temp = tanh(affine_transform({b_c, W_cx, input, W_ch, *h}));
    *c = cwise_multiply(f, *c) + cwise_multiply(i, temp);  // Update c

    Expression o = logistic(affine_transform({b_o, W_ox, input, W_oh, *h, W_oc, *c}));
    *h = cwise_multiply(o, tanh(*c));  // Update h
  }

  void GetAllHiddenUnits(const vector<Expression>& cols, const Expression& h_init,
                         const Expression& c_init, vector<Expression>* hidden) {
    Expression h = h_init, c = c_init;
    for (unsigned t = 0; t < cols.size(); ++t) {
      ComputeHC(cols[t], &h, &c);
      hidden->push_back(h);
    }
  }

  void GetLastHiddenUnit(const vector<Expression>& cols, const Expression& h_init,
                         const Expression& c_init, Expression* hidden) {
    Expression h = h_init, c = c_init;
    for (unsigned t = 0; t < cols.size(); ++t) {
      ComputeHC(cols[t], &h, &c);
    }
    *hidden = h;
  }
};

class Encoder : public LSTM {
 public:
  void EncodeInputIntoVector(const vector<Expression>& cols, Expression& h_init,
                             const Expression& c_init, Expression* hidden) {
    GetLastHiddenUnit(cols, h_init, c_init, hidden);
  }
};

class Decoder : public LSTM {
 public:
  void DecodeUntilTermFound(const Expression& encoded_word_vec, unsigned terminator,
                            const Expression& hidden_to_output,
                            LookupParameters* char_vecs,
                            unordered_map<string, unsigned>& char_to_id,
                            const Expression& h_init, const Expression& c_init,
                            vector<unsigned>* pred_target_ids,
                            ComputationGraph* cg) {
    Expression input_word_vec = lookup(*cg, char_vecs, char_to_id[BOW]);
    Expression h = h_init, c = c_init;
    while (true) {
      Expression input = concatenate({encoded_word_vec, input_word_vec});
      ComputeHC(input, &h, &c);
      Expression out = hidden_to_output * h;
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
      if (pred_index == terminator) {
        return;  // If the end is found, break from the loop and return
      }
      input_word_vec = lookup(*cg, char_vecs, pred_index);
    }
  }
};

Expression ComputeLoss(const vector<Expression>& hidden_units,
                       const vector<unsigned>& targets,
                       Expression& hidden_to_output) {
  assert(hidden_units.size() == targets.size());
  vector<Expression> losses;
  for (unsigned i = 0; i < hidden_units.size(); ++i) {
   Expression out = hidden_to_output * hidden_units[i];
   losses.push_back(pickneglogsoftmax(out, targets[i]));
  }
  return sum(losses);
}

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
  vector<float> ZERO(hidden_size, 0.0f);

  Model m;
  float regularization_strength = 1e-6f;
  AdadeltaTrainer sgd(&m, regularization_strength); 
 
  LookupParameters* char_vecs = m.add_lookup_parameters(vocab_size,
                                                        {char_size});
  Parameters* phidden_to_vocab = m.add_parameters({vocab_size, hidden_size});

  Encoder encoder;
  Decoder decoder;
  encoder.Init(char_size, hidden_size, &m);

  // The decoder takes encoded word vector with a character vector as input
  // at every step. Thus the input length is char_size + hidden_size
  decoder.Init(char_size + hidden_size, hidden_size, &m);

  // Read the training file and train the model
  for (unsigned iter = 0; iter < num_iter; ++iter) {
    ifstream train_file(train_filename);
    if (train_file.is_open()) {
      string line;
      double loss = 0;
      while (getline(train_file, line)) {
        chars = split_line(line, ' ');

        ComputationGraph cg;
        encoder.InitNewCG(&cg);
        decoder.InitNewCG(&cg);
        Expression hidden_to_vocab = parameter(cg, phidden_to_vocab);

        vector<Expression> input_vecs;
        vector<Expression> target_vecs;
        vector<unsigned> target_ids;
        input_vecs.clear(); target_vecs.clear(); target_ids.clear();
        bool reading_target = false;
        for (const string& ch : chars) {
          input_vecs.push_back(lookup(cg, char_vecs, char_to_id[ch]));
          if (reading_target) {
            target_vecs.push_back(lookup(cg, char_vecs, char_to_id[ch]));
            target_ids.push_back(char_to_id[ch]);
          }
          if (ch == EOW) {
            reading_target = true;
          }
        }

        // Read the input char vectors and obtain word representation
        Expression encoded_input_word_vec;
        Expression h = input(cg, {hidden_size}, &ZERO);
        encoder.EncodeInputIntoVector(input_vecs, h, h, &encoded_input_word_vec);

        // Use this encoded word vector to predict the transformed word
        // The input will not include '</s>'
        vector<Expression> input_target_vecs;
        for (unsigned i = 0; i < target_vecs.size() - 1; ++i) {
          input_target_vecs.push_back(concatenate({encoded_input_word_vec,
                                                   target_vecs[i]}));
        }

        // The output will not include '<s>'
        vector<unsigned> output_target_ids;
        for (unsigned i = 1; i < target_ids.size(); ++i) {
          output_target_ids.push_back(target_ids[i]);
        }

        vector<Expression> pred_target_vecs;
        decoder.GetAllHiddenUnits(input_target_vecs, h, h, &pred_target_vecs);
        Expression e = ComputeLoss(pred_target_vecs, output_target_ids,
                                   hidden_to_vocab);
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd.update(1.0f);
      }
      cerr << "Iter " << iter << " nllh: " << loss << endl;
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

      ComputationGraph cg;
      encoder.InitNewCG(&cg);
      decoder.InitNewCG(&cg);
      Expression hidden_to_vocab = parameter(cg, phidden_to_vocab);

      vector<Expression> input_vecs;
      vector<unsigned> pred_target_ids;
      input_vecs.clear(); pred_target_ids.clear();
      cerr << "Input: ";
      for (const string& ch : chars) {
        input_vecs.push_back(lookup(cg, char_vecs, char_to_id[ch]));
        cerr << ch;
      }
      cerr << endl;
      // Read the input char vectors and obtain word representation
      Expression encoded_input_word_vec;
      Expression h = input(cg, {hidden_size}, &ZERO);
      encoder.EncodeInputIntoVector(input_vecs, h, h, &encoded_input_word_vec);
 
      decoder.DecodeUntilTermFound(encoded_input_word_vec, char_to_id[EOW],
                                   hidden_to_vocab, char_vecs, char_to_id, h,
                                   h, &pred_target_ids, &cg);
      cerr << "Output: ";
      for (unsigned i = 0; i < pred_target_ids.size() - 1; ++i) {
        cerr << id_to_char[pred_target_ids[i]];
      }
      cerr << endl;
    }
  } else {
    cerr << "Test file opening failed" << endl;
  }
  return 1;
}
