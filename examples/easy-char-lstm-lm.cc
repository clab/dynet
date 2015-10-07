/*
  This program is meant to be an easy to understand tutorial on how to
  use the cnn package for training neural networks.

  This program trains a character level language model with 
  character embeddings in an LSTM neural network. The parameters
  of the LSTM model and the character embeddings are trained by minimizing
  the negative log-likelihood on an input corpus.

  Inputs: vocab.txt -- file containing the list of characters separated by
                       whitespace in a single line.
                       Ex: a b c e ... z <s> </s>
          train.txt -- file containing character sequences separated by
                       whitespace, every line is a word with spaces in between
                       every character.
                       ex: <s> c a t </s>
                           <s> c a t s </s>
                           ...
          char_size -- desired length of the character embedding.
          hidden_size -- length of the hidden layer in the LSTM.

  Run: ./build/examples/easy-char-lstm-lm vocab.txt train.txt 100 50
*/

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

/*
  Function to read a line into words separated by 'delim'
  This has nothing to be neural netowrks -- its just a utility function.
*/
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

/*
  This is the main class for implemeting LSTM based neural network.
*/
class LSTM {

public:
  /*
    Every parameter that we want to tune or every matrix/vector that we plan
    on using during computation is declared as an 'Expression' in the cnn tool.
  */
  Expression W_ix, W_ih, W_ic, W_cx, W_ch,
             W_ox, W_oh, W_oc;  // Weight matrices
  Expression b_i, b_f, b_c, b_o;  // Bias vectors
  int char_len, hidden_len;  // Length of the character embedding
                             // and internal hidden layer of the LSTM

  LSTM() {}

  /*
    Initialize the parameters of the model: specify their dimensions.
    These parameters are automatically initialized in the library.
 
    We will see what ComputationGraph and Model are in the main() function.
  */
  void Init(const int& char_length, const int& hidden_length,
            ComputationGraph *cg, Model *m) {
    char_len = char_length;
    hidden_len = hidden_length;

    /*
      Every expression that you want to use should be added as a parameter
      to the model (m, here) that you want to train. Also, it needs to be
      specified that it will be a part of the ComputationGraph (cg).
    */
    W_ix = parameter(*cg, m->add_parameters({hidden_len, char_len}));
    W_ih = parameter(*cg, m->add_parameters({hidden_len, hidden_len}));
    W_ic = parameter(*cg, m->add_parameters({hidden_len, hidden_len}));

    W_cx = parameter(*cg, m->add_parameters({hidden_len, char_len}));
    W_ch = parameter(*cg, m->add_parameters({hidden_len, hidden_len}));

    W_ox = parameter(*cg, m->add_parameters({hidden_len, char_len}));
    W_oh = parameter(*cg, m->add_parameters({hidden_len, hidden_len}));
    W_oc = parameter(*cg, m->add_parameters({hidden_len, hidden_len}));

    b_i = parameter(*cg, m->add_parameters({hidden_len, 1}));
    b_f = parameter(*cg, m->add_parameters({hidden_len, 1}));
    b_c = parameter(*cg, m->add_parameters({hidden_len, 1}));
    b_o = parameter(*cg, m->add_parameters({hidden_len, 1}));
  }

  /*
    This function reads a list of input vectors and outputs the list
    of corresponding hidden layers from the LSTM.
  */
  void GetHiddenUnits(const vector<Expression>& cols,
                      Expression& h_init,
                      vector<Expression>* hidden) {
    // Initialize the hidden unit and the cell of the LSTM.
    Expression h = h_init, c = h_init;
    for (unsigned t = 0; t < cols.size(); ++t) {  // Iterate over the input
      /*
        These are general LSTM computation equations.
        Functions like logictic(), cwise_multiply() come from cnn library.
      */
      Expression i = logistic(W_ix * cols[t] + W_ih * h + W_ic * c + b_i);
      Expression f = 1.f - i;

      Expression temp = tanh(W_cx * cols[t] + W_ch * h + b_c);
      c = cwise_multiply(f, c) + cwise_multiply(i, temp);

      Expression o = logistic(W_ox * cols[t] + W_oh * h + W_oc * c + b_o);
      h = cwise_multiply(o, tanh(c));
      hidden->push_back(h);  // Push the hidden layer at time t in the output.
    }
  }
};

/*
  This function computes the total loss while predicting the next character
  in sequence. This could be any loss function that you like. The end result
  of the loss function should be a scalar Expression.

  @param(targets) -- the correct character ids to be predicted
  @param(hidden_to_vocab) -- matrix to convert a hidden layer vector to a
                             vector of vocabulary size.
*/ 
Expression ComputeLoss(const vector<Expression>& hidden_units,
                       const vector<unsigned>& targets,
                       Expression& hidden_to_vocab) {
  vector<Expression> losses;
  for (unsigned i = 0; i < targets.size(); ++i) {
   // Project every hidden layer to a vector of size vocabulary
   Expression out = hidden_to_vocab * hidden_units[i];
   // This function computes the negative log-likelihood of obsering
   // the correct output character. 
   Expression loss = pickneglogsoftmax(out, targets[i]);
   losses.push_back(loss);
  }
  return sum(losses);  // sum all neg-llh over predictions
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // Read the vocab file and determine the size of vocabulary
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

  // Assign a unique id to every character in the vocabualry
  unordered_map<string, int> char_to_id;
  int num_chars = 0;
  for (const string& ch : chars) {
    char_to_id[ch] = num_chars++;
  }

  // Read the rest of command line parameters
  string train_filename = argv[2];  // train file
  unsigned char_size = atoi(argv[3]);
  unsigned hidden_size = atoi(argv[4]);
  vector<float> ZERO(hidden_size, 0.0f);

  Model m;  // Model that we will train
  SimpleSGDTrainer sgd(&m);  // We will train the model using stochastic
                             // gradient descent.

  // Variable that keeps track of the parameters and the computations being
  // performed, in order to perform backpropagation later.
  ComputationGraph cg;

  // Instead of representing a character vector by a one-hot vector and using
  // a projection matrix to obtain its embedding, cnn provides a more efficent
  // way of storing & updating the dense embeddings that are to be indexed 
  // frequently by their chracter ids.
  LookupParameters* char_vecs = m.add_lookup_parameters(vocab_size,
                                                        {char_size});

  // Paramter to project the hidden layer to a vocab_size vector for prediction
  Expression hidden_to_vocab = parameter(cg, m.add_parameters({vocab_size,
                                                               hidden_size}));
  LSTM lstm;

  // Initialize the parameters of the LSTM while making sure that those
  // parameters are part of the ConfigurationGraph cg and the Model m -- so
  // that they can be updated using backprop.
  lstm.Init(char_size, hidden_size, &cg, &m);

  // Read the training file and train the model
  unsigned num_iter = 100;
  for (unsigned iter = 0; iter < num_iter; ++iter) {
    ifstream train_file(train_filename);
    if (train_file.is_open()) {
      string line;
      double loss = 0;
      while (getline(train_file, line)) {
        chars = split_line(line, ' ');

        vector<Expression> input_vecs;
        vector<unsigned> targets;
        targets.clear();
        unsigned index = 0;
        for (const string& ch : chars) {
          // Construct a vector of the input vectors. The lookup function finds
          // the vector corresponding to a given id 'char_to_id[ch]'.
          if (index < chars.size() - 1) {
            // Input are all characters except the last one '</s>'
            input_vecs.push_back(lookup(cg, char_vecs, char_to_id[ch]));
          }
          if (index > 0) {
            // The characters to be predicted are all the characters except
            // the first one. '<s>'
            targets.push_back(char_to_id[ch]);
          }
          ++index;
        }

        vector<Expression> hidden_units;
        // Initialize the hidden layer of LSTM to be a zero vector
        Expression h = input(cg, {hidden_size}, &ZERO);
        // Obtain the hidden layer outputs
        lstm.GetHiddenUnits(input_vecs, h, &hidden_units);
        // Predict the output character using hidden layers and compute loss
        Expression e = ComputeLoss(hidden_units, targets, hidden_to_vocab);
        // The forward function runs all the expressions of computation we have
        // built till now in the computation graph and return the output of the
        // last step -- in this case, the prediction loss.
        loss += as_scalar(cg.forward());
        cg.backward();  // Compute graidents using backpropagation
        sgd.update(1);  // Updata parameters with learning rate 1.
      }
      cerr << "nllh: " << loss << endl;
      train_file.close();
    }
  }
}

