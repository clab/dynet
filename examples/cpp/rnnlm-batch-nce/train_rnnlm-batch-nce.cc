#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <chrono>

#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>
#include <dynet/training.h>

using namespace std;
using namespace std::chrono;
using namespace dynet;
using namespace dynet::expr;

// Read a file where each line is of the form "word1 word2 ..."
// Yields lists of the form [word1, word2, ...]
vector<vector<int> > read(const string & fname, Dict & vw) {
  ifstream fh(fname);
  if(!fh) throw std::runtime_error("Could not open file");
  string str; 
  vector<vector<int> > sents;
  while(getline(fh, str)) {
    istringstream iss(str);
    vector<int> tokens;
    while(iss >> str)
      tokens.push_back(vw.convert(str));
    tokens.push_back(vw.convert("<s>"));
    sents.push_back(tokens);
  }
  return sents;
}

struct RNNLanguageModel {
  LookupParameter p_c;
  Parameter W_sm;
  Parameter b_sm;
  VanillaLSTMBuilder builder;
  explicit RNNLanguageModel(unsigned layers, unsigned input_dim, unsigned hidden_dim, unsigned vocab_size, Model& model) : builder(layers, input_dim, hidden_dim, model) {
    p_c = model.add_lookup_parameters(vocab_size, {input_dim}, ParameterInitUniform(0.1)); 
    W_sm = model.add_parameters({vocab_size, hidden_dim}, ParameterInitUniform(0.5));
    b_sm = model.add_parameters({vocab_size}, ParameterInitUniform(0.5));
  }

  Expression calc_lm_loss(const vector<vector<int> > & sent, int pos, int mb_size, ComputationGraph & cg) {
  
    // parameters -> expressions
    Expression W_exp = parameter(cg, W_sm);
    Expression b_exp = parameter(cg, b_sm);
  
    // initialize the RNN
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
  
    // start the rnn by inputting "<s>"
    size_t tot_sents = min(sent.size()-pos, (size_t)mb_size);
    vector<unsigned> wids(tot_sents, 0);
    vector<float> masks(tot_sents);
    Expression s = builder.add_input(lookup(cg, p_c, wids)); 

    // feed word vectors into the RNN and predict the next word
    vector<Expression> losses;
    size_t j;
    for(size_t i = 0; i < sent[pos].size(); ++i) {
      // Get the words
      for(j = 0; j < tot_sents && i < sent[pos+j].size(); ++j) {
        wids[j] = sent[pos+j][i];
        masks[j] = 1.f;
      }
      // And the masks
      for(; j < tot_sents; ++j) {
        wids[j] = 0;
        masks[j] = 0.f;
      }
      // calculate the softmax and loss
      Expression score = affine_transform({b_exp, W_exp, s});
      Expression loss = pickneglogsoftmax(score, wids);
      if(0.f == *masks.rbegin())
        loss = cmult(loss, input(cg, Dim({1}, tot_sents), masks));
      losses.push_back(loss);
      // update the state of the RNN
      s = builder.add_input(lookup(cg, p_c, wids));
    }
    
    return sum_batches(sum(losses));
  }

};

struct length_greater_then {
    inline bool operator() (const vector<int> & struct1, const vector<int> & struct2) {
        return (struct1.size() > struct2.size());
    }
};

vector<int> prepare_minibatch(int mb_size, vector<vector<int> > & data) {
  stable_sort(data.begin(), data.end(), length_greater_then());
  vector<int> ids;
  for(size_t i = 0; i < data.size(); i += mb_size)
    ids.push_back(i);
  return ids;
}

int main(int argc, char** argv) {

  time_point<system_clock> start = system_clock::now();

  // format of files: each line is "word1 word2 ..."
  string train_file = "data/text/train.txt";
  string test_file = "data/text/dev.txt";

  // DyNet Starts
  dynet::initialize(argc, argv);
  Model model;

  if(argc != 6) {
    cerr << "Usage: " << argv[0] << " MB_SIZE EMBED_SIZE HIDDEN_SIZE SPARSE TIMEOUT" << endl;
    return 1;
  }
  int MB_SIZE = atoi(argv[1]);
  int EMBED_SIZE = atoi(argv[2]);
  int HIDDEN_SIZE = atoi(argv[3]);
  int SPARSE = atoi(argv[4]);
  int TIMEOUT = atoi(argv[5]);

  AdamTrainer trainer(model, 0.001);
  trainer.sparse_updates_enabled = SPARSE;
  trainer.clipping_enabled = false;

  Dict vw;
  vw.convert("<s>");
  vector<vector<int> > train = read(train_file, vw);
  vw.freeze();
  vector<vector<int> > test = read(test_file, vw);
  vector<int> train_ids = prepare_minibatch(MB_SIZE, train);
  vector<int> test_ids = prepare_minibatch(MB_SIZE, test);
  int test_words = 0;
  for(auto & sent : test) test_words += sent.size();

  int nwords = vw.size();

  RNNLanguageModel rnnlm(1, EMBED_SIZE, HIDDEN_SIZE, nwords, model);

  {
    duration<float> fs = (system_clock::now() - start);
    float startup_time = duration_cast<milliseconds>(fs).count() / float(1000);
    cout << "startup time: " << startup_time << endl;
  }

  start = system_clock::now();
  int i = 0, all_words = 0, this_words = 0;
  float this_loss = 0.f, all_time = 0.f;
  for(int iter = 0; iter < 100; iter++) {
    shuffle(train_ids.begin(), train_ids.end(), *dynet::rndeng);
    for(auto sid : train_ids) {
      i++;
      if(i % (500/MB_SIZE) == 0) {
        trainer.status();
        cout << this_loss/this_words << endl;
        all_words += this_words;
        this_loss = 0.f;
        this_words = 0;
      }
      if(i % (10000/MB_SIZE) == 0) {
        duration<float> fs = (system_clock::now() - start);
        all_time += duration_cast<milliseconds>(fs).count() / float(1000);
        float test_loss = 0;
        for(auto sentid : test_ids) {
          ComputationGraph cg;
          Expression loss_exp = rnnlm.calc_lm_loss(test, sentid, MB_SIZE, cg);
          test_loss += as_scalar(cg.forward(loss_exp));
        }
        cout << "nll=" << test_loss/test_words << ", ppl=" << exp(test_loss/test_words) << ", words=" << test_words << ", time=" << all_time << ", word_per_sec=" << all_words/all_time << endl;
        if(all_time > TIMEOUT)
          exit(0);
        start = system_clock::now();
      }

      ComputationGraph cg;
      Expression loss_exp = rnnlm.calc_lm_loss(train, sid, MB_SIZE, cg);
      this_loss += as_scalar(cg.forward(loss_exp));
      for(size_t pos = sid; pos < min((size_t)sid+MB_SIZE, train.size()); ++pos)
        this_words += train[pos].size();
      cg.backward(loss_exp);
      trainer.update();
    }
    trainer.update_epoch(1.0);
  }
}
