#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cnn;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  unsigned CONTEXT = 3;
  unsigned DIM = 100;
  unsigned VOCAB_SIZE = 29;

  // parameters
  Model model;
  SimpleSGDTrainer sgd(&model);
  LookupParameters* p_c = model.add_lookup_parameters(VOCAB_SIZE, {DIM});
  Parameters* p_C1 = model.add_parameters({DIM, DIM});
  Parameters* p_C2 = model.add_parameters({DIM, DIM});
  Parameters* p_C3 = model.add_parameters({DIM, DIM});
  Parameters* p_R = model.add_parameters({VOCAB_SIZE, DIM});
  Parameters* p_bias = model.add_parameters({VOCAB_SIZE});
  Parameters* p_hb = model.add_parameters({DIM});

  // build the graph
  ComputationGraph cg;
  unsigned in_c1, in_c2, in_c3;  // set these to set the context words
  VariableIndex i_c1 = cg.add_lookup(p_c, &in_c1);
  VariableIndex i_c2 = cg.add_lookup(p_c, &in_c2);
  VariableIndex i_c3 = cg.add_lookup(p_c, &in_c3);
  VariableIndex i_C1 = cg.add_parameters(p_C1);
  VariableIndex i_C2 = cg.add_parameters(p_C2);
  VariableIndex i_C3 = cg.add_parameters(p_C3);
  VariableIndex i_hb = cg.add_parameters(p_hb);
  VariableIndex i_R = cg.add_parameters(p_R);
  unsigned ytrue;  // set ytrue to change the value of the input
  VariableIndex i_bias = cg.add_parameters(p_bias);

  // r = hb + C1 * c1 + C2 * c2 + C3 * c3
  VariableIndex i_r = cg.add_function<AffineTransform>({i_hb, i_C1, i_c1, i_C2, i_c2, i_C3, i_c3});

  // nl = rectify(r)
  VariableIndex i_nl = cg.add_function<Rectify>({i_r});

  // o2 = bias + R * nl
  VariableIndex i_o2 = cg.add_function<AffineTransform>({i_bias, i_R, i_nl});

  // ydist = softmax(o2)
  VariableIndex i_ydist = cg.add_function<LogSoftmax>({i_o2});

  // nerr = pick(ydist, ytrue)
  VariableIndex i_nerr = cg.add_function<PickElement>({i_ydist}, &ytrue);

  // err = -nerr
  cg.add_function<Negate>({i_nerr});
  cg.PrintGraphviz();

  // load some training data
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " ngrams.txt\n";
    return 1;
  }
  ifstream in(argv[1]);
  vector<vector<unsigned>> corpus;
  string line;
  while(getline(in, line)) {
    istringstream is(line);
    vector<unsigned> x(CONTEXT+1);
    for (unsigned i = 0; i <= CONTEXT; ++i) {
      is >> x[i];
      assert(x[i] < VOCAB_SIZE);
    }
    corpus.push_back(x);
  }


  // train the parameters
  for (unsigned iter = 0; iter < 100; ++iter) {
    Timer iteration("epoch completed in");
    double loss = 0;
    unsigned n = 0;
    for (auto& ci : corpus) {
      in_c1 = ci[0];
      in_c2 = ci[1];
      in_c3 = ci[2];
      ytrue  = ci[3];
      loss += as_scalar(cg.forward());
      cg.backward();
      ++n;
      sgd.update(1.0);
      if (n == 2500) break;
    }
    loss /= n;
    cerr << "E = " << loss << ' ';
  }
}

