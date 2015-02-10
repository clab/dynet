#ifndef CNN_RNN_H_
#define CNN_RNN_H_

#include "cnn/cnn.h"
#include "cnn/edges.h"

namespace cnn {

struct Trainer;

struct RNNBuilder {
  explicit RNNBuilder(Hypergraph* g,
                      unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Trainer* trainer);
  ~RNNBuilder();

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  unsigned add_input(unsigned x);

  // hidden x hidden zero matrix
  unsigned zero;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<unsigned>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<unsigned>> h;

  Hypergraph* hg;
  const unsigned layers;

  std::vector<ParametersBase*> to_be_deleted;
};

} // namespace cnn

#endif
