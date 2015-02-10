#ifndef CNN_RNN_H_
#define CNN_RNN_H_

#include "cnn/cnn.h"
#include "cnn/edges.h"

namespace cnn {

struct Trainer;

struct RNNBuilder {
  explicit RNNBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Trainer* trainer);
  ~RNNBuilder();

  // call this before add_input
  void add_parameter_edges(Hypergraph* hg);

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  unsigned add_input(unsigned x, Hypergraph* hg);

  // hidden x hidden zero matrix
  unsigned zero;

  ConstParameters* p_z; // dummy zero parameter for starting state
  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Parameters*>> params;

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
