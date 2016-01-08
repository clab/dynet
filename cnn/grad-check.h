#ifndef CNN_GRAD_CHECK_H
#define CNN_GRAD_CHECK_H

namespace cnn {

class Model;
struct ComputationGraph;

// verbosity is zero for silence, one for only printing errors, two for everything
bool CheckGrad(Model& m, ComputationGraph& g, int verbosity = 1);

} // namespace cnn

#endif
