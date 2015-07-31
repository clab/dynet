#ifndef CNN_GRAD_CHECK_H
#define CNN_GRAD_CHECK_H

namespace cnn {

class Model;
struct ComputationGraph;

void CheckGrad(Model& m, ComputationGraph& g);

} // namespace cnn

#endif
