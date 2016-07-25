#ifndef CNN_EIGEN_INIT_H
#define CNN_EIGEN_INIT_H

namespace cnn {

extern float weight_decay_lambda;

void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

} // namespace cnn

#endif
