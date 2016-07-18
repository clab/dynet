#ifndef CNN_EIGEN_INIT_H
#define CNN_EIGEN_INIT_H

namespace cnn {

extern float weight_decay_lambda;

void Initialize(int& argc, char**& argv, bool shared_parameters = false);
void Cleanup();

} // namespace cnn

#endif
