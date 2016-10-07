#ifndef DYNET_EIGEN_INIT_H
#define DYNET_EIGEN_INIT_H

namespace dynet {

extern float weight_decay_lambda;

void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

} // namespace dynet

#endif
