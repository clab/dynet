#ifndef CNN_EIGEN_INIT_H
#define CNN_EIGEN_INIT_H

namespace cnn {

void Initialize(int& argc, char**& argv, bool shared_parameters = false);
void Cleanup();

} // namespace cnn

#endif
