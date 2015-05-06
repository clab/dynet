#ifndef CNN_SAXE_INIT_H_
#define CNN_SAXE_INIT_H_

namespace cnn {

struct Tensor;

void OrthonormalRandom(int dim, float g, Tensor& x);

}

#endif
