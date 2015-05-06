#include "cnn/backends/eigen/init.h"

#include <iostream>
#include <random>
#include <cmath>

using namespace std;

namespace cnn {

std::mt19937* rndeng = nullptr;
void Initialize(int& argc, char**& argv) {
  std::random_device rd;
  rndeng = new mt19937(rd());
//  rndeng = new mt19937(1);
}

} // namespace cnn

