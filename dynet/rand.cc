#include "dynet/rand.h"
//#include "dynet/globals.h"
#include <random>

using namespace std;

namespace dynet {

  real rand01() {
    uniform_real_distribution<real> distribution(0, 1);
    return distribution(*rndeng);
  }

  int rand0n(int n) {
    if (n <= 0) throw std::runtime_error("Integer upper bound is non-positive");
    int x = rand01() * n;
    while (n == x) { x = rand01() * n; }
    return x;
  }

  real rand_normal() {
    normal_distribution<real> distribution(0, 1);
    return distribution(*rndeng);
  }

  int draw_random_seed(){
    std::uniform_int_distribution<> seed_dist(1, 2147483647);
    return seed_dist(*rndeng);
  }
} // namespace dynet

