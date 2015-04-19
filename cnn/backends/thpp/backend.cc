#include "cnn/backends/thpp/tensor.h"

#include <random>

using namespace std;

namespace cnn {

std::mt19937* rndeng = nullptr;

void Initialize(int& argc, char**& argv) {
   std::random_device rd;
   rndeng = new mt19937(rd());
   cerr << "Created random generator: " << rndeng << endl;
}

}

