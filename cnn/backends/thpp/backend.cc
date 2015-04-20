#include "cnn/backends/thpp/tensor.h"

#include <iostream>
#include <random>

using namespace std;

namespace cnn {

std::mt19937* rndeng = nullptr;

void Initialize(int& argc, char**& argv) {
   cerr << "COMMAND:";
   for (int i = 0; i < argc; ++i)
     cerr << ' ' << argv[i];
   cerr << endl;
   std::random_device rd;
   rndeng = new mt19937(rd());
}

}

