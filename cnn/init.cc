#include "cnn/init.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"

#include <iostream>
#include <random>
#include <cmath>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


#if HAVE_CUDA
#include "cnn/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace cnn {

#define ALIGN 6
AlignedMemoryPool<ALIGN>* fxs = nullptr;
AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
mt19937* rndeng = nullptr;

void Initialize(int& argc, char**& argv) {
  cerr << "Initializing...\n";
#if HAVE_CUDA
  Initialize_GPU(argc, argv);
#else
  kSCALAR_MINUSONE = (float*) cnn_mm_malloc(1, 256);
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) cnn_mm_malloc(1, 256);
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) cnn_mm_malloc(1, 256);
  *kSCALAR_ZERO = 0;
#endif

  using namespace boost::program_options;
  variables_map vm;
  options_description opts("initialization setup");
  opts.add_options()
      ("seed,s", value<int>()->default_value(217), "seed number for random number genreation");

  store(parse_command_line(argc, argv, opts), vm);

  if (vm.count("seed")) {
      int rseed = vm["seed"].as<int>();
      rndeng = new mt19937(rseed);
  }
  else
  {
      random_device rd;
      //  rndeng = new mt19937(1);
      rndeng = new mt19937(rd());
  }
  cerr << "Allocating memory...\n";
  fxs = new AlignedMemoryPool<ALIGN>(512*(1<<20));
  dEdfs = new AlignedMemoryPool<ALIGN>(512*(1<<20));
  cerr << "Done.\n";
}

} // namespace cnn

