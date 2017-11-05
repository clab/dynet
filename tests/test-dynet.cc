#include <dynet/dynet.h>
#define BOOST_TEST_MODULE DYNETBasicTest
#include <boost/test/unit_test.hpp>
#include "test.h"

struct ConfigureDYNETTest {
  ConfigureDYNETTest() {
    // set up some dummy arguments to dynet
    for (auto x : {"ConfigureDYNETTest", "--dynet-mem", "10"}) {
      av.push_back(strdup(x));
    }
    ADD_EXTRA_ARGUMENTS(av)
    char **argv = &av[0];
    int argc = av.size();
    dynet::initialize(argc, argv);
  }
  ~ConfigureDYNETTest() {
    for (auto x : av) free(x);
  }
  std::vector<char*> av;
};

// configure DYNET
BOOST_GLOBAL_FIXTURE(ConfigureDYNETTest);

BOOST_AUTO_TEST_CASE( aligned_allocator ) {
  dynet::CPUAllocator a;
  void* mem = a.malloc(1024);
  BOOST_CHECK_EQUAL(((uintptr_t)(mem) & 0x1f), 0);
  ((char*)mem)[0] = 99;
  a.zero(mem, 1024);
  BOOST_CHECK_EQUAL(((char*)mem)[0], 0);
  a.free(mem);
}

