#include <cnn/cnn.h>
#define BOOST_TEST_MODULE CNNBasicTest
#include <boost/test/unit_test.hpp>

struct ConfigureCNNTest {
  ConfigureCNNTest() {
    // set up some dummy arguments to cnn
    for (auto x : {"ConfigureCNNTest", "--cnn-mem", "10"}) {
      av.push_back(strdup(x));
    }
    char **argv = &av[0];
    int argc = av.size();
    cnn::Initialize(argc, argv);
  }
  ~ConfigureCNNTest() {
    for (auto x : av) free(x);
  }
  std::vector<char*> av;
};

// configure CNN
BOOST_GLOBAL_FIXTURE(ConfigureCNNTest);

BOOST_AUTO_TEST_CASE( aligned_allocator ) {
  cnn::CPUAllocator a;
  void* mem = a.malloc(1024);
  BOOST_CHECK_EQUAL(((unsigned long)(mem) & 0x1f), 0);
  ((char*)mem)[0] = 99;
  a.zero(mem, 1024);
  BOOST_CHECK_EQUAL(((char*)mem)[0], 0);
  a.free(mem);
}

