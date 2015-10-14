#include <cnn/cnn.h>
#define BOOST_TEST_MODULE CNNBasicTest
#include <boost/test/unit_test.hpp>

struct ConfigureCNNTest {
  ConfigureCNNTest() {
    // set up some dummy arguments to cnn
    for (auto x : {"ConfigureCNNTest", "--cnn.mem", "1MB"}) {
      av.push_back(strdup(x));
    }
    av.push_back(nullptr);
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
    void* mem = cnn::cnn_mm_malloc(1024, 0x100);
    BOOST_CHECK_EQUAL(((unsigned long)(mem) & 0xff), 0);
}

