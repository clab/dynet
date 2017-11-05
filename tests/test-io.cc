#define BOOST_TEST_MODULE TEST_IO

#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>

#include <boost/test/unit_test.hpp>
#include <boost/version.hpp>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/gru.h>
#include <dynet/io.h>

#include "test.h"

using namespace dynet;
using namespace std;

struct IOTest {
    IOTest() {
        // initialize if necessary
        if (default_device == nullptr) {
            for (auto x : {"IOTest", "--dynet-mem", "512"}) {
                av.push_back(strdup(x));
            }
            ADD_EXTRA_ARGUMENTS(av)
            char **argv = &av[0];
            int argc = av.size();
            dynet::initialize(argc, argv);
        }
        filename = "io.dump";
    }
    ~IOTest() {
        for (auto x : av) free(x);
    }

    std::vector<char*> av;
    std::string filename;
};

class testModel {
 public:
  testModel(dynet::ParameterCollection &model) {
    lookup_param = model.add_lookup_parameters(1000, {128});
    affine_params = model.add_subcollection("affine");
    W_x = affine_params.add_parameters({40, 30});
    b_x = affine_params.add_parameters({40});
    lstm = LSTMBuilder(3, 40, 1, model);
  }
  std::string get_affine_model_name() { return affine_params.get_fullname(); }
  dynet::ParameterCollection get_affine_model() const { return affine_params; }
  dynet::ParameterCollection get_lstm_model() { return lstm.get_parameter_collection(); }
 private:
  dynet::LookupParameter lookup_param;
  dynet::Parameter W_x, b_x;
  dynet::ParameterCollection affine_params;
  dynet::LSTMBuilder lstm;
}; // class testModel

// define the test suite
BOOST_FIXTURE_TEST_SUITE(io_test, IOTest);

BOOST_AUTO_TEST_CASE ( test_save_populate_pc ) {
  ParameterCollection m, m2;
  m.add_parameters({10}, "a");
  m.add_parameters({3,7});
  m.add_lookup_parameters(10, {2});
  m2.add_parameters({10}, "a");
  m2.add_parameters({3,7});
  m2.add_lookup_parameters(10, {2});
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  {
    dynet::TextFileLoader s("test.model");
    s.populate(m2);
  }
  DYNET_CHECK_EQUAL(m2, m);
}

BOOST_AUTO_TEST_CASE ( test_save_populate_sub_pc ) {
  // Create a parameter collection with a sub collection
  ParameterCollection m, m2;
  m.add_parameters({10}, "a");
  m.add_parameters({3,7});
  m.add_lookup_parameters(10, {2});
  ParameterCollection m_sub = m.add_subcollection("model1");
  m_sub.add_parameters({5}, "x");
  m_sub.add_parameters({3,6});
  m_sub.add_lookup_parameters(5, {3});
  m.add_lookup_parameters(4, {2});
  // Create another parameter collection with the same size as the sub collection
  m2.add_parameters({5}, "x");
  m2.add_parameters({3,6});
  m2.add_lookup_parameters(5, {3});
  // Save the overall collection
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  // Load only the sub-collection
  {
    dynet::TextFileLoader s("test.model");
    s.populate(m2, m_sub.get_fullname());
  }
  // Check if the sub-collection is equal
  DYNET_CHECK_EQUAL(m2, m_sub);
}

BOOST_AUTO_TEST_CASE ( test_save_populate_parameter ) {
  ParameterCollection m, m2;
  Parameter ma = m.add_parameters({10}, "a");
  Parameter mb = m.add_parameters({3,7});
  LookupParameter mc = m.add_lookup_parameters(10, {2});
  Parameter m2a = m2.add_parameters({10}, "a");
  Parameter m2b = m2.add_parameters({3,7});
  LookupParameter m2c = m2.add_lookup_parameters(10, {2});
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  {
    dynet::TextFileLoader s("test.model");
    s.populate(m2a, ma.get_fullname());
    s.populate(m2b, mb.get_fullname());
    s.populate(m2c, mc.get_fullname());
  }
  DYNET_CHECK_EQUAL(m2a, ma);
  DYNET_CHECK_EQUAL(m2b, mb);
  DYNET_CHECK_EQUAL(m2c, mc);
}
 
BOOST_AUTO_TEST_CASE ( test_save_load_parameter ) {
  ParameterCollection m, m2;
  Parameter ma = m.add_parameters({10}, "a");
  Parameter mb = m.add_parameters({3,7});
  LookupParameter mc = m.add_lookup_parameters(10, {2});
  Parameter m2a, m2b;
  LookupParameter m2c;
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  {
    dynet::TextFileLoader s("test.model");
    m2a = s.load_param(m2, ma.get_fullname());
    m2b = s.load_param(m2, mb.get_fullname());
    m2c = s.load_lookup_param(m2, mc.get_fullname());
  }
  DYNET_CHECK_EQUAL(m2a, ma);
  DYNET_CHECK_EQUAL(m2b, mb);
  DYNET_CHECK_EQUAL(m2c, mc);
}

BOOST_AUTO_TEST_CASE ( test_save_load_parameter_nonzerograd ) {
  ParameterCollection m, m2;
  Parameter ma = m.add_parameters({10}, "a");
  ma.get_storage().nonzero_grad = true;
  Parameter mb = m.add_parameters({3,7});
  mb.get_storage().nonzero_grad = true;
  LookupParameter mc = m.add_lookup_parameters(10, {2});
  mc.get_storage().nonzero_grad = true;
  Parameter m2a, m2b;
  LookupParameter m2c;
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  {
    dynet::TextFileLoader s("test.model");
    m2a = s.load_param(m2, ma.get_fullname());
    m2b = s.load_param(m2, mb.get_fullname());
    m2c = s.load_lookup_param(m2, mc.get_fullname());
  }
  DYNET_CHECK_EQUAL(m2a, ma);
  DYNET_CHECK_EQUAL(m2b, mb);
  DYNET_CHECK_EQUAL(m2c, mc);
}

#if BOOST_VERSION >= 105900
#define BOOST_PERF_TEST_CASE(testname) \
  BOOST_AUTO_TEST_CASE ( testname, *boost::unit_test::disabled() )
#else
#define BOOST_PERF_TEST_CASE(testname) \
  BOOST_AUTO_TEST_CASE ( testname )
#endif

BOOST_PERF_TEST_CASE ( test_save1_perf ) {
  ParameterCollection m;
  for (int l = 0; l < 16; ++l)
    m.add_parameters({1024, 1024});
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "ms" << std::endl;
}

BOOST_PERF_TEST_CASE ( test_load1_perf ) {
  ParameterCollection m, m_l;
  Parameter param;
  Parameter param_l = m_l.add_parameters({1024, 1024});
  for (int l = 0; l < 16; ++l)
    param = m.add_parameters({1024, 1024});
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  {
    dynet::TextFileLoader l("test.model");
    l.populate(param_l, param.get_fullname());
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "ms" << std::endl;
  DYNET_CHECK_EQUAL(param, param_l);
}

BOOST_PERF_TEST_CASE ( test_save2_perf ) {
  ParameterCollection m;
  for (int l = 0; l < 512; ++l)
    m.add_parameters({128, 128});
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "ms" << std::endl;
}

BOOST_PERF_TEST_CASE ( test_load2_perf ) {
  ParameterCollection m, m_l;
  Parameter param;
  Parameter param_l = m_l.add_parameters({128, 128});
  for (int l = 0; l < 512; ++l)
    param = m.add_parameters({128, 128});
  {
    dynet::TextFileSaver s("test.model");
    s.save(m);
  }
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  {
    dynet::TextFileLoader l("test.model");
    l.populate(param_l, param.get_fullname());
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "ms" << std::endl;
  DYNET_CHECK_EQUAL(param, param_l);
}

BOOST_AUTO_TEST_SUITE_END()
