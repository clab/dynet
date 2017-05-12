#define BOOST_TEST_MODULE TEST_IO

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/gru.h>
#include <dynet/io.h>

#include "test.h"

using namespace dynet;
using namespace dynet::expr;
using namespace std;

struct IOTest {
    IOTest() {
        // initialize if necessary
        if (default_device == nullptr) {
            for (auto x : {"IOTest", "--dynet-mem", "512"}) {
                av.push_back(strdup(x));
            }
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
  std::string get_affine_model_name() { return affine_params.get_namespace(); }
  dynet::ParameterCollection get_affine_model() const { return affine_params; }
  dynet::ParameterCollection get_lstm_model() { return lstm.get_parameters(); }
 private:
  dynet::LookupParameter lookup_param;
  dynet::Parameter W_x, b_x;
  dynet::ParameterCollection affine_params;
  dynet::LSTMBuilder lstm;
}; // class testModel

// define the test suite
BOOST_FIXTURE_TEST_SUITE(io_test, IOTest);

BOOST_AUTO_TEST_CASE ( test_save_load_parameter_collection ) {
  {
    ParameterCollection m;
    m.add_parameters({10}, "a");
    m.add_parameters({3,7});
    m.add_lookup_parameters(10, {2});
    {
      dynet::TextFileSaver s("test.model");
      s.save(m, "model1");
      s.save(m, m.get_namespace());
    }

    {
      dynet::TextFileLoader s("test.model");
      ParameterCollection m2;
      s.populate(m2, "model1");
      DYNET_CHECK_EQUAL(m2, m);
      ParameterCollection m3;
      s.populate(m3, "/");
    }
  }
  {
    ParameterCollection collec, collec2, collec3, lstm2, affine2;
    testModel spec(collec);
    {
      TextFileSaver s1("a.model");
      s1.save(collec, "all");
    }
    {
      TextFileLoader s1("a.model");
      s1.populate(collec2);
    }
    DYNET_CHECK_EQUAL(collec2.size(), collec.size());
  
    {
      TextFileSaver s2("b.model");
      s2.save(collec, "all");
      s2.save(spec.get_lstm_model(), "lstm");
      s2.save(spec.get_affine_model(), "affine");
    }
    {
      TextFileLoader s2("b.model");
      s2.populate(affine2, "affine");
      s2.populate(collec3, "all");
      s2.populate(lstm2, "lstm");
    }
    DYNET_CHECK_EQUAL(affine2.size(), spec.get_affine_model().size());
    DYNET_CHECK_EQUAL(collec3.size(), collec.size());
    DYNET_CHECK_EQUAL(lstm2.size(), spec.get_lstm_model().size());

  }
  {
    ParameterCollection cc, ccc;
    auto cc2 = cc.add_subcollection("xx");
    cc2.add_parameters({2, 3, 4, 5});
    {
      TextFileSaver s3("d.model");
      s3.save(cc, "key");
    }
    {
      TextFileLoader s3("d.model");
      s3.populate(ccc, "key");
    }
    DYNET_CHECK_EQUAL(ccc.size(), cc.size());
  }
}

BOOST_AUTO_TEST_CASE ( test_save_load_parameter ) {
  {
    ParameterCollection model_out;
    Parameter m_out = model_out.add_parameters({100});
    LookupParameter lookup_m_out = model_out.add_lookup_parameters(10, {128});
    {
      TextFileSaver saver("f.model");
      saver.save(m_out, "m");
      saver.save(lookup_m_out, "lookup_m");
    }

    ParameterCollection model, model_in;
    Parameter m = model.add_parameters({100}), m_in;
    LookupParameter lookup_m = model_out.add_lookup_parameters(10, {128}), lookup_m_in;
    {
      TextFileLoader loader("f.model");
      loader.populate(m, "m");
      loader.populate(lookup_m, "lookup_m");
    }
    DYNET_CHECK_EQUAL(m, m_out);
    DYNET_CHECK_EQUAL(lookup_m, lookup_m_out);
    {
      TextFileLoader loader("f.model");
      m_in = loader.load_param(model_in, "m");
      lookup_m_in = loader.load_lookup_param(model_in, "lookup_m");
    }
    DYNET_CHECK_EQUAL(m_in, m_out);
    DYNET_CHECK_EQUAL(lookup_m_in, lookup_m_out);
  }
  {
    ParameterCollection model;
    Parameter m = model.add_parameters({10});
    LookupParameter lookup_m = model.add_lookup_parameters(10, {128});
    {
      TextFileSaver saver("g.model");
      saver.save(model, "model");
    }

    ParameterCollection model_in, model_in2;
    Parameter m_in = model_in.add_parameters({10});
    LookupParameter lookup_m_in = model_in.add_lookup_parameters(10, {128});

    {
      TextFileLoader loader("g.model");
      loader.populate(m_in, "/__0");
      DYNET_CHECK_EQUAL(m_in, m);
      loader.populate(lookup_m_in, "/__1");
      DYNET_CHECK_EQUAL(lookup_m_in, lookup_m);
      Parameter m_in2 = loader.load_param(model_in2, "/__0");
      DYNET_CHECK_EQUAL(m_in2, m);
      LookupParameter lookup_m_in2 = loader.load_lookup_param(model_in2, "/__1");
      DYNET_CHECK_EQUAL(lookup_m_in2, lookup_m);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
