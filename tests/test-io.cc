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

BOOST_AUTO_TEST_CASE( simple_rnn_io ) {
    dynet::ParameterCollection mod1;
    dynet::SimpleRNNBuilder rnn1(1, 10, 10, mod1);
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << mod1 << rnn1;
    out.close();

    dynet::ParameterCollection mod2;
    dynet::SimpleRNNBuilder rnn2;

    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> mod2 >> rnn2;
    in.close();
}

BOOST_AUTO_TEST_CASE( vanilla_lstm_io ) {
    dynet::ParameterCollection mod1;
    dynet::VanillaLSTMBuilder rnn1(1, 10, 10, mod1);
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << mod1 << rnn1;
    out.close();

    dynet::ParameterCollection mod2;
    dynet::VanillaLSTMBuilder rnn2;

    BOOST_CHECK(rnn2.input_dim == 0);
    BOOST_CHECK(rnn2.hid == 0);

    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> mod2 >> rnn2;
    in.close();

    BOOST_CHECK(rnn2.input_dim == 10);
    BOOST_CHECK(rnn2.hid == 10);
}

BOOST_AUTO_TEST_CASE( lstm_io ) {
    dynet::ParameterCollection mod1;
    dynet::LSTMBuilder rnn1(1, 10, 10, mod1);
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << mod1 << rnn1;
    out.close();

    dynet::ParameterCollection mod2;
    dynet::LSTMBuilder rnn2;

    BOOST_CHECK(rnn2.input_dim == 0);
    BOOST_CHECK(rnn2.hid == 0);

    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> mod2 >> rnn2;
    in.close();

    BOOST_CHECK(rnn2.input_dim == 10);
    BOOST_CHECK(rnn2.hid == 10);
}

BOOST_AUTO_TEST_CASE ( test_save_load_parameter_collection ) {
  {
    ParameterCollection m;
    Parameter a = m.add_parameters({10}, "a");
    Parameter b = m.add_parameters({3,7});
    LookupParameter c = m.add_lookup_parameters(10, {2});
    std::remove("test.model"); std::remove("test.model.meta");
    dynet::Pack s("test.model");
    s.save(m, "model1");
    s.save(m, m.get_namespace(), true);

    ParameterCollection m2;
    s.populate(m2, "model1");
    DYNET_CHECK_EQUAL(m2, m);
    ParameterCollection m3;
    s.populate(m3, "/");
  }
  {
    ParameterCollection collec;
    testModel spec(collec);
    std::remove("a.model"); std::remove("a.model.meta");
    Pack s1("a.model");
    s1.save(collec, "all");
    ParameterCollection collec2;
    s1.populate(collec2);
    DYNET_CHECK_EQUAL(collec2.size(), collec.size());
  
    std::remove("b.model"); std::remove("b.model.meta");
    Pack s2("b.model");
    s2.save(collec, "all");
    s2.save(spec.get_lstm_model(), "lstm", true);
    s2.save(spec.get_affine_model(), "affine", true);
    ParameterCollection collec3, lstm2, affine2;
    s2.populate(affine2, "affine");
    s2.populate(collec3, "all");
    s2.populate(lstm2, "lstm");
    DYNET_CHECK_EQUAL(affine2.size(), spec.get_affine_model().size());
    DYNET_CHECK_EQUAL(collec3.size(), collec.size());
    DYNET_CHECK_EQUAL(lstm2.size(), spec.get_lstm_model().size());

    std::remove("c.model"); std::remove("c.model.meta");
    s2.reinit("c.model");
    s2.save(lstm2, "lstm");
    s2.save(collec3, "all", true);
    s2.save(affine2, "affine", true);
  }
  {
    ParameterCollection cc;
    auto cc2 = cc.add_subcollection("xx");
    cc2.add_parameters({10});
    std::remove("d.model"); std::remove("d.model.meta");
    Pack s3("d.model");
    s3.save(cc, "key");

    ParameterCollection ccc;
    s3.populate(ccc, "key");
    DYNET_CHECK_EQUAL(ccc.size(), cc.size());

    std::remove("e.model"); std::remove("e.model.meta");
    s3.reinit("e.model");
    s3.save(ccc);
  }
}

BOOST_AUTO_TEST_CASE ( test_save_load_parameter ) {
  {
    std::remove("f.model.meta"); std::remove("f.model");
    Pack packer("f.model");
    ParameterCollection model_out;
    Parameter m_out = model_out.add_parameters({100});
    LookupParameter lookup_m_out = model_out.add_lookup_parameters(10, {128});
    packer.save(m_out, "m");
    packer.save(lookup_m_out, "lookup_m");

    ParameterCollection model;
    Parameter m = model.add_parameters({100});
    LookupParameter lookup_m = model_out.add_lookup_parameters(10, {128});
    packer.populate(m, "m");
    packer.populate(lookup_m, "lookup_m");
    DYNET_CHECK_EQUAL(m, m_out);
    DYNET_CHECK_EQUAL(lookup_m, lookup_m_out);

    ParameterCollection model_in;
    Parameter m_in = packer.load_param(model_in, "m");
    LookupParameter lookup_m_in = packer.load_lookup_param(model_in, "lookup_m");
    DYNET_CHECK_EQUAL(m_in, m_out);
    DYNET_CHECK_EQUAL(lookup_m_in, lookup_m_out);
  }
  {
    std::remove("g.model.meta"); std::remove("g.model");
    Pack packer("g.model");
    ParameterCollection model;
    Parameter m = model.add_parameters({10});
    LookupParameter lookup_m = model.add_lookup_parameters(10, {128});
    packer.save(model, "model");

    ParameterCollection model_in;
    Parameter m_in = model_in.add_parameters({10});
    LookupParameter lookup_m_in = model_in.add_lookup_parameters(10, {128});
    packer.populate(m_in, "model", "/__0");
    DYNET_CHECK_EQUAL(m_in, m);
    packer.populate(lookup_m_in, "model", "/__1");
    DYNET_CHECK_EQUAL(lookup_m_in, lookup_m);

    ParameterCollection model_in2;
    Parameter m_in2 = packer.load_param(model_in2, "model", "/__0");
    DYNET_CHECK_EQUAL(m_in2, m);
    LookupParameter lookup_m_in2 = packer.load_lookup_param(model_in2, "model",
                                                            "/__1");
    DYNET_CHECK_EQUAL(lookup_m_in2, lookup_m);
  }
}

BOOST_AUTO_TEST_SUITE_END()
