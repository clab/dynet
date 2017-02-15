#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/gru.h>
#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>

#include <stdexcept>

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

// define the test suite
BOOST_FIXTURE_TEST_SUITE(io_test, IOTest);

BOOST_AUTO_TEST_CASE( simple_rnn_io ) {
    dynet::Model mod1;
    dynet::SimpleRNNBuilder rnn1(1, 10, 10, mod1);
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << mod1 << rnn1;
    out.close();

    dynet::Model mod2;
    dynet::SimpleRNNBuilder rnn2;

    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> mod2 >> rnn2;
    in.close();
}

BOOST_AUTO_TEST_CASE( vanilla_lstm_io ) {
    dynet::Model mod1;
    dynet::VanillaLSTMBuilder rnn1(1, 10, 10, mod1);
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << mod1 << rnn1;
    out.close();

    dynet::Model mod2;
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
    dynet::Model mod1;
    dynet::LSTMBuilder rnn1(1, 10, 10, mod1);
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << mod1 << rnn1;
    out.close();

    dynet::Model mod2;
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


BOOST_AUTO_TEST_SUITE_END()