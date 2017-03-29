#include "dynet/model.h"
#include "dynet/cfsm-builder.h"
#include "dynet/deep-lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/gru.h"
#include "dynet/hsm-builder.h"
#include "dynet/lstm.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

namespace pydynet {

// Wrappers for the templated boost interface so that it is accessible
// form cython. Needs to add a method for every type we would like to
// load / save.
struct ParameterCollectionSaver {
    ParameterCollectionSaver(std::string filename, dynet::ParameterCollection *model) :
        ofs(filename),
        oa(ofs)
    {
        oa << *model;
    };

    ParameterCollectionSaver* add_parameter(dynet::Parameter &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_lookup_parameter(dynet::LookupParameter &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_lstm_builder(dynet::LSTMBuilder &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_vanilla_lstm_builder(dynet::VanillaLSTMBuilder &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_srnn_builder(dynet::SimpleRNNBuilder &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_gru_builder(dynet::GRUBuilder &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_hsm_builder(dynet::HierarchicalSoftmaxBuilder &p) {
        oa << p; return this;
    }


    ParameterCollectionSaver* add_fast_lstm_builder(dynet::FastLSTMBuilder &p) {
        oa << p; return this;
    }

    // TODO what is this?
    ParameterCollectionSaver* add_deep_lstm_builder(dynet::DeepLSTMBuilder &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_cfsm_builder(dynet::ClassFactoredSoftmaxBuilder &p) {
        oa << p; return this;
    }

    ParameterCollectionSaver* add_sm_builder(dynet::StandardSoftmaxBuilder &p) {
        oa << p; return this;
    }

    void done() { ofs.close(); }


    private:
        std::ofstream ofs;
        boost::archive::text_oarchive oa;

};

struct ParameterCollectionLoader {
    ParameterCollectionLoader(std::string filename, dynet::ParameterCollection *model) :
        ifs(filename),
        ia(ifs)
    {
        printf("loading model\n");
        ia >> *model;
        printf("done %zu\n", model->parameters_list().size());
    };

    ParameterCollectionLoader* fill_parameter(dynet::Parameter &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_lookup_parameter(dynet::LookupParameter &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_lstm_builder(dynet::LSTMBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_vanilla_lstm_builder(dynet::VanillaLSTMBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_srnn_builder(dynet::SimpleRNNBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_gru_builder(dynet::GRUBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_hsm_builder(dynet::HierarchicalSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_fast_lstm_builder(dynet::FastLSTMBuilder &p) {
        ia >> p; return this;
    }

    // TODO what is this?
    ParameterCollectionLoader* fill_deep_lstm_builder(dynet::DeepLSTMBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_cfsm_builder(dynet::ClassFactoredSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    ParameterCollectionLoader* fill_sm_builder(dynet::StandardSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    void done() { ifs.close(); }

    private:
        std::ifstream ifs;
        boost::archive::text_iarchive ia;

};
}

