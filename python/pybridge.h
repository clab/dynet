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
struct ModelSaver {
    ModelSaver(std::string filename, dynet::Model *model) :
        ofs(filename),
        oa(ofs)
    {
        oa << *model;
    };

    ModelSaver* add_parameter(dynet::Parameter &p) {
        oa << p; return this;
    }

    ModelSaver* add_lookup_parameter(dynet::LookupParameter &p) {
        oa << p; return this;
    }

    ModelSaver* add_lstm_builder(dynet::LSTMBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_srnn_builder(dynet::SimpleRNNBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_gru_builder(dynet::GRUBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_hsm_builder(dynet::HierarchicalSoftmaxBuilder &p) {
        oa << p; return this;
    }


    ModelSaver* add_fast_lstm_builder(dynet::FastLSTMBuilder &p) {
        oa << p; return this;
    }

    // TODO what is this?
    ModelSaver* add_deep_lstm_builder(dynet::DeepLSTMBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_cfsm_builder(dynet::ClassFactoredSoftmaxBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_sm_builder(dynet::StandardSoftmaxBuilder &p) {
        oa << p; return this;
    }

    void done() { ofs.close(); }


    private:
        std::ofstream ofs;
        boost::archive::text_oarchive oa;

};

struct ModelLoader {
    ModelLoader(std::string filename, dynet::Model *model) :
        ifs(filename),
        ia(ifs)
    {
        printf("loading model\n");
        ia >> *model;
        printf("done %d\n", model->parameters_list().size());
    };

    ModelLoader* fill_parameter(dynet::Parameter &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_lookup_parameter(dynet::LookupParameter &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_lstm_builder(dynet::LSTMBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_srnn_builder(dynet::SimpleRNNBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_gru_builder(dynet::GRUBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_hsm_builder(dynet::HierarchicalSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_fast_lstm_builder(dynet::FastLSTMBuilder &p) {
        ia >> p; return this;
    }

    // TODO what is this?
    ModelLoader* fill_deep_lstm_builder(dynet::DeepLSTMBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_cfsm_builder(dynet::ClassFactoredSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_sm_builder(dynet::StandardSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    void done() { ifs.close(); }

    private:
        std::ifstream ifs;
        boost::archive::text_iarchive ia;

};
}

