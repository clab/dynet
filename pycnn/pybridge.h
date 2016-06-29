#include "cnn/model.h"
#include "cnn/cfsm-builder.h"
#include "cnn/deep-lstm.h"
#include "cnn/fast-lstm.h"
#include "cnn/fast-lstm.h"
#include "cnn/gru.h"
#include "cnn/hsm-builder.h"
#include "cnn/lstm.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

namespace pycnn {

// Wrappers for the templated boost interface so that it is accessible
// form cython. Needs to add a method for every type we would like to
// load / save.
struct ModelSaver {
    ModelSaver(std::string filename, cnn::Model *model) :
        ofs(filename),
        oa(ofs)
    {
        oa << *model;
    };

    ModelSaver* add_parameter(cnn::Parameter &p) {
        oa << p; return this;
    }

    ModelSaver* add_lookup_parameter(cnn::LookupParameter &p) {
        oa << p; return this;
    }

    ModelSaver* add_lstm_builder(cnn::LSTMBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_srnn_builder(cnn::SimpleRNNBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_gru_builder(cnn::GRUBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_hsm_builder(cnn::HierarchicalSoftmaxBuilder &p) {
        oa << p; return this;
    }


    ModelSaver* add_fast_lstm_builder(cnn::FastLSTMBuilder &p) {
        oa << p; return this;
    }

    // TODO what is this?
    ModelSaver* add_deep_lstm_builder(cnn::DeepLSTMBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_cfsm_builder(cnn::ClassFactoredSoftmaxBuilder &p) {
        oa << p; return this;
    }

    ModelSaver* add_sm_builder(cnn::StandardSoftmaxBuilder &p) {
        oa << p; return this;
    }

    void done() { ofs.close(); }


    private:
        std::ofstream ofs;
        boost::archive::text_oarchive oa;

};

struct ModelLoader {
    ModelLoader(std::string filename, cnn::Model *model) :
        ifs(filename),
        ia(ifs)
    {
        printf("loading model\n");
        ia >> *model;
        printf("done %d\n", model->parameters_list().size());
    };

    ModelLoader* fill_parameter(cnn::Parameter &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_lookup_parameter(cnn::LookupParameter &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_lstm_builder(cnn::LSTMBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_srnn_builder(cnn::SimpleRNNBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_hsm_builder(cnn::HierarchicalSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_fast_lstm_builder(cnn::FastLSTMBuilder &p) {
        ia >> p; return this;
    }

    // TODO what is this?
    ModelLoader* fill_deep_lstm_builder(cnn::DeepLSTMBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_cfsm_builder(cnn::ClassFactoredSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    ModelLoader* fill_sm_builder(cnn::StandardSoftmaxBuilder &p) {
        ia >> p; return this;
    }

    void done() { ifs.close(); }

    private:
        std::ifstream ifs;
        boost::archive::text_iarchive ia;

};
}

