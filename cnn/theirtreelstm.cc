/*
 * mytreelstm.cc
 *
 *  Created on: Apr 14, 2016
 *      Author: swabha
 *      TODO: add attention to the sum of hidden state
 */

#include "cnn/theirtreelstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum {
    X2I, H2I, BI, X2F, H2F, BF, X2O, H2O, BO, X2C, H2C, BC
};

TheirTreeLSTMBuilder::TheirTreeLSTMBuilder(unsigned layers, unsigned input_dim,
        unsigned hidden_dim, Model* model) :
        layers(layers) {

    unsigned layer_input_dim = input_dim;

    for (unsigned i = 0; i < layers; ++i) {
        // i
        Parameters* p_x2i = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2i = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bi = model->add_parameters( { hidden_dim });

        //f
        Parameters* p_x2f = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2f = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bf = model->add_parameters( { hidden_dim });

        // o
        Parameters* p_x2o = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2o = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bo = model->add_parameters( { hidden_dim });

        // c
        Parameters* p_x2c = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2c = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bc = model->add_parameters( { hidden_dim });
        layer_input_dim = hidden_dim; // output (hidden) from 1st layer is input to next

        vector<Parameters*> ps = { p_x2i, p_h2i, p_bi, p_x2f, p_h2f, p_bf,
                p_x2o, p_h2o, p_bo, p_x2c, p_h2c, p_bc };
        params.push_back(ps);
    }  // layers
    dropout_rate = 0.0f;
}

void TheirTreeLSTMBuilder::initialize_structure(unsigned sent_len) {
    for (unsigned i = 0; i < sent_len; i++) {
        h.push_back(vector < Expression > (layers));
        c.push_back(vector < Expression > (layers));
    }
}

void TheirTreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {
    param_vars.clear();

    for (unsigned i = 0; i < layers; ++i) {
        auto& p = params[i];

        //i
        Expression i_x2i = parameter(cg, p[X2I]);
        Expression i_h2i = parameter(cg, p[H2I]);
        Expression i_bi = parameter(cg, p[BI]);
        //f
        Expression i_x2f = parameter(cg, p[X2F]);
        Expression i_h2f = parameter(cg, p[H2F]);
        Expression i_bf = parameter(cg, p[BF]);
        //o
        Expression i_x2o = parameter(cg, p[X2O]);
        Expression i_h2o = parameter(cg, p[H2O]);
        Expression i_bo = parameter(cg, p[BO]);
        //c
        Expression i_x2c = parameter(cg, p[X2C]);
        Expression i_h2c = parameter(cg, p[H2C]);
        Expression i_bc = parameter(cg, p[BC]);

        vector<Expression> vars = {i_x2i, i_h2i, i_bi, i_x2f, i_h2f, i_bf, i_x2o, i_h2o, i_bo, i_x2c, i_h2c, i_bc};
        param_vars.push_back(vars);
    }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void TheirTreeLSTMBuilder::start_new_sequence_impl(
        const vector<Expression>& hinit) {
    h.clear();
    c.clear();
    if (hinit.size() > 0) {
        assert(layers * 2 == hinit.size());
        h0.resize(layers);
        c0.resize(layers);
        for (unsigned i = 0; i < layers; ++i) {
            c0[i] = hinit[i];
            h0[i] = hinit[i + layers];
        }
        has_initial_state = true;
    } else {
        has_initial_state = false;
    }
}

Expression TheirTreeLSTMBuilder::add_input(int idx, vector<unsigned> children,
        const Expression& x) {
    //    h.push_back(vector < Expression > (layers)); In the header now !!
    //    c.push_back(vector < Expression > (layers)); In the header now !!

    vector < Expression > &ht = h[idx]; // ht is a vector of size layers
    vector < Expression > &ct = c[idx]; // ct is a vector of size layers

    Expression in = x;
    for (unsigned layer = 0; layer < layers; ++layer) {
        const vector<Expression>& vars = param_vars[layer];
        vector<Expression> i_h_k, i_c_k; // hidden layer and cell state of children

        bool has_children = (children.size() > 0);
        if (has_children == false) { // leaf node
            if (has_initial_state) {
                // intial value for h and c at timestep 0 in layer i
                // defaults to zero matrix input if not set in add_parameter_edges
                i_h_k.push_back(h0[layer]);
                i_c_k.push_back(c0[layer]);
                // TODO: how to initialize non-leaf nodes?
            } else {
                Expression i_h_leaf, i_c_leaf;
                i_h_k.push_back(i_h_leaf);
                i_c_k.push_back(i_c_leaf); // let's just hope for the best...
            }
        } else {  // parent node
            for (int k : children) {
                i_h_k.push_back(h[k][layer]);
                i_c_k.push_back(c[k][layer]);
            }
        }

        Expression i_h_k_sum;
        if (has_children) {
            i_h_k_sum = i_h_k[0];
            for (unsigned k = 1; k < children.size(); k++) {
                i_h_k_sum = i_h_k_sum + i_h_k[k];
            }
        } // else not used

        // apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
        if (dropout_rate) {
            in = dropout(in, dropout_rate);
        }

        // input
        Expression i_at;
        if (has_children) {
            i_at = affine_transform( { vars[BI], vars[X2I], in, vars[H2I],
                    i_h_k_sum });
        } else {
            i_at = affine_transform( { vars[BI], vars[X2I], in });
        }

        Expression i_it = logistic(i_at);

        // forget
        vector < Expression > i_ftk;
        if (has_children) {
            for (unsigned k = 0; k < children.size(); k++) {
                Expression i_aftk = affine_transform( { vars[BF], vars[X2F], in,
                        vars[H2F], i_h_k[k] });
                i_ftk.push_back(logistic(i_aftk));
            }
        } // else not used

        // memory cell
        Expression i_act;
        if (has_children) {
            i_act = affine_transform( { vars[BC], vars[X2C], in, vars[H2C],
                    i_h_k_sum });
        } else {
            i_act = affine_transform( { vars[BC], vars[X2C], in });
        }

        Expression i_tct = tanh(i_act);
        Expression i_cit = cwise_multiply(i_it, i_tct);

        if (has_children) {
            Expression i_cft = cwise_multiply(i_ftk[0], i_c_k[0]);
            for (unsigned k = 1; k < children.size(); k++) {
                i_cft = i_cft + cwise_multiply(i_ftk[k], i_c_k[k]);
            }
            ct[layer] = i_cft + i_cit;
        } else {
            ct[layer] = i_cit;
        }

        // output
        Expression i_aot;
        if (has_children) {
            i_aot = affine_transform( { vars[BO], vars[X2O], in, vars[H2O],
                    i_h_k_sum });
        } else {
            i_aot = affine_transform( { vars[BO], vars[X2O], in });
        }

        Expression i_ot = logistic(i_aot);
        Expression ph_t = tanh(ct[layer]);

        in = ht[layer] = cwise_multiply(i_ot, ph_t);
        if (dropout_rate) {
            ht[layer] = dropout(ht[layer], dropout_rate); // TODO: not sure is correct
        }
    }
    return ht.back();
}

Expression TheirTreeLSTMBuilder::add_input_impl(int prev, const Expression& x) {
    assert(false);
    return x;
}

void TheirTreeLSTMBuilder::copy(const RNNBuilder & rnn) {
    const TheirTreeLSTMBuilder & rnn_treelstm =
            (const TheirTreeLSTMBuilder&) rnn;
    assert(params.size() == rnn_treelstm.params.size());
    for (size_t i = 0; i < params.size(); ++i)
        for (size_t j = 0; j < params[i].size(); ++j)
            params[i][j]->copy(*rnn_treelstm.params[i][j]);
}

} // namespace cnn

