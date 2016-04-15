/*
 * mytreelstm.cc
 *
 *  Created on: Apr 14, 2016
 *      Author: swabha
 */

#include "cnn/peepholetreelstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum {
    X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC
};

TreeLSTMBuilder::TreeLSTMBuilder(unsigned layers, unsigned input_dim,
        unsigned hidden_dim, unsigned sent_len, Model* model) :
        layers(layers) {

    unsigned layer_input_dim = input_dim;
    for (unsigned i = 0; i < sent_len; i++) {
        h.push_back(vector < Expression > (layers));
        c.push_back(vector < Expression > (layers));
    }

    for (unsigned i = 0; i < layers; ++i) {
        // i
        Parameters* p_x2i = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2i = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_c2i = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bi = model->add_parameters( { hidden_dim });

        // o
        Parameters* p_x2o = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2o = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_c2o = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bo = model->add_parameters( { hidden_dim });

        // c
        Parameters* p_x2c = model->add_parameters(
                { hidden_dim, layer_input_dim });
        Parameters* p_h2c = model->add_parameters( { hidden_dim, hidden_dim });
        Parameters* p_bc = model->add_parameters( { hidden_dim });
        layer_input_dim = hidden_dim; // output (hidden) from 1st layer is input to next

        vector<Parameters*> ps = { p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o,
                p_c2o, p_bo, p_x2c, p_h2c, p_bc };
        params.push_back(ps);
    }  // layers
    dropout_rate = 0.0f;
}

void TreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {
    param_vars.clear();

    for (unsigned i = 0; i < layers; ++i) {
        auto& p = params[i];

        //i
        Expression i_x2i = parameter(cg, p[X2I]);
        Expression i_h2i = parameter(cg, p[H2I]);
        Expression i_c2i = parameter(cg, p[C2I]);
        Expression i_bi = parameter(cg, p[BI]);
        //o
        Expression i_x2o = parameter(cg, p[X2O]);
        Expression i_h2o = parameter(cg, p[H2O]);
        Expression i_c2o = parameter(cg, p[C2O]);
        Expression i_bo = parameter(cg, p[BO]);
        //c
        Expression i_x2c = parameter(cg, p[X2C]);
        Expression i_h2c = parameter(cg, p[H2C]);
        Expression i_bc = parameter(cg, p[BC]);

        vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
        param_vars.push_back(vars);
    }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void TreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
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

Expression TreeLSTMBuilder::add_input_impl(int idx, vector<int> children,
        const Expression& x) {
//    h.push_back(vector < Expression > (layers)); In the header now !!
//    c.push_back(vector < Expression > (layers)); In the header now !!

    vector < Expression > &ht = h[idx]; // ht is a vector of size layers
    vector < Expression > &ct = c[idx]; // ct is a vector of size layers

    Expression in = x;
    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];
        vector<Expression> i_h_k, i_c_k; // hidden layer and cell state of children

        bool has_children = (children.size() > 0);
        if (has_children == false) { // leaf node
            if (has_initial_state) {
                // intial value for h and c at timestep 0 in layer i
                // defaults to zero matrix input if not set in add_parameter_edges
                i_h_k.push_back(h0[i]);
                i_c_k.push_back(c0[i]);
            }
        } else {  // parent node
            for (int k : children) {
                i_h_k.push_back(h[k][i]);
                i_c_k.push_back(c[k][i]);
            }
        }

        Expression i_h_k_sum = i_h_k[0]; // TODO: what happens if this is empty?
        for (unsigned k = 1; k < children.size(); k++) {
            i_h_k_sum = i_h_k_sum + i_h_k[k];
        }

        // apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
        if (dropout_rate) {
            in = dropout(in, dropout_rate);
        }
        // input
        vector < Expression > i_aitk;
        if (has_children) {
            for (unsigned k = 0; k < children.size(); k++) {
                i_aitk.push_back(affine_transform( { vars[BI], vars[X2I], in,
                        vars[H2I], i_h_k[k], vars[C2I], i_c_k[k] }));
            }
        } else {
            i_aitk.push_back(affine_transform( { vars[BI], vars[X2I], in }));
        }

        vector<Expression> i_itk, i_ftk;
        Expression i_itk_sum;
        for (unsigned k = 0; k < children.size(); k++) {
            i_itk.push_back(logistic(i_aitk[k]));
            if (k == 0) {
                i_itk_sum = i_itk[k];
            } else {
                i_itk_sum = i_itk_sum + i_itk[k];
            }
            // forget
            i_ftk.push_back(1.f - i_itk[k]);
        }

        // write memory cell
        Expression i_awt;
        if (has_children) {
            i_awt = affine_transform( { vars[BC], vars[X2C], in, vars[H2C],
                    i_h_k_sum });
        } else {
            i_awt = affine_transform( { vars[BC], vars[X2C], in });
        }
        Expression i_wt = tanh(i_awt);

        // output
        if (has_children) {
            Expression i_crtk = cwise_multiply(i_ftk[0], i_c_k[0]);
            for (unsigned k = 1; k < children.size(); k++) {
                i_crtk = i_crtk + cwise_multiply(i_ftk[k], i_c_k[k]);
            }
            Expression i_nwt = cwise_multiply(i_itk_sum, i_wt);
            ct[i] = i_crtk + i_nwt;
        } else {
            ct[i] = cwise_multiply(i_itk[0], i_wt);
        }

        Expression i_aot;
        if (has_children) {
            i_aot = affine_transform( { vars[BO], vars[X2O], in, vars[H2O],
                    i_h_k_sum, vars[C2O], ct[i] });
        } else {
            i_aot = affine_transform( { vars[BO], vars[X2O], in, vars[C2O],
                    ct[i] });
        }

        Expression i_ot = logistic(i_aot);
        Expression ph_t = tanh(ct[i]);
        in = ht[i] = cwise_multiply(i_ot, ph_t);
    }
    if (dropout_rate) {
        return dropout(ht.back(), dropout_rate);
    } else {
        return ht.back();
    }
}

void TreeLSTMBuilder::copy(const RNNBuilder & rnn) {
    const TreeLSTMBuilder & rnn_treelstm = (const TreeLSTMBuilder&) rnn;
    assert(params.size() == rnn_treelstm.params.size());
    for (size_t i = 0; i < params.size(); ++i)
        for (size_t j = 0; j < params[i].size(); ++j)
            params[i][j]->copy(*rnn_treelstm.params[i][j]);
}

} // namespace cnn

