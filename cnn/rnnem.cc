#include "cnn/rnnem.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;


namespace cnn {

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

    enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC, WK, WKB, EXTMEM, WBETA, WG };

    RNNEMBuilder::RNNEMBuilder(long layers,
        long input_dim,
        long hidden_dim,
        Model* model) : layers(layers), m_mem_dim(hidden_dim) 
    {
        m_mem_size = RNNEM_MEM_SIZE;

        unsigned mem_dim = m_mem_dim;
        long mem_size = m_mem_size;
        unsigned layer_input_dim = input_dim;
        for (unsigned i = 0; i < layers; ++i) {
            // i
            Parameters* p_x2i = model->add_parameters({ (long)hidden_dim, (long)(layer_input_dim + mem_dim)});
            Parameters* p_h2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_c2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_bi = model->add_parameters({ (long)hidden_dim });

            // o
            Parameters* p_x2o = model->add_parameters({ (long)hidden_dim, (long)(layer_input_dim + mem_dim)});
            Parameters* p_h2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_c2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_bo = model->add_parameters({ (long)hidden_dim });

            // c
            Parameters* p_x2c = model->add_parameters({ (long)hidden_dim, (long)(layer_input_dim + mem_dim)});
            Parameters* p_h2c = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_bc = model->add_parameters({ (long)hidden_dim });

            // for key generation
            Parameters* p_wk = model->add_parameters({ (long)mem_dim, (long)layer_input_dim });
            Parameters* p_wkb = model->add_parameters({ (long)mem_dim });

            // for memory at this layer
            Parameters* p_mem = model->add_parameters({ (long)mem_dim, (long)mem_size });

            // for scaling
            Parameters* p_beta = model->add_parameters({ (long)1, (long)layer_input_dim });

            /// for interploation
            Parameters* p_interpolation = model->add_parameters({ (long)1, (long)layer_input_dim });

            vector<Parameters*> ps = { p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc, p_wk, p_wkb, p_mem, p_beta, p_interpolation };

            layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

            params.push_back(ps);
        }  // layers
    }

    void RNNEMBuilder::new_graph_impl(ComputationGraph& cg){
        param_vars.clear();

        for (unsigned i = 0; i < layers; ++i){
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

            // key
            Expression i_wk = parameter(cg, p[WK]);
            Expression i_wkb = parameter(cg, p[WKB]);

            // memory 
            Expression i_mem = parameter(cg, p[EXTMEM]);

            // memory 
            Expression i_beta = parameter(cg, p[WBETA]);

            // memory 
            Expression i_interpolation = parameter(cg, p[WG]);

            vector<Expression> vars = { i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc, i_wk, i_wkb, i_mem, i_beta, i_interpolation };
            param_vars.push_back(vars);

        }
    }

    // layout: 0..layers = c
    //         layers+1..2*layers = h
    //                           2layers + 1..3*layers  = w
    void RNNEMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
        h.clear();
        c.clear();
        w.clear();
        if (hinit.size() > 0) {
            assert(layers * 3 == hinit.size());
            h0.resize(layers);
            c0.resize(layers);
            w0.resize(layers);
            for (unsigned i = 0; i < layers; ++i) {
                w0[i] = hinit[i];
                c0[i] = hinit[i + layers];
                h0[i] = hinit[i + 2 * layers];
            }
            has_initial_state = true;
        }
        else {
            has_initial_state = false;
        }
    }

    /**
    retrieve content from m_external_memory from an input which is usually the hidden layer activity at time t
    */
    std::vector<Expression> RNNEMBuilder::read_memory(const size_t& t, const Expression & x_t, const size_t layer)
    {
        vector<Expression> ret; 


        Expression i_w_tm1;
        if (t == 0) {
            if (has_initial_state) {
                i_w_tm1 = w0[layer];
            }
        }
        else {  // t > 0
            i_w_tm1 = w[t - 1][layer];
        }

        const vector<Expression>& vars = param_vars[layer];

        Expression M_t = vars[EXTMEM];

        /// do affine transformation to have a key
        Expression key_t = vars[WK] * x_t + vars[WKB];

        Expression raw_weight = transpose(M_t) * key_t;

        Expression beta_t = log(1.0 + exp(vars[WBETA] * x_t));

        Expression v_beta = concatenate(std::vector<Expression>(m_mem_size, beta_t));

        Expression raise_by_beta = cwise_multiply(v_beta , raw_weight);
        Expression i_alpha_t = softmax(raise_by_beta); /// get the weight to each column slice in n x 1

        Expression i_w_t;
        /// interpolation
        if (has_initial_state || t > 0)
        {
            Expression g_t = logistic(vars[WG] * x_t);
            Expression g_v = concatenate(std::vector<Expression>(m_mem_size, g_t));
            Expression f_v = concatenate(std::vector<Expression>(m_mem_size, 1.0 - g_t));
            Expression w_f_v = cwise_multiply(f_v, i_w_tm1);
            i_w_t = w_f_v + cwise_multiply(g_v, i_alpha_t);
        }
        else
        {
            i_w_t = i_alpha_t;
        }

        ret.push_back(i_w_t); /// update memory weight

        Expression retrieved_content = M_t * i_w_t;

        ret.push_back(retrieved_content);

        return ret;
    }

    Expression RNNEMBuilder::add_input_impl(const Expression& x) {
        const unsigned t = h.size();
        h.push_back(vector<Expression>(layers));
        c.push_back(vector<Expression>(layers));
        w.push_back(vector<Expression>(layers));
        vector<Expression>& ht = h.back();
        vector<Expression>& ct = c.back();
        vector<Expression>& wt = w.back();
        Expression in = x;
        for (unsigned i = 0; i < layers; ++i) {
            const vector<Expression>& vars = param_vars[i];
            Expression i_h_tm1, i_c_tm1, i_w_tm1;
            bool has_prev_state = (t > 0 || has_initial_state);
            if (t == 0) {
                if (has_initial_state) {
                    // intial value for h and c at timestep 0 in layer i
                    // defaults to zero matrix input if not set in add_parameter_edges
                    i_h_tm1 = h0[i];
                    i_c_tm1 = c0[i];
                    i_w_tm1 = w0[i];
                }
            }
            else {  // t > 0
                i_h_tm1 = h[t - 1][i];
                i_c_tm1 = c[t - 1][i];
                i_w_tm1 = w[t - 1][i];
            }

            vector<Expression> mem_ret = read_memory(t, in, i);
            Expression mem_wgt = mem_ret[0];
            Expression mem_c_t = mem_ret[1];

            vector<Expression> new_in; 
            new_in.push_back(in);
            new_in.push_back(mem_c_t);
            Expression x_and_past_content = concatenate(new_in);

            // input
            Expression i_ait;
            if (has_prev_state)
                //      i_ait = vars[BI] + vars[X2I] * in + vars[H2I]*i_h_tm1 + vars[C2I] * i_c_tm1;
                i_ait = affine_transform({ vars[BI], vars[X2I], x_and_past_content, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1 });
            else
                //      i_ait = vars[BI] + vars[X2I] * in;
                i_ait = affine_transform({ vars[BI], vars[X2I], x_and_past_content });
            Expression i_it = logistic(i_ait);
            // forget
            Expression i_ft = 1.f - i_it;
            // write memory cell
            Expression i_awt;
            if (has_prev_state)
                //      i_awt = vars[BC] + vars[X2C] * in + vars[H2C]*i_h_tm1;
                i_awt = affine_transform({ vars[BC], vars[X2C], x_and_past_content, vars[H2C], i_h_tm1 });
            else
                //      i_awt = vars[BC] + vars[X2C] * in;
                i_awt = affine_transform({ vars[BC], vars[X2C], x_and_past_content });
            Expression i_wt = tanh(i_awt);
            // output
            if (has_prev_state) {
                Expression i_nwt = cwise_multiply(i_it, i_wt);
                Expression i_crt = cwise_multiply(i_ft, i_c_tm1);
                ct[i] = i_crt + i_nwt;
            }
            else {
                ct[i] = cwise_multiply(i_it, i_wt);
            }

            Expression i_aot;
            if (has_prev_state)
                //      i_aot = vars[BO] + vars[X2O] * in + vars[H2O] * i_h_tm1 + vars[C2O] * ct[i];
                i_aot = affine_transform({ vars[BO], vars[X2O], x_and_past_content, vars[H2O], i_h_tm1, vars[C2O], ct[i] });
            else
                //      i_aot = vars[BO] + vars[X2O] * in;
                i_aot = affine_transform({ vars[BO], vars[X2O], x_and_past_content });
            Expression i_ot = logistic(i_aot);
            Expression ph_t = tanh(ct[i]);
            in = ht[i] = cwise_multiply(i_ot, ph_t);

            wt[i] = mem_wgt;
        }
        return ht.back();
    } // namespace cnn


#undef WTF
#undef KTHXBYE
#undef LOLCAT
}
