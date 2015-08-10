#include "rnnem.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <cnn/grad-check.h>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

inline bool is_close(float a, float b) {
    /// to-do use CNTK's isclose function
    return (fabs(a - b) < 1e-7);
}

namespace cnn {

    enum { X2I, H2I, C2I, BI, X2F, H2F, C2F, BF, X2O, H2O, C2O, BO, X2C, H2C, BC, WK, WKB, EXTMEM, WG, WA, WV, H2E, BE, WO, OB, WGAMMA, BGAMMA, WGAMMAWTM1, IMPORTB, WOXI, WEXI, BXI, 
        W0,
        MEM_INIT
        /// for initial memory weight and memory content
    };

    NMNBuilder::NMNBuilder(long layers,
        long input_dim,
        long hidden_dim,
        Model* model) : layers(layers), m_mem_dim(hidden_dim), m_align_dim(RNNEM_ALIGN_DIM)
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

            // f
            Parameters* p_x2f = model->add_parameters({ (long)hidden_dim, (long)(layer_input_dim + mem_dim) });
            Parameters* p_h2f = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_c2f = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
            Parameters* p_bf = model->add_parameters({ (long)hidden_dim });

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
            Parameters* p_wk = model->add_parameters({ (long)m_align_dim, (long)layer_input_dim });
            Parameters* p_wkb = model->add_parameters({ (long)m_align_dim});

            // for memory at this layer
            Parameters* p_mem = model->add_parameters({ (long)mem_dim, (long)mem_size });
            Parameters* p_mem_w0 = model->add_parameters({ (long)mem_size });  /// for initial memory weight
            p_mem_w0->reset_to_zero();
            Parameters* p_mem_M0= model->add_parameters({ (long)mem_dim, (long)mem_size }); /// for initial memory content
            p_mem_M0->reset_to_zero();

            // for memory matching
            Parameters* p_wa = model->add_parameters({ (long)m_align_dim, (long)m_mem_dim });
            Parameters* p_wv = model->add_parameters({ (long)1, (long)m_align_dim });
            Parameters* p_gamma = model->add_parameters({ (long)1, (long)layer_input_dim });
            Parameters* p_gamma_b = model->add_parameters({ (long)1 });
            Parameters* p_gamma_wtm1 = model->add_parameters({ (long)1, (long)m_mem_size });

            // for memory update
            Parameters* p_h2e = model->add_parameters({ (long)m_mem_dim, (long)layer_input_dim });
            Parameters* p_be = model->add_parameters({ (long)m_mem_dim });
            Parameters* p_wo = model->add_parameters({ (long)m_mem_dim, (long)layer_input_dim });
            Parameters* p_ob = model->add_parameters({ (long)m_mem_dim });

            /// for interploation
            Parameters* p_interpolation = model->add_parameters({ (long)1, (long)layer_input_dim });

            /// for importance factor
            Parameters* p_important_b = model->add_parameters({ (long)1, (long)m_mem_dim });
            Parameters* p_woxi = model->add_parameters({ (long)m_mem_dim, (long)m_mem_dim });
            Parameters* p_wexi = model->add_parameters({ (long)m_mem_dim, (long)m_mem_dim });
            Parameters* p_bxi = model->add_parameters({ (long)m_mem_dim });
            vector<Parameters*> ps = { p_x2i, p_h2i, p_c2i, p_bi, p_x2f, p_h2f, p_c2f, p_bf, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc, p_wk, p_wkb, p_mem, p_interpolation, p_wa, p_wv, p_h2e, p_be, p_wo, p_ob, p_gamma, p_gamma_b, p_gamma_wtm1, p_important_b, p_woxi, p_wexi, p_bxi, p_mem_w0, p_mem_M0 };

            layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

            params.push_back(ps);
        }  // layers
    }

    void NMNBuilder::new_graph_impl(ComputationGraph& cg){
        param_vars.clear();

        for (unsigned i = 0; i < layers; ++i){
            auto& p = params[i];

            //i
            Expression i_x2i = parameter(cg, p[X2I]);
            Expression i_h2i = parameter(cg, p[H2I]);
            Expression i_c2i = parameter(cg, p[C2I]);
            Expression i_bi = parameter(cg, p[BI]);
            //f
            Expression i_x2f = parameter(cg, p[X2F]);
            Expression i_h2f = parameter(cg, p[H2F]);
            Expression i_c2f = parameter(cg, p[C2F]);
            Expression i_bf = parameter(cg, p[BF]);
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
            Expression i_mem_w0 = parameter(cg, p[W0]);
            Expression i_mem_M0 = parameter(cg, p[MEM_INIT]);

            // memory 
            Expression i_wa = parameter(cg, p[WA]);
            Expression i_wv = parameter(cg, p[WV]);
            Expression i_gamma = parameter(cg, p[WGAMMA]);
            Expression i_gamma_b = parameter(cg, p[BGAMMA]);
            Expression i_gamma_wtm1 = parameter(cg, p[WGAMMAWTM1]);

            // memory upate
            Expression i_h2e = parameter(cg, p[H2E]);
            Expression i_be = parameter(cg, p[BE]);
            Expression i_wo = parameter(cg, p[WO]);
            Expression i_ob = parameter(cg, p[OB]);

            // memory 
            Expression i_interpolation = parameter(cg, p[WG]);

            Expression i_important_b = parameter(cg, p[IMPORTB]);
            Expression i_woxi = parameter(cg, p[WOXI]);
            Expression i_wexi = parameter(cg, p[WEXI]);
            Expression i_bxi = parameter(cg, p[BXI]);
            vector<Expression> vars = { i_x2i, i_h2i, i_c2i, i_bi, i_x2f, i_h2f, i_c2f, i_bf, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc, i_wk, i_wkb, i_mem, i_interpolation, i_wa, i_wv, i_h2e, i_be, i_wo, i_ob, i_gamma, i_gamma_b, i_gamma_wtm1, i_important_b, i_woxi, i_wexi, i_bxi ,
                i_mem_w0, i_mem_M0
            };
            param_vars.push_back(vars);

        }
    }

    void NMNBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
        h.clear();
        c.clear();
        w.clear();
        M.clear();

        if (hinit.size() > 0) {
            assert(layers * 4 == hinit.size());
            h0.resize(layers);
            c0.resize(layers);
            w0.resize(layers);
            M0.resize(layers);
            for (unsigned i = 0; i < layers; ++i) {
                w0[i] = hinit[i];
                M0[i] = hinit[i+layers];
                c0[i] = hinit[i + 2*layers];
                h0[i] = hinit[i + 3 * layers];
            }
            has_initial_state = true;
        }
        else {
            has_initial_state = false;
        }
    }

    /**
    retrieve content from m_external_memory from an input which is usually the hidden layer activity at time t
    @x_t : should from target, such as next word in the target language or the next word for language modeling
    */
    std::vector<Expression> NMNBuilder::read_memory(const int& t, const Expression & x_t, const size_t layer)
    {
        const vector<Expression>& vars = param_vars[layer];

        vector<Expression> ret;

        Expression M_tm1; 
        Expression i_w_tm1;
        if (t <= 0) {
            if (has_initial_state)
            {
                i_w_tm1 = w0[layer];
                M_tm1 = M0[layer];
            }
            else
            {
                i_w_tm1 = vars[W0];
                M_tm1 = vars[MEM_INIT];
            }
        }
        else {  // t > 0
            i_w_tm1 = w[t - 1][layer];
            M_tm1 = M[t - 1][layer];
        }

        /// do affine transformation to have a key
        Expression key_t = tanh(vars[WK] * x_t + vars[WKB]);

        vector<Expression> key_t_to_compare_all_mem = std::vector<Expression>(m_mem_size, key_t);
        Expression combine_with_memory;
        if (has_initial_state || t > 0)
        {
            combine_with_memory = vars[WA] * M_tm1 + concatenate_cols(key_t_to_compare_all_mem);
        }
        else
        {
            combine_with_memory = concatenate_cols(key_t_to_compare_all_mem);
        }
        Expression raw_weight = vars[WV] * combine_with_memory;
        Expression raised_coef = log(1.0 + exp(vars[WGAMMA] * x_t + vars[BGAMMA] + vars[WGAMMAWTM1] * i_w_tm1));
        Expression w_tilde = softmax(transpose(raw_weight) * raised_coef);

        Expression i_w_t;
        Expression retrieved_content;
        /// interpolation
        Expression g_t = logistic(vars[WG] * x_t);
        Expression g_v = concatenate(std::vector<Expression>(m_mem_size, g_t));
        Expression f_v = concatenate(std::vector<Expression>(m_mem_size, 1.0 - g_t));
        Expression w_f_v = cwise_multiply(f_v, i_w_tm1);
        i_w_t = w_f_v + cwise_multiply(g_v, w_tilde);
        retrieved_content = M_tm1 * i_w_t;

        ret.push_back(i_w_t); /// update memory weight
        ret.push_back(retrieved_content);

        return ret;
    }

    /**
    update memory 
    */
    vector<Expression> NMNBuilder::update_memory(const int& t, const Expression & x_t, 
        const size_t layer,
        vector<Expression>& Mt)
    {
        vector<Expression> ret;
        bool has_prev_state = (t > 0 || has_initial_state);

        const vector<Expression>& vars = param_vars[layer];

        Expression M_tm1;
        vector<Expression> vw = final_w();
        Expression i_w_t = vw[layer];

        if (t <= 0) {
            if (has_initial_state)
                M_tm1 = M0[layer];
            else
                M_tm1 = vars[MEM_INIT];
        }
        else {  // t > 0
            M_tm1 = M[t - 1][layer];
        }

        // erase vector
        Expression i_ait;
        i_ait = affine_transform({ vars[BE], vars[H2E], x_t });
        Expression i_e_t = logistic(i_ait);  /// erase vector
        Expression new_cnt_t = tanh(vars[WO] * x_t + vars[OB]);

        vector<Expression> v_new_cnt, v_try_to_erase;
        for (auto k = 0; k < m_mem_size; k++)
        {
            Expression p_w_i = pick(i_w_t, k);
            vector<Expression> v_pw = vector<Expression>(m_mem_dim, p_w_i);
            Expression v_pw_i = concatenate(v_pw);

            Expression try_to_erase = 1.0 - cwise_multiply(v_pw_i ,  i_e_t); 
            v_try_to_erase.push_back(try_to_erase);

            Expression try_to_add = cwise_multiply(v_pw_i, new_cnt_t);
            v_new_cnt.push_back(try_to_add);
        }

        Expression M_tilde = cwise_multiply(M_tm1, concatenate_cols(v_try_to_erase));
        Expression M_to_update = concatenate_cols(v_new_cnt); 
        
        Mt[layer] = M_tilde + M_to_update; 

        ret.push_back(i_e_t);
        ret.push_back(new_cnt_t);

        return ret;
    }

    /**
    whether uses past history
    */
    Expression NMNBuilder::compute_importance_factor(const Expression & r_t, /// retrieved memory content 
        const Expression & e_t, /// erase vector
        const Expression & o_t , /// new content vector
        const size_t layer)
    {
        const vector<Expression>& vars = param_vars[layer];

        /// importance factor
        Expression imp_factor = logistic(vars[IMPORTB] * tanh(vars[WOXI] * o_t +vars[WEXI] * e_t + vars[BXI]));
        Expression retrieved_content = r_t * imp_factor;

        return retrieved_content; 
    }

    Expression NMNBuilder::add_input_impl(int prev, const Expression& x) {
        int t = prev; 
        h.push_back(vector<Expression>(layers));
        c.push_back(vector<Expression>(layers));
        w.push_back(vector<Expression>(layers));
        M.push_back(vector<Expression>(layers));
        vector<Expression>& ht = h.back();
        vector<Expression>& ct = c.back();
        vector<Expression>& wt = w.back();
        vector<Expression>& Mt = M.back();
        
        Expression in = x;
        for (unsigned i = 0; i < layers; ++i) {
            const vector<Expression>& vars = param_vars[i];
            Expression i_h_tm1, i_c_tm1, i_w_tm1, i_M_tm1;
            bool has_prev_state = (t > 0 || has_initial_state);
            if (t <= 0) {
                if (has_initial_state) {
                    // intial value for h and c at timestep 0 in layer i
                    // defaults to zero matrix input if not set in add_parameter_edges
                    i_h_tm1 = h0[i];
                    i_c_tm1 = c0[i];
                    i_w_tm1 = w0[i];
                    i_M_tm1 = M0[i];
                }
                else
                {
                    i_w_tm1 = vars[W0];
                    i_M_tm1 = vars[MEM_INIT];
                }
            }
            else {  // t > 0
                i_h_tm1 = h[t - 1][i];
                i_c_tm1 = c[t - 1][i];
                i_w_tm1 = w[t - 1][i];
                i_M_tm1 = M[t - 1][i];
            }

            vector<Expression> mem_ret = read_memory(t, in, i);
            Expression mem_wgt = mem_ret[0];
            Expression mem_c_t = mem_ret[1];

            wt[i] = mem_wgt;  /// update memory weight

            vector<Expression> v_e_o = update_memory(t, in, i, Mt);

            Expression modulated_content = compute_importance_factor(mem_ret[1], v_e_o[0], v_e_o[1], i);

            vector<Expression> new_in;
            new_in.push_back(in);
            new_in.push_back(modulated_content);
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
            // input
            Expression i_aft;
            if (has_prev_state)
                //      i_ait = vars[BI] + vars[X2I] * in + vars[H2I]*i_h_tm1 + vars[C2I] * i_c_tm1;
                i_aft = affine_transform({ vars[BF], vars[X2F], x_and_past_content, vars[H2F], i_h_tm1, vars[C2F], i_c_tm1 });
            else
                //      i_ait = vars[BI] + vars[X2I] * in;
                i_aft = affine_transform({ vars[BF], vars[X2F], x_and_past_content });
            Expression i_ft = logistic(i_aft);

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
            Expression i_nwt = cwise_multiply(i_it, i_wt);
            if (has_prev_state) {
                Expression i_crt = cwise_multiply(i_ft, i_c_tm1);
                ct[i] = i_crt + i_nwt;
            }
            else {
                ct[i] = i_nwt;
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
        }
        return ht.back();
    } // namespace cnn


}
