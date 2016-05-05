/* Header file for tree LSTM as defined under the Dynamic TreeLSTM project
 * mytreelstm.h
 *
 *  Created on: Apr 14, 2016
 *      Author: swabha
 */

#ifndef CNN_TREELSTM_H_
#define CNN_TREELSTM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

class Model;

struct TheirTreeLSTMBuilder: public RNNBuilder {

    TheirTreeLSTMBuilder() = default;

    explicit TheirTreeLSTMBuilder(unsigned layers, unsigned input_dim,
            unsigned hidden_dim, Model* model);

    void set_dropout(float d) {
        dropout_rate = d;
    }

    // in general, you should disable dropout at test time
    void disable_dropout() {
        dropout_rate = 0;
    }

    Expression back() const override {
        return (cur == -1 ? h0.back() : h[cur].back());
    }

    vector<Expression> final_h() const override {
        return (h.size() == 0 ? h0 : h.back());
    }

    vector<Expression> final_s() const override {
        vector < Expression > ret = (c.size() == 0 ? c0 : c.back());
        for (auto my_h : final_h())
            ret.push_back(my_h);
        return ret;
    }

    unsigned num_h0_components() const override {
        return 2 * layers;
    }

    vector<Expression> get_h(RNNPointer i) const override {
        return (i == -1 ? h0 : h[i]);
    }

    vector<Expression> get_s(RNNPointer i) const override {
        vector < Expression > ret = (i == -1 ? c0 : c[i]);
        for (auto my_h : get_h(i))
            ret.push_back(my_h);
        return ret;
    }

    void copy(const RNNBuilder & params) override;

    Expression add_input(int idx, vector<unsigned> children,
            const Expression& x);

    void initialize_structure(unsigned sent_len);

protected:
    void new_graph_impl(ComputationGraph& cg) override;
    void start_new_sequence_impl(const vector<Expression>& h0) override;
    Expression add_input_impl(int idx, const Expression& x) override;

public:
    // first index is layer, then ...
    vector<vector<Parameters*>> params;

    // first index is layer, then ...
    vector<vector<Expression>> param_vars;

    // first index is time, second is layer
    vector<vector<Expression>> h, c;

    // initial values of h and c at each layer
    // - both default to zero matrix input
    bool has_initial_state; // if this is false, treat h0 and c0 as 0
    vector<Expression> h0;
    vector<Expression> c0;
    unsigned layers;
    float dropout_rate;
};

} // namespace cnn

#endif
