#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "expr-xtra.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace cnn {

template <class Builder>
struct AttentionalModel {
    explicit AttentionalModel(Model& model, 
            unsigned layers, unsigned vocab_size_src, unsigned vocab_size_tgt, 
            unsigned hidden_dim, unsigned align_dim, bool rnn_src_embeddings,
	    bool giza_extensions, unsigned hidden_replicates=1, 
            LookupParameters* cs=0, LookupParameters *ct=0);

    Expression BuildGraph(const std::vector<int> &source, const std::vector<int>& target, 
            ComputationGraph& cg, Expression *alignment=0);

    void display(const std::vector<int> &source, const std::vector<int>& target, 
            ComputationGraph& cg, const Expression& alignment, Dict &sd, Dict &td);

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, 
            int beam_width, Dict &tdict);

    std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, Dict &tdict);

    LookupParameters* p_cs;
    LookupParameters* p_ct;
    std::vector<Parameters*> p_h0;
    Parameters* p_R;
    Parameters* p_bias;
    //Parameters* p_M;
    Parameters* p_Wa;
    Parameters* p_Ua;
    Parameters* p_va;
    Parameters* p_Ta;
    Builder builder;
    Builder builder_src_fwd;
    Builder builder_src_bwd;
    bool rnn_src_embeddings;
    bool giza_extensions;

    // statefull functions for incrementally creating computation graph, one
    // target word at a time
    void start_new_instance(const std::vector<int> &src, ComputationGraph &cg);
    Expression add_input(int tgt_tok, int t, ComputationGraph &cg);

    // state variables used in the above two methods
    Expression src;
    Expression i_R;
    Expression i_bias;
    Expression i_Wa;
    Expression i_Ua;
    Expression i_va;
    Expression i_uax;
    std::vector<Expression> i_h0;
    Expression i_Ta;
    Expression i_src_idx;
    Expression i_src_len;
    std::vector<Expression> aligns;
    unsigned slen;
};

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class Builder>
AttentionalModel<Builder>::AttentionalModel(cnn::Model& model,
    unsigned vocab_size_src, unsigned vocab_size_tgt, unsigned layers, unsigned hidden_dim, 
    unsigned align_dim, bool _rnn_src_embeddings, bool _giza_extentions, unsigned hidden_replicates, 
    LookupParameters* cs, LookupParameters *ct)
: builder(layers, (_rnn_src_embeddings) ? 3*hidden_dim : 2*hidden_dim, hidden_dim, &model),
  builder_src_fwd(1, hidden_dim, hidden_dim, &model),
  builder_src_bwd(1, hidden_dim, hidden_dim, &model),
  rnn_src_embeddings(_rnn_src_embeddings), 
  giza_extensions(_giza_extentions)
{
    p_cs = (cs) ? cs : model.add_lookup_parameters(long(vocab_size_src), {long(hidden_dim)}); 
    p_ct = (ct) ? ct : model.add_lookup_parameters(vocab_size_tgt, {long(hidden_dim)}); 
    p_R = model.add_parameters({long(vocab_size_tgt), long(hidden_dim)});
    p_bias = model.add_parameters({long(vocab_size_tgt)});
    for (auto l = 0; l < layers * hidden_replicates; ++l) 
        p_h0.push_back(model.add_parameters({long(hidden_dim)}));
    p_Wa = model.add_parameters({long(align_dim), long(hidden_dim)});
    if (rnn_src_embeddings) {
        p_Ua = model.add_parameters({long(align_dim), 2*long(hidden_dim)});
    } else {
        p_Ua = model.add_parameters({long(align_dim), long(hidden_dim)});
    }
    if (giza_extensions) {
        p_Ta = model.add_parameters({long(align_dim), 9});
    }
    p_va = model.add_parameters({long(align_dim)});
}

template <class Builder>
void AttentionalModel<Builder>::start_new_instance(const std::vector<int> &source, ComputationGraph &cg)
{
    //slen = source.size() - 1; 
    slen = source.size(); 
    if (!rnn_src_embeddings) {
	std::vector<Expression> source_embeddings;
	for (unsigned s = 0; s < slen; ++s) 
	    source_embeddings.push_back(lookup(cg, p_cs, source[s]));
	src = concatenate_cols(source_embeddings); 
    } else {
	std::vector<Expression> source_embeddings;
	// run a RNN backward and forward over the source sentence
	// and stack the top-level hidden states from each model as 
	// the representation at each position
	std::vector<Expression> src_fwd(slen);
	builder_src_fwd.new_graph(cg);
	builder_src_fwd.start_new_sequence();
	for (unsigned i = 0; i < slen; ++i) 
	    src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_cs, source[i]));

	std::vector<Expression> src_bwd(slen);
	builder_src_bwd.new_graph(cg);
	builder_src_bwd.start_new_sequence();
	for (int i = slen; i > 0; --i) {
	    // offset by one position to the right, to catch </s> and generally
	    // not duplicate the w_t already captured in src_fwd[t]
	    src_bwd[i-1] = builder_src_bwd.add_input(lookup(cg, p_cs, source[i]));
	}

	for (unsigned i = 0; i < slen; ++i) 
	    source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
	src = concatenate_cols(source_embeddings); 
	//WTF(src);
    }

    // now for the target sentence
    builder.new_graph(cg); 
    i_h0.clear();
    for (const auto &p: p_h0)
        i_h0.push_back(parameter(cg, p));
    builder.start_new_sequence(i_h0);
    i_R = parameter(cg, p_R); // hidden -> word rep parameter
    i_bias = parameter(cg, p_bias);  // word bias
    i_Wa = parameter(cg, p_Wa); 
    i_Ua = parameter(cg, p_Ua);
    i_va = parameter(cg, p_va);
    //WTF(i_Ua);
    //WTF(src);
    i_uax = i_Ua * src;

    if (giza_extensions) {
	i_Ta = parameter(cg, p_Ta);   
	i_src_idx = arange(cg, 0, slen, true);
	i_src_len = repeat(cg, slen, log(1.0 + slen));
    }

    aligns.clear();
    aligns.push_back(repeat(cg, slen, 0.0f));
}

template <class Builder>
Expression AttentionalModel<Builder>::add_input(int trg_tok, int t, ComputationGraph &cg)
{
    // alignment input -- FIXME: just done for top layer
    auto i_h_tm1 = (t == 0) ? i_h0.back() : builder.final_h().back();
    //WTF(i_h_tm1);
    //Expression i_e_t = tanh(i_src_M * i_h_tm1); 
    Expression i_wah = i_Wa * i_h_tm1;
    //WTF(i_wah);
    // want numpy style broadcasting, but have to do this manually
    Expression i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
    //WTF(i_wah_rep);
    Expression i_e_t;
    if (giza_extensions) {
	std::vector<Expression> alignment_context;
	if (t >= 1) {
	    auto i_aprev = concatenate_cols(aligns);
	    auto i_asum = sum_cols(i_aprev);
	    auto i_asum_pm = dither(cg, i_asum);
	    //WTF(i_asum_pm);
	    alignment_context.push_back(i_asum_pm);
	    auto i_alast_pm = dither(cg, aligns.back());
	    //WTF(i_alast_pm);
	    alignment_context.push_back(i_alast_pm);
	} else {
	    // just 6 repeats of the 0 vector
	    auto zeros = repeat(cg, slen, 0);
	    //WTF(zeros);
	    alignment_context.push_back(zeros); 
	    alignment_context.push_back(zeros);
	    alignment_context.push_back(zeros);
	    alignment_context.push_back(zeros);
	    alignment_context.push_back(zeros);
	    alignment_context.push_back(zeros);
	}
	//WTF(i_src_idx);
	alignment_context.push_back(i_src_idx);
	//WTF(i_src_len);
	alignment_context.push_back(i_src_len);
	auto i_tgt_idx = repeat(cg, slen, log(1.0 + t));
	//WTF(i_tgt_idx);
	alignment_context.push_back(i_tgt_idx);
	auto i_context = concatenate_cols(alignment_context);
	//WTF(i_context);

	auto i_e_t_input = i_wah_rep + i_uax + i_Ta * transpose(i_context); 
	//WTF(i_e_t_input);
	i_e_t = transpose(tanh(i_e_t_input)) * i_va;
	//WTF(i_e_t);
    } else {
	i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
	//WTF(i_e_t);
    }
    Expression i_alpha_t = softmax(i_e_t);
    //WTF(i_alpha_t);
    aligns.push_back(i_alpha_t);
    Expression i_c_t = src * i_alpha_t; 
    //WTF(i_c_t);
    // word input
    Expression i_x_t = lookup(cg, p_ct, trg_tok);
    //WTF(i_x_t);
    Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); // vstack/hstack?
    //WTF(input);
    // y_t = RNN([x_t, a_t])
    Expression i_y_t = builder.add_input(input);
    //WTF(i_y_t);
    Expression i_r_t = i_bias + i_R * i_y_t;
    //WTF(i_r_t);

    return i_r_t;
}

template <class Builder>
Expression AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
        const std::vector<int>& target, ComputationGraph& cg, Expression *alignment) 
{
    //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
    start_new_instance(source, cg);

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1; 
    for (unsigned t = 0; t < tlen; ++t) {
        Expression i_r_t = add_input(target[t], t, cg);
	//WTF(i_r_t);
        Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
        errs.push_back(i_err);
    }
    // save the alignment for later
    if (alignment != 0) {
	// pop off the last alignment column
        *alignment = concatenate_cols(aligns);
    }

    Expression i_nerr = sum(errs);
    return i_nerr;
}

template <class Builder>
void 
AttentionalModel<Builder>::display(const std::vector<int> &source, const std::vector<int>& target, 
                          ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td)
{
    using namespace std;

    // display the alignment
    //float I = target.size() - 1;
    //float J = source.size() - 1;
    float I = target.size();
    float J = source.size();
    //vector<string> symbols{"\u2588","\u2589","\u258A","\u258B","\u258C","\u258D","\u258E","\u258F"};
    vector<string> symbols{".","o","*","O","@"};
    int num_symbols = symbols.size();
    vector<float> thresholds;
    thresholds.push_back(0.8/I);
    float lgap = (0 - log(thresholds.back())) / (num_symbols - 1);
    for (auto rit = symbols.begin(); rit != symbols.end(); ++rit) {
        float thr = exp(log(thresholds.back()) + lgap);
        thresholds.push_back(thr);
    }
    // FIXME: thresholds > 1, what's going on?
    //cout << thresholds.back() << endl;

    const Tensor &a = cg.get_value(alignment.i);
    //WTF(alignment);
    //cout << "I = " << I << " J = " << J << endl;

    cout.setf(ios_base::adjustfield, ios_base::left);
    cout << setw(12) << "source" << "  ";
    cout.setf(ios_base::adjustfield, ios_base::right);
    for (int j = 0; j < J; ++j) 
        cout << setw(2) << j << ' ';
    cout << endl;

    for (int i = 0; i < I; ++i) {
        cout.setf(ios_base::adjustfield, ios_base::left);
        //cout << setw(12) << td.Convert(target[i+1]) << "  ";
        cout << setw(12) << td.Convert(target[i]) << "  ";
        cout.setf(ios_base::adjustfield, ios_base::right);
        float max_v = 0;
        int max_j = -1;
        for (int j = 0; j < J; ++j) {
            float v = TensorTools::AccessElement(a, Dim(j, i));
            string symbol;
            for (int s = 0; s <= num_symbols; ++s) {
                if (s == 0) 
                    symbol = ' ';
                else
                    symbol = symbols[s-1];
                if (s != num_symbols && v < thresholds[s])
                    break;
            }
            cout << setw(2) << symbol << ' ';
            if (v >= max_v) {
                max_v = v;
                max_j = j;
            }
        }
        cout << setw(20) << "max Pr=" << setprecision(3) << setw(5) << max_v << " @ " << max_j << endl;
    }
    cout << resetiosflags(ios_base::adjustfield);
    for (int j = 0; j < J; ++j) 
        cout << j << ":" << sd.Convert(source[j]) << ' ';
    cout << endl;
}

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
{
    assert(beam_width == 1); // beam search not implemented 
    const int sos_sym = tdict.Convert("<s>");
    const int eos_sym = tdict.Convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym); 

    std::cerr << tdict.Convert(target.back());
    int t = 0;
    start_new_instance(source, cg);
    while (target.back() != eos_sym) 
    {
        Expression i_scores = add_input(target.back(), t, cg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 100) {
            w = eos_sym;
            pr_w = dist[w];
        }

        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }
    std::cerr << std::endl;

    return target;
}

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict &tdict)
{
    const int sos_sym = tdict.Convert("<s>");
    const int eos_sym = tdict.Convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym); 

    std::cerr << tdict.Convert(target.back());
    int t = 0;
    start_new_instance(source, cg);
    while (target.back() != eos_sym) 
    {
        Expression i_scores = add_input(target.back(), t, cg);
        Expression ydist = softmax(i_scores);

	// in rnnlm.cc there's a loop around this block -- why? can incremental_forward fail?
        auto dist = as_vector(cg.incremental_forward());
	double p = rand01();
        unsigned w = 0;
        for (; w < dist.size(); ++w) {
	    p -= dist[w];
	    if (p < 0) break;
        }
	// this shouldn't happen
	if (w == dist.size()) w = eos_sym;

        std::cerr << " " << tdict.Convert(w) << " [p=" << dist[w] << "]";
        t += 1;
        target.push_back(w);
    }
    std::cerr << std::endl;

    return target;
}


#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace cnn
