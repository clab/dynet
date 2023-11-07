/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#pragma once

// DyNet
#include "dynet/globals.h"
#include "dynet/nodes.h"
#include "dynet/param-init.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

// STL
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

// Boost
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

// Utilities
#include "utils.h"

using namespace std;
using namespace dynet;

typedef dynet::ParameterCollection DyNetModel;
typedef std::shared_ptr<DyNetModel> DyNetModelPointer;

namespace transformer {

#define MULTI_HEAD_ATTENTION_PARALLEL // to use pseudo-batching for multi-head attention computing (faster)
#define USE_COLWISE_DROPOUT // use col-wise dropout
#define USE_LECUN_DIST_PARAM_INIT // use Le Cun's uniform distribution for LinearLayer params initialisation (arguably faster convergence)
#define USE_KEY_QUERY_MASKINGS // use key and query maskings in multi-head attention
#define USE_LINEAR_TRANSFORMATION_BROADCASTING // use linear transformation broadcasting at final output layer (much faster)

enum ATTENTION_TYPE { DOT_PRODUCT=1, ADDITIVE_MLP=2 };
enum FFL_ACTIVATION_TYPE { RELU=1, SWISH=2, SWISH_LEARNABLE_BETA=3 };

//---
struct SentinelMarkers{
	int _kSRC_SOS = -1;
	int _kSRC_EOS = -1;
	int _kSRC_UNK = -1;
	int _kTGT_SOS = -1;
	int _kTGT_EOS = -1;
	int _kTGT_UNK = -1;

	SentinelMarkers(){}

	SentinelMarkers(int kSRC_SOS, int kSRC_EOS, int kSRC_UNK
		, int kTGT_SOS, int kTGT_EOS, int kTGT_UNK)
	{
		_kSRC_SOS = kSRC_SOS;
		_kSRC_EOS = kSRC_EOS;
		_kSRC_UNK = kSRC_UNK;
		_kTGT_SOS = kTGT_SOS;
		_kTGT_EOS = kTGT_EOS;
		_kTGT_UNK = kTGT_UNK;
	}
};
//---

//---
struct TransformerConfig{
	unsigned _src_vocab_size = 0;
	unsigned _tgt_vocab_size = 0;
	
	unsigned _num_units = 512;

	unsigned _nheads = 8;

	unsigned _nlayers = 6;

	unsigned _n_ff_units_factor = 4;

	bool _use_dropout = true;
	float _encoder_emb_dropout_rate = 0.1f;
	float _encoder_sublayer_dropout_rate = 0.1f;
	float _decoder_emb_dropout_rate = 0.1f;
	float _decoder_sublayer_dropout_rate = 0.1f;
	float _attention_dropout_rate = 0.1f;
	float _ff_dropout_rate = 0.1f;

	bool _use_label_smoothing = false;
	float _label_smoothing_weight = 0.1f;

	unsigned _position_encoding = 1; // 1: learned positional embedding ; 2: sinusoidal positional encoding ; 0: none
	unsigned _max_length = 500;// for learned positional embedding

	SentinelMarkers _sm;

	unsigned _attention_type = ATTENTION_TYPE::DOT_PRODUCT;

	unsigned _ffl_activation_type = FFL_ACTIVATION_TYPE::RELU;

	bool _use_hybrid_model = false;// RNN encoding over word embeddings instead of word embeddings + positional encoding

	bool _is_training = true;

	TransformerConfig(){}

	TransformerConfig(unsigned src_vocab_size
		, unsigned tgt_vocab_size
		, unsigned num_units
		, unsigned nheads
		, unsigned nlayers
		, unsigned n_ff_units_factor
		, float encoder_emb_dropout_rate
		, float encoder_sublayer_dropout_rate
		, float decoder_emb_dropout_rate
		, float decoder_sublayer_dropout_rate
		, float attention_dropout_rate
		, float ff_dropout_rate
		, bool use_label_smoothing
		, float label_smoothing_weight
		, unsigned position_encoding
		, unsigned max_length
		, SentinelMarkers sm
		, unsigned attention_type
		, unsigned ffl_activation_type
		, bool use_hybrid_model=false
		, bool is_training=true)
	{
		_src_vocab_size = src_vocab_size;
		_tgt_vocab_size = tgt_vocab_size;
		_num_units = num_units;
		_nheads = nheads;
		_nlayers = nlayers;
		_n_ff_units_factor = n_ff_units_factor;
		_encoder_emb_dropout_rate = encoder_emb_dropout_rate;
		_encoder_sublayer_dropout_rate = encoder_sublayer_dropout_rate;
		_decoder_emb_dropout_rate = decoder_emb_dropout_rate;
		_decoder_sublayer_dropout_rate = decoder_sublayer_dropout_rate;
		_attention_dropout_rate = attention_dropout_rate;
		_ff_dropout_rate = ff_dropout_rate;
		_use_label_smoothing = use_label_smoothing;
		_label_smoothing_weight = label_smoothing_weight;
		_position_encoding = position_encoding;
		_max_length = max_length;
		_sm = sm;
		_attention_type = attention_type;
		_ffl_activation_type = ffl_activation_type;
		_use_hybrid_model = use_hybrid_model;
		_is_training = is_training;
		_use_dropout = _is_training;
	}

	TransformerConfig(const TransformerConfig& tfc){
		_src_vocab_size = tfc._src_vocab_size;
		_tgt_vocab_size = tfc._tgt_vocab_size;
		_num_units = tfc._num_units;
		_nheads = tfc._nheads;
		_nlayers = tfc._nlayers;
		_n_ff_units_factor = tfc._n_ff_units_factor;
		_encoder_emb_dropout_rate = tfc._encoder_emb_dropout_rate;
		_encoder_sublayer_dropout_rate = tfc._encoder_sublayer_dropout_rate;
		_decoder_emb_dropout_rate = tfc._decoder_emb_dropout_rate;
		_decoder_sublayer_dropout_rate = tfc._decoder_sublayer_dropout_rate;
		_attention_dropout_rate = tfc._attention_dropout_rate;
		_ff_dropout_rate = tfc._ff_dropout_rate;
		_use_label_smoothing = tfc._use_label_smoothing;
		_label_smoothing_weight = tfc._label_smoothing_weight;
		_position_encoding = tfc._position_encoding;
		_max_length = tfc._max_length;
		_sm = tfc._sm;
		_attention_type = tfc._attention_type;
		_ffl_activation_type = tfc._ffl_activation_type;
		_use_hybrid_model = tfc._use_hybrid_model;
		_is_training = tfc._is_training;
		_use_dropout = _is_training;
	}
};
//---

// --- 
struct ModelStats {
	double _losses[2] = {0.f, 0.f};// If having additional loss, resize this array!
	unsigned int _words_src = 0;
	unsigned int _words_tgt = 0;
	unsigned int _words_src_unk = 0;
	unsigned int _words_tgt_unk = 0;

	ModelStats(){}
};
// --- 

// --- Le Cun's uniform distribution
/*
 * Initialize parameters with samples from a Le Cun's uniform distribution
 * Reference: LeCun 98, Efficient Backprop
 * http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 * Code: https://github.com/neulab/xnmt/blob/TF_new/xnmt/initializer.py
 */
struct ParameterInitLeCunUniform : public ParameterInit {
	ParameterInitLeCunUniform(float fan_in, float scale=1.f) 
		: fan_in(fan_in), scale(scale) { 
		if (scale == 0.0f) throw std::domain_error("Scale of the Le Cun uniform distribution cannot be 0 in ParameterInitLeCunUniform"); 
	}

	virtual void initialize_params(Tensor & values) const override;

private:
	float fan_in, scale;
};

void ParameterInitLeCunUniform::initialize_params(Tensor & values) const {
	float s = scale * std::sqrt(3.f / fan_in);
	TensorTools::randomize_uniform(values, -s, s);
}
// --- 

//--- Simple Linear Layer (w/ or w/o bias)
struct LinearLayer{
	explicit LinearLayer(DyNetModel* mod, unsigned input_dim, unsigned output_dim, bool have_bias=true, bool initLC=false)
		: _have_bias(have_bias)
	{		
		_p_W = (initLC == false)?mod->add_parameters({output_dim, input_dim}):mod->add_parameters({output_dim, input_dim}, ParameterInitLeCunUniform(input_dim));
		if (_have_bias)
			_p_b = (initLC == false)?mod->add_parameters({output_dim}):mod->add_parameters({output_dim}, ParameterInitLeCunUniform(output_dim));
	}

	dynet::Expression apply(dynet::ComputationGraph& cg, const dynet::Expression& i_x, bool reconstruct_shape=true, bool time_distributed=false){
		dynet::Expression i_W = dynet::parameter(cg, _p_W);
		dynet::Expression i_b; 
		if (_have_bias)
			i_b = dynet::parameter(cg, _p_b);
	
		dynet::Expression i_x_in = (!time_distributed)?make_time_distributed(i_x)/*((input_dim, 1), batch_size * seq_len)*/:i_x/*((input_dim, seq_len), batch_size)*/;

		dynet::Expression i_x_out;
		if (_have_bias) i_x_out = dynet::affine_transform({i_b, i_W, i_x_in});// dim of i_x_out depends on i_x
		else i_x_out = i_W * i_x_in;

		if (!reconstruct_shape) return i_x_out;

		auto& d = i_x.dim();
		auto b = d.batch_elems();
		return make_reverse_time_distributed(i_x_out, d[1], b);// ((input_dim, seq_len), batch_size)
	}

	~LinearLayer(){}

	dynet::Parameter _p_W;
	dynet::Parameter _p_b;
	bool _have_bias = true;
};

//--- Highway Network Layer
/* Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wx + b)
    z = t * g(x) + (1 - t) * x
    where g is nonlinearity over x (e.g., a block of feedforward networks), t is transform gate, and (1 - t) is carry gate.
*/
struct HighwayNetworkLayer{
	explicit HighwayNetworkLayer(DyNetModel* mod, unsigned input_dim, bool have_bias=true)
#ifdef USE_LECUN_DIST_PARAM_INIT
		: _l_layer(mod, input_dim, input_dim, have_bias, true)
#else
		: _l_layer(mod, input_dim, input_dim, have_bias)
#endif
	{}

	dynet::Expression apply(dynet::ComputationGraph& cg
		, const dynet::Expression& i_x, const dynet::Expression& i_g_x
		, bool reconstruct_shape=true, bool time_distributed=false)
	{
		dynet::Expression i_l = _l_layer.apply(cg, i_x, reconstruct_shape, time_distributed);
		dynet::Expression i_t = dynet::logistic(i_l);
		dynet::Expression i_z = dynet::cmult(i_t, i_g_x) + dynet::cmult(1.f - i_t, i_x);
		return i_z;
	}

	LinearLayer _l_layer;

	~HighwayNetworkLayer(){}
};

struct FeedForwardLayer{
	explicit FeedForwardLayer(DyNetModel* mod, TransformerConfig& tfc)
		: _l_inner(mod, tfc._num_units, tfc._num_units * tfc._n_ff_units_factor/*4 by default according to the paper*/)
		, _l_outer(mod, tfc._num_units * tfc._n_ff_units_factor/*4 by default according to the paper*/, tfc._num_units)
	{		
		_p_tfc = &tfc;

		if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::SWISH_LEARNABLE_BETA)
			_p_beta = mod->add_parameters({1});
	}	

	~FeedForwardLayer(){}	

	dynet::Parameter _p_beta;// learnable \beta for Swish activation function (work in progress!)

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;	

	dynet::Expression build_graph(dynet::ComputationGraph& cg, const dynet::Expression& i_inp/*num_units x L*/){
		// FFN(x) = relu(x * W1 + b1) * W2 + b2
		dynet::Expression i_inner = _l_inner.apply(cg, i_inp, false, true);// x * W1 + b1

		if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::RELU)
			i_inner = dynet::rectify(i_inner);
		// use Swish from https://arxiv.org/pdf/1710.05941.pdf
		else if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::SWISH) 
			i_inner = dynet::silu(i_inner);
		else if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::SWISH_LEARNABLE_BETA){
			//dynet::Expression i_beta = dynet::parameter(cg, _p_beta);
			// FIXME: requires this: i_inner = dynet::silu(i_inner, i_beta); ? Not supported in DyNet yet!
			assert("Not implemented yet!");
		}
		else assert("Unknown feed-forward activation type!");

		dynet::Expression i_outer = _l_outer.apply(cg, i_inner, false, true);// relu(x * W1 + b1) * W2 + b2

		// dropout for feed-forward layer
		if (_p_tfc->_use_dropout && _p_tfc->_ff_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_outer = dynet::dropout_dim(i_outer, 1/*col-major*/, _p_tfc->_ff_dropout_rate);
#else
			i_outer = dynet::dropout(i_outer, _p_tfc->_ff_dropout_rate);
#endif

		return i_outer;
	}

	LinearLayer _l_inner;
	LinearLayer _l_outer;
};

//---

// ---
// This MaskBase consists of all functions for maskings (both padding positions and future blinding)
struct MaskBase{
	explicit MaskBase(){}

	~MaskBase(){}

	void create_future_blinding_mask(dynet::ComputationGraph& cg, unsigned l){
		_i_mask_fb = create_triangle_mask(cg, l, false);	
	}

	void create_seq_mask_expr(dynet::ComputationGraph& cg
		, const std::vector<vector<float>>& v_seq_masks
		, bool self=true/*self-attention?*/)
	{
		unsigned l = v_seq_masks[0].size();

		std::vector<dynet::Expression> v_i_seq_masks;
		for (unsigned i = 0; i < v_seq_masks.size(); i++){
			dynet::Expression i_mask = (self) ? dynet::input(cg, {l, 1}, v_seq_masks[i]) : dynet::input(cg, {1, l}, v_seq_masks[i]);// ((l, 1), 1) or ((1, l), 1)
			v_i_seq_masks.push_back((self) ? i_mask * PSEUDO_MIN_VALUE : 1.f - i_mask);
		}
		_i_seq_mask = dynet::concatenate_to_batch(v_i_seq_masks);// ((l, 1), batch_size) or ((1, l), batch_size)
	}

	void create_padding_positions_masks(unsigned nheads) // for self-attention
	{
		unsigned l = _i_seq_mask.dim()[0];
		
		// key mask
		_i_mask_pp_k = dynet::concatenate_to_batch(std::vector<dynet::Expression>(nheads, dynet::concatenate_cols(std::vector<dynet::Expression>(l, _i_seq_mask))));// ((l, l), batch_size*nheads)
		
		// query mask
		_i_mask_pp_q = 1.f - _i_mask_pp_k / PSEUDO_MIN_VALUE;// ((l, l), batch_size*nheads)
	}

	void create_padding_positions_masks(const dynet::Expression& i_src_seq_mask, unsigned nheads) // for source-attention
	{
		unsigned ly = _i_seq_mask.dim()[1];
		unsigned lx = i_src_seq_mask.dim()[0];

		// key mask
		_i_mask_pp_k = dynet::concatenate_to_batch(std::vector<dynet::Expression>(nheads, dynet::concatenate_cols(std::vector<dynet::Expression>(ly, i_src_seq_mask))));// ((lx, ly), batch_size*nheads)

		// query mask
		_i_mask_pp_q = dynet::concatenate_to_batch(std::vector<dynet::Expression>(nheads, dynet::concatenate(std::vector<dynet::Expression>(lx, _i_seq_mask))));// ((lx, ly), batch_size*nheads)
	}

	// sequence mask
	dynet::Expression _i_seq_mask;

	// 2 masks for padding positions
	dynet::Expression _i_mask_pp_k;// for keys
	dynet::Expression _i_mask_pp_q;// for queries

	// 1 mask for future blinding
	dynet::Expression _i_mask_fb;
};
// ---

//--- Multi-Head Attention Layer
struct MultiHeadAttentionLayer{
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
	explicit MultiHeadAttentionLayer(DyNetModel* mod, TransformerConfig& tfc, bool is_future_blinding=false)
#ifdef USE_LECUN_DIST_PARAM_INIT
		: _l_W_Q(mod, tfc._num_units, tfc._num_units, false/*linear layer w/o bias*/, true)
		, _l_W_K(mod, tfc._num_units, tfc._num_units, false, true)
		, _l_W_V(mod, tfc._num_units, tfc._num_units, false, true)
		, _l_W_O(mod, tfc._num_units, tfc._num_units, false, true)
#else
		: _l_W_Q(mod, tfc._num_units, tfc._num_units, false/*linear layer w/o bias*/)
		, _l_W_K(mod, tfc._num_units, tfc._num_units, false)
		, _l_W_V(mod, tfc._num_units, tfc._num_units, false)
		, _l_W_O(mod, tfc._num_units, tfc._num_units, false)
#endif
	{
		_att_scale = 1.f / sqrt(tfc._num_units / tfc._nheads);

		_is_future_blinding = is_future_blinding;

		_p_tfc = &tfc;
	}

	~MultiHeadAttentionLayer(){}

	// linear projection matrices
	LinearLayer _l_W_Q;
	LinearLayer _l_W_K;
	LinearLayer _l_W_V;
	LinearLayer _l_W_O;// finishing linear layer

	// attention scale factor
	float _att_scale = 0.f;

	bool _is_future_blinding = false;

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph& cg
		, const dynet::Expression& i_y/*queries*/
		, const dynet::Expression& i_x/*keys and values. i_x is equal to i_y if using self_attention*/
		, const MaskBase& i_mask)
	{
		dynet::Expression i_Q = _l_W_Q.apply(cg, i_y, false, true);// ((num_units, Ly), batch_size)
		dynet::Expression i_K = _l_W_K.apply(cg, i_x, false, true);// ((num_units, Lx), batch_size)
		dynet::Expression i_V = _l_W_V.apply(cg, i_x, false, true);// ((num_units, Lx), batch_size)

		// Note: this will be done in parallel for efficiency!
		// e.g., utilising pseudo-batching
		dynet::Expression i_batch_Q = dynet::concatenate_to_batch(split_rows(i_Q, _p_tfc->_nheads));// ((num_units/nheads, Ly), batch_size*nheads)
		dynet::Expression i_batch_K = dynet::concatenate_to_batch(split_rows(i_K, _p_tfc->_nheads));// ((num_units/nheads, Lx), batch_size*nheads)
		dynet::Expression i_batch_V = dynet::concatenate_to_batch(split_rows(i_V, _p_tfc->_nheads));// ((num_units/nheads, Lx), batch_size*nheads)

		dynet::Expression i_atts;
		if (_p_tfc->_attention_type == ATTENTION_TYPE::DOT_PRODUCT){// Luong attention type
			dynet::Expression i_batch_alphas = (dynet::transpose(i_batch_K) * i_batch_Q) * _att_scale;// ((Lx, Ly),  batch_size*nheads)) (unnormalised) 

#ifdef USE_KEY_QUERY_MASKINGS
			// key masking
			i_batch_alphas = i_batch_alphas + i_mask._i_mask_pp_k;
#endif

			// future blinding masking
			if (_is_future_blinding)
				i_batch_alphas = dynet::softmax(i_batch_alphas + i_mask._i_mask_fb);// ((Lx, Ly),  batch_size*nheads)) (normalised, col-major)
			else
				i_batch_alphas = dynet::softmax(i_batch_alphas);// ((Lx, Ly),  batch_size*nheads)) (normalised, col-major)
			// FIXME: save the soft alignment in i_batch_alphas if necessary!
			
#ifdef USE_KEY_QUERY_MASKINGS
			// query masking
			i_batch_alphas = dynet::cmult(i_batch_alphas, i_mask._i_mask_pp_q);
#endif
					
			// attention dropout (col-major or full?)
			if (_p_tfc->_use_dropout && _p_tfc->_attention_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
				i_batch_alphas = dynet::dropout_dim(i_batch_alphas, 1/*col-major*/, _p_tfc->_attention_dropout_rate);// col-wise dropout
#else
				i_batch_alphas = dynet::dropout(i_batch_alphas, _p_tfc->_attention_dropout_rate);// full matrix
#endif

			i_batch_alphas = i_batch_V/*((num_units/nheads, Lx), batch_size*nheads)*/ * i_batch_alphas/*((Lx, Ly), batch_size*nheads))*/;// ((num_units/nheads, Ly), batch_size*nheads)

			i_atts = dynet::concatenate(split_batch(i_batch_alphas, _p_tfc->_nheads));// ((num_units, Ly), batch_size)			
		}
		else if (_p_tfc->_attention_type == ATTENTION_TYPE::ADDITIVE_MLP){// Bahdanau attention type
			// FIXME
			assert("MultiHeadAttentionLayer: Bahdanau attention type not yet implemented!");
		}
		else assert("MultiHeadAttentionLayer: Unknown attention type!");
		
		// linear projection
		dynet::Expression i_proj_atts = _l_W_O.apply(cg, i_atts, false, true);// ((num_units, Ly), batch_size)

		return i_proj_atts;
	}
#else // without using pseudo-batching
	explicit MultiHeadAttentionLayer(DyNetModel* mod, TransformerConfig& tfc, bool is_future_blinding=false)
	{
		_p_WQ.resize(tfc._nheads);
		_p_WK.resize(tfc._nheads);
		_p_WV.resize(tfc._nheads);
		for (unsigned h = 0; h < tfc._nheads; h++){
			_p_WQ[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dk = num_units/nheads
			_p_WK[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dk
			_p_WV[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dv = num_units/nheads
		}

		_p_WO = mod->add_parameters({tfc._num_units, tfc._num_units});
		
		_att_scale = 1.f / sqrt(tfc._num_units / tfc._nheads);

		_is_future_blinding = is_future_blinding;

		_p_tfc = &tfc;
	}

	~MultiHeadAttentionLayer(){}

	// linear projection matrices
	std::vector<dynet::Parameter> _p_WQ;
	std::vector<dynet::Parameter> _p_WK;
	std::vector<dynet::Parameter> _p_WV;
	dynet::Parameter _p_WO;

	// attention scale factor
	float _att_scale = 0.f;

	bool _is_future_blinding = false;

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph& cg
		, const dynet::Expression& i_y/*queries*/
		, const dynet::Expression& i_x/*keys and values. i_x is equal to i_y if using self_attention*/
		, const MaskBase& i_mask)
	{
		// Note: this should be done in parallel for efficiency!
		// e.g., utilising pseudo-batching?	
		std::vector<dynet::Expression> v_atts(_p_tfc->_nheads);
		for (unsigned h = 0; h < _p_tfc->_nheads; h++){
			dynet::Expression i_Q/*queries*/ = dynet::parameter(cg, _p_WQ[h])/*dk x num_units*/ * i_x/*num_units x Ly*/;// ((dk, Ly), batch_size)
			dynet::Expression i_K/*keys*/ = dynet::parameter(cg, _p_WK[h])/*dk x num_units*/ * i_y/*num_units x Lx*/;// ((dk, Lx), batch_size)
			dynet::Expression i_V/*values*/ = dynet::parameter(cg, _p_WV[h])/*dv x num_units*/ * i_y/*num_units x Lx*/;// ((dk, Lx), batch_size)

			dynet::Expression i_att_h;
			if (_p_tfc->_attention_type == ATTENTION_TYPE::DOT_PRODUCT){// Luong attention type
				dynet::Expression i_alpha_pre = (dynet::transpose(i_K) * i_Q) * _att_scale;// ((Lx, Ly), batch_size) (unnormalised) 

#ifdef USE_KEY_QUERY_MASKINGS
				// key masking
				i_alpha_pre = i_alpha_pre + i_mask._i_mask_pp_k);
#endif

				dynet::Expression i_alpha;
				if (_is_future_blinding)
					i_alpha = dynet::softmax(i_alpha_pre + i_mask._i_mask_fb);// ((Lx, Ly), batch_size) (normalised, col-major)
				else
					i_alpha = dynet::softmax(i_alpha_pre);// ((Lx, Ly), batch_size) (normalised, col-major)
				// FIXME: save the soft alignment in i_alpha if necessary!

#ifdef USE_KEY_QUERY_MASKINGS
				// query masking
				i_alpha = dynet::cmult(i_alpha, i_mask._i_mask_pp_q));
#endif
						
				// attention dropout (col-major or full?)
				if (_p_tfc->_use_dropout && _p_tfc->_attention_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
					i_alpha = dynet::dropout_dim(i_alpha, 1/*col-major*/, _p_tfc->_attention_dropout_rate);// col-wise dropout
#else
					i_alpha = dynet::dropout(i_alpha, _p_tfc->_attention_dropout_rate);// full dropout
#endif

				i_att_h = i_V * i_alpha;// ((dk, Ly), batch_size)
			}
			else if (_p_tfc->_attention_type == ATTENTION_TYPE::ADDITIVE_MLP){// Bahdanau attention type
				assert("MultiHeadAttentionLayer: Bahdanau attention type not yet implemented!");
			}
			else assert("MultiHeadAttentionLayer: Unknown attention type!");

			v_atts[h] = i_att_h;
		}

		// joint all head attentions
		dynet::Expression i_atts = dynet::concatenate(v_atts);// ((dk*nheads=num_units, Ly), batch_size)

		// linear projection
		dynet::Expression i_proj_atts = dynet::parameter(cg, _p_WO) * i_atts;// ((num_units, Ly), batch_size)

		return i_proj_atts;
	}
#endif
};
//---

// --- Sinusoidal Positional Encoding
// Note: think effective way to make this much faster!
dynet::Expression make_sinusoidal_position_encoding(dynet::ComputationGraph &cg, const dynet::Dim& dim){
	unsigned nUnits = dim[0];
	unsigned nWords = dim[1];

	float num_timescales = nUnits / 2;
	float log_timescale_increment = std::log(10000.f) / (num_timescales - 1.f);

	std::vector<float> vSS(nUnits * nWords, 0.f);
	for(unsigned p = 0; p < nWords; ++p) {
		for(int i = 0; i < num_timescales; ++i) {
			float v = p * std::exp(i * -log_timescale_increment);
			vSS[p * nUnits + i] = std::sin(v);
			vSS[p * nUnits + num_timescales + i] = std::cos(v);
		}
	}

	return dynet::input(cg, {nUnits, nWords}, vSS);
}
// ---

//--- Encoder Layer
struct EncoderLayer{
	explicit EncoderLayer(DyNetModel* mod, TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc)
		, _feed_forward_sublayer(mod, tfc)
	{		
		// for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

		_p_tfc = &tfc;
	}

	~EncoderLayer(){}

	// multi-head attention sub-layer
	MultiHeadAttentionLayer _self_attention_sublayer;

	// position-wise feed forward sub-layer
	FeedForwardLayer _feed_forward_sublayer;

	// for layer normalisation
	dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
	dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src
		, const MaskBase& self_mask)
	{	
		// get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
		dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
		dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
		dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
		dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

		dynet::Expression i_encl = i_src;
		
		// multi-head attention sub-layer
		dynet::Expression i_mh_att = _self_attention_sublayer.build_graph(cg, i_encl, i_encl, self_mask);// ((num_units, Lx), batch_size)	

		// dropout to the above sub-layer
		if (_p_tfc->_use_dropout && _p_tfc->_encoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_mh_att = dynet::dropout_dim(i_mh_att, 1/*col-major*/, _p_tfc->_encoder_sublayer_dropout_rate);// col-wise dropout
#else
			i_mh_att = dynet::dropout(i_mh_att, _p_tfc->_encoder_sublayer_dropout_rate);// full dropout
#endif

		// w/ residual connection
		i_encl = i_encl + i_mh_att;// ((num_units, Lx), batch_size)

		// position-wise layer normalisation 1
		i_encl = layer_norm_colwise_3(i_encl, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_encl);// ((num_units, Lx), batch_size)

		// w/ residual connection
		i_encl = i_encl + i_ff;// ((num_units, Lx), batch_size)

		// position-wise layer normalisation 2
		i_encl = layer_norm_colwise_3(i_encl, i_ln2_g, i_ln2_b);// ((num_units, Lx), batch_size)

		return i_encl;
	}
};

struct Encoder{
	explicit Encoder(DyNetModel* mod, TransformerConfig& tfc)
	{
		_p_embed_s = mod->add_lookup_parameters(tfc._src_vocab_size, {tfc._num_units});

		if (!tfc._use_hybrid_model && tfc._position_encoding == 1){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		if (tfc._use_hybrid_model){
			_v_p_src_rnns.resize(3);// first twos are forward and backward RNNs; third is forward RNN.
			for (unsigned l = 0; l < 3; l++){
				if (l == 2)
					_v_p_src_rnns[l].reset(new dynet::LSTMBuilder(1, tfc._num_units * 2, tfc._num_units, *mod, true/*w/ layer norm*/));
				else
					_v_p_src_rnns[l].reset(new dynet::LSTMBuilder(1, tfc._num_units, tfc._num_units, *mod, true/*w/ layer norm*/));
			}
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_enc_layers.push_back(EncoderLayer(mod, tfc));
		}

		_scale_emb = sqrt(tfc._num_units);

		_p_tfc = &tfc;
	}

	~Encoder(){}

	dynet::LookupParameter _p_embed_s;// source embeddings

	dynet::LookupParameter _p_embed_pos;// position embeddings
	unsigned _position_encoding = 1;

	// hybrid architecture: use LSTM-based RNNs over word embeddings instead of word embeddings + positional encodings
	std::vector<std::shared_ptr<dynet::LSTMBuilder>> _v_p_src_rnns;
	
	std::vector<EncoderLayer> _v_enc_layers;// stack of identical encoder layers

	// --- intermediate variables
	float _scale_emb = 0.f;
	unsigned _batch_slen = 0;
	MaskBase _self_mask;
	// ---

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const WordIdSentences& sents/*batch of sentences*/
		, ModelStats &stats)
	{
		// compute embeddings
		// get max length within a batch
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < sents.size(); i++) max_len = std::max(max_len, sents[i].size());
		_batch_slen = max_len;

		std::vector<dynet::Expression> source_embeddings;   
		std::vector<unsigned> words(sents.size());
		std::vector<std::vector<float>> v_seq_masks(sents.size());
		dynet::Expression i_src;
		if (_p_tfc->_use_hybrid_model){
			// first 2 layers (forward and backward)
			// run a RNN backward and forward over the source sentence
			// and stack the top-level hidden states from each model 
			// and feed them into yet another forward RNN as 
			// the representation at each position
			// inspired from Google NMT system (https://arxiv.org/pdf/1609.08144.pdf)

			// first RNN (forward)
			std::vector<Expression> src_fwd(max_len);
			_v_p_src_rnns[0]->new_graph(cg);
			_v_p_src_rnns[0]->start_new_sequence();
			for (unsigned l = 0; l < max_len; l++){
				for (unsigned bs = 0; bs < sents.size(); ++bs){
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kSRC_EOS;		
					if (l < sents[bs].size()){ 
						words[bs] = (unsigned)sents[bs][l];
						stats._words_src++; 
						if (sents[bs][l] == _p_tfc->_sm._kSRC_UNK) stats._words_src_unk++;
						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kSRC_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				src_fwd[l] = _v_p_src_rnns[0]->add_input(dynet::lookup(cg, _p_embed_s, words));
			}

			// second RNN (backward)
			std::vector<Expression> src_bwd(max_len);
			_v_p_src_rnns[1]->new_graph(cg);
			_v_p_src_rnns[1]->start_new_sequence();
			for (int l = max_len - 1; l >= 0; --l) { // int instead of unsigned for negative value of l
				// offset by one position to the right, to catch </s> and generally
				// not duplicate the w_t already captured in src_fwd[t]
				for (unsigned bs = 0; bs < sents.size(); ++bs) 
					words[bs] = ((unsigned)l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kSRC_EOS;
				src_bwd[l] = _v_p_src_rnns[1]->add_input(dynet::lookup(cg, _p_embed_s, words));
			}

			// third RNN (yet another forward)
			_v_p_src_rnns[2]->new_graph(cg);
			_v_p_src_rnns[2]->start_new_sequence();
			for (unsigned l = 0; l < max_len; l++){
				source_embeddings.push_back(_v_p_src_rnns[2]->add_input(dynet::concatenate(std::vector<dynet::Expression>({src_fwd[l], src_bwd[l]}))));
			}
			
			i_src = dynet::concatenate_cols(source_embeddings);
		}
		else{
			// source embeddings
			for (unsigned l = 0; l < max_len; l++){
				for (unsigned bs = 0; bs < sents.size(); ++bs){
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kSRC_EOS;		
					if (l < sents[bs].size()){ 
						words[bs] = (unsigned)sents[bs][l];
						stats._words_src++; 
						if (sents[bs][l] == _p_tfc->_sm._kSRC_UNK) stats._words_src_unk++;
						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kSRC_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				source_embeddings.push_back(dynet::lookup(cg, _p_embed_s, words));
			}
			i_src = dynet::concatenate_cols(source_embeddings);// ((num_units, Lx), batch_size)

			i_src = i_src * _scale_emb;// scaled embeddings

			// + postional encoding
			if (_p_tfc->_position_encoding == 1){// learned positional embedding 
				std::vector<dynet::Expression> pos_embeddings;  
				std::vector<unsigned> positions(sents.size());
				for (unsigned l = 0; l < max_len; l++){
					for (unsigned bs = 0; bs < sents.size(); ++bs){
						if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to (_p_tfc._max_length - 1).
						else
							positions[bs] = l;
				}

					pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
				}
				dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// ((num_units, Lx), batch_size)

				i_src = i_src + i_pos;
			}
			else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
				dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_src.dim());

				i_src = i_src + i_pos;
			}
			else assert("Unknown positional encoding type!");
		}	

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_encoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_src = dynet::dropout_dim(i_src, 1/*col-major*/, _p_tfc->_encoder_emb_dropout_rate);// col-wise dropout
#else
			i_src = dynet::dropout(i_src, _p_tfc->_encoder_emb_dropout_rate);// full dropout
#endif

		// create maskings
		_self_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_self_mask.create_padding_positions_masks(_p_tfc->_nheads);
#else
		_self_mask.create_padding_positions_masks(1);
#endif

		return i_src;
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg, const WordIdSentences& ssents/*batch of sentences*/, ModelStats &stats){
		// compute source (+ postion) embeddings
		dynet::Expression i_src_rep = compute_embeddings_and_masks(cg, ssents, stats);// ((num_units, Lx), batch_size)
		
		// compute stacked encoder layers
		dynet::Expression i_enc_l_out = i_src_rep;
		for (auto enc : _v_enc_layers){
			// stacking approach
			i_enc_l_out = enc.build_graph(cg, i_enc_l_out, _self_mask);// each position in the encoder can attend to all positions in the previous layer of the encoder.
		}

		return i_enc_l_out;// ((num_units, Lx), batch_size)
	}
};
typedef std::shared_ptr<Encoder> EncoderPointer;
//---

//--- Decoder Layer
struct DecoderLayer{
	explicit DecoderLayer(DyNetModel* mod, TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc, true)
		, _src_attention_sublayer(mod, tfc)
		, _feed_forward_sublayer(mod, tfc)
	{	
		// initialisation for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln3_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln3_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

		_p_tfc = &tfc;
	}

	~DecoderLayer(){}	

	// multi-head attention sub-layers
	MultiHeadAttentionLayer _self_attention_sublayer;// self-attention
	MultiHeadAttentionLayer _src_attention_sublayer;// source attention

	// position-wise feed forward sub-layer
	FeedForwardLayer _feed_forward_sublayer;

	// for layer normalisation
	dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
	dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2
	dynet::Parameter _p_ln3_g, _p_ln3_b;// layer normalisation 3

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const dynet::Expression& i_enc_inp
		, const dynet::Expression& i_dec_inp
		, const MaskBase& self_mask
		, const MaskBase& src_mask)
	{	
		// get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b, i_ln3_g, i_ln3_b
		dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
		dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
		dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
		dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);
		dynet::Expression i_ln3_g = dynet::parameter(cg, _p_ln3_g);
		dynet::Expression i_ln3_b = dynet::parameter(cg, _p_ln3_b);
	
		dynet::Expression i_decl = i_dec_inp;
		
		// multi-head self attention sub-layer
		dynet::Expression i_mh_self_att = _self_attention_sublayer.build_graph(cg, i_decl, i_decl, self_mask);// ((num_units, Ly), batch_size)

		// dropout to the output of sub-layer
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_mh_self_att = dynet::dropout_dim(i_mh_self_att, 1/*col-major*/, _p_tfc->_decoder_sublayer_dropout_rate);// col-wise dropout
#else
			i_mh_self_att = dynet::dropout(i_mh_self_att, _p_tfc->_decoder_sublayer_dropout_rate);// full dropout
#endif

		// w/ residual connection
		i_decl = i_decl + i_mh_self_att;// ((num_units, Ly), batch_size)

		// layer normalisation 1
		i_decl = layer_norm_colwise_3(i_decl, i_ln1_g, i_ln1_b);// ((num_units, Ly), batch_size)

		// multi-head source attention sub-layer
		dynet::Expression i_mh_src_att = _src_attention_sublayer.build_graph(cg, i_decl, i_enc_inp, src_mask);// ((num_units, Ly), batch_size)

		// dropout to the output of sub-layer
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_mh_src_att = dynet::dropout_dim(i_mh_src_att, 1/*col-major*/, _p_tfc->_decoder_sublayer_dropout_rate);// col-wise dropout
#else
			i_mh_src_att = dynet::dropout(i_mh_src_att, _p_tfc->_decoder_sublayer_dropout_rate);// full dropout
#endif

		// w/ residual connection
		i_decl = i_decl + i_mh_src_att;

		// layer normalisation 2
		i_decl = layer_norm_colwise_3(i_decl, i_ln2_g, i_ln2_b);// ((num_units, Ly), batch_size)

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_decl);// ((num_units, Ly), batch_size)

		// w/ residual connection
		i_decl = i_decl + i_ff;

		// layer normalisation 3
		i_decl = layer_norm_colwise_3(i_decl, i_ln3_g, i_ln3_b);// ((num_units, Ly), batch_size)

		return i_decl;
	}
};

struct Decoder{
	explicit Decoder(DyNetModel* mod, TransformerConfig& tfc, Encoder* p_encoder)
	{
		_p_embed_t = mod->add_lookup_parameters(tfc._tgt_vocab_size, {tfc._num_units});

		if (!tfc._use_hybrid_model && tfc._position_encoding == 1){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_dec_layers.push_back(DecoderLayer(mod, tfc));
		}

		if (tfc._use_hybrid_model){
			_p_tgt_rnn.reset(new dynet::LSTMBuilder(1/*shallow*/, tfc._num_units, tfc._num_units, *mod, true/*w/ layer norm*/));
		}

		_scale_emb = std::sqrt(tfc._num_units);

		_p_tfc = &tfc;

		_p_encoder = p_encoder;
	}

	~Decoder(){}

	dynet::LookupParameter _p_embed_t;// source embeddings
	dynet::LookupParameter _p_embed_pos;// position embeddings

	// hybrid architecture: use LSTM-based RNN over word embeddings instead of word embeddings + positional encodings
	std::shared_ptr<dynet::LSTMBuilder> _p_tgt_rnn;

	std::vector<DecoderLayer> _v_dec_layers;// stack of identical decoder layers

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	// encoder object pointer
	Encoder* _p_encoder = nullptr;

	// --- intermediate variables
	float _scale_emb = 0.f;
	unsigned _batch_tlen = 0;
	// decoder masks
	MaskBase _self_mask;
	MaskBase _src_mask;
	// ---

	dynet::Expression get_wrd_embedding_matrix(dynet::ComputationGraph &cg){
		return dynet::parameter(cg, _p_embed_t);// target word embedding matrix (num_units x |V_T|)
	}

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const WordIdSentences& sents/*batch of target sentences*/)
	{
		// compute embeddings			
		// get max length in a batch
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < sents.size(); i++) max_len = std::max(max_len, sents[i].size());
		_batch_tlen = max_len;

		std::vector<dynet::Expression> target_embeddings;   
		std::vector<unsigned> words(sents.size());
		std::vector<std::vector<float>> v_seq_masks(sents.size());
		dynet::Expression i_tgt;
		if (_p_tfc->_use_hybrid_model){
			// target embeddings via RNN
			_p_tgt_rnn->new_graph(cg);
			_p_tgt_rnn->start_new_sequence();
			for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
				for (unsigned bs = 0; bs < sents.size(); ++bs)
				{
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kTGT_EOS;
					if (l < sents[bs].size()){
						words[bs] = (unsigned)sents[bs][l];
						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kTGT_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				target_embeddings.push_back(_p_tgt_rnn->add_input(dynet::lookup(cg, _p_embed_t, words)));
			}
			i_tgt = dynet::concatenate_cols(target_embeddings);// ((num_units, Ly), batch_size)
		}
		else{
			// target embeddings
			for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
				for (unsigned bs = 0; bs < sents.size(); ++bs)
				{
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kTGT_EOS;
					if (l < sents[bs].size()){
						words[bs] = (unsigned)sents[bs][l];
						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kTGT_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				target_embeddings.push_back(dynet::lookup(cg, _p_embed_t, words));
			}
			i_tgt = dynet::concatenate_cols(target_embeddings);// ((num_units, Ly), batch_size)

			// scale
			i_tgt = i_tgt * _scale_emb;// scaled embeddings

			// + postional encoding
			if (_p_tfc->_position_encoding == 1){// learned positional embedding 
				std::vector<dynet::Expression> pos_embeddings;  
				std::vector<unsigned> positions(sents.size());
				for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
					for (unsigned bs = 0; bs < sents.size(); ++bs){
						if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to _p_tfc._max_length.
						else
							positions[bs] = l;
				}

					pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
				}
				dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// // ((num_units, Ly), batch_size)

				i_tgt = i_tgt + i_pos;
			}
			else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
				dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_tgt.dim());

				i_tgt = i_tgt + i_pos;
			}
			else assert("Unknown positional encoding type!");
		}

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_tgt = dynet::dropout_dim(i_tgt, 1/*col-major*/, _p_tfc->_decoder_emb_dropout_rate);// col-wise dropout
#else
			i_tgt = dynet::dropout(i_tgt, _p_tfc->_decoder_emb_dropout_rate);// full dropout		
#endif

		// create maskings
		// self-attention
		// for future blinding
		_self_mask.create_future_blinding_mask(cg, i_tgt.dim()[1]);

		// for padding positions blinding
		_self_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_self_mask.create_padding_positions_masks(_p_tfc->_nheads);
#else 
		_self_mask.create_padding_positions_masks(1);
#endif
		// source-attention
		_src_mask.create_seq_mask_expr(cg, v_seq_masks, false);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_src_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else 
		_src_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

		return i_tgt;
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& tsents/*batch of sentences*/
		, const dynet::Expression& i_src_rep)
	{		
		// compute source (+ postion) embeddings
		dynet::Expression i_tgt_rep = compute_embeddings_and_masks(cg, tsents);// ((num_units, Ly), batch_size)
			
		dynet::Expression i_dec_l_out = i_tgt_rep;
		for (auto dec : _v_dec_layers){
			// stacking approach
			i_dec_l_out = dec.build_graph(cg, i_src_rep, i_dec_l_out, _self_mask, _src_mask);// each position in the decoder can attend to all positions (up to and including the current position) in the previous layer of the decoder.
		}
	
		return i_dec_l_out;// ((num_units, Ly), batch_size)
	}
};
typedef std::shared_ptr<Decoder> DecoderPointer;
//---

//--- Transformer Model
struct TransformerModel {

public:
	explicit TransformerModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	explicit TransformerModel();

	~TransformerModel(){}

	// for training
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents/*batched*/
		, const WordIdSentences& tsents/*batched*/
		, ModelStats &stats
		, bool is_eval_on_dev=false);
	// for decoding
	dynet::Expression compute_source_rep(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents/*pseudo batch*/);// source representation
	dynet::Expression step_forward(dynet::ComputationGraph & cg
		, const dynet::Expression& i_src_rep
		, const WordIdSentence &partial_sent
		, bool log_prob
		, std::vector<dynet::Expression> &aligns);// forward step to get softmax scores
	std::string sample(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target);// sampling
	std::string greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target);// greedy decoding
	std::string beam_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target, unsigned beam_width);// beam search decoding

	dynet::ParameterCollection& get_model_parameters();
	void initialise_params_from_file(const string &params_file);
	void save_params_to_file(const string &params_file);

	void set_dropout(bool is_activated = true);

	dynet::Dict& get_source_dict();
	dynet::Dict& get_target_dict();

	TransformerConfig& get_config();

protected:

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!

	EncoderPointer _encoder;// encoder

	DecoderPointer _decoder;// decoder

	std::pair<dynet::Dict, dynet::Dict> _dicts;// pair of source and target vocabularies

	dynet::Parameter _p_Wo_bias;// bias of final linear projection layer

	TransformerConfig _tfc;// local configuration storage
};

TransformerModel::TransformerModel(){
	_all_params = nullptr;
	_encoder = nullptr;
	_decoder = nullptr;
}

TransformerModel::TransformerModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
: _tfc(tfc)
{
	_all_params.reset(new DyNetModel());// create new model parameter object

	_encoder.reset(new Encoder(_all_params.get(), _tfc));// create new encoder object

	_decoder.reset(new Decoder(_all_params.get(), _tfc, _encoder.get()));// create new decoder object

	// final output projection layer
	_p_Wo_bias = _all_params.get()->add_parameters({tfc._tgt_vocab_size});// optional

	// dictionaries
	_dicts.first = sd;
	_dicts.second = td;
}

dynet::Expression TransformerModel::compute_source_rep(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents)// for decoding only
{
	// encode source
	ModelStats stats;// unused
	dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, stats);// ((num_units, Lx), batch_size)

	return i_src_ctx;
}

dynet::Expression TransformerModel::step_forward(dynet::ComputationGraph &cg
	, const dynet::Expression& i_src_rep
	, const WordIdSentence &partial_sent
	, bool log_prob
	, std::vector<dynet::Expression> &aligns)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, WordIdSentences(1, partial_sent), i_src_rep);
	dynet::Expression i_tgt_t;
	if (partial_sent.size() == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(partial_sent.size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation

	// compute softmax prediction
	if (log_prob)
		return dynet::log_softmax(i_r_t);
	else
		return dynet::softmax(i_r_t);
}

dynet::Expression TransformerModel::build_graph(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents
	, const WordIdSentences& tsents
	, ModelStats &stats
	, bool is_eval_on_dev)
{
	// encode source
	dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, stats);// ((num_units, Lx), batch_size)
	
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents, i_src_ctx);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING 
	// Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t) {
				stats._words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) stats._words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)
	
		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else 
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}
#else // Note: this way is much faster!
	// compute the logit and linear projections
	dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_ctx});// ((|V_T|, (Ly-1)), batch_size)

	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t) {
				stats._words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) stats._words_tgt_unk++;
			}
		}

		// get the prediction at timestep t
		//dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
		dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// shifted right, ((|V_T|, 1), batch_size)
	
		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else 
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}
#endif

	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

std::string TransformerModel::sample(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;
	Dict& tdict = _dicts.second;

	// start of sentence
	target.clear();
	target.push_back(sos_sym); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);

	std::vector<dynet::Expression> aligns;// FIXME: unused
	std::stringstream ss;
	ss << "<s>";
	unsigned t = 0;
	while (target.back() != eos_sym) 
	{
		cg.checkpoint();
				
		dynet::Expression ydist = this->step_forward(cg, i_src_rep, target, false, aligns);

		auto dist = dynet::as_vector(cg.incremental_forward(ydist));
		double p = rand01();
		WordId w = 0;
		for (; w < (WordId)dist.size(); ++w) {
			p -= dist[w];
			if (p < 0.f) break;
		}

		// this shouldn't happen
		if (w == (WordId)dist.size()) w = eos_sym;

		ss << " " << tdict.convert(w) << " [p=" << dist[w] << "]";
		target.push_back(w);

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();
	}

	_tfc._is_training = true;

	return ss.str();
}

std::string TransformerModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;
	Dict& tdict = _dicts.second;

	// start of sentence
	target.clear();
	target.push_back(sos_sym); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);
	
	std::vector<dynet::Expression> aligns;// FIXME: unused
	std::stringstream ss;
	ss << "<s>";
	unsigned t = 0;
	while (target.back() != eos_sym) 
	{
		cg.checkpoint();
			
		dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, false, aligns);

		// find the argmax next word (greedy)
		unsigned w = 0;
		auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));
		auto pr_w = ydist[w];
		for (unsigned x = 1; x < ydist.size(); ++x) {
			if (ydist[x] > pr_w) {
				w = x;
				pr_w = ydist[x];
			}
		}

		// break potential infinite loop
		if (t > 2 * source.size()) {
			w = eos_sym;
			pr_w = ydist[w];
		}

		ss << " " << tdict.convert(w) << " [p=" << pr_w << "]";// translation with individual scores
		target.push_back(w);

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();
	}

	_tfc._is_training = true;
	
	return ss.str();
}

struct Hypothesis {
	Hypothesis() {};
	Hypothesis(int tgt, float cst, std::vector<Expression> &al)
		: target({tgt}), cost(cst), costs({1.f}), aligns(al) {}
	Hypothesis(int tgt, float cst, Hypothesis &last, std::vector<Expression> &al)
		: target(last.target), costs(last.costs), aligns(al) {
		target.push_back(tgt);
		cost = last.cost - std::log(cst);
		costs.push_back(cst); 
	}

	std::vector<int> target;
	float cost;
	std::vector<float> costs;
	std::vector<Expression> aligns;
};

std::string TransformerModel::beam_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target, unsigned beam_width)// FIXME: to be tested?
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;
	Dict& tdict = _dicts.second;

	// start of sentence
	target.clear();
	target.push_back(sos_sym); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);
	
	std::vector<dynet::Expression> aligns;// FIXME: unused

	std::vector<Hypothesis> chart;
	chart.push_back(Hypothesis(sos_sym, 0.0f, aligns));

	std::vector<unsigned> vocab(boost::copy_range<std::vector<unsigned>>(boost::irange(0u, tdict.size())));
	std::vector<Hypothesis> completed;

	for (unsigned steps = 0; completed.size() < beam_width && steps < 2*source.size(); ++steps) {
		std::vector<Hypothesis> new_chart;

		for (auto &hprev: chart) {
			cg.checkpoint();
		
			dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, hprev.target, false, aligns);

			// find the top k best next words
			auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));
			std::partial_sort(vocab.begin(), vocab.begin()+beam_width, vocab.end(), 
				[&ydist](unsigned v1, unsigned v2) { return ydist[v1] > ydist[v2]; });

			// add to chart
			for (auto vi = vocab.begin(); vi < vocab.begin() + beam_width; ++vi) {
				//if (new_chart.size() < beam_width) {
					Hypothesis hnew(*vi, ydist[*vi]/*hprev.cost - std::log(ydist[*vi])*/, hprev, aligns);
					if (*vi == (unsigned int)eos_sym)
						completed.push_back(hnew);
					else
						new_chart.push_back(hnew);
				//} 
			}
	
			cg.revert();
		}

		if (new_chart.size() > beam_width) {
			// sort new_chart by score, to get kbest candidates
			std::partial_sort(new_chart.begin(), new_chart.begin()+beam_width, new_chart.end(),
				[](Hypothesis &h1, Hypothesis &h2) { return h1.cost < h2.cost; });
			new_chart.resize(beam_width);
		}
		chart.swap(new_chart);
	}

	// sort completed by score, adjusting for length -- not very effective, too short!
	auto best = std::min_element(completed.begin(), completed.end(),
			[](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });
	assert(best != completed.end());

	target = best->target;
	std::stringstream ss;
	ss << "<s>";
	for (unsigned i = 1; i < target.size(); i++){
		ss << " " << tdict.convert(target[i]) << " [p=" << best->costs[i] << "]";// translation with individual scores
	}

	_tfc._is_training = true;

	return ss.str();
}

dynet::ParameterCollection& TransformerModel::get_model_parameters(){
	return *_all_params.get();
}

void TransformerModel::initialise_params_from_file(const string &params_file)
{
	dynet::load_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerModel::save_params_to_file(const string &params_file)
{
	dynet::save_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerModel::set_dropout(bool is_activated){
	_tfc._use_dropout = is_activated;
}

dynet::Dict& TransformerModel::get_source_dict()
{
	return _dicts.first;
}
dynet::Dict& TransformerModel::get_target_dict()
{
	return _dicts.second;
}

TransformerConfig& TransformerModel::get_config(){
	return _tfc;
}

//---

}; // namespace transformer



