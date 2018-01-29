/* This is an implementation of ensemble decoder for Transformer, adapted from ensemble-decoder.{h,cc} of lamtram toolkit (https://github.com/neubig/lamtram).
 * Adapted by Cong Duy Vu Hoang (duyvuleo@gmail.com, vhoang2@student.unimelb.edu.au) 
 */

#pragma once

#include <dynet/tensor.h>
#include <dynet/dynet.h>

#include <vector>
#include <cfloat>

#include "transformer.h"

using namespace std;
using namespace dynet;
using namespace transformer;

#define USE_BEAM_SEARCH_LENGTH_NORMALISATION

class EnsembleDecoderHyp {
public:
	EnsembleDecoderHyp(float score, const WordIdSentence & sent, const WordIdSentence & align) :
		_score(score), _sent(sent), _align(align) { }

	float get_score() const { return _score; }
	const WordIdSentence & get_sentence() const { return _sent; }
	const WordIdSentence & get_alignment() const { return _align; }

protected:

	float _score;
	WordIdSentence _sent;
	WordIdSentence _align;
};

typedef std::shared_ptr<EnsembleDecoderHyp> EnsembleDecoderHypPtr;

inline bool operator<(const EnsembleDecoderHypPtr & lhs, const EnsembleDecoderHypPtr & rhs) {
#ifndef USE_BEAM_SEARCH_LENGTH_NORMALISATION
	if(lhs->get_score() != rhs->get_score()) return lhs->get_score() > rhs->get_score();
	return lhs->get_sentence() < rhs->get_sentence();
#else
	//score with word-based length normalization. better?
	float score_l = lhs->get_score()/lhs->get_sentence().size(), score_r = rhs->get_score()/rhs->get_sentence().size();
	if( score_l != score_r) return score_l > score_r;
	return lhs->get_sentence() < rhs->get_sentence();
#endif
}

typedef tuple<float,int,int,int> Beam_Info;

class EnsembleDecoder {
public:
	EnsembleDecoder(dynet::Dict& td);
	~EnsembleDecoder() {}

	EnsembleDecoderHypPtr generate(dynet::ComputationGraph& cg
		, const WordIdSentence & sent_src
		, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models);
	std::vector<EnsembleDecoderHypPtr> generate_nbest(dynet::ComputationGraph& cg
		, const WordIdSentence & sent_src
		, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
		, unsigned nbest_size);

	// Ensemble together probabilities or log probabilities for a single word
	Expression ensemble_probs(const std::vector<Expression> & in, dynet::ComputationGraph & cg);
	Expression ensemble_logprobs(const std::vector<Expression> & in, dynet::ComputationGraph & cg);
	
	float get_word_pen() const { return _word_pen; }
	float get_unk_pen() const { return _unk_pen; }
	std::string get_ensemble_operation() const { return _ensemble_operation; }
	void set_word_pen(float word_pen) { _word_pen = word_pen; }
	void set_unk_pen(float unk_pen) { _unk_pen = unk_pen; }
	void set_ensemble_operation(const std::string & ensemble_operation) { _ensemble_operation = ensemble_operation; }

	int get_beam_size() const { return _beam_size; }
	void set_beam_size(int beam_size) { _beam_size = beam_size; }
	int get_size_limit() const { return _size_limit; }
	void set_size_limit(int size_limit) { _size_limit = size_limit; }

protected:

	float _word_pen;
	float _unk_pen, _unk_log_prob;
	WordId _unk_id;
	int _size_limit;
	int _beam_size;
	std::string _ensemble_operation;

	bool _verbose;
};

EnsembleDecoder::EnsembleDecoder(dynet::Dict& td)
	: _word_pen(0.f), _unk_pen(0.f), _size_limit(500), _beam_size(1), _ensemble_operation("sum"), _verbose(false) 
{
	_unk_id = td.convert("<unk>");
	_unk_log_prob = -std::log(td.size());// penalty score for <unk>
}

dynet::Expression EnsembleDecoder::ensemble_probs(const std::vector<dynet::Expression> & v_ins, dynet::ComputationGraph & cg) {
	if(v_ins.size() == 1) return v_ins[0];
	return dynet::average(v_ins);
}

dynet::Expression EnsembleDecoder::ensemble_logprobs(const std::vector<dynet::Expression> & v_ins, dynet::ComputationGraph & cg) {
	if(v_ins.size() == 1) return v_ins[0];
	dynet::Expression i_average = dynet::average(v_ins);
	return dynet::log_softmax({i_average});
}

inline int max_len(const WordIdSentence & sent) { return sent.size(); }
inline int max_len(const std::vector<WordIdSentence> & sent) {
	size_t val = 0;
	for (const auto & s : sent){
		val = std::max(val, s.size()); 
	}
	return val;
}

inline int get_word(const std::vector<WordIdSentence> & vec, int t) { return vec[0][t]; }
inline int get_word(const WordIdSentence & vec, int t) { return vec[t]; }

EnsembleDecoderHypPtr EnsembleDecoder::generate(dynet::ComputationGraph& cg
	, const WordIdSentence & sent_src
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models) 
{
	auto nbest = generate_nbest(cg, sent_src, v_models, 1);
	return (nbest.size() > 0 ? nbest[0] : EnsembleDecoderHypPtr());
}

std::vector<EnsembleDecoderHypPtr> EnsembleDecoder::generate_nbest(dynet::ComputationGraph& cg
	, const WordIdSentence & sent_src
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned nbest_size/*FIXME: segmentation fault error with nbest_size <= 40?*/) 
{ 
	// Sentinel symbols
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;
	  
	// compute source representation
	//cerr << "GenerateNbest::(1)" << endl;
	std::vector<dynet::Expression> v_src_reps;
	for (auto & tf : v_models){
		v_src_reps.push_back(tf.get()->compute_source_rep(cg, WordIdSentences(1, sent_src)/*pseudo batch (1)*/));
	}

	// The n-best hypotheses
	std::vector<EnsembleDecoderHypPtr> nbest;

	// Create the initial hypothesis
	std::vector<EnsembleDecoderHypPtr> curr_beam(1, EnsembleDecoderHypPtr(new EnsembleDecoderHyp(0.0, WordIdSentence(1, sm._kTGT_SOS), WordIdSentence(1, 0))));

	int bid;
	Expression empty_idx;

	// limit the output length
	_size_limit = sent_src.size() * 3/*x*/;// not generating target with length "x times" the source length

	// Perform decoding
	for (int sent_len = 0; sent_len <= _size_limit; sent_len++) {
		// This vector will hold the best IDs
		std::vector<Beam_Info> next_beam_id(_beam_size+1, Beam_Info(-DBL_MAX,-1,-1,-1));

		// Go through all the hypothesis IDs
		//cerr << "GenerateNbest::(2)" << endl;
		for (int hypid = 0; hypid < (int)curr_beam.size(); hypid++) {
			EnsembleDecoderHypPtr curr_hyp = curr_beam[hypid];
			const WordIdSentence& sent = curr_beam[hypid]->get_sentence();// partial generated sentence from current hypo in the beam

			if (sent_len != 0 && *sent.rbegin() == sm._kTGT_EOS) continue;

			cg.checkpoint();

			// Perform the forward step on all models
			//cerr << "GenerateNbest::(2)::(a,Forward) ";
			std::vector<Expression> i_softmaxes, i_aligns;
			for(int j : boost::irange(0, (int)v_models.size())){
				i_softmaxes.push_back(v_models[j].get()->step_forward(cg, v_src_reps[j]
					, sent
					, _ensemble_operation == "logsum"
					, i_aligns));
			}

			// Ensemble and calculate the likelihood
			//cerr << "GenerateNbest::(2)::(b,Ensemble) ";
			Expression i_softmax, i_logprob;
			if (_ensemble_operation == "sum") {
				i_softmax = ensemble_probs(i_softmaxes, cg);
				i_logprob = dynet::log({i_softmax});
			}
			else if (_ensemble_operation == "logsum") {
				i_logprob = ensemble_logprobs(i_softmaxes, cg);
			}
			else
				assert(string("Bad ensembling operation: " + _ensemble_operation).c_str());

			// Get the (log) softmax predictions
			//cerr << "GenerateNbest::(2)::(c,softmax) ";
			std::vector<float> softmaxes = dynet::as_vector(cg.incremental_forward(i_logprob));

			// Add the word/unk penalties
			//  - word penalty
			if (_word_pen != 0.f) {
				for(size_t i = 0; i < softmaxes.size(); i++)
					softmaxes[i] += _word_pen;
			}
			//  - unk penalty
			if (_unk_id >= 0) softmaxes[_unk_id] += _unk_pen * _unk_log_prob;

			// Find the best aligned source, if any alignments exists
			//cerr << "GenerateNbest::(2)::(d,Align) ";
			WordId best_align = -1;
			if (i_aligns.size() != 0) {
				dynet::Expression ens_align = dynet::sum(i_aligns);
				std::vector<float> align = dynet::as_vector(cg.incremental_forward(ens_align));
				best_align = 0;
				for(size_t aid = 0; aid < align.size(); aid++)
					if(align[aid] > align[best_align])
						best_align = aid;
			}

			// Find the best IDs in the beam
			//cerr << "GenerateNbest::(2)::(e,ID) ";
			for (int wid = 0; wid < (int)softmaxes.size(); wid++) {
				float my_score = curr_hyp->get_score() + softmaxes[wid];
				for (bid = _beam_size; bid > 0 && my_score > std::get<0>(next_beam_id[bid-1]); bid--)
					next_beam_id[bid] = next_beam_id[bid-1];
				next_beam_id[bid] = Beam_Info(my_score, hypid, wid, best_align);
			}

			cg.revert();
		}

		// Create the new hypotheses
		//cerr << endl << "GenerateNbest::(3) " << endl;
		std::vector<EnsembleDecoderHypPtr> next_beam;
		for (int i = 0; i < _beam_size; i++) {
			float score = std::get<0>(next_beam_id[i]);
			int hypid = std::get<1>(next_beam_id[i]);
			int wid = std::get<2>(next_beam_id[i]);
			int aid = std::get<3>(next_beam_id[i]);

			if (hypid == -1) break;

			WordIdSentence next_sent = curr_beam[hypid]->get_sentence();
			next_sent.push_back(wid);

			WordIdSentence next_align = curr_beam[hypid]->get_alignment();
			next_align.push_back(aid);

			EnsembleDecoderHypPtr hyp(new EnsembleDecoderHyp(score, next_sent, next_align));

			if (wid == sm._kTGT_EOS && hyp->get_sentence().size() == 2) //as of 26 April 2017: excluding: <s> </s>
				continue;

			if (wid == sm._kTGT_EOS || sent_len == _size_limit)
				nbest.push_back(hyp);

			next_beam.push_back(hyp);
		}

		curr_beam = next_beam;

		// Check if we're done with search
		//cerr << "GenerateNbest::(4) " << endl;
		if(nbest.size() != 0) {
			sort(nbest.begin(), nbest.end());

			if(nbest.size() > nbest_size) nbest.resize(nbest_size);
			if(nbest.size() == nbest_size && (curr_beam.size() == 0 || (*nbest.rbegin())->get_score() >= next_beam[0]->get_score()))
				return nbest;
		}

		//if current beam size is 0, stop!
		if(curr_beam.size() == 0) break;
	}

	if (_verbose) cerr << "WARNING: Generated sentence size exceeded " << _size_limit << ". Truncating." << endl;

	return nbest;
}

