/**
 * \file encdec.h
 * \defgroup seq2seqbuilders seq2seqbuilders
 * \brief Sequence to sequence models
 * 
 * An example implementation of a simple sequence to sequence model based on lstm encoder/decoder
 *
 */

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"

#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int kSOS;
int kEOS;


unsigned INPUT_VOCAB_SIZE;
unsigned OUTPUT_VOCAB_SIZE;

unordered_map<unsigned, vector<float>> src_pret, tgt_pret;

/**
 * \ingroup seq2seqbuilders
 * 
 * \struct EncoderDecoder
 * \brief This structure is a "vanilla" encoder decoder model
 * \details This sequence to sequence network models the conditional probability
 * \f$p(y_1,\dots,y_m\vert x_1,\dots,x_n)=\prod_{i=1}^m p(y_i\vert \textbf{e},y_1,\dots,y_{i-1})\f$
 * where \f$\textbf{e}=ENC(x_1,\dots,x_n)\f$ is an encoding of the input sequence
 * produced by a recurrent neural network.
 *
 * Typically \f$\textbf{e}\f$ is the concatenated cell and output vector of a (multilayer) LSTM.
 *
 * Sequence to sequence models were introduced in [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) .
 *
 * Our implementation is more akin to the one from [Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) .
 *
 * \tparam Builder This can theoretically be any RNNbuilder. It's only been tested with an LSTM as
 * of now
 */
template <class Builder>
struct EncoderDecoder {
private:

    LookupParameter p_c; // output embedding lookup
    LookupParameter p_ec;  // map input to embedding (used in fwd and rev models)
    LookupParameter p_pt_src;  // pretrained word embedding for the source language 
    LookupParameter p_pt_tgt;  // pretrained word embedding for the target language
    Parameter p_ie2oe; // encoder to decoder connection
    Parameter p_boe; // encoder to decoder bias
    Parameter p_input_pret_src; // mapping from pretrained dim to input dim
    Parameter p_input_pret_tgt; // mapping from pretrained dim to input dim
    Builder dec_builder; // LSTM decoder
    Builder fwd_enc_builder; // LSTM forward encoder
    Builder rev_enc_builder; // LSTM backward encoder
    Builder tgt_builder; // LSTM decoder
    // initialize the expressions
    Expression i_ie2oe;
    Expression i_bie;
    Expression input_pret_src;
    Expression input_pret_tgt;

public:
    // Hyperparameters
    unsigned LAYERS;
    unsigned OUT_LAYERS;
    unsigned INPUT_DIM;
    unsigned HIDDEN_DIM;
    unsigned PRET_SRC_DIM;
    unsigned PRET_TGT_DIM;
    //bool bidirectional;
    float DROPOUT;
    /**
     * \brief Default builder
     */
    EncoderDecoder() {}

    /**
     * \brief Creates an EncoderDecoder
     *
     * \param model Model holding the parameters
     * \param num_layers Number of layers (same in the ecoder and decoder)
     * \param input_dim Dimension of the word/char embeddings
     * \param hidden_dim Dimension of the hidden states
     * \param bwd Set to `true` to make the encoder bidirectional. This doubles the number
     * of parameters in the encoder. This will also add parameters for an affine transformation
     * from the bidirectional encodings (of size num_layers * 2 * hidden_dim) to encodings
     * of size num_layers * hidden_dim compatible with the decoder
     *
     */
    explicit EncoderDecoder(Model& model,
                            unsigned num_layers,
                            unsigned num_out_layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
			    unsigned pret_src_dim,
			    unsigned pret_tgt_dim,
                            float dropout) :
        LAYERS(num_layers), OUT_LAYERS(num_out_layers), INPUT_DIM(input_dim), HIDDEN_DIM(hidden_dim), PRET_SRC_DIM(pret_src_dim), PRET_TGT_DIM(pret_tgt_dim), DROPOUT(dropout),
        dec_builder(num_out_layers, input_dim, hidden_dim, model),
        fwd_enc_builder(num_layers, input_dim, hidden_dim, model),
        rev_enc_builder(num_layers, input_dim, hidden_dim, model), 
        tgt_builder(num_out_layers, input_dim, hidden_dim, model) {
        
        p_ie2oe = model.add_parameters({unsigned(HIDDEN_DIM * OUT_LAYERS), // maps the memory cells of the encoder to the memory cells of the decoder
                                        unsigned(HIDDEN_DIM * LAYERS)
                                       });
        p_boe = model.add_parameters({unsigned(HIDDEN_DIM * OUT_LAYERS)});

        p_c = model.add_lookup_parameters(OUTPUT_VOCAB_SIZE, {INPUT_DIM});
        p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM});

        if(src_pret.size() > 0) {
          p_pt_src = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {PRET_SRC_DIM});
          p_input_pret_src = model.add_parameters({INPUT_DIM,
                                                PRET_SRC_DIM
                                                });

          vector<float> zeros;
	  for(unsigned i = 0; i < PRET_SRC_DIM; ++i) {
	    zeros.push_back(0.0f);
	  }
	  assert(zeros.size() == PRET_SRC_DIM);
          for (unsigned j = 0; j < INPUT_VOCAB_SIZE; ++j) {
	    p_pt_src.initialize(j, zeros);
	  } 
   	  for (auto it: src_pret) {
	    p_pt_src.initialize(it.first, it.second); 
	  }
	}
        if(tgt_pret.size() > 0) {
          p_pt_tgt = model.add_lookup_parameters(OUTPUT_VOCAB_SIZE, {PRET_TGT_DIM});
	  p_input_pret_tgt = model.add_parameters({INPUT_DIM,
                                                PRET_TGT_DIM
                                                });
            vector<float> zeros;
            for(unsigned i = 0; i < PRET_TGT_DIM; ++i) {
              zeros.push_back(0.0f);
            }
	    assert(zeros.size() == PRET_TGT_DIM);
            for (unsigned j = 0; j < OUTPUT_VOCAB_SIZE; ++j) {
              p_pt_tgt.initialize(j, zeros);
            }
            for (auto it: tgt_pret) {
              p_pt_tgt.initialize(it.first, it.second);
            }
	}
    }

    // returns the pair<bidirectional encoding of each word at the source sentence, the concatenated memory states of the bidirectional encoder>
    // DO NOT USE, use unidirectional encoding (i.e. the encode method)
    pair<vector<Expression>, Expression> encode_bidir(const vector<int>& src, ComputationGraph& cg, bool apply_dropout) {
       fwd_enc_builder.new_graph(cg);
       rev_enc_builder.new_graph(cg);
       fwd_enc_builder.start_new_sequence();
       rev_enc_builder.start_new_sequence();
       // Process dropout
       if(apply_dropout) {
         assert(DROPOUT >= 0.0);
         fwd_enc_builder.set_dropout(DROPOUT);
         rev_enc_builder.set_dropout(DROPOUT);
       } else {
	 fwd_enc_builder.disable_dropout();
	 rev_enc_builder.disable_dropout();
       }

       fwd_enc_builder.add_input(lookup(cg, p_ec, kSOS));
       rev_enc_builder.add_input(lookup(cg, p_ec, kEOS));

       unsigned slen = src.size();
       
       vector<Expression> fwd;
       deque<Expression> rev;

       for (unsigned i = 0; i < src.size(); ++i) {
         fwd_enc_builder.add_input(lookup(cg, p_ec, src[i]));
         fwd.push_back(fwd_enc_builder.back());
         rev_enc_builder.add_input(lookup(cg, p_ec, src[slen - i - 1]));
         rev.push_front(rev_enc_builder.back());
       }
       assert(fwd.size() == rev.size() && fwd.size() == slen);

       // now concatenate the forward and backward
       vector<Expression> concatenated;
       for(unsigned i = 0; i < fwd.size(); ++i) {
	  vector<Expression> temp;
          temp.push_back(fwd[i]); 
	  temp.push_back(rev[i]);
	  assert(temp.size() == 2);
	  concatenated.push_back(concatenate(temp));
       }
       assert(concatenated.size() == slen && concatenated.size() == fwd.size());

       // get the final memory cells of the forward and backward
       auto final_s_fwd = fwd_enc_builder.final_s();
       auto final_s_rev = rev_enc_builder.final_s();
       assert(final_s_fwd.size() == 2 && final_s_rev.size() == 2); // the forward and backward encoders are single-layered
       // get the memory cells
       vector<Expression> s_concatenated;
       s_concatenated.push_back(final_s_fwd[0]);
       s_concatenated.push_back(final_s_rev[0]);
       assert(s_concatenated.size() == 2);
       Expression final_s = concatenate(s_concatenated); 

      Expression i_ie2oe = parameter(cg, p_ie2oe);
      Expression i_bie = parameter(cg, p_boe);
      //i_nc = i_bie + i_ie2oe * i_combined;
      Expression i_nc = affine_transform({i_bie, i_ie2oe, final_s}); 
       
       return make_pair(concatenated, i_nc);
    }

   void resetCG(ComputationGraph& cg, bool apply_dropout) {
      // First the fwd_enc_builder
      fwd_enc_builder.new_graph(cg);
       if(apply_dropout) {
         assert(DROPOUT >= 0.0);
         fwd_enc_builder.set_dropout(DROPOUT);
       } else {
         fwd_enc_builder.disable_dropout();
       }

       // Next the decbuilder 
       dec_builder.new_graph(cg);
       // Initialize new sequence with encoded states
       if(apply_dropout) dec_builder.set_dropout(DROPOUT);
       else dec_builder.disable_dropout();

       // Next the tgt builder
       tgt_builder.new_graph(cg);
       if(apply_dropout) tgt_builder.set_dropout(DROPOUT);
       else tgt_builder.disable_dropout();

       // initialize the Expression for this computation graph
       input_pret_src = parameter(cg, p_input_pret_src);
       i_ie2oe = parameter(cg, p_ie2oe);
       i_bie = parameter(cg, p_boe);
       input_pret_tgt = parameter(cg, p_input_pret_tgt);
   }

    pair<vector<Expression>, Expression> encode(const vector<int>& src, ComputationGraph& cg) {
       //fwd_enc_builder.new_graph(cg);
       //if(apply_dropout) {
       //  assert(DROPOUT >= 0.0);
       //  fwd_enc_builder.set_dropout(DROPOUT);
       //} else {
       //  fwd_enc_builder.disable_dropout();
       //}

       fwd_enc_builder.start_new_sequence();
       //cerr << "Starting the dropout" << endl;
       // Process dropout
       //if(apply_dropout) {
       //  assert(DROPOUT >= 0.0);
       //  fwd_enc_builder.set_dropout(DROPOUT);
       //} else {
       //  fwd_enc_builder.disable_dropout();
       //}
       //cerr << "Ending the dropout" << endl;

       fwd_enc_builder.add_input(lookup(cg, p_ec, kSOS));

       //cerr << "Adding <s>" << endl;

       unsigned slen = src.size();

       vector<Expression> fwd;
       fwd.push_back(fwd_enc_builder.back());


       for (unsigned i = 0; i < src.size(); ++i) {
         Expression i_i = lookup(cg, p_ec, src[i]);
         if(src_pret.count(src[i])) {
           Expression pret = const_lookup(cg, p_pt_src, src[i]);
	   i_i = affine_transform({i_i, input_pret_src, pret});
         }
         fwd_enc_builder.add_input(i_i);
         fwd.push_back(fwd_enc_builder.back());
       }
       //cerr << "Practically done encoding the sentence" << endl;
       assert(fwd.size() == (slen + 1));

       // get the final memory cells of the forward and backward
       auto final_s_fwd = fwd_enc_builder.final_s();
       assert(final_s_fwd.size() == 2); // the forward and backward encoders are single-layered
       // get the memory cells
       Expression final_s = final_s_fwd[0];

      //i_nc = i_bie + i_ie2oe * i_combined;
      Expression i_nc = affine_transform({i_bie, i_ie2oe, final_s});

      //cerr << "Done encoding the whole sentence!" << endl;

       return make_pair(fwd, i_nc);
    }

    vector<Expression> decode(const Expression i_nc, // concatenated memory states of the {forward, backward} encoders
                      const vector<int>& osent,
                      ComputationGraph & cg
                      ) {
        // Reconstruct input states from encodings -------------------------------------------------
        // List of input states for decoder 
        vector<Expression> oein;
        // Add input cell states
        for (unsigned i = 0; i < OUT_LAYERS; ++i) {
            oein.push_back(pick_range(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
        }
        // Add input output states
        for (unsigned i = 0; i < OUT_LAYERS; ++i) {
            oein.push_back(tanh(oein[i]));
        }
        assert(oein.size() == 2 * OUT_LAYERS); // the memory cells and the layer after tanh
        // Initialize graph for decoder
        //dec_builder.new_graph(cg);
        // Initialize new sequence with encoded states

        //if(apply_dropout) dec_builder.set_dropout(DROPOUT);
        //else dec_builder.disable_dropout();
        dec_builder.start_new_sequence(oein);
        // Run decoder -----------------------------------------------------------------------------
        // Set start of sentence
        dec_builder.add_input(lookup(cg, p_c, kSOS));

        vector<Expression> output;
        output.push_back(dec_builder.back());
  	for (unsigned i = 0; i < osent.size(); ++i) {
	  Expression i_i = lookup(cg, p_c, osent[i]);
          if(tgt_pret.count(osent[i])) {
	    Expression pret = const_lookup(cg, p_pt_tgt, osent[i]); 
	    i_i = affine_transform({i_i, input_pret_tgt, pret});
	  }

	  dec_builder.add_input(i_i); 
          output.push_back(dec_builder.back());    
	}
        assert(output.size() == (osent.size() + 1));
        return output;
    }

    vector<Expression> encode_tgt(const vector<vector<int>>& cands, ComputationGraph& cg) {
      vector<Expression> candsEnc;
      //tgt_builder.new_graph(cg);
      //if(apply_dropout) tgt_builder.set_dropout(DROPOUT);
      //else tgt_builder.disable_dropout();
  
      for(unsigned i = 0; i < cands.size(); ++i) {
	tgt_builder.start_new_sequence();
        assert(cands[i].size() > 0);
        for(unsigned j = 0; j < cands[i].size(); ++j) {
	  Expression i_i = lookup(cg, p_c, cands[i][j]);
          if(tgt_pret.count(cands[i][j])) {
	     Expression pret = const_lookup(cg, p_pt_tgt, cands[i][j]);
	      i_i = affine_transform({i_i, input_pret_tgt, pret});
          }
          tgt_builder.add_input(i_i);
        }	
	candsEnc.push_back(tgt_builder.back());
      }
      assert(candsEnc.size() == cands.size());
      return candsEnc;
    }


    /**
     * \brief Batched encoding
     * \details Encodes a batch of sentences of the same size (don't forget to pad them)
     *
     * \param isents Whole dataset
     * \param id Index of the start of the batch
     * \param bsize Batch size
     * \param chars Number of tokens processed (used to compute loss per characters)
     * \param cg Computation graph
     * \return Returns the expression for the negative (batched) encoding
     */
    /*
    Expression encode(const vector<vector<int>>& isents,
                      unsigned id,
                      unsigned bsize,
                      unsigned & chars,
                      ComputationGraph & cg) {
        // Set variables for the input sentence
        const unsigned islen = isents[id].size();
        vector<unsigned> x_t(bsize);

        // Forward encoder -------------------------------------------------------------------------

        // Initialize parameters in fwd_enc_builder
        fwd_enc_builder.new_graph(cg);
        // Initialize the sequence
        fwd_enc_builder.start_new_sequence();

        // Run the forward encoder on the batch
        for (unsigned t = 0; t < islen; ++t) {
            // Fill x_t with the characters at step t in the batch
            for (unsigned i = 0; i < bsize; ++i) {
                x_t[i] = isents[id + i][t];
                if (x_t[i] != *isents[id].rbegin()) chars++; // if x_t is non-EOS, count a char
            }
            // Get embedding
            Expression i_x_t = lookup(cg, p_ec, x_t);
            // Run a step in the forward encoder
            fwd_enc_builder.add_input(i_x_t);
        }

        // Backward encoder ------------------------------------------------------------------------
        if (bidirectional) {
            // Initialize parameters in bwd_enc_builder
            rev_enc_builder.new_graph(cg);
            // Initialize the sequence
            rev_enc_builder.start_new_sequence();
            // Fill x_t with the characters at step t in the batch (in reverse order)
            for (int t = islen - 1; t >= 0; --t) {
                for (int i = 0; i < bsize; ++i) {
                    x_t[i] = isents[id + i][t];
                }
                // Get embedding (could be mutualized with fwd_enc_builder)
                Expression i_x_t = lookup(cg, p_ec, x_t);
                // Run a step in the forward encoder
                rev_enc_builder.add_input(i_x_t);
            }
        }

        // Collect encodings -----------------------------------------------------------------------
        vector<Expression> to;
        // Get states from forward encoder
        for (auto s_l : fwd_enc_builder.final_s()) to.push_back(s_l);
        // Get states from backward encoder
        if (bidirectional)
            for (auto s_l : rev_enc_builder.final_s()) to.push_back(s_l);

        // Put it as a vector (matrix because it's batched)
        Expression i_combined = concatenate(to);
        Expression i_nc;
        if (bidirectional) {
            // Perform an affine transformation for rescaling in case of bidirectional encoder
            Expression i_ie2oe = parameter(cg, p_ie2oe);
            Expression i_bie = parameter(cg, p_boe);
            i_nc = i_bie + i_ie2oe * i_combined;
        } else {
            // Otherwise just copy the states
            i_nc = i_combined;
        }

        return i_nc;
    } */

    /**
     * \brief Single sentence version of `encode`
     * \details Note : this just creates a trivial dataset and feed it to the batched version with
     * batch_size 1. It's not very effective so don't use it for training.
     *
     * \param insent Input sentence
     * \param cg Computation graph
     *
     * \return Expression of the encoding
     */
    /*Expression encode(const vector<int>& insent, ComputationGraph & cg) {
        vector<vector<int>> isents;
        isents.push_back(insent);
        unsigned chars = 0;
        return encode(isents, 0, 1, chars, cg);
    } */

    /**
     * \brief Batched decoding
     * \details [long description]
     *
     * \param i_nc Encoding (should be batched)
     * \param osents Output sentences dataset
     * \param id Start index of the batch
     * \param bsize Batch size (should be consistent with the shape of `i_nc`)
     * \param cg Computation graph
     * \return Expression for the negative log likelihood
     */
    /*Expression decode(const Expression i_nc,
                      const vector<vector<int>>& osents,
                      int id,
                      int bsize,
                      ComputationGraph & cg) {
        // Reconstruct input states from encodings -------------------------------------------------
        // List of input states for decoder 
        vector<Expression> oein;
        // Add input cell states
        for (unsigned i = 0; i < LAYERS; ++i) {
            oein.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM ));
        }
        // Add input output states
        for (unsigned i = 0; i < LAYERS; ++i) {
            oein.push_back(pickrange(i_nc, HIDDEN_DIM * LAYERS + i * HIDDEN_DIM,
                                     HIDDEN_DIM * LAYERS + (i + 1) * HIDDEN_DIM));
        }
        // Initialize graph for decoder
        dec_builder.new_graph(cg);
        // Initialize new sequence with encoded states
        dec_builder.start_new_sequence(oein);
        // Run decoder -----------------------------------------------------------------------------
        // Add parameters to the graph
        Expression i_R = parameter(cg, p_R);
        Expression i_bias = parameter(cg, p_bias);
        // Initialize errors and input vectors
        vector<Expression> errs;
        vector<unsigned> x_t(bsize);
        // Set start of sentence
        for (int i = 0; i < bsize; ++i) {
            x_t[i] = osents[id + i][0];
        }
        vector<unsigned> next_x_t(bsize);
        const unsigned oslen = osents[id].size();
        // Run on output sentence
        for (unsigned t = 1; t < oslen; ++t) {
            // Retrieve input
            for (int i = 0; i < bsize; ++i) {
                next_x_t[i] = osents[id + i][t];
            }
            // Embed token
            Expression i_x_t = lookup(cg, p_c, x_t);
            // Run decoder step
            Expression i_y_t = dec_builder.add_input(i_x_t);
            // Project from output dim to dictionary dimension
            Expression i_r_t = i_bias + i_R * i_y_t;
            // Compute softmax and negative log
            Expression i_err = pickneglogsoftmax(i_r_t, next_x_t);
            errs.push_back(i_err);
            x_t = next_x_t;
        }
        // Sum loss over batch
        Expression i_nerr = sum_batches(sum(errs));
        return i_nerr;
    } */

    /**
     * \brief Single sentence version of `decode`
     * \details For similar reasons as `encode`, this is not really efficient. USed the batched 
     * version directly for training
     * 
     * \param i_nc Encoding
     * \param osent Output sentence
     * \param cg Computation graph
     * \return Expression for the negative log likelihood
     */
     /*
    Expression decode(const Expression i_nc,
                      const vector<int>& osent,
                      ComputationGraph & cg) {
        vector<vector<int>> osents;
        osents.push_back(osent);
        return decode(i_nc, osents, 0, 1, cg);
    } */

    /**
     * \brief Generate a sentence from an input sentence
     * \details Samples at each timestep ducring decoding. Possible variations are greedy decoding
     * and beam search for better performance
     * 
     * \param insent Input sentence
     * \param cg Computation Graph
     * 
     * \return Generated sentence (indices in the dictionary)
     */
    /*
    vector<int> generate(const vector<int>& insent, ComputationGraph & cg) {
        return generate(encode(insent, cg, false), 2 * insent.size() - 1, cg);
    } */

    /**
     * @brief Generate a sentence from an encoding
     * @details You can use this directly to generate random sentences
     * 
     * @param i_nc Input encoding
     * @param oslen Maximum length of output
     * @param cg Computation graph
     * @return Generated sentence (indices in the dictionary)
     */
    /*vector<int> generate(Expression i_nc, unsigned oslen, ComputationGraph & cg) {

        vector<Expression> oein1, oein2, oein;
        for (unsigned i = 0; i < LAYERS; ++i) {
            oein1.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
            oein2.push_back(tanh(oein1[i]));
        }
        for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein1[i]);
        for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein2[i]);

        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(oein);

        // decoder
        Expression i_R = parameter(cg, p_R);
        Expression i_bias = parameter(cg, p_bias);
        vector<int> osent;
        osent.push_back(kSOS);

        for (unsigned t = 0; t < oslen; ++t) {
            Expression i_x_t = lookup(cg, p_c, osent[t]);
            Expression i_y_t = dec_builder.add_input(i_x_t);
            Expression i_r_t = i_bias + i_R * i_y_t;
            Expression i_ydist = softmax(i_r_t);
            vector<float> probs = as_vector(i_ydist.value());
            osent.push_back(sample(probs));
            if (osent[t + 1] == kEOS) break;
        }
        return osent;
    } */

/*
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int) {
        //ar & bidirectional;
        ar & LAYERS & INPUT_DIM & HIDDEN_DIM;
        ar & p_c & p_ec & p_R & p_bias;
        //if (bidirectional)
            ar & p_ie2oe & p_boe;
        //if (bidirectional)
            ar & dec_builder & rev_enc_builder & fwd_enc_builder;
        //else
           // ar & dec_builder & fwd_enc_builder;
    } */
     /*
    inline int sample(const vector<float>& v) {
        float p = (float)rand() / (float) RAND_MAX;
        float cumul = 0.f;
        int idx = 0;
        while (idx < v.size() && p > cumul) {
            cumul += v[idx];
            idx += 1;
        }
        return idx ? idx - 1 : 0;
    } */
};
