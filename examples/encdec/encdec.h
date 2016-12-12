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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int kSOS;
int kEOS;


unsigned INPUT_VOCAB_SIZE;
unsigned OUTPUT_VOCAB_SIZE;

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
    // Hyperparameters
    unsigned LAYERS;
    unsigned INPUT_DIM;
    unsigned HIDDEN_DIM;
    bool bidirectional;

    LookupParameter p_c;
    LookupParameter p_ec;  // map input to embedding (used in fwd and rev models)
    Parameter p_ie2oe;
    Parameter p_boe;
    Parameter p_R;
    Parameter p_bias;
    Builder dec_builder;
    Builder fwd_enc_builder;
    Builder rev_enc_builder;

public:
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
                            unsigned input_dim,
                            unsigned hidden_dim,
                            bool bwd = false) :
        LAYERS(num_layers), INPUT_DIM(input_dim), HIDDEN_DIM(hidden_dim), bidirectional(bwd),
        dec_builder(num_layers, input_dim, hidden_dim, model),
        fwd_enc_builder(num_layers, input_dim, hidden_dim, model) {
        
        if (bidirectional) {
            rev_enc_builder = Builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
            p_ie2oe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 2),
                                            unsigned(HIDDEN_DIM * LAYERS * 4)
                                           });
            p_boe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 2)});
        }

        p_c = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM});
        p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM});
        p_R = model.add_parameters({OUTPUT_VOCAB_SIZE, HIDDEN_DIM});
        p_bias = model.add_parameters({OUTPUT_VOCAB_SIZE});

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
    }

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
    Expression encode(const vector<int>& insent, ComputationGraph & cg) {
        vector<vector<int>> isents;
        isents.push_back(insent);
        unsigned chars = 0;
        return encode(isents, 0, 1, chars, cg);
    }

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
    Expression decode(const Expression i_nc,
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

    }

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
    Expression decode(const Expression i_nc,
                      const vector<int>& osent,
                      ComputationGraph & cg) {
        vector<vector<int>> osents;
        osents.push_back(osent);
        return decode(i_nc, osents, 0, 1, cg);
    }

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
    vector<int> generate(const vector<int>& insent, ComputationGraph & cg) {
        return generate(encode(insent, cg), 2 * insent.size() - 1, cg);
    }

    /**
     * @brief Generate a sentence from an encoding
     * @details You can use this directly to generate random sentences
     * 
     * @param i_nc Input encoding
     * @param oslen Maximum length of output
     * @param cg Computation graph
     * @return Generated sentence (indices in the dictionary)
     */
    vector<int> generate(Expression i_nc, unsigned oslen, ComputationGraph & cg) {

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
    }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & bidirectional;
        ar & LAYERS & INPUT_DIM & HIDDEN_DIM;
        ar & p_c & p_ec & p_R & p_bias;
        if (bidirectional)
            ar & p_ie2oe & p_boe;
        if (bidirectional)
            ar & dec_builder & rev_enc_builder & fwd_enc_builder;
        else
            ar & dec_builder & fwd_enc_builder;
    }
    inline int sample(const vector<float>& v) {
        float p = (float)rand() / (float) RAND_MAX;
        float cumul = 0.f;
        int idx = 0;
        while (idx < v.size() && p > cumul) {
            cumul += v[idx];
            idx += 1;
        }
        return idx ? idx - 1 : 0;
    }
};
