#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/model.h"
#include "cnn/rnn.h"
#include "cnn/timing.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include "cnn/treelstm.h"
#include "conll_helper.cc"

using namespace cnn;

//TODO: add code for POS tag dictionary and dependency relation dictionary
cnn::Dict tokdict, sentitagdict, depreldict;
vector<unsigned> sentitaglist; // a list of all sentiment tags
unordered_map<unsigned, vector<float>> pretrained;

const string UNK_STR = "UNK";

unsigned VOCAB_SIZE = 0, DEPREL_SIZE = 0, SENTI_TAG_SIZE = 0;

unsigned LAYERS = 1;
unsigned PRETRAINED_DIM = 300;
unsigned DEPREL_DIM = 100;
unsigned LSTM_INPUT_DIM = 300;
unsigned HIDDEN_DIM = 168;

template<class Builder>
struct OurSentimentModel {
    LookupParameters* p_x;
    LookupParameters* p_e;
    LookupParameters* p_emb;

    Parameters* p_emb2l;
    Parameters* p_tok2l;
    Parameters* p_dep2l;
    Parameters* p_inp_bias;

    Parameters* p_root2senti;
    Parameters* p_sentibias;

    Builder treebuilder;

    explicit OurSentimentModel(Model &model) :
            treebuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, &model) {
        p_x = model.add_lookup_parameters(VOCAB_SIZE, { PRETRAINED_DIM });
        p_e = model.add_lookup_parameters(DEPREL_SIZE, { DEPREL_DIM });

        p_tok2l = model.add_parameters( { LSTM_INPUT_DIM, PRETRAINED_DIM });
        p_dep2l = model.add_parameters( { LSTM_INPUT_DIM, DEPREL_DIM });
        p_inp_bias = model.add_parameters( { LSTM_INPUT_DIM });
        // TODO: Change to add a regular BiLSTM below the tree

        p_root2senti = model.add_parameters( { SENTI_TAG_SIZE, HIDDEN_DIM });
        p_sentibias = model.add_parameters( { SENTI_TAG_SIZE });

        if (pretrained.size() > 0) {
            p_emb = model.add_lookup_parameters(VOCAB_SIZE, { PRETRAINED_DIM });
            for (auto it : pretrained) {
                p_emb->Initialize(it.first, it.second);
            }
            p_emb2l = model.add_parameters( { LSTM_INPUT_DIM, PRETRAINED_DIM });
        } else {
            p_emb = nullptr;
            p_emb2l = nullptr;
        }
    }

    Expression BuildTreeCompGraph(const DepTree& tree,
            const vector<int>& sentilabel, ComputationGraph* cg,
            int* prediction) {
        bool is_training = true;
        if (sentilabel.size() == 0) {
            is_training = false;
        }
        vector < Expression > errs; // the sum of this is to be returned...

        treebuilder.new_graph(*cg);
        treebuilder.start_new_sequence();
        treebuilder.initialize_structure(tree.nummsgs + tree.numnodes - 1);

        Expression tok2l = parameter(*cg, p_tok2l);
        Expression dep2l = parameter(*cg, p_dep2l);
        Expression inp_bias = parameter(*cg, p_inp_bias);

        Expression emb2l;
        if (p_emb2l) {
            emb2l = parameter(*cg, p_emb2l);
        }

        Expression root2senti = parameter(*cg, p_root2senti);
        Expression senti_bias = parameter(*cg, p_sentibias);

        vector<Expression> i_words, i_deprels, inputs;
        for (unsigned n = 0; n < tree.numnodes; n++) {
            Expression i_word = lookup(*cg, p_x, tree.sent[n]);
            if (p_emb && pretrained.count(tree.sent[n])) {
                Expression pre = const_lookup(*cg, p_emb, tree.sent[n]);
                i_word = affine_transform( { i_word, emb2l, pre });
            }
            i_words.push_back(i_word);
            i_deprels.push_back(lookup(*cg, p_e, tree.deprels[n]));

            inputs.push_back(affine_transform( { inp_bias, tok2l,
                    i_words.back(), dep2l, i_deprels.back() })); // TODO: add POS
        }

//        cerr << "Full graph size = " << (tree.nummsgs + tree.numnodes - 1)
//                << endl;
//        cerr << "Tree num msgs = " << tree.nummsgs << endl;
        for (DepEdge edge : tree.dfo_msgs) { // Bottom up and top down
            unsigned node = edge.head;

            // find id of edge
            auto edgeid_pos = tree.msgdict.find(edge);
            assert(edgeid_pos != tree.msgdict.end());
            unsigned edgeid = edgeid_pos->second;

            auto nbrs_vec = tree.msg_nbrs.find(edgeid);
            assert(nbrs_vec != tree.msg_nbrs.end());
            vector<unsigned> msg_neighbors = nbrs_vec->second;
            treebuilder.add_input(edgeid, msg_neighbors, inputs[node]);
//            cerr << "processing " << edgeid->second << " ";
//            edge.print();
//            cerr << endl;
        }

        for (unsigned node : tree.dfo) {

            auto nbrs_vec = tree.node_msg_nbrs.find(node);
            assert(nbrs_vec != tree.node_msg_nbrs.end());
            vector<unsigned> edge_neighbors = nbrs_vec->second;

            unsigned nodeincg = tree.nummsgs + node - 1;
            Expression z_node = treebuilder.add_input(nodeincg, edge_neighbors,
                    inputs[node]);
//            cerr << "finalizing " << n << " graphnode at "
//                    << tree.nummsgs + n - 1 << endl;

            Expression i_node = affine_transform( { senti_bias, root2senti,
                    z_node });

            Expression prob_dist = log_softmax(i_node, sentitaglist);
            vector<float> prob_dist_vec = as_vector(cg->incremental_forward());

            int chosen_sentiment;
            if (is_training) {
                chosen_sentiment = sentilabel[node];
            } else { // the argmax

                double best_score = prob_dist_vec[0];
                chosen_sentiment = 0;
                for (unsigned i = 1; i < prob_dist_vec.size(); ++i) {
                    if (prob_dist_vec[i] > best_score) {
                        best_score = prob_dist_vec[i];
                        chosen_sentiment = i;
                    }
                }
                if (node == tree.root) {  // -- only for the root
                    *prediction = chosen_sentiment;
                }
            }
            Expression logprob = pick(prob_dist, chosen_sentiment);
            errs.push_back(logprob);
        }
        assert(errs.size() == tree.dfo.size());
        return -sum(errs);
    }
}
;
