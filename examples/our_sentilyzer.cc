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

const string UNK_STR = "UNK";

unsigned VOCAB_SIZE = 0, DEPREL_SIZE, SENTI_TAG_SIZE = 0;

unsigned LAYERS = 1;
unsigned GLOVE_DIM = 300;
unsigned DEPREL_DIM = 100;
unsigned LSTM_INPUT_DIM = 300;
unsigned HIDDEN_DIM = 168;

template<class Builder>
struct OurSentimentModel {
    LookupParameters* p_x;
    LookupParameters* p_e;

    Parameters* p_tok2l;
    Parameters* p_dep2l;
    Parameters* p_inp_bias;

    Parameters* p_root2senti;
    Parameters* p_sentibias;

    Builder treebuilder;

    explicit OurSentimentModel(Model &model) :
            treebuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, &model) {
        p_x = model.add_lookup_parameters(VOCAB_SIZE, { GLOVE_DIM });
        p_e = model.add_lookup_parameters(DEPREL_SIZE, { DEPREL_DIM });

        p_tok2l = model.add_parameters( { LSTM_INPUT_DIM, GLOVE_DIM });
        p_dep2l = model.add_parameters( { LSTM_INPUT_DIM, DEPREL_DIM });
        p_inp_bias = model.add_parameters( { LSTM_INPUT_DIM });
        // TODO: Change to add a regular BiLSTM below the tree

        p_root2senti = model.add_parameters( { SENTI_TAG_SIZE, HIDDEN_DIM });
        p_sentibias = model.add_parameters( { SENTI_TAG_SIZE });
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

        Expression root2senti = parameter(*cg, p_root2senti);
        Expression senti_bias = parameter(*cg, p_sentibias);

        vector<Expression> i_words, i_deprels, inputs;
        for (unsigned n = 0; n < tree.numnodes; n++) {
            i_words.push_back(lookup(*cg, p_x, tree.sent[n]));
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

void EvaluateTags(DepTree tree, vector<int>& gold, int& predicted, double* corr,
        double* tot) {
    if (gold[tree.root] == predicted) {
        (*corr)++;
    }
    (*tot)++;
}

void RunTest(string fname, Model& model, vector<pair<DepTree, vector<int>>>& test, OurSentimentModel<TreeLSTMBuilder>& mytree) {
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;

    double cor = 0;
    double tot = 0;

    auto time_begin = chrono::high_resolution_clock::now();
    for (auto& test_ex : test) {
        ComputationGraph test_cg;
        int predicted_sentiment;
        mytree.BuildTreeCompGraph(test_ex.first, vector<int>(), &test_cg, &predicted_sentiment);
        EvaluateTags(test_ex.first, test_ex.second, predicted_sentiment, &cor, &tot);
    }

    double acc = cor/tot;
    auto time_end = chrono::high_resolution_clock::now();
    cerr << "TEST accuracy: " << acc << "\t[" << test.size() << " sents in "
    << std::chrono::duration<double, std::milli>(time_end - time_begin).count()
    << " ms]" << endl;
}

void RunTraining(Model& model, Trainer* sgd,
        OurSentimentModel<TreeLSTMBuilder>& mytree,
        vector<pair<DepTree, vector<int>>>& training,vector<pair<DepTree, vector<int>>>& dev) {
    ostringstream os;
    os << "sentanalyzer" << '_' << LAYERS << '_' << LSTM_INPUT_DIM << '_'
    << HIDDEN_DIM << "-pid" << getpid() << ".params";
    const string savedmodelfname = os.str();
    cerr << "Parameters will be written to: " << savedmodelfname << endl;
    bool soft_link_created = false;

    unsigned report_every_i = 100;
    unsigned dev_every_i_reports = 25;
    unsigned si = training.size();

    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    double tot_seen = 0;
    bool first = true;
    int report = 0;
    unsigned trs = 0;
    double llh = 0;
    double best_acc = 0.0;
    int iter = -1;

    while (1) {
        ++iter;

        Timer iteration("completed in");
        double llh = 0;

        for (unsigned tr_idx = 0; tr_idx < report_every_i; ++tr_idx) {
            if (si == training.size()) {
                si = 0;
                if (first) {
                    first = false;
                } else {
                    sgd->update_epoch();
                }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance

            auto& sent = training[order[si]];
            //sent.first.printTree(tokdict, depreldict);
            int predicted_sentiment;

            ComputationGraph cg;
            mytree.BuildTreeCompGraph(sent.first, sent.second, &cg, &predicted_sentiment);

            llh += as_scalar(cg.incremental_forward());
            cg.backward();
            sgd->update(1.0);

            ++si;
            ++trs;
            ++tot_seen;
        }
        sgd->status();
        cerr << "update #" << iter << " (epoch " << (tot_seen / training.size()) << ")\t" << " llh: " << llh << " ppl = " << exp(llh / trs);

        // show score on dev data
        if (report % dev_every_i_reports == 0) {
            //double dloss = 0;
            double dcor = 0;
            double dtags = 0;
            //lm.p_th2t->scale_parameters(pdrop);
            for (auto& dev_ex : dev) {
                ComputationGraph dev_cg;
                int dev_predicted_sentiment;

                mytree.BuildTreeCompGraph(dev_ex.first, vector<int>(), &dev_cg,
                &dev_predicted_sentiment);
                //dloss += as_scalar(dev_cg.forward());
                EvaluateTags(dev_ex.first, dev_ex.second, dev_predicted_sentiment, &dcor, &dtags);
            }
            cerr << "\n***DEV [epoch=" << (tot_seen / training.size())
            << "]" << " accuracy = " << (dcor/dtags);

            double acc = dcor/dtags;

            if (acc > best_acc) {
                best_acc = acc;
                ofstream out(savedmodelfname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                cerr << "Updated model! " << endl;

                if (soft_link_created == false) {
                    string softlink = string(" latest_model_");
                    if (system((string("rm -f ") + softlink).c_str()) == 0
                    && system((string("ln -s ") + savedmodelfname + softlink).c_str()) == 0) {
                        cerr << "Created " << softlink << " as a soft link to "
                        << savedmodelfname << " for convenience.";
                    }
                    soft_link_created = true;
                }
            }
        }
        report++;
    }
    delete sgd;
}

void test_tree(DepTree t, cnn::Dict depreldict) {
    assert(t.msg_nbrs.size() == t.dfo_msgs.size());
    if (t.numnodes <= 10) {
        t.printTree(tokdict, depreldict);
        cerr << "\nEdge DFO:\n";
        for (DepEdge e : t.dfo_msgs) {
            e.print(depreldict);
            cerr << "(";
            auto nid = t.msgdict.find(e);
            assert(nid != t.msgdict.end());
            cerr << nid->second << ")\t";
            cerr << endl;
        }

        cerr << "\nEdge Neighbors:\n";
        for (DepEdge e : t.dfo_msgs) {
            e.print();
            cerr << " has neighbors: ";
            auto mid = t.msgdict.find(e);
            assert(mid != t.msgdict.end());

            vector<unsigned> neighbors = t.msg_nbrs[mid->second];

            for (unsigned i : neighbors) {
                DepEdge ee = t.dfo_msgs[i];
                ee.print();
                cerr << "(";
                auto nid = t.msgdict.find(ee);
                assert(nid != t.msgdict.end());
                cerr << nid->second << ")\t";
            }
            cerr << endl;
        }
        cerr << endl;

        cerr << "\nNode - Incoming Messages:\n";
        for (unsigned n = 1; n < t.numnodes; ++n) {
            cerr << n << " will receive  ";
            vector<unsigned> neighbors = t.node_msg_nbrs[n];

            for (unsigned i : neighbors) {
                DepEdge ee = t.dfo_msgs[i];
                ee.print();
                cerr << "(";
                auto nid = t.msgdict.find(ee);
                assert(nid != t.msgdict.end());
                cerr << nid->second << ")\t";
            }
            cerr << endl;
        }
        cerr << endl;
    }
}

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0]
                << " train.conll dev.conll [trained.model]\n";
        return 1;
    }

    vector<pair<DepTree, vector<int>>> training, dev;

    cerr << "Reading training data from " << argv[1] << "...\n";
    ReadCoNLLFile(argv[1], training, &tokdict, &depreldict, &sentitagdict);

//    for (pair<DepTree, vector<int>> ex : training) {
//        DepTree tree = ex.first;
//        if (tree.numnodes <= 10) {
//            test_tree(tree, depreldict);
//            break;
//        }
//    }

    tokdict.Freeze(); // no new word types allowed
    tokdict.SetUnk(UNK_STR);
    sentitagdict.Freeze(); // no new tag types allowed
    for (unsigned i = 0; i < sentitagdict.size(); ++i) {
        sentitaglist.push_back(i);
    }
    depreldict.Freeze();

    VOCAB_SIZE = tokdict.size();
    DEPREL_SIZE = depreldict.size();
    SENTI_TAG_SIZE = sentitagdict.size();

    cerr << "Reading dev data from " << argv[2] << "...\n";
    ReadCoNLLFile(argv[2], dev, &tokdict, &depreldict, &sentitagdict);

    Model model;
    bool use_momentum = true;
    Trainer* sgd = nullptr;
    if (use_momentum)
        sgd = new MomentumSGDTrainer(&model);
    else
        sgd = new AdamTrainer(&model);

    OurSentimentModel<TreeLSTMBuilder> mytree(model);
    if (argc == 4) { // test mode
        string model_fname = argv[3];
        RunTest(model_fname, model, dev, mytree);
        exit(1);
    }

    RunTraining(model, sgd, mytree, training, dev);
}
