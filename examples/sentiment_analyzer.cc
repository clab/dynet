#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/model.h"
#include "cnn/peepholetreelstm.h"
#include "cnn/rnn.h"
#include "cnn/timing.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

using namespace cnn;

//TODO: add code for POS tag dictionary
cnn::Dict tokdict, sentimenttagdict, depreldict;
vector<unsigned> sentiments;
const unsigned int ROOT = 0;
const string R = "ROOT";
const string UNK = "UNK";

unsigned VOCAB_SIZE = 0, DEPREL_SIZE = 0, SENTI_TAG_SIZE = 0;

unsigned LAYERS = 1;
unsigned INPUT_DIM = 100;
unsigned HIDDEN_DIM = 100;
unsigned TAG_HIDDEN_DIM = 100;

struct DepTree {
    unsigned numnodes;
    vector<unsigned> parents;
    vector<unsigned> sent;
    vector<unsigned> deprels;
    set<unsigned> leaves;

    vector<unsigned> dfo; // depth-first ordering of the nodes
    map<unsigned, vector<unsigned>> children;

    explicit DepTree(vector<unsigned> parents, vector<unsigned> deprels,
            vector<unsigned> sent) {
        this->numnodes = parents.size(); // = (length of the sentence + 1) to accommodate root
        this->parents = parents;
        this->sent = sent;
        this->deprels = deprels;

        set_children();
        set_leaves();
        set_dfo();
    }

    void printTree() {
        cerr << "Tree for sentence \"";
        for (unsigned int i = 0; i < numnodes; i++) {
            cerr << tokdict.Convert(sent[i]) << " ";
        }
        cerr << "\"" << endl;

        for (unsigned int i = 0; i < numnodes; i++) {
            cerr << i << "<-" << depreldict.Convert(deprels[i]) << "-"
                    << parents[i] << endl;
        }
        cerr << "Leaves: ";
        for (unsigned leaf : leaves)
            cerr << leaf << " ";
        cerr << endl;
        cerr << "Depth-first Ordering:" << endl;
        for (unsigned node : dfo) {
            cerr << node << "->";
        }
        cerr << endl;
    }

    vector<unsigned> get_children(unsigned node) {
        vector<unsigned> clist;
        if (children.find(node) == children.end()) {
            return clist;
        } else {
            return children[node];
        }
    }

private:

    void set_children() {
        for (unsigned child = 1; child < numnodes; child++) {
            unsigned parent = parents[child];
            vector<unsigned> clist;
            if (children.find(parent) != children.end()) {
                clist = children[parent];
            }
            clist.push_back(child);
            children[parent] = clist;
        }
    }

    void set_leaves() {
        for (unsigned node = 0; node < numnodes; ++node) {
            if (get_children(node).size() == 0)
                leaves.insert(node);
        }
    }

    void set_dfo() {
        vector<unsigned> stack;
        set<unsigned> seen;
        stack.push_back(ROOT);

        while (!stack.empty()) {
            int top = stack.back();

            if (children.find(top) != children.end()
                    && seen.find(top) == seen.end()) {
                vector<unsigned> clist = children[top];
                for (auto itr2 = clist.rbegin(); itr2 != clist.rend(); ++itr2) {
                    stack.push_back(*itr2);
                }
                seen.insert(top);
            } else if (children.find(top) != children.end()
                    && seen.find(top) != seen.end()) {
                unsigned tobepopped = stack.back();
                dfo.push_back(tobepopped);
                stack.pop_back();
            } else {
                unsigned tobepopped = stack.back();
                dfo.push_back(tobepopped);
                stack.pop_back();
            }
        }
        // TODO: should we maintain root in dfo? No
        assert(dfo.back() == ROOT);
        dfo.pop_back();
    }
};

template<class Builder>
struct StanfordTreeLSTMModel {
    LookupParameters* p_w;
    // TODO: input should also contain deprel to parent
    //LookupParameters* p_d;

    Parameters* p_tok2l;
    Parameters* p_dep2l;
    Parameters* p_inp_bias;

    Parameters* p_root2senti;
    Parameters* p_sentibias;

    Builder treebuilder;

    explicit StanfordTreeLSTMModel(Model &model) :
            treebuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
        p_w = model.add_lookup_parameters(VOCAB_SIZE, { INPUT_DIM });
        //p_d = model.add_lookup_parameters(DEPREL_SIZE, { INPUT_DIM });

        p_tok2l = model.add_parameters( { HIDDEN_DIM, INPUT_DIM });
        p_dep2l = model.add_parameters( { HIDDEN_DIM, INPUT_DIM });
        p_inp_bias = model.add_parameters( { HIDDEN_DIM });
        // TODO: Change to add a regular BiLSTM below the tree

        p_root2senti = model.add_parameters( { SENTI_TAG_SIZE, HIDDEN_DIM });
        p_sentibias = model.add_parameters( { SENTI_TAG_SIZE });
    }

};

// Split string str on any delimiting character in delim, and write the result
// as a vector of strings.
void StringSplit(const string &str, const string &delim,
        vector<string> *results, bool ignore_multiple_separators) {
    size_t cutAt;
    string tmp = str;
    while ((cutAt = tmp.find_first_of(delim)) != tmp.npos) {
        if ((ignore_multiple_separators && cutAt > 0)
                || (!ignore_multiple_separators)) {
            // Note: this "if" guarantees that every field is not empty.
            // This complies with multiple consecutive occurrences of the
            // delimiter (e.g. several consecutive occurrences of a whitespace
            // will count as a single delimiter).
            // To allow empty fields, this if-condition should be removed.
            results->push_back(tmp.substr(0, cutAt));
        }
        tmp = tmp.substr(cutAt + 1);
    }
    if (tmp.length() > 0)
        results->push_back(tmp);
}

void ReadCoNLL09Line(string& line, int* id, unsigned *token, unsigned* parent,
        unsigned* deprel, int *sentiment) {
    if (line.length() <= 0) {
        *id = -1; // end of sentence in CoNLL file
        return;
    }
    vector < string > fields;
    StringSplit(line, "\t", &fields, true);
    *id = stoi(fields[0]);
    *token = tokdict.Convert(fields[1]); // LEMMA
    *parent = stoi(fields[6]); // PHEAD
    *deprel = depreldict.Convert(fields[7]); // PDEPREL
    *sentiment = sentimenttagdict.Convert(fields[2]);
}

void ReadCoNLLFile(const string& conll_fname, vector<pair<DepTree, vector<int>>>& dataset) {
    int numex = 0;
    int ttoks = 0;
    string line;

    vector<unsigned> parents, deprels, sentence;
    parents.push_back(ROOT);
    deprels.push_back(depreldict.Convert(R));
    sentence.push_back(tokdict.Convert(R));

    vector<int> sentiments;
    sentiments.push_back(sentimenttagdict.Convert(R)); // dummy root doesn't have a sentiment

    ifstream in(conll_fname);
    assert(in);
    while (getline(in, line)) {
        int id;
        unsigned token, parent, deprel;
        int sentiment;
        ReadCoNLL09Line(line, &id, &token, &parent, &deprel, &sentiment);
        ttoks += 1;
        if (id == -1) { // end of conll
            ttoks -= 1;
            numex += 1;
            DepTree tree(parents, deprels, sentence);
            dataset.push_back(make_pair(tree, sentiments));

            parents.clear();
            parents.push_back(ROOT);

            deprels.clear();
            deprels.push_back(depreldict.Convert(R));

            sentence.clear();
            sentence.push_back(tokdict.Convert(R));

            sentiments.clear();
            sentiments.push_back(sentimenttagdict.Convert(R));
        } else {
            assert(id == parents.size());
            parents.push_back(parent);
            deprels.push_back(deprel);
            sentence.push_back(token);
            sentiments.push_back(sentiment);
        }
    }
    cerr << numex << " sentences, " << ttoks << " tokens, " << tokdict.size()
    << " types -- " << sentimenttagdict.size() << " tags in data set"
    << endl;
}

void RunTest(string fname, Model& model, vector<pair<DepTree, vector<int>>>& test, StanfordTreeLSTMModel<TreeLSTMBuilder>& mytree) {

}

void RunTraining(Model& model, Trainer* sgd,
        StanfordTreeLSTMModel<TreeLSTMBuilder>& mytree,
        vector<pair<DepTree, vector<int>>>& training,vector<pair<DepTree, vector<int>>>& dev) {
//    ostringstream os;
//    os << "sentanalyzer" << '_' << LAYERS << '_' << INPUT_DIM << '_'
//    << HIDDEN_DIM << "-pid" << getpid() << ".params";
//    const string savedmodelfname = os.str();
//    cerr << "Parameters will be written to: " << savedmodelfname << endl;
//    bool soft_link_created = false;
}

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0]
                << " train.conll dev.conll [model.params]\n";
        return 1;
    }

    vector<pair<DepTree, vector<int>>> training, dev;

    cerr << "Reading training data from " << argv[1] << "...\n";
    ReadCoNLLFile(argv[1], training);

    tokdict.Freeze(); // no new word types allowed
    tokdict.SetUnk(UNK);
    sentimenttagdict.Freeze(); // no new tag types allowed
    for (unsigned i = 0; i < sentimenttagdict.size(); ++i) {
        sentiments.push_back(i);
    }
    depreldict.Freeze();

    VOCAB_SIZE = tokdict.size();
    DEPREL_SIZE = depreldict.size();
    SENTI_TAG_SIZE = sentimenttagdict.size();

    cerr << "Reading dev data from " << argv[2] << "...\n";
    ReadCoNLLFile(argv[2], dev);

    Model model;
    bool use_momentum = true;
    Trainer* sgd = nullptr;
    if (use_momentum)
        sgd = new MomentumSGDTrainer(&model);
    else
        sgd = new SimpleSGDTrainer(&model);

    StanfordTreeLSTMModel<TreeLSTMBuilder> mytree(model);
    if (argc == 4) { // test mode
        string model_fname = argv[3];
        RunTest(model_fname, model, dev, mytree);
        exit(1);
    }

    RunTraining(model, sgd, mytree, training, dev);
}
