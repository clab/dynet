#include "cnn/peepholetreelstm.h"
#include "cnn/dict.h"
#include "cnn/training.h"

#include <iostream>
#include <fstream>
#include <sstream>

cnn::Dict tokdict, tagdict, depreldict;
int VOCAB_SIZE = 0, DEPREL_SIZE = 0, TAG_SIZE = 0;

struct DepTree {
    unsigned numnodes;
    vector<unsigned> parents;
    vector<unsigned> sent;
    vector<unsigned> deprels;
    // TODO: fill the data structures below
//    set<unsigned> leaves;
//    vector<unsigned> dfo; // depth-first ordering of the nodes
    explicit DepTree(vector<unsigned> parents, vector<unsigned> deprels,
            vector<unsigned> sent) {
        this->numnodes = parents.size();
        this->parents = parents;
        this->sent = sent;
        this->deprels = deprels;
    }

    void printTree() {
        cerr << "Tree for sentence \"";
        for (unsigned int i = 0; i < numnodes; i++) {
            cerr << tokdict.Convert(sent[i]) << " ";
        }
        cerr << "\"" << endl;

        for (unsigned int i = 0; i < numnodes; i++) {
            cerr << i + 1 << "<-" << depreldict.Convert(deprels[i]) << "-"
                    << parents[i] << endl;
        }
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

void ReadCoNLL09Line(string& line, int* id, int *token, int* parent,
        int* deprel, int *sentiment) {
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
    if (*parent == 0) {
        *sentiment = tagdict.Convert(fields[2]);
    }
}

void ReadCoNLLFile(const string& conll_fname,
        vector<pair<DepTree, int>>& dataset) {
    int numex = 0;
    int ttoks = 0;
    string line;
    int label; // sentiment label of root
    vector<unsigned> parents, deprels, sentence;

    ifstream in(conll_fname);
    assert(in);
    while (getline(in, line)) {
        int id, token, parent, deprel, sentiment;
        ReadCoNLL09Line(line, &id, &token, &parent, &deprel, &sentiment);
        ttoks += 1;
        if (id == -1) { // end of conll
            ttoks -= 1;
            numex += 1;
            DepTree tree(parents, deprels, sentence);
            assert(label != -1);
            dataset.push_back(make_pair(tree, label));
            label = -1;
            parents.clear();
            deprels.clear();
            sentence.clear();
        } else {
            parents.push_back(parent);
            deprels.push_back(deprel);
            sentence.push_back(token);
            if (parent == 0) {
                label = sentiment;
            }
        }
    }
    cerr << numex << " sentences, " << ttoks << " tokens, " << tokdict.size()
            << " types -- " << tagdict.size() << " tags in data set" << endl;
}

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0]
                << " train.conll dev.conll [model.params]\n";
        return 1;
    }

    vector<pair<DepTree, int>> training, dev;

    cerr << "Reading training data from " << argv[1] << "...\n";
    ReadCoNLLFile(argv[1], training);

    tokdict.Freeze(); // no new word types allowed
    tokdict.SetUnk("UNK");
    tagdict.Freeze(); // no new tag types allowed
    depreldict.Freeze();
    VOCAB_SIZE = tokdict.size();
    DEPREL_SIZE = depreldict.size();
    TAG_SIZE = tagdict.size();

    cerr << "Reading dev data from " << argv[2] << "...\n";
    ReadCoNLLFile(argv[2], dev);

}
