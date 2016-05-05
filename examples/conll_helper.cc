#include <iostream>
#include <fstream>
#include <set>

#include "deptree.cc"

using namespace std;

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
        unsigned* deprel, int *sentiment, cnn::Dict* tokdict,
        cnn::Dict* depreldict, cnn::Dict* sentitagdict) {
    if (line.length() <= 0) {
        *id = -1; // end of sentence in CoNLL file
        return;
    }
    vector < string > fields;
    StringSplit(line, "\t", &fields, true);
    *id = stoi(fields[0]);
    string mixedcase_token = fields[1];
    transform(mixedcase_token.begin(), mixedcase_token.end(),
            mixedcase_token.begin(), ::tolower); // need to lower case the token, to associate with
    // word embeddings
    *token = tokdict->Convert(mixedcase_token); // LEMMA
    *parent = stoi(fields[6]); // PHEAD
    *deprel = depreldict->Convert(fields[7]); // PDEPREL
    *sentiment = sentitagdict->Convert(fields[2]);
}

void ReadCoNLLFile(const string& conll_fname, vector<pair<DepTree, vector<int>>>&dataset, cnn::Dict* tokdict,
cnn::Dict* depreldict, cnn::Dict* sentitagdict) {
    int numex = 0;
    int ttoks = 0;
    string line;

    vector<unsigned> parents, deprels, sentence;
    parents.push_back(DUMMY_ROOT);
    deprels.push_back(depreldict->Convert(DUMMY_ROOT_STR));
    sentence.push_back(tokdict->Convert(DUMMY_ROOT_STR));

    vector<int> sentiments;
    sentiments.push_back(sentitagdict->Convert(DUMMY_ROOT_STR)); // dummy root doesn't have a sentiment

    ifstream in(conll_fname);
    assert(in);
    while (getline(in, line)) {
        int id;
        unsigned token, parent, deprel;
        int sentiment;
        ReadCoNLL09Line(line, &id, &token, &parent, &deprel, &sentiment, tokdict, depreldict, sentitagdict);
        ttoks += 1;
        if (id == -1) { // end of conll
            ttoks -= 1;
            numex += 1;
            DepTree tree(parents, deprels, sentence);
            dataset.push_back(make_pair(tree, sentiments));

            parents.clear();
            parents.push_back(DUMMY_ROOT);

            deprels.clear();
            deprels.push_back(depreldict->Convert(DUMMY_ROOT_STR));

            sentence.clear();
            sentence.push_back(tokdict->Convert(DUMMY_ROOT_STR));

            sentiments.clear();
            sentiments.push_back(sentitagdict->Convert(DUMMY_ROOT_STR));
        } else {
            assert(id == parents.size());
            parents.push_back(parent);
            deprels.push_back(deprel);
            sentence.push_back(token);
            sentiments.push_back(sentiment);
        }
    }
    cerr << numex << " sentences, " << ttoks << " tokens, " << tokdict->size()
    << " types -- " << sentitagdict->size() << " tags in data set"
    << endl;
}

void ReadTestFileVocab(const string& conll_fname,
        unordered_set<string>* test_vocab) {
    string line;
    ifstream in(conll_fname);
    assert(in);
    while (getline(in, line)) {
        vector < string > fields;
        StringSplit(line, "\t", &fields, true);
        if (fields.size() > 2)
            test_vocab->insert(fields[1]);
    }
    in.close();
}

/** add words to vocabulary if pretrained embeddings are known, but word is not seen at train time*/
void PreReadPretrainedVectors(const string& pretrainedfile,
        const unordered_set<string>& test_vocab, cnn::Dict* tokdict) {
    ifstream in(pretrainedfile);
    if (!in.is_open()) {
        cerr << "Pretrained embeddings FILE NOT FOUND!" << endl;
        exit(1);
    }

    string line;
    getline(in, line);
    string word;
    while (getline(in, line)) {
        istringstream lin(line);
        lin >> word;
        if (test_vocab.find(word) != test_vocab.end()
                && tokdict->Contains(word) == false) {
            tokdict->Convert(word);
        }
    }
    in.close();
}
