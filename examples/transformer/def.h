#pragma once

#include <vector>

using namespace std;

typedef int WordId;// word Id
typedef std::vector<WordId> WordIdSentence;// word Id sentence
typedef std::vector<WordIdSentence> WordIdSentences;// batches of sentences
typedef tuple<WordIdSentence, WordIdSentence> WordIdSentencePair; // Note: can be extended to include additional information (e.g., document ID)
typedef vector<WordIdSentencePair> WordIdCorpus;


