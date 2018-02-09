#pragma once

#include <vector>
#include <numeric>

// --------------------------------------------------------------------------------------------------------------------------------
// The batching strategy here is similar to one used in lamtram toolkit (https://github.com/neubig/lamtram).

inline size_t calc_size(const WordIdSentence & src, const WordIdSentence & trg);
inline size_t create_minibatches(const WordIdCorpus& cor
	, size_t max_size
	, std::vector<WordIdSentences> & train_src_minibatch
	, std::vector<WordIdSentences> & train_trg_minibatch
	, std::vector<size_t> & train_ids_minibatch);

struct DoubleLength
{
  DoubleLength(const WordIdCorpus & cor_) : cor(cor_) { }
  bool operator() (int i1, int i2);
  const WordIdCorpus & cor;
};

bool DoubleLength::operator() (int i1, int i2) {
  if(std::get<0>(cor[i2]).size() != std::get<0>(cor[i1]).size()) return (std::get<0>(cor[i2]).size() < std::get<0>(cor[i1]).size());
  return (std::get<1>(cor[i2]).size() < std::get<1>(cor[i1]).size());
}

inline size_t calc_size(const WordIdSentence & src, const WordIdSentence & trg) {
  return src.size()+trg.size();
}

inline size_t create_minibatches(const WordIdCorpus& cor
	, size_t max_size
	, std::vector<WordIdSentences> & train_src_minibatch
	, std::vector<WordIdSentences> & train_trg_minibatch
	, std::vector<size_t> & train_ids_minibatch) 
{
	cerr << endl << "Creating minibatches for training data (using minibatch_size=" << max_size << ")..." << endl;

	train_src_minibatch.clear();
	train_trg_minibatch.clear();

	std::vector<size_t> train_ids(cor.size());
	std::iota(train_ids.begin(), train_ids.end(), 0);
	if(max_size > 1)
		sort(train_ids.begin(), train_ids.end(), DoubleLength(cor));

	std::vector<WordIdSentence> train_src_next;
	std::vector<WordIdSentence> train_trg_next;

	size_t max_len = 0;
	for(size_t i = 0; i < train_ids.size(); i++) {
		max_len = std::max(max_len, calc_size(std::get<0>(cor[train_ids[i]]), std::get<1>(cor[train_ids[i]])));
		train_src_next.push_back(std::get<0>(cor[train_ids[i]]));
		train_trg_next.push_back(std::get<1>(cor[train_ids[i]]));

		if((train_trg_next.size()+1) * max_len > max_size) {
			train_src_minibatch.push_back(train_src_next);
			train_src_next.clear();
			train_trg_minibatch.push_back(train_trg_next);
			train_trg_next.clear();
			max_len = 0;
		}
	}

	if(train_trg_next.size()) {
		train_src_minibatch.push_back(train_src_next);
		train_trg_minibatch.push_back(train_trg_next);
	}

	// Create a sentence list for this minibatch
	train_ids_minibatch.resize(train_src_minibatch.size());
	std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

	return train_ids.size();
}
// --------------------------------------------------------------------------------------------------------------------------------

