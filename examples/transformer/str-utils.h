#pragma once

#include <sstream>
#include <vector>

using namespace std;

std::vector<std::string> split_words(const std::string & str);

std::vector<std::string> split_words(const std::string &line) {
	std::istringstream in(line);
	std::string word;
	std::vector<std::string> res;
	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		res.push_back(word);
	}
	return res;
}

