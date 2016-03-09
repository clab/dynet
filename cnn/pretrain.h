#ifndef CNN_PRETRAIN_H
#define CNN_PRETRAIN_H

#include <string>
#include <vector>
#include <unordered_map>
#include "cnn/dict.h"
#include "cnn/model.h"

namespace cnn {

void SavePretrainedEmbeddings(const std::string& fname,
    const Dict& d,
    const LookupParameter& lp);

void ReadPretrainedEmbeddings(const std::string& fname,
    Dict* d,
    std::unordered_map<int, std::vector<float>>* vectors);

} // namespace cnn

#endif
