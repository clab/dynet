using namespace std;
#include "deptree.h"

void EvaluateTags(DepTree tree, vector<int>& gold, int& predicted, double* corr,
        double* tot) {
    if (gold[tree.root] == predicted) {
        (*corr)++;
    }
    (*tot)++;
}
