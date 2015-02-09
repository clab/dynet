#include <Eigen/Sparse>

#include <iostream>

using namespace std;

int main() {
  Eigen::SparseMatrix<double> m(1000000,1000);
  cerr << "ok!\n";
}

