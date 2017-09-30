#ifndef DATA_IO_H
#define DATA_IO_H

#include <iostream>
#include <sstream>
#include <vector>

using namespace std;



int ReverseInt (int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Code from https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/ to read MNIST
void read_mnist(string data_file,  vector<vector<float>> &arr) {
  ifstream file (data_file, ios::binary);

  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);
    for (int i = 0; i < number_of_images; ++i) {
      arr.push_back(vector<float>());
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char*)&temp, sizeof(temp));
          arr[i].push_back((float)temp);
        }
      }
    }
  } else {
    stringstream ss;
    ss << "Error with input file " << data_file;
    throw runtime_error(ss.str());
  }
}

void read_mnist_labels(string label_file, vector<unsigned> &labels) {
  ifstream file (label_file, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    for (int i = 0; i < number_of_images; ++i) {
      unsigned char temp = 0;
      file.read((char*)&temp, sizeof(temp));
      labels.push_back((unsigned)temp);
    }
  }else {
    stringstream ss;
    ss << "Error with input file " << label_file;
    throw runtime_error(ss.str());
  }
}

#endif