all: graph1 xor nlm rnnlm

graph1: graph1.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I/Users/cdyer/software/eigen-eigen-36fd1ba04c12 $< ../cnn/libcnn.a -o $@

xor: xor.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I/Users/cdyer/software/eigen-eigen-36fd1ba04c12 $< ../cnn/libcnn.a -o $@

nlm: nlm.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I/Users/cdyer/software/eigen-eigen-36fd1ba04c12 $< ../cnn/libcnn.a -o $@

rnnlm: rnnlm.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I/Users/cdyer/software/eigen-eigen-36fd1ba04c12 $< ../cnn/libcnn.a -o $@

