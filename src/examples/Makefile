all: xor nlm rnnlm

xor: xor.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< ../cnn/libcnn.a -o $@

nlm: nlm.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< ../cnn/libcnn.a -o $@

rnnlm: rnnlm.cc
	g++ -g -O4 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< ../cnn/libcnn.a -o $@

