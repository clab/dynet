all: xor xor-xent nlm rnnlm

xor: xor.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< $(LINKER_STUFF) -o $@

xor-xent: xor-xent.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< $(LINKER_STUFF) -o $@

nlm: nlm.cc
	g++ -g -O3 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< $(LINKER_STUFF) -o $@

rnnlm: rnnlm.cc
	g++ -g -O4 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< $(LINKER_STUFF) -o $@

