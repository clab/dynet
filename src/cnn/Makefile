all: libcnn.a

OBJ_FILES = \
  cnn.o \
  edges.o \
  lstm.o \
  param-edges.o \
  model.o \
  rnn.o \
  training.o

clean:
	$(RM) -f *.o libcnn.a

libcnn.a: $(OBJ_FILES)
	ar rvs $@ $(OBJ_FILES)

.cc.o:
	g++ -c -g -O3 -Wall -std=c++11 -I.. -I$(EIGEN_PATH) $< -o $@
