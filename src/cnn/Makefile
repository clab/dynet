all: libcnn.a

OBJ_FILES = \
  cnn.o \
  cnn-edges.o

clean:
	$(RM) -f *.o libcnn.a

libcnn.a: $(OBJ_FILES)
	ar rvs $@ $(OBJ_FILES)

.cc.o:
	g++ -c -g -O3 -Wall -std=c++11 -I.. -I/Users/cdyer/software/eigen-eigen-36fd1ba04c12 $< -o $@
