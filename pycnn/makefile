
PYTHON := python
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
	DYSUF := dylib
else
	DYSUF := so
endif

pycnn.${DYSUF}: ../build/cnn/libcnn_shared.${DYSUF} pycnn.pyx pycnn.pxd setup.py
	cp ../build/cnn/libcnn_shared.${DYSUF} .
	${PYTHON} setup.py build_ext --inplace

install: pycnn.${DYSUF}
	${PYTHON} setup.py install --user

clean:
	rm *.${DYSUF} *.cpp


