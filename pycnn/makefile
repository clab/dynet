
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

gpycnn.${DYSUF}: ../build/cnn/libgcnn_shared.${DYSUF} pycnn.pyx pycnn.pxd setup_gpu.py
	cp ../build/cnn/libgcnn_shared.${DYSUF} .
	cp ../build/cnn/libcnncuda_shared.${DYSUF} .
	cp pycnn.pyx gpycnn.pyx
	cp pycnn.pxd gpycnn.pxd
	${PYTHON} setup_gpu.py build_ext --inplace

ginstall: gpycnn.${DYSUF}
	${PYTHON} setup_gpu.py install --user

install: pycnn.${DYSUF}
	${PYTHON} setup.py install --user

clean:
	rm *.${DYSUF} *.cpp


