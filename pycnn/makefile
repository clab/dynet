pycnn.so: ../build/cnn/libcnn_shared.so pycnn.pyx pycnn.pxd setup.py
	cp ../build/cnn/libcnn_shared.so .
	python2 setup.py build_ext --inplace

install: pycnn.so
	python2 setup.py install --user

clean:
	rm *.so *.cpp


