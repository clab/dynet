all: cnn examples

# EIGEN_PATH=/Users/cdyer/software/eigen-eigen-36fd1ba04c12
LINKER_STUFF=../cnn/libcnn.a -lboost_program_options -lboost_serialization
EIGEN_PATH=/usr0/home/cdyer/eigen-eigen-10219c95fe65

export EIGEN_PATH
export LINKER_STUFF

SUBDIRS = cnn examples parser

.PHONY: subdirs $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@


