all: cnn examples

EIGEN_PATH=/usr0/home/cdyer/eigen-eigen-10219c95fe65

export EIGEN_PATH

SUBDIRS = cnn examples

.PHONY: subdirs $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@


