# ########## Project setup ##########
PROJECT(cnn CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# ########## cnn library ##########
# Sources:
SET(cnn_library_SRCS
    cnn.cc
    edges.cc
    eigen-backend.cc
    lstm.cc
    lstm-fast.cc
    model.cc
    param-edges.cc
    rnn.cc
    saxe_init.cc
    training.cc
)

# Headers:
SET(cnn_library_HDRS
    cnn.h
    edges.h
    eigen-backend.h
    eigen-serialization.h
    lstm.h
    lstm-fast.h
    model.h
    param-edges.h
    rnn.h
    saxe_init.h
    tensor.h
    timing.h
    training.h
)


# actual target:
ADD_LIBRARY(cnn ${cnn_library_SRCS} ${cnn_library_HDRS})
# target_compile_features(cnn PRIVATE cxx_range_for)

