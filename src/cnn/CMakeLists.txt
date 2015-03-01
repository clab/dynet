# ########## Project setup ##########
PROJECT(cnn CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

# ########## cnn library ##########
# Sources:
SET(cnn_library_SRCS
    cnn.cc
    edges.cc
    lstm.cc
    lstm2.cc
    model.cc
    param-edges.cc
    rnn.cc
    training.cc
)

# Headers:
SET(cnn_library_HDRS
    cnn.h
    edges.h
    eigen-serialization.h
    lstm.h
    lstm2.h
    model.h
    param-edges.h
    rnn.h
    tensor.h
    timing.h
    training.h
)


# actual target:
ADD_LIBRARY(cnn ${cnn_library_SRCS} ${cnn_library_HDRS})
# target_compile_features(cnn PRIVATE cxx_range_for)

