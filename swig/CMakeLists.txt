cmake_minimum_required(VERSION 2.8)

# Find required dependencies
find_package(SWIG REQUIRED)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)
include(UseSWIG)
include_directories(${JNI_INCLUDE_DIRS})

# Include dynet C++ sources
include_directories("../dynet")

# Set java package
set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java/src/edu/cmu/dynet")
set(CMAKE_SWIG_FLAGS -package edu.cmu.dynet)

# Run swig
set_source_files_properties(dynet_swig.i PROPERTIES CPLUSPLUS ON)
swig_add_module(dynet_swig java dynet_swig.i)

# Link with dynet library
swig_link_libraries(dynet_swig dynet)

# Create jar file
add_jar(
    dynet_swigJNI
    SOURCES
    "${CMAKE_SWIG_OUTDIR}/ComputationGraph.java"
    "${CMAKE_SWIG_OUTDIR}/Dim.java"
    "${CMAKE_SWIG_OUTDIR}/DoubleVector.java"
    "${CMAKE_SWIG_OUTDIR}/dynet_swig.java"
    "${CMAKE_SWIG_OUTDIR}/dynet_swigJNI.java"
    "${CMAKE_SWIG_OUTDIR}/Expression.java"
    "${CMAKE_SWIG_OUTDIR}/ExpressionVector.java"
    "${CMAKE_SWIG_OUTDIR}/FloatVector.java"
    "${CMAKE_SWIG_OUTDIR}/IntVector.java"
    "${CMAKE_SWIG_OUTDIR}/LongVector.java"
    "${CMAKE_SWIG_OUTDIR}/LookupParameter.java"
    "${CMAKE_SWIG_OUTDIR}/LSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/Model.java"
    "${CMAKE_SWIG_OUTDIR}/Parameter.java"
    "${CMAKE_SWIG_OUTDIR}/RNNBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/SimpleSGDTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/StringVector.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_dynet__LookupParameterStorage.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_dynet__VariableIndex.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_float.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_int.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_p_p_char.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__Node_p_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__Tensor_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__VariableIndex_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_unsigned_int_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_unsigned_int.java"
    "${CMAKE_SWIG_OUTDIR}/Tensor.java"
    "${CMAKE_SWIG_OUTDIR}/Trainer.java"
)

# TODO(joelgrus): This is probably not defensive or robust enough.
option(INCLUDE_SCALA_HELPERS "INCLUDE_SCALA_HELPERS" ON)
if(INCLUDE_SCALA_HELPERS)

  # Find sbt
  find_program(SBT sbt)

  # Run sbt assembly
  add_custom_command(
    COMMENT "Running sbt"
    OUTPUT scala_helper_uberjar
    DEPENDS dynet_swigJNI
    COMMAND ${SBT} assembly -Dbuildpath=${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  )

  add_custom_target(scala_helper ALL DEPENDS scala_helper_uberjar)

endif()


