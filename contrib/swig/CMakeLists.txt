cmake_minimum_required(VERSION 2.8)

# Find required dependencies
find_package(SWIG 3.0.11 REQUIRED)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)
include(UseSWIG)
include_directories(${JNI_INCLUDE_DIRS})

# Include dynet C++ sources
include_directories("../../dynet")

# Set java output dir
set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java/src/edu/cmu/dynet")

# the exact scala version used for compilation
set(SCALA_VERSION "2.11.11" CACHE STRING "Scala version to compile the swig wrapper for")

# the binary compatibility, e.g. 2.11 or 2.12
string(REGEX MATCH "[0-9]+\\.[0-9]+" SCALA_BIN_VERSION "${SCALA_VERSION}")

# Set java package (+cuda flag, if appropriate)
if(WITH_CUDA_BACKEND)
  set(CMAKE_SWIG_FLAGS -package edu.cmu.dynet.internal -DSWIG_USE_CUDA)
else(WITH_CUDA_BACKEND)
  set(CMAKE_SWIG_FLAGS -package edu.cmu.dynet.internal)
endif(WITH_CUDA_BACKEND)

# Run swig
set_source_files_properties(dynet_swig.i PROPERTIES CPLUSPLUS ON)
swig_add_module(dynet_swig java dynet_swig.i)

# add C++ compiler flags
if(WITH_CUDA_BACKEND)
  set_target_properties(dynet_swig PROPERTIES
                        COMPILE_DEFINITIONS HAVE_CUDA)
endif(WITH_CUDA_BACKEND)

# Link with dynet library
if(WITH_CUDA_BACKEND)
  MESSAGE("-- swig link with GPU library")
  swig_link_libraries(dynet_swig dynet)
else(WITH_CUDA_BACKEND)
  MESSAGE("-- swig link with CPU library")
  swig_link_libraries(dynet_swig dynet)
endif(WITH_CUDA_BACKEND)

# Create jar file
add_jar(
    dynet_swigJNI
    SOURCES
    "${CMAKE_SWIG_OUTDIR}/AdadeltaTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/AdagradTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/AdamTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/AmsgradTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/BidirectionalTreeLSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/ComputationGraph.java"
    "${CMAKE_SWIG_OUTDIR}/CoupledLSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/CompactVanillaLSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/CyclicalSGDTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/Device.java"
    "${CMAKE_SWIG_OUTDIR}/DeviceMempool.java"
    "${CMAKE_SWIG_OUTDIR}/DeviceMempoolSizes.java"
    "${CMAKE_SWIG_OUTDIR}/DeviceType.java"
    "${CMAKE_SWIG_OUTDIR}/Dim.java"
    "${CMAKE_SWIG_OUTDIR}/DoubleVector.java"
    "${CMAKE_SWIG_OUTDIR}/dynet_swig.java"
    "${CMAKE_SWIG_OUTDIR}/dynet_swigJNI.java"
    "${CMAKE_SWIG_OUTDIR}/DynetParams.java"
    # "${CMAKE_SWIG_OUTDIR}/EGTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/Expression.java"
    "${CMAKE_SWIG_OUTDIR}/ExpressionVector.java"
    "${CMAKE_SWIG_OUTDIR}/ExpressionVectorVector.java"
    "${CMAKE_SWIG_OUTDIR}/FastLSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/FloatVector.java"
    "${CMAKE_SWIG_OUTDIR}/GRUBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/IntVector.java"
    "${CMAKE_SWIG_OUTDIR}/Loader.java"
    "${CMAKE_SWIG_OUTDIR}/LongVector.java"
    "${CMAKE_SWIG_OUTDIR}/LookupParameter.java"
    "${CMAKE_SWIG_OUTDIR}/LookupParameterStorage.java"
    "${CMAKE_SWIG_OUTDIR}/LookupParameterStorageVector.java"
    "${CMAKE_SWIG_OUTDIR}/MomentumSGDTrainer.java"
	"${CMAKE_SWIG_OUTDIR}/NaryTreeLSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/Parameter.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterCollection.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInit.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInitNormal.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInitUniform.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInitConst.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInitIdentity.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInitGlorot.java"
    # "${CMAKE_SWIG_OUTDIR}/ParameterInitSaxe.java" (I AM NOT ACTUALLY IMPLEMENTED IN DYNET)
    "${CMAKE_SWIG_OUTDIR}/ParameterInitFromFile.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterInitFromVector.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterStorage.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterStorageBase.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterStorageVector.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterVector.java"
    "${CMAKE_SWIG_OUTDIR}/ParameterVectorVector.java"
    "${CMAKE_SWIG_OUTDIR}/RMSPropTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/RNNBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/Saver.java"
    "${CMAKE_SWIG_OUTDIR}/SimpleRNNBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/SimpleSGDTrainer.java"
    "${CMAKE_SWIG_OUTDIR}/StringVector.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_dynet__MemAllocator.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_dynet__VariableIndex.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_float.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_int.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_p_p_char.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_size_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__AlignedMemoryPool_p_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__Node_p_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__Tensor_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_std__vectorT_dynet__VariableIndex_t.java"
    "${CMAKE_SWIG_OUTDIR}/SWIGTYPE_p_unsigned_int.java"
    "${CMAKE_SWIG_OUTDIR}/Tensor.java"
    "${CMAKE_SWIG_OUTDIR}/TensorTools.java"
    "${CMAKE_SWIG_OUTDIR}/TextFileLoader.java"
    "${CMAKE_SWIG_OUTDIR}/TextFileSaver.java"
    "${CMAKE_SWIG_OUTDIR}/Trainer.java"
    "${CMAKE_SWIG_OUTDIR}/TreeLSTMBuilder.java"
	"${CMAKE_SWIG_OUTDIR}/UnidirectionalTreeLSTMBuilder.java"
    "${CMAKE_SWIG_OUTDIR}/UnsignedVector.java"
    "${CMAKE_SWIG_OUTDIR}/UnsignedVectorVector.java"
    "${CMAKE_SWIG_OUTDIR}/VanillaLSTMBuilder.java"
)

# Add dynamic library files into the Jar
add_custom_target(
    dylib_into_jar
    COMMAND zip -j ${CMAKE_CURRENT_BINARY_DIR}/dynet_swigJNI_dylib.jar ${CMAKE_CURRENT_BINARY_DIR}/libdynet_swig.*
)
add_dependencies(dylib_into_jar dynet_swigJNI)
add_dependencies(dynet_swigJNI dynet_swig)

# TODO(joelgrus): This is probably not defensive or robust enough.
option(INCLUDE_SCALA "INCLUDE_SCALA" ON)
if(INCLUDE_SCALA)

  # make doesn't know about the dynet_scala jar, so it doesn't clean it unless we tell it about it
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "dynet_swigJNI_scala_${SCALA_BIN_VERSION}.jar")

  # Find sbt
  find_program(SBT sbt)

  # Run sbt assembly
  add_custom_command(
    COMMENT "Running sbt"
    OUTPUT scala_helper_uberjar
    DEPENDS dylib_into_jar
    COMMAND ${SBT} -Dbuildpath=${CMAKE_CURRENT_BINARY_DIR} -Dscalaversion=${SCALA_VERSION} assembly
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  )

  add_custom_target(scala_helper ALL DEPENDS scala_helper_uberjar)

endif()
