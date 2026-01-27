# Common boilerplate
cmake_minimum_required(VERSION 3.12)

project(fom-fpr LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent in-source builds
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory and run CMake from there")
endif()

set(BOOST_MIN_VERSION 1.65)
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED iostreams)


find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  # If this is the master project, update submodules
  message(STATUS "Submodules update")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update failed")
  endif()

  # Get the current working branch
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE FP_GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Get the latest abbreviated commit hash of the working branch
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --always --dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE FP_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

find_package(CUDAToolkit REQUIRED)

set(COPTDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/COPT" CACHE PATH "Path to COPT")
set(CPLEXDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/cplex" CACHE PATH "Path to CPLEX")
set(GUROBIDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/gurobi" CACHE PATH "Path to GUROBI")
set(XPRESSDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/xpress" CACHE PATH "Path to XPRESS")

message(STATUS "Solver lookup directories:")
message("   COPT directory: ${COPTDIR}")
message("   Cplex directory: ${CPLEXDIR}")
message("   Gurobi directory: ${GUROBIDIR}")
message("   Xpress directory: ${XPRESSDIR}")

# Create a library with all common source files
add_library(fp_lib STATIC
  src/cliquecover.cpp
  src/cliquetable.cpp
  src/impltable.cpp
  src/linear_propagator.cpp
  src/lp_algorithm_type.cpp
  src/mip.cpp
  src/preset_type.cpp
  src/propagation.cpp
  src/ranker_type.cpp
  src/solver_type.cpp
  src/table_propagators.cpp
  src/value_chooser_type.cpp
  src/vectorlist.cpp
  src/walkmip.cpp
)

target_link_libraries(fp_lib Utils::Lib fmt::fmt)

target_include_directories(fp_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define fp_main executable
add_executable(fp_main
  src/fp_main.cpp
)

target_link_libraries(fp_main fp_lib)

# MIP competition executable
add_executable(mipcomp_main
  src/mipcomp_main.cpp
  src/evolution_search.cu
  src/gpu_data.cu
)

# TODO: which exactly?
set_target_properties(mipcomp_main PROPERTIES CUDA_ARCHITECTURES "75")#;86;90")
set_target_properties(mipcomp_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(mipcomp_main PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G;-g>
)

target_compile_options(mipcomp_main PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Include CCCL headers
target_include_directories(mipcomp_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cccl/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/cccl/thrust
    ${CMAKE_CURRENT_SOURCE_DIR}/cccl/libcudacxx/include
)

target_link_libraries(mipcomp_main fp_lib CUDA::cudart Boost::iostreams)

add_subdirectory(extern/fmt)
add_subdirectory(extern/utils)
