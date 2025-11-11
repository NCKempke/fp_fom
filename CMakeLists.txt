# Common boilerplate
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent in-source builds
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory and run CMake from there")
endif()

# Determine if this is built as a subproject (using add_subdirectory)
# or if it is the master project.
set(MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MASTER_PROJECT ON)
  message(STATUS "Utils is MASTER PROJECT")
  message(STATUS "CMake version: ${CMAKE_VERSION}")
endif()

find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  # If this is the master project, update submodules
  if (MASTER_PROJECT)
    message(STATUS "Submodules update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update failed")
    endif()
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

project(fom-fpr)

set(COPTDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/COPT" CACHE PATH "Path to COPT")
set(CPLEXDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/cplex" CACHE PATH "Path to CPLEX")
set(GUROBIDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/gurobi" CACHE PATH "Path to GUROBI")
set(XPRESSDIR "${CMAKE_CURRENT_SOURCE_DIR}/solvers/xpress" CACHE PATH "Path to XPRESS")

message(STATUS "Solver lookup directories:")
message("   COPT directory: ${COPTDIR}")
message("   Cplex directory: ${CPLEXDIR}")
message("   Gurobi directory: ${GUROBIDIR}")
message("   Xpress directory: ${XPRESSDIR}")

# Define fp_main executable
add_executable(fp_main
  src/fp_main.cpp
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

target_include_directories(fp_main PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(fp_main PRIVATE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)
target_link_libraries(fp_main Utils::Lib fmt::fmt)

# Add subprojects if this is master project
if (MASTER_PROJECT)
  add_subdirectory(extern/fmt)
  add_subdirectory(extern/utils)
endif()
