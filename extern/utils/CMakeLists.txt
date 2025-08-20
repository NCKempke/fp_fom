# Common boilerplate
cmake_minimum_required(VERSION 3.6)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CPLEX library
find_package(CPLEX)

# Find XPRESS library
find_package(XPRESS)

# Find COPT library
find_package(COPT)

# Find Gurobi library.
find_package(GUROBI)

# Find Threads
find_package(Threads)

# Include zlib as an imported target
find_package(ZLIB REQUIRED)
add_library(ZLIB::Lib STATIC IMPORTED)
set_target_properties(ZLIB::Lib PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIRS}
  IMPORTED_LOCATION ${ZLIB_LIBRARIES}
)

# Define libutils
add_library(utils STATIC src/tool_app.cpp src/tool_assert.cpp src/maths.cpp
                         src/path.cpp src/args_parser.cpp src/timer.cpp src/fileconfig.cpp)

target_include_directories(utils
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(utils PUBLIC fmt::fmt ZLIB::Lib)

add_library(Utils::Lib ALIAS utils)

# Deal with optional dependencies
if (CPLEX_FOUND)
  target_compile_definitions(utils PUBLIC HAS_CPLEX=1)
  target_sources(utils PRIVATE src/cpxmodel.cpp)
  target_link_libraries(utils PUBLIC Cplex::Cplex)
endif()

if (XPRESS_FOUND)
  target_compile_definitions(utils PUBLIC HAS_XPRESS=1)
  target_sources(utils PRIVATE src/xprsmodel.cpp)
  target_link_libraries(utils PUBLIC Xpress::Xpress)
endif()

if (COPT_FOUND)
  target_compile_definitions(utils PUBLIC HAS_COPT=1)
  target_sources(utils PRIVATE src/coptmodel.cpp)
  target_link_libraries(utils PUBLIC Copt::Copt)
endif()

if (GUROBI_FOUND)
  target_compile_definitions(utils PUBLIC HAS_GUROBI=1)
  target_sources(utils PRIVATE src/gurobimodel.cpp)
  target_link_libraries(utils PUBLIC Gurobi::Gurobi)
endif()
