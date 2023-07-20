project(crps_scores)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(thirdparty/pybind11)
set(Boost_INCLUDE_DIR thirdparty/boost)
set(Boost_LIBRARY_DIR  thirdparty/boost/libs)
find_package(Boost)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})


pybind11_add_module(crps_lims crps_scores.cpp)

