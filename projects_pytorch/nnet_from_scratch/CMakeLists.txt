cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nnet_from_scratch)
find_package(Torch REQUIRED)

set_property(TARGET nnet_from_scratch PROPERTY CXX_STANDARD 11)

add_executable(nnet_from_scratch nnet_from_scratch.cpp)
target_link_libraries(nnet_from_scratch "${TORCH_LIBRARIES}")
set_property(TARGET nnet_from_scratch PROPERTY CXX_STANDARD 11)
