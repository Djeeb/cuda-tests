cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_nnet)
find_package(Torch REQUIRED)

add_executable(custom_nnet custom_nnet.cpp)
target_link_libraries(custom_nnet "${TORCH_LIBRARIES}")
set_property(TARGET custom_nnet PROPERTY CXX_STANDARD 11)
