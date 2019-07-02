cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SAGA_nnet)
find_package(Torch REQUIRED)

add_executable(SAGA_nnet SAGA_nnet.cpp)
target_link_libraries(SAGA_nnet "${TORCH_LIBRARIES}")
set_property(TARGET SAGA_nnet PROPERTY CXX_STANDARD 11)
