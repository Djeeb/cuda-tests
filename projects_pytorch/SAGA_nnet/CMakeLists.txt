cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SAGA_nnet)
find_package(Torch REQUIRED)

add_executable(test_nn test_nn.cpp)
target_link_libraries(test_nn "${TORCH_LIBRARIES}")
set_property(TARGET test_nn PROPERTY CXX_STANDARD 11 )

add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear "${TORCH_LIBRARIES}")
set_property(TARGET test_linear PROPERTY CXX_STANDARD 11 )
