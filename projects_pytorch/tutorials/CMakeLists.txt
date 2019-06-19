cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tutorials)
find_package(Torch REQUIRED)

add_executable(tensor_example tensor_example.cpp)
target_link_libraries(tensor_example "${TORCH_LIBRARIES}")
set_property(TARGET tensor_example PROPERTY CXX_STANDARD 11)


add_executable(autograd_example autograd_example.cpp)
target_link_libraries(autograd_example "${TORCH_LIBRARIES}")
set_property(TARGET autograd_example PROPERTY CXX_STANDARD 11)

add_executable(GPU_CPU_example GPU_CPU_example.cpp)
target_link_libraries(GPU_CPU_example "${TORCH_LIBRARIES}")
set_property(TARGET GPU_CPU_example PROPERTY CXX_STANDARD 11)
