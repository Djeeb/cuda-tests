cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(first_example)

find_package(Torch REQUIRED)

add_executable(first_example first_example.cpp)
target_link_libraries(first_example "${TORCH_LIBRARIES}")
set_property(TARGET first_example PROPERTY CXX_STANDARD 11)
