cmake_minimum_required(VERSION 3.14)
project(gpu-playground)

find_package(OpenCL REQUIRED)

add_library(cl_context INTERFACE)
target_include_directories(cl_context INTERFACE .)

add_executable(process_array process_array.cpp)
target_link_libraries(process_array cl_context OpenCL)
