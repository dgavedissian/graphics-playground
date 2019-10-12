cmake_minimum_required(VERSION 3.14)
project(gpu-playground)

find_package(OpenCL REQUIRED)

add_executable(mandelbrot-set mandelbrot.cpp)
target_link_libraries(mandelbrot-set OpenCL)
