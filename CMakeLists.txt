cmake_minimum_required(VERSION 3.14)
project(gpu-playground)

include(FetchContent)

# stb
FetchContent_Declare(
    stb
    URL https://github.com/nothings/stb/archive/f67165c2bb2af3060ecae7d20d6f731173485ad0.tar.gz
    URL_MD5 c879fbb6bfffdc759b5650f37c8fe7e0
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
endif()

find_package(OpenCL REQUIRED)

add_library(cl_context INTERFACE)
target_include_directories(cl_context INTERFACE .)

add_executable(mandelbrot mandelbrot.cpp)
target_link_libraries(mandelbrot cl_context OpenCL stb)

add_executable(process_array process_array.cpp)
target_link_libraries(process_array cl_context OpenCL)
