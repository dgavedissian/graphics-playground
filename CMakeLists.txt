cmake_minimum_required(VERSION 3.14)
project(graphics-playground)

include(FetchContent)

# stb
FetchContent_Declare(
    stb
    URL https://github.com/nothings/stb/archive/f58f558c120e9b32c217290b80bad1a0729fbb2c.tar.gz
    URL_MD5 a69d2b9417c1b30a84f4537eb5d868ca
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# glm
FetchContent_Declare(
    glm
    URL https://github.com/g-truc/glm/releases/download/1.0.1/glm-1.0.1-light.zip
    URL_MD5 7d235d4813a2e7b1e10cc711b8e25213
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(glm)
add_library(glm-hdr INTERFACE)
target_include_directories(glm-hdr INTERFACE ${glm_SOURCE_DIR})

add_executable(cpu_raytracing cpu_raytracing/raytracing.cpp)
target_link_libraries(cpu_raytracing PRIVATE stb glm-hdr)
target_compile_features(cpu_raytracing PUBLIC cxx_std_17)

find_package(OpenCL)
if(NOT OpenCL_FOUND)
    message(STATUS "OpenCL not found, disabling OpenCL targets")
else()
    add_library(cl_context INTERFACE)
    target_include_directories(cl_context INTERFACE opencl)

    add_executable(process_array opencl/process_array.cpp)
    target_link_libraries(process_array cl_context OpenCL)

    add_executable(mandelbrot opencl/mandelbrot.cpp)
    target_link_libraries(mandelbrot cl_context OpenCL stb)

    add_executable(numerical_reduction opencl/numerical_reduction.cpp)
    target_link_libraries(numerical_reduction cl_context OpenCL stb)
    target_compile_features(numerical_reduction PUBLIC cxx_std_17)
endif()
