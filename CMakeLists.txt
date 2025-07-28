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

# GLFW
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    URL_MD5 7d1e90c5c0011eb10eac280b6093d0d0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(glfw)

find_package(OpenGL REQUIRED)

set(RAYTRACING_SOURCE_FILES
    raytracing/glad/gl.cpp
    raytracing/cpu_renderer.cpp
    raytracing/main.cpp
    raytracing/math.cpp
    raytracing/single_image_window.cpp
)

if(APPLE)
    # metal-cpp
    FetchContent_Declare(
        metalcpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15.2_iOS18.2.zip
        URL_MD5 861a9b0447071c08fa43951bf68d4b99
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(metalcpp)

    set(RAYTRACING_SOURCE_FILES
        ${RAYTRACING_SOURCE_FILES}
        raytracing/metal_impl.cpp
        raytracing/metal_renderer.mm
        raytracing/metal_renderer_shader.metal
    )
endif()

add_executable(raytracing ${RAYTRACING_SOURCE_FILES})
target_link_libraries(raytracing PRIVATE stb glm-hdr glfw OpenGL::GL)
target_compile_features(raytracing PUBLIC cxx_std_20)

if(APPLE)
    target_include_directories(raytracing PRIVATE ${metalcpp_SOURCE_DIR})
    target_link_libraries(raytracing PUBLIC
        "$<LINK_LIBRARY:FRAMEWORK,Foundation>"
        "$<LINK_LIBRARY:FRAMEWORK,QuartzCore>"
        "$<LINK_LIBRARY:FRAMEWORK,Metal>"
    )
endif()

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
