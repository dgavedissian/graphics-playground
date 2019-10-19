#pragma once

#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <iostream>

#define CL_CHECK(x) __CL_CHECK(x, #x, __FILE__, __LINE__)
#define __CL_CHECK(x, str, file, line)                                         \
  do {                                                                         \
    int err = (x);                                                             \
    if (err != CL_SUCCESS) {                                                   \
      throw std::runtime_error(str " failed at " file ":" +                    \
                               std::to_string(line) + ": " +                   \
                               std::to_string(err));                           \
    }                                                                          \
  } while (false)

struct CLContext {
  cl::Device device;
  cl::Context context;

  static CLContext create() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(!platforms.empty());
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(!devices.empty());
    auto device = devices.front();

    std::cout << "Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "Version: " << device.getInfo<CL_DEVICE_VERSION>()
              << std::endl;

    return CLContext{device, cl::Context{device}};
  }

  cl::Program compileProgram(const std::string &source, const std::string& additional_options = "") {
    cl::Program program(context, source);
    auto options = "-cl-std=CL1.2 " + additional_options;
    auto err = program.build(options.c_str());
    if (err != CL_SUCCESS) {
      std::string error = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      throw std::runtime_error("Error when building CL kernel: " + error);
    }
    return program;
  }

  cl::Kernel createKernel(cl::Program program, const char *kernel_name) {
    int err;
    cl::Kernel kernel(program, kernel_name, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Error when creating CL kernel: " +
                               std::to_string(err));
    }
    return kernel;
  }
};

std::string loadTextFile(const char *filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    throw std::runtime_error("Could not find " + std::string{filename});
  }
  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}
