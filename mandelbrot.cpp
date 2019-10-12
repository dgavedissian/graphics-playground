#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <iostream>

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

  cl::Program compileProgram(const std::string &source) {
    cl::Program::Sources sources = {{source.c_str(), source.size()}};
    cl::Program program(context, sources);

    auto err = program.build("-cl-std=CL1.2");
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
    throw std::runtime_error("Could not find hello_world.cl");
  }
  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

int main() {
  auto ctx = CLContext::create();
  auto program = ctx.compileProgram(loadTextFile("../hello_world.cl"));

  char buf[14];
  cl::Buffer membuf(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                    sizeof(buf));
  cl::Kernel kernel = ctx.createKernel(program, "hello_world");
  kernel.setArg(0, membuf);

  cl::CommandQueue queue(ctx.context, ctx.device);
  queue.enqueueTask(kernel);
  queue.enqueueReadBuffer(membuf, true, 0, sizeof(buf), buf);

  std::cout << buf << std::endl;

  return 0;
}