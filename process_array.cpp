#include "cl_context.h"

int main() {
  auto ctx = CLContext::create();
  auto program = ctx.compileProgram(loadTextFile("../process_array.cl"));

  std::vector<int> vec(1024);
  std::fill(vec.begin(), vec.end(), 1);

  cl::Buffer in_buf(ctx.context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS |
                        CL_MEM_USE_HOST_PTR,
                    sizeof(int) * vec.size(), vec.data());
  cl::Buffer out_buf(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     sizeof(int) * vec.size());

  cl::Kernel kernel = ctx.createKernel(program, "process_array");
  CL_CHECK(kernel.setArg(0, in_buf));
  CL_CHECK(kernel.setArg(1, out_buf));

  cl::CommandQueue queue(ctx.context, ctx.device);
  CL_CHECK(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange{vec.size()}));
  CL_CHECK(queue.enqueueReadBuffer(out_buf, CL_FALSE, 0,
                                   sizeof(int) * vec.size(), vec.data()));

  // Wait on pending operations.
  cl::finish();

  int sum = 0;
  for (int i : vec) {
    sum += i;
  }
  std::cout << sum << std::endl;

  return 0;
}