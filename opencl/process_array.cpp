#include "cl_context.h"

const char source[] = R"""(
__kernel void process_array(__global const int* in_data, __global int* out_data) {
    out_data[get_global_id(0)] = in_data[get_global_id(0)] * 2;
}
)""";

int main() {
  auto ctx = CLContext::create();
  auto program = ctx.compileProgram(source);

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

  CL_CHECK(ctx.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange{vec.size()}));
  CL_CHECK(ctx.queue.enqueueReadBuffer(out_buf, CL_FALSE, 0,
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