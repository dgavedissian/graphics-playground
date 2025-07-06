#include "cl_context.h"
#include <chrono>
#include <numeric>
#include <tuple>
#include <type_traits>

const char source[] = R"""(
__kernel void numerical_reduction(__global const int* in_data, __local int* local_data, __global int* out_data) {
  size_t global_id = get_global_id(0);
  size_t local_size = get_local_size(0);
  size_t local_id = get_local_id(0);

  local_data[local_id] = in_data[global_id];

  // wait until all other items in work group reach this point.
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = local_size >> 1; i > 0; i >>= 1) {
    if (local_id < i) {
      local_data[local_id] += local_data[local_id + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    out_data[get_group_id(0)] = local_data[0];
  }
}
)""";

int accumulateUsingOpenCL(CLContext &ctx, cl::Kernel kernel,
                          std::vector<int> values) {
  auto work_group_size =
      kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(ctx.device);
  auto num_work_groups = values.size() / work_group_size;

  if (values.size() % work_group_size != 0) {
    throw std::runtime_error(
        "Values to reduce must be a multiple of the work group size: " +
        std::to_string(work_group_size));
  }

  // Setup input/outputs.
  cl::Buffer in_buf(ctx.context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS |
                        CL_MEM_COPY_HOST_PTR,
                    sizeof(int) * values.size(), values.data());
  cl::Buffer out_buf(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     sizeof(int) * num_work_groups);
  CL_CHECK(kernel.setArg(0, in_buf));
  CL_CHECK(kernel.setArg(1, sizeof(int) * work_group_size, nullptr));
  CL_CHECK(kernel.setArg(2, out_buf));

  // Run kernel.
  std::vector<int> out_vec(num_work_groups);
  CL_CHECK(ctx.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                          cl::NDRange{values.size()},
                                          cl::NDRange{work_group_size}));
  CL_CHECK(ctx.queue.enqueueReadBuffer(
      out_buf, CL_TRUE, 0, sizeof(int) * out_vec.size(), out_vec.data()));

  // If we have more data to reduce, then recurse.
  if (out_vec.size() <= work_group_size) {
    return std::accumulate(out_vec.begin(), out_vec.end(), 0);
  } else {
    return accumulateUsingOpenCL(ctx, kernel, std::move(out_vec));
  }
}

int main() {
  auto ctx = CLContext::create();

  std::vector<int> vec(2u << 27u);
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }

  auto time = [&](auto &&f) {
    auto start = std::chrono::steady_clock::now();
    auto result = f();
    std::chrono::duration<float> duration =
        std::chrono::steady_clock::now() - start;
    return std::make_tuple(result, duration.count());
  };

  std::cout << "Timing sum of " << vec.size() << " elements." << std::endl;

  auto kernel =
      ctx.createKernel(ctx.compileProgram(source), "numerical_reduction");
  auto [sum_cl, time_cl] =
      time([&] { return accumulateUsingOpenCL(ctx, kernel, vec); });
  std::cout << "OpenCL" << std::endl;
  std::cout << "Sum: " << sum_cl << " Time Taken: " << time_cl << "s"
            << std::endl;

  auto [sum_cpu, time_cpu] =
      time([&] { return std::accumulate(vec.begin(), vec.end(), 0); });
  std::cout << "std::accumulate" << std::endl;
  std::cout << "Sum: " << sum_cpu << " Time Taken: " << time_cpu << "s"
            << std::endl;

  return 0;
}