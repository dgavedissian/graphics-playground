#include "cl_context.h"
#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "./mandelbrot <size_x> <size_y> <max_iterations> <png_output_file>"
              << std::endl;
    return 0;
  }

  std::uint32_t width = std::strtol(argv[1], nullptr, 0);
  std::uint32_t height = std::strtol(argv[2], nullptr, 0);
  std::uint32_t max_iterations = std::strtol(argv[3], nullptr, 0);
  const char *output_filename = argv[4];
  if (width == 0) {
    std::cout << "Width must not be 0" << std::endl;
    return 1;
  }
  if (height == 0) {
    std::cout << "Height must not be 0" << std::endl;
    return 1;
  }
  if (max_iterations == 0) {
    std::cout << "Max iterations must not be 0" << std::endl;
    return 1;
  }

  auto ctx = CLContext::create();
  auto program =
      ctx.compileProgram(loadTextFile("../mandelbrot.cl"),
                         "-DMAX_ITERATIONS=" + std::to_string(max_iterations));

  int bytes_per_pixel = 3;
  std::vector<char> pixel_data((width * height * bytes_per_pixel));
  cl::Buffer out_buf(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     pixel_data.size());

  cl::Kernel kernel = ctx.createKernel(program, "generate_mandelbrot_set");
  CL_CHECK(kernel.setArg(0, out_buf));

  cl::CommandQueue queue(ctx.context, ctx.device);
  CL_CHECK(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange{width, height}));
  CL_CHECK(queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, pixel_data.size(),
                                   pixel_data.data()));

  // Write image.
  stbi_write_png(output_filename, width, height, 3, pixel_data.data(), 0);

  return 0;
}