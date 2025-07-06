#include "cl_context.h"
#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

const char source[] = R"""(
typedef struct ComplexStruct {
  double x;
  double y;
} Complex;

Complex cmul(Complex a, Complex b) {
  Complex result = {
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  };
  return result;
}

Complex cadd(Complex a, Complex b) {
  Complex result = {
    a.x + b.x,
    a.y + b.y
  };
  return result;
}

float cabs(Complex c) {
  return sqrt(c.x * c.x + c.y * c.y);
}

int mandelbrot_iterations(Complex c) {
  Complex z = {0.0, 0.0};
  int n = 0;
  while (cabs(z) <= 2 && n < MAX_ITERATIONS) {
    z = cadd(cmul(z, z), c);
    n++;
  }

  if (n == MAX_ITERATIONS) {
    return 0;
  }
  return n + 1 - log(log2(cabs(z)));
}

__kernel void generate_mandelbrot_set(__global char* out_data) {
  int i = get_global_id(1) * get_global_size(0) + get_global_id(0);

  const double x = (double)get_global_id(0);
  const double y = (double)get_global_id(1);
  const double width = (double)get_global_size(0);
  const double height = (double)get_global_size(1);

  // Convert pixel coordinates to complex number.
  const double kReStart = -4.0;
  const double kReEnd = 2.0;
  const double kImStart = -2.0;
  const double kImEnd = 2.0;
  Complex c = {
    kReStart + (x / width) * (kReEnd - kReStart),
    kImStart + (y / height) * (kImEnd - kImStart)
  };

  // Compute mandelbrot iterations, and map to RGB value.
  int n = mandelbrot_iterations(c);
  char val = (char)(255.0f * (double)n / MAX_ITERATIONS);
  out_data[i * 3 + 0] = val;
  out_data[i * 3 + 1] = val;
  out_data[i * 3 + 2] = val;
}
)""";

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout
        << "./mandelbrot <size_x> <size_y> <max_iterations> <png_output_file>"
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
  auto program = ctx.compileProgram(source, "-DMAX_ITERATIONS=" +
                                                std::to_string(max_iterations));

  int bytes_per_pixel = 3;
  std::vector<char> pixel_data((width * height * bytes_per_pixel));
  cl::Buffer out_buf(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     pixel_data.size());

  cl::Kernel kernel = ctx.createKernel(program, "generate_mandelbrot_set");
  CL_CHECK(kernel.setArg(0, out_buf));

  // Generate image.
  CL_CHECK(ctx.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                          cl::NDRange{width, height}));
  CL_CHECK(ctx.queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, pixel_data.size(),
                                       pixel_data.data()));

  // Write image.
  stbi_write_png(output_filename, width, height, 3, pixel_data.data(), 0);

  return 0;
}