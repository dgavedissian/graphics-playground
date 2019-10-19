struct Complex {
    double x;
    double y;
};

struct Complex cmul(struct Complex a, struct Complex b) {
    struct Complex result = {
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    };
    return result;
}

struct Complex cadd(struct Complex a, struct Complex b) {
    struct Complex result = {
        a.x + b.x,
        a.y + b.y
    };
    return result;
}

float cabs(struct Complex c) {
    return sqrt(c.x * c.x + c.y * c.y);
}

int mandelbrot_iterations(struct Complex c) {
    struct Complex z = {0.0, 0.0};
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
    struct Complex c = {
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