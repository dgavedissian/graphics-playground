#include <iostream>
#include <vector>
#include <random>
#include <thread>

#include <glm.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using color = glm::dvec3;
using point3 = glm::dvec3;
using vec3 = glm::dvec3;

struct interval {
    double min;
    double max;

    interval() : min(std::numeric_limits<double>::min()), max(std::numeric_limits<double>::max()) {}
    interval(double min_val, double max_val) : min(min_val), max(max_val) {}

    double length() const {
        return max - min;
    }

    bool contains(double value) const {
        return value >= min && value <= max;
    }
    
    bool surrounds(double value) const {
        return value > min && value < max;
    }

    double clamp(double x) const {
        return (x < min) ? min : (x > max) ? max : x;
    }

    static const interval empty;
};

const interval interval::empty = interval(std::numeric_limits<double>::max(), std::numeric_limits<double>::min());

inline double randomDouble() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*randomDouble();
}

inline vec3 randomUnitVector() {
    while (true) {
        auto p = vec3{randomDouble(-1, 1), randomDouble(-1, 1), randomDouble(-1, 1)};
        auto lengthSq = glm::dot(p, p);
        if (1e-160 < lengthSq && lengthSq <= 1) {
            return p / sqrt(lengthSq);
        }
    }
}

inline vec3 randomOnHemisphere(const vec3& normal) {
    vec3 onUnitSphere = randomUnitVector();
    return dot(onUnitSphere, normal) < 0.0 ? -onUnitSphere : onUnitSphere;
}

class Ray {
public:
    Ray(const vec3& origin, const vec3& direction) : origin_(origin), direction_(direction) {}

    const vec3& origin() const { return origin_; }
    const vec3& direction() const { return direction_; }

    point3 at(double t) const {
        return origin_ + t * direction_;
    } 

private:
    vec3 origin_;
    vec3 direction_;
};

struct HitResult {
    point3 point;
    vec3 normal;
    double t;
    bool front_face;

    void setNormal(const Ray& r, const vec3& outward_normal) {
        front_face = glm::dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    virtual ~Hittable() = default;

    virtual bool hit(const Ray& r, interval t, HitResult& result) const = 0;
};

class Sphere : public Hittable {
public:
    Sphere(const point3& centre, double radius) : centre_(centre), radius_(std::fmax(0.0, radius)) {}

    bool hit(const Ray& r, interval t, HitResult& result) const override {
        vec3 oc = centre_ - r.origin();
        auto a = glm::dot(r.direction(), r.direction());
        auto h = glm::dot(r.direction(), oc);
        auto c = glm::dot(oc, oc) - radius_ * radius_;
        auto discriminant = h * h - a * c;
        
        if (discriminant < 0) {
            return false;
        }

        auto sqrt_discriminant = std::sqrt(discriminant);
        auto root = (h - sqrt_discriminant) / a;
        if (!t.surrounds(root)) {
            // Try the other root.
            root = (h + sqrt_discriminant) / a;
            if (!t.surrounds(root)) {
                // Both roots are outside the range.
                return false;
            }
        }

        result.t = root;
        result.point = r.at(root);
        result.setNormal(r, (result.point - centre_) / radius_);
        return true;
    }

private:
    point3 centre_;
    double radius_;
};

class Scene : public Hittable {
public:
    void add(std::unique_ptr<Hittable> hittable) {
        hittables_.emplace_back(std::move(hittable));
    }

    bool hit(const Ray& r, interval t, HitResult& result) const override {
        bool hitAnything = false;
        double tmax = t.max;

        for (const auto& hittable : hittables_) {
            HitResult tempResult;
            if (hittable->hit(r, interval(t.min, tmax), tempResult)) {
                hitAnything = true;
                result = tempResult;

                // The maximum the ray can travel now, is to where this object was hit.
                tmax = tempResult.t;
            }
        }
        return hitAnything;
    }

private:
    std::vector<std::unique_ptr<Hittable>> hittables_;
};

color rayColour(const Ray& r, int depth, const Scene& scene) {
    if (depth == 0) {
        // If we've run out of rays, then return no colour.
        return color(0, 0, 0);
    }

    HitResult result;
    if (scene.hit(r, interval(0.001, std::numeric_limits<double>::max()), result)) {
        vec3 direction = result.normal + randomUnitVector();
        return 0.5 * rayColour(Ray(result.point, direction), depth - 1, scene);
    }
    
    vec3 unitDirection = glm::normalize(r.direction());
    auto a = 0.5 * (unitDirection.y + 1.0);
    return ((1.0 - a) * color(1.0, 1.0, 1.0)) + (a * color(0.5, 0.7, 1.0));
}

void writeColour(uint8_t* data, int x, int y, int width, const color& pixel_color) {
    int offset = (x + (y * width)) * 3;
    data[offset] = uint8_t(255 * pixel_color.x);
    data[offset + 1] = uint8_t(255 * pixel_color.y);
    data[offset + 2] = uint8_t(255 * pixel_color.z);
}

int main() {
    const int imageWidth = 960;
    const int imageHeight = 540;
    const int maxDepth = 5;

    // Scene
    Scene scene;
    scene.add(std::make_unique<Sphere>(point3(0, 0, -1), 0.5));
    scene.add(std::make_unique<Sphere>(point3(0, -100.5, -1), 100));

    // Camera
    auto focalLength = 1.0;
    auto viewportHeight = 2.0;
    auto viewportWidth = viewportHeight * (double(imageWidth)/imageHeight);
    auto cameraCentre = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewportU = vec3(viewportWidth, 0, 0);
    auto viewportV = vec3(0, -viewportHeight, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixelDeltaU = viewportU / double(imageWidth);
    auto pixelDeltaV = viewportV / double(imageHeight);

    // Calculate the location of the upper left pixel.
    auto viewportUpperLeft = cameraCentre
                             - vec3(0, 0, focalLength) - viewportU/2.0 - viewportV/2.0;
    auto pixelUpperLeft = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

    const int samples = 10;

    const int numWorkers = 16;

    // Render.
    std::vector<uint8_t> pixels;
    pixels.resize(imageWidth * imageHeight * 3);
    uint8_t* pixelData = pixels.data();

    std::atomic_int scanlineCounter(imageHeight);
    std::vector<std::thread> workers;

    for (int t = 0; t < numWorkers; ++t) {
        // Divide the work into threads.
        workers.emplace_back([&, t]() {
            int numScanlines = imageHeight / numWorkers;
            int start = numScanlines * t;
            int end = (t == numWorkers - 1) ? imageHeight : start + numScanlines;
            for (int j = start; j < end; j++) {
                for (int i = 0; i < imageWidth; i++) {
                    auto pixelCentre = pixelUpperLeft + (double(i) * pixelDeltaU) + (double(j) * pixelDeltaV);
                    auto rayDirection = pixelCentre - cameraCentre;

                    // Take the average of N samples for this pixel.
                    color finalColour = color(0, 0, 0);
                    for (int i = 0; i < samples; ++i) {
                        finalColour += rayColour(Ray{cameraCentre, rayDirection}, maxDepth, scene);
                    }
                    finalColour /= double(samples);
                    writeColour(pixelData, i, j, imageWidth, finalColour);
                }
                scanlineCounter--;
            }
        });
    }
    
    // Kick off an additional worker thread that monitors the progress of the render.
    workers.emplace_back([&]() {
        int prevCounter = -1;
        while (scanlineCounter > 0) {
            if (scanlineCounter != prevCounter) {
                std::clog << "\rScanlines remaining: " << scanlineCounter << ' ' << std::flush;
            }
            stbi_write_png("output.png", imageWidth, imageHeight, 3, pixels.data(), 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    for (auto& w : workers) {
        w.join();
    }

    std::clog << std::endl << "Done" << std::endl;

    // Write the image to a file.
    stbi_write_png("output.png", imageWidth, imageHeight, 3, pixels.data(), 0);
    return 0;
}