#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <iomanip>
#include <chrono>

#include <glm.hpp>
#include <gtc/epsilon.hpp>

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

inline vec3 reflect(const vec3& vec, const vec3& normal) {
    return vec - 2.0 * glm::dot(vec, normal) * normal;
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

struct HitResult;

class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(const Ray& ray, const HitResult& result, color& attenuation, Ray& scattered) const {
        return false;
    }
};

struct HitResult {
    point3 point;
    vec3 normal;
    double t;
    bool front_face;
    Material* material;

    void setNormal(const Ray& r, const vec3& outward_normal) {
        front_face = glm::dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class LambertianMaterial : public Material {
public:
    LambertianMaterial(color albedo) : albedo_(albedo) {}

    bool scatter(const Ray& ray, const HitResult& result, color& attenuation, Ray& scattered) const override {
        vec3 direction = result.normal + randomUnitVector();

        // Prevent degenerate rays.
        if (glm::all(glm::epsilonEqual(direction, vec3(0.0), 1e-8))) {
            direction = result.normal;
        }

        scattered = Ray(result.point, direction);
        attenuation = albedo_;
        return true;
    }

private:
    color albedo_;

};

class MetalMaterial : public Material {
public:
    MetalMaterial(color albedo) : albedo_(albedo) {}

    bool scatter(const Ray& ray, const HitResult& result, color& attenuation, Ray& scattered) const override {
        vec3 reflected = reflect(ray.direction(), result.normal);
        scattered = Ray(result.point, reflected);
        attenuation = albedo_;
        return true;
    }

private:
    color albedo_;

};

class Hittable {
public:
    virtual ~Hittable() = default;

    virtual bool hit(const Ray& r, interval t, HitResult& result) const = 0;
};

class Sphere : public Hittable {
public:
    Sphere(const point3& centre, double radius, std::unique_ptr<Material> material) :
        centre_(centre),
        radius_(std::fmax(0.0, radius)),
        material_(std::move(material))
    {
    }

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
        result.material = material_.get();
        return true;
    }

private:
    point3 centre_;
    double radius_;
    std::unique_ptr<Material> material_;

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

double linearToGammaSpace(double value, double invGamma) {
    return std::pow(value, invGamma);
}

void writeColour(uint8_t* data, int x, int y, int width, double invGamma, const color& pixelColour) {
    int offset = (x + y * width) * 3;
    data[offset] = uint8_t(255 * linearToGammaSpace(pixelColour.r, invGamma));
    data[offset + 1] = uint8_t(255 * linearToGammaSpace(pixelColour.g, invGamma));
    data[offset + 2] = uint8_t(255 * linearToGammaSpace(pixelColour.b, invGamma));
}

class Renderer {
public:
    Renderer(const Scene& scene, int imageWidth, int imageHeight, int maxDepth, int samples, double gamma, point3 cameraCentre) :
        scene_(scene),
        imageWidth_(imageWidth),
        imageHeight_(imageHeight),
        maxDepth_(maxDepth),
        samples_(samples),
        invGamma_(1.0 / gamma),
        cameraCentre_(cameraCentre)
    {
        auto focalLength = 1.0;
        auto viewportHeight = 2.0;
        auto viewportWidth = viewportHeight * (double(imageWidth)/imageHeight);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewportU = vec3(viewportWidth, 0, 0);
        auto viewportV = vec3(0, -viewportHeight, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixelDeltaU_ = viewportU / double(imageWidth);
        pixelDeltaV_ = viewportV / double(imageHeight);

        // Calculate the location of the upper left pixel.
        auto viewportUpperLeft = cameraCentre
                                - vec3(0, 0, focalLength) - viewportU/2.0 - viewportV/2.0;
        pixelUpperLeft_ = viewportUpperLeft + 0.5 * (pixelDeltaU_ + pixelDeltaV_);
    }

    void renderPixel(uint8_t* data, int x, int y, int width) {
        // Take the average of N samples for this pixel, with slight random offsets for anti-aliasing.
        color pixelColour = color(0, 0, 0);
        for (int i = 0; i < samples_; ++i) {
            pixelColour += rayColour(getRay(x, y), maxDepth_);
        }
        pixelColour /= double(samples_);
        writeColour(data, x, y, imageWidth_, invGamma_, pixelColour);
    }

    int imageWidth() const { return imageWidth_; }
    int imageHeight() const { return imageHeight_; }

private:
    const Scene& scene_;

    int imageWidth_;
    int imageHeight_;
    int maxDepth_;
    int samples_;
    double invGamma_;

    point3 cameraCentre_;

    vec3 pixelUpperLeft_;
    vec3 pixelDeltaU_;
    vec3 pixelDeltaV_;

    Ray getRay(int x, int y) {
        // Returns the vector to a random point in the [-0.5, -0.5] - [+0.5, +0.5] unit square.
        vec3 offset(randomDouble() - 0.5, randomDouble() - 0.5, 0);

        auto pixelSample = pixelUpperLeft_ +
            ((double(x) + offset.x) * pixelDeltaU_) +
            ((double(y) + offset.y) * pixelDeltaV_);

        auto rayDirection = pixelSample - cameraCentre_;

        return Ray{cameraCentre_, rayDirection};
    }

    color rayColour(const Ray& r, int depth) {
        if (depth == 0) {
            // If we've run out of rays, then return no colour.
            return color(0, 0, 0);
        }

        HitResult result;
        if (scene_.hit(r, interval(0.001, std::numeric_limits<double>::max()), result)) {
            Ray scattered{vec3(0.0), vec3(0.0)};
            color attenuation;
            if (result.material->scatter(r, result, attenuation, scattered)) {
                return attenuation * rayColour(scattered, depth-1);
            }
            return color(0, 0, 0);
        }
        
        vec3 unitDirection = glm::normalize(r.direction());
        auto a = 0.5 * (unitDirection.y + 1.0);
        return ((1.0 - a) * color(1.0, 1.0, 1.0)) + (a * color(0.5, 0.7, 1.0));
    }
};

struct RenderJob {
    int i, j;
};

int main() {
    const int numWorkers = 8;
    const int samples = 10;

    Scene scene;
    scene.add(std::make_unique<Sphere>(point3(0, -100.5, -1), 100, std::make_unique<LambertianMaterial>(color(0.5))));
    scene.add(std::make_unique<Sphere>(point3(0, 0, -1), 0.5, std::make_unique<LambertianMaterial>(color(0.1, 0.2, 0.5))));
    scene.add(std::make_unique<Sphere>(point3(1, -0.25, -1), 0.25, std::make_unique<LambertianMaterial>(color(1.0, 0.2, 0.5))));
    scene.add(std::make_unique<Sphere>(point3(-1, -0.5 + 0.4, -1), 0.4, std::make_unique<MetalMaterial>(color(0.8))));

    Renderer renderer(scene, 960, 540, 10, samples, 2.0, point3(0, 0, 0.25));

    // Kick off rendering.
    auto startTime = std::chrono::system_clock::now();

    const int imageWidth = renderer.imageWidth();
    const int imageHeight = renderer.imageHeight();
    const int numPixels = imageWidth * imageHeight;

    std::vector<uint8_t> pixels;
    pixels.resize(numPixels * 3);
    uint8_t* pixelData = pixels.data();

    // Spawn workers.
    std::vector<std::thread> workers;
    std::atomic<int> nextPixelIndex{0};
    for (int t = 0; t < numWorkers; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                auto pixelIndex = nextPixelIndex++;

                // Ran out of jobs, abort.
                if (pixelIndex >= numPixels) {
                    break;
                }
                
                int x = pixelIndex % imageWidth;
                int y = pixelIndex / imageWidth;
                renderer.renderPixel(pixelData, x, y, imageWidth);
            }
        });
    }
    
    // Monitor progress.
    auto lastWritten = std::chrono::system_clock::now();
    int totalPixels = imageWidth * imageHeight;
    const int numBars = 50;
    while (nextPixelIndex < numPixels) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        float progress = float(nextPixelIndex) / totalPixels;
        std::string bars = std::string(int(progress * numBars), '#') + std::string(int((1.0 - progress) * numBars), '.');
        std::clog << "\rProgress: " << bars << " " << std::fixed << std::setprecision(1) << (progress * 100.0f) << "% " << std::flush;

        // Periodically write the image.
        auto now = std::chrono::system_clock::now();
        if ((now - lastWritten) > std::chrono::milliseconds(500)) {
            stbi_write_png("output.png", imageWidth, imageHeight, 3, pixels.data(), 0);
            lastWritten = now;
        }
    }

    // Wait for all workers to complete.
    for (auto& w : workers) {
        w.join();
    }

    auto duration = std::chrono::system_clock::now() - startTime;
    std::clog << std::endl << "Done. Took " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::duration<float>>(duration).count() << " seconds." << std::endl;

    // Write the final image to a file.
    stbi_write_png("output.png", imageWidth, imageHeight, 3, pixels.data(), 0);
    return 0;
}
