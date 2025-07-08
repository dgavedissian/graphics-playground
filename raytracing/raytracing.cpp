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
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    thread_local std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    return min + (max-min) * randomDouble();
}

inline vec3 randomUnitVector() {
    while (true) {
        auto p = vec3{randomDouble(-1, 1), randomDouble(-1, 1), randomDouble(-1, 1)};
        auto lengthSq = glm::dot(p, p);
        if (1e-160 < lengthSq) {
            return p / sqrt(lengthSq);
        }
    }
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
    MetalMaterial(color albedo, double fuzz) : albedo_(albedo), fuzz_(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const Ray& ray, const HitResult& result, color& attenuation, Ray& scattered) const override {
        vec3 reflected = glm::reflect(ray.direction(), result.normal);
        reflected = glm::normalize(reflected) + fuzz_ * randomUnitVector();
        scattered = Ray(result.point, reflected);
        attenuation = albedo_;
        return glm::dot(reflected, result.normal) > 0;
    }

private:
    color albedo_;
    double fuzz_;

};

class GlassMaterial : public Material {
public:
    GlassMaterial(double refractionIndex) : refractionIndex_(refractionIndex) {}

    bool scatter(const Ray& ray, const HitResult& result, color& attenuation, Ray& scattered) const override {
        attenuation = color(1.0);

        double ri = result.front_face ? (1.0 / refractionIndex_) : refractionIndex_;

        vec3 unit_direction = glm::normalize(ray.direction());

        double cos_theta = std::fmin(glm::dot(-unit_direction, result.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;

        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, ri) > randomDouble()) {
            direction = glm::reflect(unit_direction, result.normal);
        } else {
            direction = glm::refract(unit_direction, result.normal, ri);
        }

        scattered = Ray{result.point, direction};
        return true;
    }

private:
    double refractionIndex_;
    
    static double reflectance(double cosine, double refractionIndex) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
        r0 *= r0;
        return r0 + (1 - r0) * std::pow(1 - cosine, 5);
    }
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
        invGamma_(1.0 / gamma)
    {
        setCamera(cameraCentre, point3(0, 0, -1), vec3(0, 1, 0), 90.0);
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

    void setCamera(point3 lookfrom, point3 lookat, vec3 vup, double fovDegrees) {
        lookfrom_ = lookfrom;
        lookat_ = lookat;
        vup_ = vup;
        fovDegrees_ = fovDegrees;

        // Calculate the camera basis vectors.
        vec3 w = glm::normalize(lookfrom_ - lookat_);
        vec3 u = glm::normalize(glm::cross(vup_, w));
        vec3 v = glm::cross(w, u);

        auto focalLength = glm::length(lookfrom - lookat);
        auto theta = glm::radians(fovDegrees_);
        auto h = std::tan(theta / 2);
        auto viewportHeight = 2 * h * focalLength;
        auto viewportWidth = viewportHeight * (double(imageWidth_)/imageHeight_);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewportU = viewportWidth * u;
        auto viewportV = viewportHeight * -v;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixelDeltaU_ = viewportU / double(imageWidth_);
        pixelDeltaV_ = viewportV / double(imageHeight_);
        
        // Calculate the location of the upper left pixel.
        auto viewportUpperLeft = lookfrom_ - focalLength * w - viewportU / 2.0 - viewportV / 2.0;
        pixelUpperLeft_ = viewportUpperLeft + 0.5 * (pixelDeltaU_ + pixelDeltaV_);
    }

private:
    const Scene& scene_;

    int imageWidth_;
    int imageHeight_;
    int maxDepth_;
    int samples_;
    double invGamma_;

    vec3 pixelUpperLeft_;
    vec3 pixelDeltaU_;
    vec3 pixelDeltaV_;
    
    point3 lookfrom_;
    point3 lookat_;
    vec3 vup_;
    double fovDegrees_;

    Ray getRay(int x, int y) {
        // Returns the vector to a random point in the [-0.5, -0.5] - [+0.5, +0.5] unit square.
        vec3 offset(randomDouble() - 0.5, randomDouble() - 0.5, 0);

        auto pixelSample = pixelUpperLeft_ +
            ((double(x) + offset.x) * pixelDeltaU_) +
            ((double(y) + offset.y) * pixelDeltaV_);

        auto rayDirection = pixelSample - lookfrom_;

        return Ray{lookfrom_, rayDirection};
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

void renderImage(Renderer& renderer, uint8_t* pixelData, int numPixels, int numWorkers, int imageWidth) {
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
    const int numBars = 50;
    while (nextPixelIndex < numPixels) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        float progress = float(nextPixelIndex) / numPixels;
        std::string bars = std::string(int(progress * numBars), '#') + std::string(int((1.0 - progress) * numBars), '.');
        std::clog << "\rProgress: " << bars << " " << std::fixed << std::setprecision(1) << (progress * 100.0f) << "% " << std::flush;
    }

    // Wait for all workers to complete.
    for (auto& w : workers) {
        w.join();
    }
}

int main() {
    const int numWorkers = std::max(1u, std::thread::hardware_concurrency() - 1);
    const int samples = 20;

    Scene scene;
    scene.add(std::make_unique<Sphere>(point3(0, -100.5, -1.25), 100, std::make_unique<LambertianMaterial>(color(0.5))));
    scene.add(std::make_unique<Sphere>(point3(0, 0, -1.25), 0.5, std::make_unique<LambertianMaterial>(color(0.1, 0.2, 0.5))));
    scene.add(std::make_unique<Sphere>(point3(1, -0.25, -1.25), 0.25, std::make_unique<GlassMaterial>(1.5)));
    scene.add(std::make_unique<Sphere>(point3(-1, -0.5 + 0.4, -1.25), 0.4, std::make_unique<MetalMaterial>(color(0.8), 0.2)));

    Renderer renderer(scene, 960, 540, 10, samples, 2.0, point3(0, 0, 0));
    renderer.setCamera(point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0), 60.0);

    // Kick off rendering.
    auto startTime = std::chrono::system_clock::now();

    const int imageWidth = renderer.imageWidth();
    const int imageHeight = renderer.imageHeight();
    const int numPixels = imageWidth * imageHeight;

    std::vector<uint8_t> pixels;
    pixels.resize(numPixels * 3);
    uint8_t* pixelData = pixels.data();

    std::clog << "Rendering " << imageWidth << "x" << imageHeight << " with " << numWorkers << " workers." << std::endl;

    renderImage(renderer, pixels.data(), numPixels, numWorkers, imageWidth);

    auto duration = std::chrono::system_clock::now() - startTime;
    std::clog << std::endl << "Done. Took " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::duration<float>>(duration).count() << " seconds." << std::endl;

    // Write the final image to a file.
    stbi_write_png("output.png", imageWidth, imageHeight, 3, pixels.data(), 0);
    return 0;
}
