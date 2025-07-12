#include "cpu_renderer.h"
#include "material.h"

#include <thread>

double reflectance(double cosine, double refractionIndex) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
    r0 *= r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}

bool materialScatter(const Material& mat, const Ray& ray, const HitResult& result, Vec3& attenuation, Ray& scattered) {
    switch (mat.type) {
    case MAT_LAMBERTIAN: {
        Vec3 direction = result.normal + randomUnitVector();

        // Prevent degenerate rays.
        if (glm::all(glm::epsilonEqual(direction, Vec3(0.0), 1e-8))) {
            direction = result.normal;
        }

        scattered = Ray(result.point, glm::normalize(direction));
        attenuation = mat.albedo;
        return true;
    }
    case MAT_METAL: {
        Vec3 reflected = glm::reflect(ray.direction(), result.normal);
        reflected = glm::normalize(reflected) + mat.fuzz * randomUnitVector();
        scattered = Ray(result.point, glm::normalize(reflected));
        attenuation = mat.albedo;
        return glm::dot(reflected, result.normal) > 0;
    }
    case MAT_GLASS: {
        attenuation = Vec3(1.0);

        double ri = result.frontFace ? (1.0 / mat.refractionIndex) : mat.refractionIndex;

        Vec3 unitDirection = glm::normalize(ray.direction());

        double cosTheta = std::fmin(glm::dot(-unitDirection, result.normal), 1.0);
        double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = ri * sinTheta > 1.0;

        Vec3 direction;
        if (cannotRefract || reflectance(cosTheta, ri) > randomDouble()) {
            direction = glm::reflect(unitDirection, result.normal);
        } else {
            direction = glm::refract(unitDirection, result.normal, ri);
        }

        scattered = Ray{result.point, glm::normalize(direction)};
        return true;
    }
    default:
        return false;
    }
}

Vec3 rayColour(const BVHTree& bvhTree, const Ray& r, int depth) {
    if (depth == 0) {
        // If we've run out of rays, then return no colour.
        return Vec3(0, 0, 0);
    }

    // Perform ray test.
    HitResult result;
    bool hit = bvhTree.hit(r, Interval(0.001, std::numeric_limits<double>::max()), result);
    if (!hit) {
        return Vec3(0.4, 0.6, 0.9);
    }

    Ray scattered;
    Vec3 attenuation;
    Vec3 emission = result.material->emit;

    if (materialScatter(*result.material, r, result, attenuation, scattered)) {
        return emission + attenuation * rayColour(bvhTree, scattered, depth - 1);
    }
    return emission;
}

inline double linearToGammaSpace(double value, double invGamma) {
    return std::pow(value, invGamma);
}

inline void writeColour(uint8_t* data, int x, int y, int width, double invGamma, const Vec3& pixelColour) {
    int offset = (x + y * width) * 3;
    data[offset] = uint8_t(255 * linearToGammaSpace(glm::clamp(pixelColour.r, 0.0, 1.0), invGamma));
    data[offset + 1] = uint8_t(255 * linearToGammaSpace(glm::clamp(pixelColour.g, 0.0, 1.0), invGamma));
    data[offset + 2] = uint8_t(255 * linearToGammaSpace(glm::clamp(pixelColour.b, 0.0, 1.0), invGamma));
}

CPURenderer::CPURenderer(
    const std::vector<std::unique_ptr<RTObject>>& scene,
    int imageWidth,
    int imageHeight,
    int maxDepth,
    int samples,
    double gamma,
    int numWorkers,
    const char* title
) :
    bvhTree_(generateBVHTree(scene)),
    imageWidth_(imageWidth),
    imageHeight_(imageHeight),
    maxDepth_(maxDepth),
    samples_(samples),
    invGamma_(1.0 / gamma),
    numWorkers_(numWorkers),
    window_(imageWidth, imageHeight, title),
    buffer_(imageWidth * imageHeight * 3)
{
    setCamera(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0), 90.0);
}

void CPURenderer::setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees) {
    lookfrom_ = lookfrom;
    lookat_ = lookat;
    vup_ = vup;
    fovDegrees_ = fovDegrees;

    // Calculate the camera basis vectors.
    Vec3 w = glm::normalize(lookfrom_ - lookat_);
    Vec3 u = glm::normalize(glm::cross(vup_, w));
    Vec3 v = glm::cross(w, u);

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

bool CPURenderer::shouldClose() const {
    return window_.shouldClose();
}

void CPURenderer::draw() {
    renderImage(buffer_.data());
    window_.updateImage(buffer_.data());
    window_.draw();
}

void CPURenderer::renderImage( uint8_t* pixelData) {
    int numPixels = imageWidth_ * imageHeight_;

    // Spawn workers.
    std::vector<std::thread> workers;
    std::atomic<int> nextPixelIndex{0};
    for (int t = 0; t < numWorkers_; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                auto pixelIndex = nextPixelIndex++;

                // Ran out of jobs, abort.
                if (pixelIndex >= numPixels) {
                    break;
                }

                int x = pixelIndex % imageWidth_;
                int y = pixelIndex / imageWidth_;
                renderPixel(pixelData, x, y, imageWidth_);
            }
        });
    }
    
    // Wait for all workers to complete.
    for (auto& w : workers) {
        w.join();
    }
}

void CPURenderer::renderPixel(uint8_t* data, int x, int y, int width) {
    // Take the average of N samples for this pixel, with slight random offsets for anti-aliasing.
    Vec3 pixelColour = Vec3(0, 0, 0);
    for (int i = 0; i < samples_; ++i) {
        pixelColour += rayColour(bvhTree_, getRay(x, y), maxDepth_);
    }
    pixelColour /= double(samples_);
    writeColour(data, x, y, imageWidth_, invGamma_, pixelColour);
}

Ray CPURenderer::getRay(int x, int y) {
    // Returns the vector to a random point in the [-0.5, -0.5] - [+0.5, +0.5] unit square.
    Vec3 offset(randomDouble() - 0.5, randomDouble() - 0.5, 0);

    auto pixelSample = pixelUpperLeft_ +
        ((double(x) + offset.x) * pixelDeltaU_) +
        ((double(y) + offset.y) * pixelDeltaV_);

    auto rayDirection = glm::normalize(pixelSample - lookfrom_);

    return Ray{lookfrom_, rayDirection};
}
