#include <iostream>
#include <vector>
#include <glm.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using color = glm::dvec3;
using point3 = glm::dvec3;
using vec3 = glm::dvec3;

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

    virtual bool hit(const Ray& r, double tmin, double tmax, HitResult& result) const = 0;
};

class Sphere : public Hittable {
public:
    Sphere(const point3& centre, double radius) : centre_(centre), radius_(std::fmax(0.0, radius)) {}

    bool hit(const Ray& r, double tmin, double tmax, HitResult& result) const override {
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
        if (root <= tmin || root >= tmax) {
            // Try the other root.
            root = (h + sqrt_discriminant) / a;
            if (root <= tmin || root >= tmax) {
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

    bool hit(const Ray& r, double tmin, double tmax, HitResult& result) const override {
        bool hitAnything = false;

        for (const auto& hittable : hittables_) {
            HitResult tempResult;
            if (hittable->hit(r, tmin, tmax, tempResult)) {
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

color ray_color(const Ray& r, const Scene& scene) {
    HitResult result;
    if (scene.hit(r, 0.0, std::numeric_limits<double>::max(), result)) {
        return 0.5 * color(result.normal.x + 1, result.normal.y + 1, result.normal.z + 1);
    }
    
    vec3 unitDirection = glm::normalize(r.direction());
    auto a = 0.5 * (unitDirection.y + 1.0);
    return ((1.0 - a) * color(1.0, 1.0, 1.0)) + (a * color(0.5, 0.7, 1.0));
}

void write_color(std::vector<uint8_t>& out, const color& pixel_color) {
    out.emplace_back(uint8_t(255 * pixel_color.x));
    out.emplace_back(uint8_t(255 * pixel_color.y));
    out.emplace_back(uint8_t(255 * pixel_color.z));
}

int main() {
    int image_width = 1280;
    int image_height = 720;

    // Scene
    Scene scene;
    scene.add(std::make_unique<Sphere>(point3(0, 0, -1), 0.5));
    scene.add(std::make_unique<Sphere>(point3(0, -100.5, -1), 100));

    // Camera
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / double(image_width);
    auto pixel_delta_v = viewport_v / double(image_height);

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2.0 - viewport_v/2.0;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Render.
    std::vector<uint8_t> pixels;
    pixels.reserve(image_width * image_height * 3);
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_center = pixel00_loc + (double(i) * pixel_delta_u) + (double(j) * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            write_color(pixels, ray_color(Ray{camera_center, ray_direction}, scene));
        }
    }

    std::clog << std::endl << "\rDone" << std::endl;

    // Write the image to a file.
    stbi_write_png("output.png", image_width, image_height, 3, pixels.data(), 0);
    return 0;
}