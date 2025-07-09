#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <iomanip>
#include <chrono>

#include <glm.hpp>
#include <gtc/epsilon.hpp>

#define GLAD_GL_IMPLEMENTATION
#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include "math.h"

struct HitResult;

class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(const Ray& ray, const HitResult& result, color& attenuation, Ray& scattered) const {
        return false;
    }
};

struct HitResult {
    vec3 point;
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

    virtual AABB boundingBox() const = 0;
};

class Sphere : public Hittable {
public:
    Sphere(const vec3& centre, double radius, std::unique_ptr<Material> material) :
        centre_(centre),
        radius_(std::fmax(0.0, radius)),
        material_(std::move(material))
    {
        auto rvec = vec3(radius);
        bbox_ = AABB(centre - rvec, centre + rvec);
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

    AABB boundingBox() const override { return bbox_; }

private:
    vec3 centre_;
    double radius_;
    std::unique_ptr<Material> material_;

    AABB bbox_;

};

class BVHNode : public Hittable {
  public:
    BVHNode(std::vector<Hittable*>& objects, std::vector<std::unique_ptr<BVHNode>>& nodes, size_t start, size_t end) {
        // Build the bounding box of the span of source objects.
        bbox_ = objects[start]->boundingBox();
        for (size_t i = start + 1; i < end; i++) {
            bbox_ = AABB(bbox_, objects[i]->boundingBox());
        }

        int axis = bbox_.longestAxis();

        auto comparator = (axis == 0) ? boxXCompare : ((axis == 1) ? boxYCompare : boxZCompare);
        size_t objectSpan = end - start;

        if (objectSpan == 1) {
            left_ = right_ = objects[start];
        } else if (objectSpan == 2) {
            left_ = objects[start];
            right_ = objects[start + 1];
        } else {
            std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

            auto mid = start + objectSpan / 2;
            auto left = std::make_unique<BVHNode>(objects, nodes, start, mid);
            auto right = std::make_unique<BVHNode>(objects, nodes, mid, end);
            left_ = left.get();
            right_ = right.get();
            nodes.push_back(std::move(left));
            nodes.push_back(std::move(right));
        }
    }

    bool hit(const Ray& r, interval t, HitResult& result) const override {
        if (!bbox_.hit(r, t)) {
            return false;
        }

        bool hitLeft = left_->hit(r, t, result);
        bool hitRight = right_->hit(r, interval(t.min, hitLeft ? result.t : t.max), result);

        return hitLeft || hitRight;
    }

    AABB boundingBox() const override { return bbox_; }

  private:
    Hittable* left_;
    Hittable* right_;
    AABB bbox_;

    static bool boxCompare(const Hittable* a, const Hittable* b, int axisIndex) {
        auto aAxisInterval = a->boundingBox().axisInterval(axisIndex);
        auto bAxisInterval = b->boundingBox().axisInterval(axisIndex);
        return aAxisInterval.min < bAxisInterval.min;
    }

    static bool boxXCompare(const Hittable* a, const Hittable* b) {
        return boxCompare(a, b, 0);
    }

    static bool boxYCompare(const Hittable* a, const Hittable* b) {
        return boxCompare(a, b, 1);
    }

    static bool boxZCompare(const Hittable* a, const Hittable* b) {
        return boxCompare(a, b, 2);
    }
};

class BVHTree : public Hittable {
public:
    BVHTree(std::unique_ptr<BVHNode> root, std::vector<std::unique_ptr<BVHNode>> storage) :
        root_(std::move(root)),
        storage_(std::move(storage))
    {
    }

    bool hit(const Ray& r, interval t, HitResult& result) const override {
        return root_->hit(r, t, result);
    }

    AABB boundingBox() const override {
        return root_->boundingBox();
    }

private:
    std::unique_ptr<BVHNode> root_;
    std::vector<std::unique_ptr<BVHNode>> storage_;

};

class Scene : public Hittable {
public:
    void add(std::unique_ptr<Hittable> hittable) {
        bbox_ = AABB(bbox_, hittable->boundingBox());
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

    AABB boundingBox() const override { return bbox_; }

    std::unique_ptr<BVHTree> generateBVHTree() const {
        std::vector<Hittable*> hittables;
        std::transform(
            hittables_.begin(), hittables_.end(),
            std::back_inserter(hittables),
            [](const auto& p) { return p.get(); }
        );

        std::vector<std::unique_ptr<BVHNode>> nodes;
        auto root = std::make_unique<BVHNode>(hittables, nodes, 0, hittables.size());
        return std::make_unique<BVHTree>(std::move(root), std::move(nodes));
    }

private:
    std::vector<std::unique_ptr<Hittable>> hittables_;
    AABB bbox_;
};

class Renderer {
public:
    Renderer(const Hittable& scene, int imageWidth, int imageHeight, int maxDepth, int samples, double gamma, vec3 cameraCentre) :
        scene_(scene),
        imageWidth_(imageWidth),
        imageHeight_(imageHeight),
        maxDepth_(maxDepth),
        samples_(samples),
        invGamma_(1.0 / gamma)
    {
        setCamera(cameraCentre, vec3(0, 0, -1), vec3(0, 1, 0), 90.0);
    }

    int imageWidth() const { return imageWidth_; }
    int imageHeight() const { return imageHeight_; }

    void setCamera(vec3 lookfrom, vec3 lookat, vec3 vup, double fovDegrees) {
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

    void renderImage(int numWorkers, uint8_t* pixelData) {
        int numPixels = imageWidth_ * imageHeight_;

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

    double linearToGammaSpace(double value, double invGamma) {
        return std::pow(value, invGamma);
    }

    void writeColour(uint8_t* data, int x, int y, int width, double invGamma, const color& pixelColour) {
        int offset = (x + y * width) * 3;
        data[offset] = uint8_t(255 * linearToGammaSpace(pixelColour.r, invGamma));
        data[offset + 1] = uint8_t(255 * linearToGammaSpace(pixelColour.g, invGamma));
        data[offset + 2] = uint8_t(255 * linearToGammaSpace(pixelColour.b, invGamma));
    }

private:
    const Hittable& scene_;

    int imageWidth_;
    int imageHeight_;
    int maxDepth_;
    int samples_;
    double invGamma_;

    vec3 pixelUpperLeft_;
    vec3 pixelDeltaU_;
    vec3 pixelDeltaV_;
    
    vec3 lookfrom_;
    vec3 lookat_;
    vec3 vup_;
    double fovDegrees_;

    void renderPixel(uint8_t* data, int x, int y, int width) {
        // Take the average of N samples for this pixel, with slight random offsets for anti-aliasing.
        color pixelColour = color(0, 0, 0);
        for (int i = 0; i < samples_; ++i) {
            pixelColour += rayColour(getRay(x, y), maxDepth_);
        }
        pixelColour /= double(samples_);
        writeColour(data, x, y, imageWidth_, invGamma_, pixelColour);
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

    Ray getRay(int x, int y) {
        // Returns the vector to a random point in the [-0.5, -0.5] - [+0.5, +0.5] unit square.
        vec3 offset(randomDouble() - 0.5, randomDouble() - 0.5, 0);

        auto pixelSample = pixelUpperLeft_ +
            ((double(x) + offset.x) * pixelDeltaU_) +
            ((double(y) + offset.y) * pixelDeltaV_);

        auto rayDirection = pixelSample - lookfrom_;

        return Ray{lookfrom_, rayDirection};
    }
};

const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

class Window {
public:
    Window(int width, int height, const char* title)
        : width_(width), height_(height) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwMakeContextCurrent(window_);
        int version = gladLoadGL(glfwGetProcAddress);
        if (version == 0) {
            throw std::runtime_error("Failed to initialize OpenGL context");
        }

        // Texture
        glGenTextures(1, &tex_);
        glBindTexture(GL_TEXTURE_2D, tex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        // Fullscreen quad.
        float quadVertices[] = {
            -1.0f,  1.0f,  0.0f, 0.0f,
            -1.0f, -1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 1.0f,
             1.0f,  1.0f,  1.0f, 0.0f
        };
        glGenVertexArrays(1, &quadVAO_);
        glGenBuffers(1, &quadVBO_);
        glBindVertexArray(quadVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        // Shaders.
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        shaderProgram_ = glCreateProgram();
        glAttachShader(shaderProgram_, vertexShader);
        glAttachShader(shaderProgram_, fragmentShader);
        glLinkProgram(shaderProgram_);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    ~Window() {
        glDeleteVertexArrays(1, &quadVAO_);
        glDeleteBuffers(1, &quadVBO_);
        glDeleteTextures(1, &tex_);
        glDeleteProgram(shaderProgram_);
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    bool shouldClose() const { return glfwWindowShouldClose(window_); }
    
    void updateTexture(const uint8_t* pixelData) {
        glBindTexture(GL_TEXTURE_2D, tex_);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, pixelData);
    }

    void draw() {
        glUseProgram(shaderProgram_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_);
        glUniform1i(glGetUniformLocation(shaderProgram_, "screenTexture"), 0);
        glBindVertexArray(quadVAO_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        glfwPollEvents();
        glfwSwapBuffers(window_);
    }

private:
    int width_, height_;
    GLFWwindow* window_;
    GLuint tex_;
    GLuint quadVAO_, quadVBO_;
    GLuint shaderProgram_;

};

int main() {
    const int numWorkers = std::max(1u, std::thread::hardware_concurrency() - 1);
    const int samples = 2;

    Scene scene;
    scene.add(std::make_unique<Sphere>(vec3(0, -1000, 0), 1000, std::make_unique<LambertianMaterial>(color(0.5))));
    // scene.add(std::make_unique<Sphere>(vec3(0, 0, -1.25), 0.5, std::make_unique<GlassMaterial>(1.5)));
    // scene.add(std::make_unique<Sphere>(vec3(1, -0.25, -1.25), 0.25, std::make_unique<LambertianMaterial>(color(0.1, 0.2, 0.5))));
    // scene.add(std::make_unique<Sphere>(vec3(-1, -0.5 + 0.4, -1.25), 0.4, std::make_unique<MetalMaterial>(color(0.8), 0.2)));

    const int size = 20;
    for (int x = -size; x <= size; ++x) {
        for (int y = -size; y <= size; ++y) {
            auto size = randomDouble(0.1, 0.3);
            vec3 centre{x + 0.9 * randomDouble(), size, y + 0.9 * randomDouble()};

            std::unique_ptr<Material> material;
            switch (randomInt(0, 2)) {
            case 0: {
                material = std::make_unique<LambertianMaterial>(randomVec3() * randomVec3());
                break;
            }
            case 1: {
                auto albedo = randomVec3(0.5, 1);
                auto fuzz = randomDouble(0, 0.5);
                material = std::make_unique<MetalMaterial>(albedo, fuzz);
                break;
            }
            case 2:
                material = std::make_unique<GlassMaterial>(1.5);
                break;
            }
            scene.add(std::make_unique<Sphere>(centre, size, std::move(material)));
        }
    }

    auto bvh_tree = scene.generateBVHTree();

    const int imageWidth = 960;
    const int imageHeight = 540;

    Renderer renderer(*bvh_tree, imageWidth, imageHeight, 10, samples, 2.0, vec3(0, 0, 0));
    Window window(imageWidth, imageHeight, "Raytracing");

    const int numPixels = imageWidth * imageHeight;
    std::vector<uint8_t> pixels(numPixels * 3);
    uint8_t* pixelData = pixels.data();

    auto start = std::chrono::system_clock::now();
    while (!window.shouldClose()) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::system_clock::now() - start
            ).count();

        renderer.setCamera(vec3(std::sin(elapsed * 0.5) * 4, 1.5, std::cos(elapsed * 0.5) * 4 - 1.0), vec3(0, 0.5, -1), vec3(0, 1, 0), 60.0);
        renderer.renderImage(numWorkers, pixelData);
        window.updateTexture(pixelData);

        window.draw();
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
        std::cout << "\rFrame time: " << frameTimeMs << " ms" << std::flush;
    }
    return 0;
}
