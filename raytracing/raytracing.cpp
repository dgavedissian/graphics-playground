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
#include "material.h"
#include "hittable.h"
#include "bvh_tree.h"

class Renderer {
public:
    Renderer(const Hittable& scene, int imageWidth, int imageHeight, int maxDepth, int samples, double gamma, Vec3 cameraCentre) :
        scene_(scene),
        imageWidth_(imageWidth),
        imageHeight_(imageHeight),
        maxDepth_(maxDepth),
        samples_(samples),
        invGamma_(1.0 / gamma)
    {
        setCamera(cameraCentre, Vec3(0, 0, -1), Vec3(0, 1, 0), 90.0);
    }

    int imageWidth() const { return imageWidth_; }
    int imageHeight() const { return imageHeight_; }

    void setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees) {
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

    void writeColour(uint8_t* data, int x, int y, int width, double invGamma, const Vec3& pixelColour) {
        int offset = (x + y * width) * 3;
        data[offset] = uint8_t(255 * linearToGammaSpace(glm::clamp(pixelColour.r, 0.0, 1.0), invGamma));
        data[offset + 1] = uint8_t(255 * linearToGammaSpace(glm::clamp(pixelColour.g, 0.0, 1.0), invGamma));
        data[offset + 2] = uint8_t(255 * linearToGammaSpace(glm::clamp(pixelColour.b, 0.0, 1.0), invGamma));
    }

private:
    const Hittable& scene_;

    int imageWidth_;
    int imageHeight_;
    int maxDepth_;
    int samples_;
    double invGamma_;

    Vec3 pixelUpperLeft_;
    Vec3 pixelDeltaU_;
    Vec3 pixelDeltaV_;
    
    Vec3 lookfrom_;
    Vec3 lookat_;
    Vec3 vup_;
    double fovDegrees_;

    void renderPixel(uint8_t* data, int x, int y, int width) {
        // Take the average of N samples for this pixel, with slight random offsets for anti-aliasing.
        Vec3 pixelColour = Vec3(0, 0, 0);
        for (int i = 0; i < samples_; ++i) {
            pixelColour += rayColour(getRay(x, y), maxDepth_);
        }
        pixelColour /= double(samples_);
        writeColour(data, x, y, imageWidth_, invGamma_, pixelColour);
    }

    Vec3 rayColour(const Ray& r, int depth) {
        if (depth == 0) {
            // If we've run out of rays, then return no colour.
            return Vec3(0, 0, 0);
        }

        // Perform ray test.
        HitResult result;
        bool hit = scene_.hit(r, Interval(0.001, std::numeric_limits<double>::max()), result);

        if (!hit) {
            return Vec3(0.4, 0.6, 0.9);
        }

        Ray scattered{Vec3(0.0), Vec3(0.0)};
        Vec3 attenuation;
        Vec3 emission = result.material->emitted(result.point);

        if (result.material->scatter(r, result, attenuation, scattered)) {
            return emission + attenuation * rayColour(scattered, depth-1);
        }
        return emission;
    }

    Ray getRay(int x, int y) {
        // Returns the vector to a random point in the [-0.5, -0.5] - [+0.5, +0.5] unit square.
        Vec3 offset(randomDouble() - 0.5, randomDouble() - 0.5, 0);

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
    const int samples = 10;

    Scene scene;
    scene.add(std::make_unique<Sphere>(Vec3(0, -1000, 0), 1000, std::make_unique<LambertianMaterial>(Vec3(0.5))));
    // scene.add(std::make_unique<Sphere>(Vec3(0, 0, -1.25), 0.5, std::make_unique<GlassMaterial>(1.5)));
    // scene.add(std::make_unique<Sphere>(Vec3(1, -0.25, -1.25), 0.25, std::make_unique<LambertianMaterial>(Vec3(0.1, 0.2, 0.5))));
    // scene.add(std::make_unique<Sphere>(Vec3(-1, -0.5 + 0.4, -1.25), 0.4, std::make_unique<MetalMaterial>(Vec3(0.8), 0.2)));

    const int size = 10;
    for (int x = -size; x <= size; ++x) {
        for (int y = -size; y <= size; ++y) {
            auto size = randomDouble(0.15, 0.3);
            Vec3 centre{x + 0.9 * randomDouble(), size, y + 0.9 * randomDouble()};

            std::unique_ptr<Material> material;
            switch (randomInt(0, 3)) {
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

    scene.add(std::make_unique<Sphere>(Vec3(3, 5, 0), 2, std::make_unique<LightMaterial>(Vec3(3))));

    const int imageWidth = 960;
    const int imageHeight = 540;

    auto bvhTree = generateBVHTree(scene);
    Renderer renderer(*bvhTree, imageWidth, imageHeight, 10, samples, 2.0, Vec3(0, 0, 0));
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

        renderer.setCamera(Vec3(std::sin(elapsed * 0.5) * 4, 1.5, std::cos(elapsed * 0.5) * 4 - 1.0), Vec3(0, 0.5, -1), Vec3(0, 1, 0), 60.0);
        renderer.renderImage(numWorkers, pixelData);
        window.updateTexture(pixelData);

        window.draw();
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
        std::cout << "\rFrame time: " << frameTimeMs << " ms" << std::flush;
    }
    return 0;
}
