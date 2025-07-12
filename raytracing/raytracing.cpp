#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <iomanip>
#include <chrono>

#include <glm.hpp>
#include <gtc/epsilon.hpp>

#include "math.h"
#include "material.h"
#include "object.h"
#include "cpu_renderer.h"

int main() {
    const int numWorkers = std::max(1u, std::thread::hardware_concurrency() - 1);
    const int samples = 10;

    std::vector<std::unique_ptr<Object>> scene;
    scene.push_back(std::make_unique<Sphere>(Vec3(0, -1000, 0), 1000, Material{MAT_LAMBERTIAN, Vec3(0.5), 0, 0, Vec3()}));
    // scene.push_back(std::make_unique<Sphere>(Vec3(0, 0, -1.25), 0.5, std::make_unique<GlassMaterial>(1.5)));
    // scene.push_back(std::make_unique<Sphere>(Vec3(1, -0.25, -1.25), 0.25, std::make_unique<LambertianMaterial>(Vec3(0.1, 0.2, 0.5))));
    // scene.push_back(std::make_unique<Sphere>(Vec3(-1, -0.5 + 0.4, -1.25), 0.4, std::make_unique<MetalMaterial>(Vec3(0.8), 0.2)));

    const int size = 10;
    for (int x = -size; x <= size; ++x) {
        for (int y = -size; y <= size; ++y) {
            auto size = randomDouble(0.15, 0.3);
            Vec3 centre{x + 0.9 * randomDouble(), size, y + 0.9 * randomDouble()};

            Material mat;
            mat.type = randomInt(0, 3) + 1;
            switch (mat.type) {
            case MAT_LAMBERTIAN: {
                mat.albedo = randomVec3() * randomVec3();
                break;
            }
            case MAT_METAL: {
                mat.albedo = randomVec3(0.5, 1);
                mat.fuzz = randomDouble(0, 0.5);
                break;
            }
            case MAT_GLASS:
                mat.refractionIndex = 1.5;
                break;
            }
            scene.push_back(std::make_unique<Sphere>(centre, size, std::move(mat)));
        }
    }

    scene.push_back(std::make_unique<Sphere>(Vec3(3, 5, 0), 2, Material{MAT_LIGHT, Vec3(), 0.0, 0.0, Vec3(3)}));

    const int imageWidth = 960;
    const int imageHeight = 540;

    CPURenderer renderer(scene, imageWidth, imageHeight, 10, samples, 2.0, numWorkers, "Raytracing");

    auto start = std::chrono::system_clock::now();
    while (!renderer.shouldClose()) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::system_clock::now() - start
            ).count();

        renderer.setCamera(Vec3(std::sin(elapsed * 0.5) * 4, 1.5, std::cos(elapsed * 0.5) * 4 - 1.0), Vec3(0, 0.5, -1), Vec3(0, 1, 0), 60.0);
        renderer.draw();
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
        std::cout << "\rFrame time: " << frameTimeMs << " ms" << std::flush;
    }
    return 0;
}
