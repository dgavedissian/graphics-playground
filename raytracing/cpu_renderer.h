#pragma once

#include <vector>

#include "math.h"
#include "object.h"
#include "bvh_tree.h"
#include "single_image_window.h"

class CPURenderer {
public:
    CPURenderer(const std::vector<std::unique_ptr<RTObject>>& scene, int imageWidth, int imageHeight, int maxDepth, int samples, double gamma, int numWorkers, const char* title);

    void setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees);
    bool shouldClose() const;
    void draw();

private:
    BVHTree bvhTree_;

    int imageWidth_;
    int imageHeight_;
    int maxDepth_;
    int samples_;
    double invGamma_;

    int numWorkers_;
    SingleImageWindow window_;

    std::vector<uint8_t> buffer_;

    Vec3 pixelUpperLeft_;
    Vec3 pixelDeltaU_;
    Vec3 pixelDeltaV_;
    
    Vec3 lookfrom_;
    Vec3 lookat_;
    Vec3 vup_;
    double fovDegrees_;

    void renderImage(uint8_t* pixelData);
    void renderPixel(uint8_t* data, int x, int y, int width);
    Ray getRay(int x, int y);
};
