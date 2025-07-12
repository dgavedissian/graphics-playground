#pragma once

#include <vector>

#include "math.h"
#include "object.h"
#include "bvh_tree.h"

class MetalRenderer {
public:
    MetalRenderer(const std::vector<std::unique_ptr<RTObject>>& scene, int imageWidth, int imageHeight, int maxDepth, int samples, double gamma, const char* title);
    ~MetalRenderer();

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
    
    Vec3 lookfrom_;
    Vec3 lookat_;
    Vec3 vup_;
    double fovDegrees_;

    std::unique_ptr<struct MetalImpl> impl_;
};
