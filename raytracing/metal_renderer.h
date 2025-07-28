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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
