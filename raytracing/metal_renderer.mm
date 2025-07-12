#include "metal_renderer.h"

#define GLFW_INCLUDE_NONE
#import <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3native.h>

#include <Foundation/NSTypes.hpp>
#include <Metal/Metal.hpp>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.hpp>
#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/QuartzCore.hpp>

struct MetalImpl {
    MTL::Device* metalDevice;
    GLFWwindow* glfwWindow;
    NSWindow* metalWindow;
    CAMetalLayer* metalLayer;
};

MetalRenderer::MetalRenderer(
    const std::vector<std::unique_ptr<RTObject>>& scene,
    int imageWidth,
    int imageHeight,
    int maxDepth,
    int samples,
    double gamma,
    const char* title
) :
    bvhTree_(generateBVHTree(scene)),
    imageWidth_(imageWidth),
    imageHeight_(imageHeight),
    maxDepth_(maxDepth),
    samples_(samples),
    invGamma_(1.0 / gamma)
{
    setCamera(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0), 90.0);

    impl_ = std::make_unique<MetalImpl>();
    impl_->metalDevice = MTL::CreateSystemDefaultDevice();

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    impl_->glfwWindow = glfwCreateWindow(800, 600, "Metal Engine", NULL, NULL);
    if (!impl_->glfwWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    impl_->metalWindow = glfwGetCocoaWindow(impl_->glfwWindow);
    impl_->metalLayer = [CAMetalLayer layer];
    impl_->metalLayer.device = (__bridge id<MTLDevice>)impl_->metalDevice;
    impl_->metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    impl_->metalWindow.contentView.layer = impl_->metalLayer;
    impl_->metalWindow.contentView.wantsLayer = YES;
}

MetalRenderer::~MetalRenderer() {
    glfwTerminate();
}

void MetalRenderer::setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees) {

}

bool MetalRenderer::shouldClose() const {
    return glfwWindowShouldClose(impl_->glfwWindow);
}

void MetalRenderer::draw() {
    glfwPollEvents();
}
