#include "metal_renderer.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <Foundation/NSTypes.hpp>
#include <Metal/Metal.hpp>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.hpp>
#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <fstream>

class MetalRenderer::Impl {
public:
    Impl(const std::vector<std::unique_ptr<RTObject>>& scene, int imageWidth, int imageHeight, int maxDepth, int samples, double gamma, const char* title);
    ~Impl();

    void setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees);
    bool shouldClose() const;
    void draw();
    
    void sendRenderCommand();
    void encodeRenderCommand(MTL::RenderCommandEncoder* renderEncoder);

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

    MTL::Device* metalDevice_;
    GLFWwindow* glfwWindow_;
    NSWindow* metalWindow_;
    CAMetalLayer* metalLayer_;
    CA::MetalDrawable* metalDrawable_;

    MTL::Library* metalDefaultLibrary_;
    MTL::CommandQueue* metalCommandQueue_;
    MTL::CommandBuffer* metalCommandBuffer_;
    MTL::RenderPipelineState* metalRenderPSO_;
    MTL::Buffer* triangleVertexBuffer_;
};

MetalRenderer::MetalRenderer(
    const std::vector<std::unique_ptr<RTObject>>& scene,
    int imageWidth,
    int imageHeight,
    int maxDepth,
    int samples,
    double gamma,
    const char* title
) : impl_(std::make_unique<Impl>(scene, imageWidth, imageHeight, maxDepth, samples, gamma, title)) {}

MetalRenderer::~MetalRenderer() = default;

void MetalRenderer::setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees) {
    impl_->setCamera(lookfrom, lookat, vup, fovDegrees);
}

bool MetalRenderer::shouldClose() const {
    return impl_->shouldClose();
}

void MetalRenderer::draw() {
    impl_->draw();
}

MetalRenderer::Impl::Impl(
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

    metalDevice_ = MTL::CreateSystemDefaultDevice();

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindow_ = glfwCreateWindow(800, 600, "Metal Engine", NULL, NULL);
    if (!glfwWindow_) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    int width, height;
    glfwGetFramebufferSize(glfwWindow_, &width, &height);

    metalWindow_ = glfwGetCocoaWindow(glfwWindow_);
    metalLayer_ = [CAMetalLayer layer];
    metalLayer_.device = (__bridge id<MTLDevice>)metalDevice_;
    metalLayer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalLayer_.drawableSize = CGSizeMake(width, height);
    metalWindow_.contentView.layer = metalLayer_;
    metalWindow_.contentView.wantsLayer = YES;

    glm::vec4 triangleVertices[] = {
        {-0.5f, -0.5f, 0.0f, 0.0f},
        { 0.5f, -0.5f, 0.0f, 0.0f},
        { 0.0f,  0.5f, 0.0f, 0.0f}
    };
    triangleVertexBuffer_ = metalDevice_->newBuffer(&triangleVertices, sizeof(triangleVertices), MTL::ResourceStorageModeShared);
    
    NS::Error* error = nullptr;
    
//    std::ifstream file("raytracing/shaders/metal/raytracing.metal");
//    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
//    NS::String* nsSource = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
//    metalDefaultLibrary_ = metalDevice_->newLibrary(nsSource, nullptr, &error);
    metalDefaultLibrary_ = metalDevice_->newDefaultLibrary();
    if (!metalDefaultLibrary_) {
        std::cerr << "Failed to load default library.";
        std::exit(-1);
    }

    metalCommandQueue_ = metalDevice_->newCommandQueue();

    MTL::Function* vertexShader = metalDefaultLibrary_->newFunction(NS::String::string("vertexShader", NS::ASCIIStringEncoding));
    assert(vertexShader);
    MTL::Function* fragmentShader = metalDefaultLibrary_->newFunction(NS::String::string("fragmentShader", NS::ASCIIStringEncoding));
    assert(fragmentShader);

    MTL::RenderPipelineDescriptor* renderPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    renderPipelineDescriptor->setLabel(NS::String::string("Triangle Rendering Pipeline", NS::ASCIIStringEncoding));
    renderPipelineDescriptor->setVertexFunction(vertexShader);
    renderPipelineDescriptor->setFragmentFunction(fragmentShader);
    assert(renderPipelineDescriptor);
    MTL::PixelFormat pixelFormat = (MTL::PixelFormat)metalLayer_.pixelFormat;
    renderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(pixelFormat);

    metalRenderPSO_ = metalDevice_->newRenderPipelineState(renderPipelineDescriptor, &error);
    if (!metalRenderPSO_) {
        std::cerr << "Failed to create pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        std::exit(-1);
    }
    
    renderPipelineDescriptor->release();
}

MetalRenderer::Impl::~Impl() {
    glfwTerminate();
}

void MetalRenderer::Impl::setCamera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double fovDegrees) {

}

bool MetalRenderer::Impl::shouldClose() const {
    return glfwWindowShouldClose(glfwWindow_);
}

void MetalRenderer::Impl::draw() {
    @autoreleasepool {
        metalDrawable_ = (__bridge CA::MetalDrawable*)[metalLayer_ nextDrawable];
        sendRenderCommand();
    }
    glfwPollEvents();
}

void MetalRenderer::Impl::sendRenderCommand() {
    metalCommandBuffer_ = metalCommandQueue_->commandBuffer();

    MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    MTL::RenderPassColorAttachmentDescriptor* cd = renderPassDescriptor->colorAttachments()->object(0);
    cd->setTexture(metalDrawable_->texture());
    cd->setLoadAction(MTL::LoadActionClear);
    cd->setClearColor(MTL::ClearColor(41.0f / 255.0f, 42.0f / 255.0f, 48.0f / 255.0f, 1.0));
    cd->setStoreAction(MTL::StoreActionStore);

    MTL::RenderCommandEncoder* renderCommandEncoder = metalCommandBuffer_->renderCommandEncoder(renderPassDescriptor);
    encodeRenderCommand(renderCommandEncoder);
    renderCommandEncoder->endEncoding();

    metalCommandBuffer_->presentDrawable(metalDrawable_);
    metalCommandBuffer_->commit();
    metalCommandBuffer_->waitUntilCompleted();

    renderPassDescriptor->release();
}

void MetalRenderer::Impl::encodeRenderCommand(MTL::RenderCommandEncoder* renderCommandEncoder) {
    renderCommandEncoder->setRenderPipelineState(metalRenderPSO_);
    renderCommandEncoder->setVertexBuffer(triangleVertexBuffer_, 0, 0);
    MTL::PrimitiveType typeTriangle = MTL::PrimitiveTypeTriangle;
    NS::UInteger vertexStart = 0;
    NS::UInteger vertexCount = 3;
    renderCommandEncoder->drawPrimitives(typeTriangle, vertexStart, vertexCount);
}
