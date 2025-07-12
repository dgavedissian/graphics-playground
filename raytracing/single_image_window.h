#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

class SingleImageWindow {
public:
    SingleImageWindow(int width, int height, const char* title);
    ~SingleImageWindow();

    bool shouldClose() const;
    void updateImage(const uint8_t* pixelData);
    void draw();

private:
    int width_, height_;
    GLFWwindow* window_;
    GLuint tex_;
    GLuint quadVAO_, quadVBO_;
    GLuint shaderProgram_;

};
