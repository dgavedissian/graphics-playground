#pragma once

#include "math.h"

struct Material {
    int type;
    Vec3 albedo;
    double fuzz; // for metal
    double refractionIndex; // for glass
    Vec3 emit; // for light
};

struct GPUMaterial {
    std::uint32_t type;
    glm::vec3 albedo;
    glm::vec3 emit;
    float fuzz;
    float refractionIndex;
};

enum MaterialType {
    MAT_LAMBERTIAN = 1,
    MAT_METAL,
    MAT_GLASS,
    MAT_LIGHT
};
