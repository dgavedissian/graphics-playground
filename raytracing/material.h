#pragma once

#include "math.h"

struct Material {
    int type;
    Vec3 albedo;
    double fuzz; // for metal
    double refractionIndex; // for glass
    Vec3 emit; // for light
};

enum MaterialType {
    MAT_LAMBERTIAN = 1,
    MAT_METAL,
    MAT_GLASS,
    MAT_LIGHT
};
