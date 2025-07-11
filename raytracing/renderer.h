#pragma once

#include "math.h"
#include "material.h"
#include "object.h"

double reflectance(double cosine, double refractionIndex) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
    r0 *= r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}

bool materialScatter(const Material& mat, const Ray& ray, const HitResult& result, Vec3& attenuation, Ray& scattered) {
    switch (mat.type) {
    case MAT_LAMBERTIAN: {
        Vec3 direction = result.normal + randomUnitVector();

        // Prevent degenerate rays.
        if (glm::all(glm::epsilonEqual(direction, Vec3(0.0), 1e-8))) {
            direction = result.normal;
        }

        scattered = Ray(result.point, glm::normalize(direction));
        attenuation = mat.albedo;
        return true;
    }
    case MAT_METAL: {
        Vec3 reflected = glm::reflect(ray.direction(), result.normal);
        reflected = glm::normalize(reflected) + mat.fuzz * randomUnitVector();
        scattered = Ray(result.point, glm::normalize(reflected));
        attenuation = mat.albedo;
        return glm::dot(reflected, result.normal) > 0;
    }
    case MAT_GLASS: {
        attenuation = Vec3(1.0);

        double ri = result.frontFace ? (1.0 / mat.refractionIndex) : mat.refractionIndex;

        Vec3 unitDirection = glm::normalize(ray.direction());

        double cosTheta = std::fmin(glm::dot(-unitDirection, result.normal), 1.0);
        double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = ri * sinTheta > 1.0;

        Vec3 direction;
        if (cannotRefract || reflectance(cosTheta, ri) > randomDouble()) {
            direction = glm::reflect(unitDirection, result.normal);
        } else {
            direction = glm::refract(unitDirection, result.normal, ri);
        }

        scattered = Ray{result.point, glm::normalize(direction)};
        return true;
    }
    default:
        return false;
    }
}

Vec3 rayColour(const Object& scene, const Ray& r, int depth) {
    if (depth == 0) {
        // If we've run out of rays, then return no colour.
        return Vec3(0, 0, 0);
    }

    // Perform ray test.
    HitResult result;
    bool hit = scene.hit(r, Interval(0.001, std::numeric_limits<double>::max()), result);
    if (!hit) {
        return Vec3(0.4, 0.6, 0.9);
    }

    Ray scattered;
    Vec3 attenuation;
    Vec3 emission = result.material->emit;

    if (materialScatter(*result.material, r, result, attenuation, scattered)) {
        return emission + attenuation * rayColour(scene, scattered, depth - 1);
    }
    return emission;
}
