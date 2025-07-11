#pragma once

#include "math.h"

struct HitResult;

class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(const Ray& ray, const HitResult& result, Vec3& attenuation, Ray& scattered) const {
        return false;
    }

    virtual Vec3 emitted(const Vec3& p) const {
        return Vec3(0);
    }
};

struct HitResult {
    Vec3 point;
    Vec3 normal;
    double t;
    bool front_face;
    Material* material;

    void setNormal(const Ray& r, const Vec3& outward_normal) {
        front_face = glm::dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class LambertianMaterial : public Material {
public:
    LambertianMaterial(Vec3 albedo) : albedo_(albedo) {}

    bool scatter(const Ray& ray, const HitResult& result, Vec3& attenuation, Ray& scattered) const override {
        Vec3 direction = result.normal + randomUnitVector();

        // Prevent degenerate rays.
        if (glm::all(glm::epsilonEqual(direction, Vec3(0.0), 1e-8))) {
            direction = result.normal;
        }

        scattered = Ray(result.point, direction);
        attenuation = albedo_;
        return true;
    }

private:
    Vec3 albedo_;

};

class MetalMaterial : public Material {
public:
    MetalMaterial(Vec3 albedo, double fuzz) : albedo_(albedo), fuzz_(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const Ray& ray, const HitResult& result, Vec3& attenuation, Ray& scattered) const override {
        Vec3 reflected = glm::reflect(ray.direction(), result.normal);
        reflected = glm::normalize(reflected) + fuzz_ * randomUnitVector();
        scattered = Ray(result.point, reflected);
        attenuation = albedo_;
        return glm::dot(reflected, result.normal) > 0;
    }

private:
    Vec3 albedo_;
    double fuzz_;

};

class GlassMaterial : public Material {
public:
    GlassMaterial(double refractionIndex) : refractionIndex_(refractionIndex) {}

    bool scatter(const Ray& ray, const HitResult& result, Vec3& attenuation, Ray& scattered) const override {
        attenuation = Vec3(1.0);

        double ri = result.front_face ? (1.0 / refractionIndex_) : refractionIndex_;

        Vec3 unit_direction = glm::normalize(ray.direction());

        double cos_theta = std::fmin(glm::dot(-unit_direction, result.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;

        Vec3 direction;
        if (cannot_refract || reflectance(cos_theta, ri) > randomDouble()) {
            direction = glm::reflect(unit_direction, result.normal);
        } else {
            direction = glm::refract(unit_direction, result.normal, ri);
        }

        scattered = Ray{result.point, direction};
        return true;
    }

private:
    double refractionIndex_;
    
    static double reflectance(double cosine, double refractionIndex) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
        r0 *= r0;
        return r0 + (1 - r0) * std::pow(1 - cosine, 5);
    }
};

class LightMaterial : public Material {
public:
    LightMaterial(const Vec3& emit) : emit_(emit) {}

    Vec3 emitted(const Vec3& p) const override {
        return emit_;
    }

private:
    Vec3 emit_;
};
