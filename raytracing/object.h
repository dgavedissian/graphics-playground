#pragma once

#include "math.h"
#include "material.h"

struct HitResult {
    Vec3 point;
    Vec3 normal;
    double t;
    bool frontFace;
    const Material* material;

    void setNormal(const Ray& r, const Vec3& outwardNormal) {
        frontFace = glm::dot(r.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

struct GPUObject {
    glm::vec3 centre;
    float radius;
    std::uint32_t materialIndex;
};

class RTObject {
public:
    virtual ~RTObject() = default;

    virtual bool hit(const Ray& r, Interval t, HitResult& result) const = 0;
    virtual AABB boundingBox() const = 0;
    virtual GPUObject asGPUObject(std::vector<GPUMaterial>& materialStorage) const = 0;
};

class Sphere : public RTObject {
public:
    Sphere(const Vec3& centre, double radius, Material material) :
        centre_(centre),
        radius_(std::fmax(0.0, radius)),
        material_(std::move(material))
    {
        auto rvec = Vec3(radius);
        bbox_ = AABB(centre - rvec, centre + rvec);
    }

    // r.direction() must be normalised.
    bool hit(const Ray& r, Interval t, HitResult& result) const override {
        Vec3 oc = centre_ - r.origin();
        auto h = glm::dot(r.direction(), oc);
        auto c = glm::dot(oc, oc) - radius_ * radius_;
        auto discriminant = h * h - c;
        
        if (discriminant < 0) {
            return false;
        }

        auto sqrt_discriminant = std::sqrt(discriminant);
        auto root = h - sqrt_discriminant;
        if (!t.surrounds(root)) {
            // Try the other root.
            root = h + sqrt_discriminant;
            if (!t.surrounds(root)) {
                // Both roots are outside the range.
                return false;
            }
        }

        result.t = root;
        result.point = r.at(root);
        result.setNormal(r, (result.point - centre_) / radius_);
        result.material = &material_;
        return true;
    }

    AABB boundingBox() const override { return bbox_; }
    
    GPUObject asGPUObject(std::vector<GPUMaterial>& materialStorage) const override {
        GPUMaterial mat;
        mat.type = material_.type;
        mat.albedo = glm::vec3(material_.albedo);
        mat.fuzz = material_.fuzz;
        mat.refractionIndex = material_.refractionIndex;
        mat.emit = material_.emit;
        materialStorage.push_back(mat);
        
        GPUObject object;
        object.centre = glm::vec3(centre_);
        object.radius = radius_;
        object.materialIndex = std::uint32_t(materialStorage.size() - 1);
        return object;
    }

private:
    Vec3 centre_;
    double radius_;
    Material material_;

    AABB bbox_;

};
