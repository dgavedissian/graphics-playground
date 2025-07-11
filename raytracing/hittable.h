#pragma once

#include "math.h"
#include "material.h"

class Hittable {
public:
    virtual ~Hittable() = default;

    virtual bool hit(const Ray& r, Interval t, HitResult& result) const = 0;

    virtual AABB boundingBox() const = 0;

    virtual bool isLeaf() const { return true; }
};

class Sphere : public Hittable {
public:
    Sphere(const Vec3& centre, double radius, std::unique_ptr<Material> material) :
        centre_(centre),
        radius_(std::fmax(0.0, radius)),
        material_(std::move(material))
    {
        auto rvec = Vec3(radius);
        bbox_ = AABB(centre - rvec, centre + rvec);
    }

    bool hit(const Ray& r, Interval t, HitResult& result) const override {
        Vec3 oc = centre_ - r.origin();
        auto a = glm::dot(r.direction(), r.direction());
        auto h = glm::dot(r.direction(), oc);
        auto c = glm::dot(oc, oc) - radius_ * radius_;
        auto discriminant = h * h - a * c;
        
        if (discriminant < 0) {
            return false;
        }

        auto sqrt_discriminant = std::sqrt(discriminant);
        auto root = (h - sqrt_discriminant) / a;
        if (!t.surrounds(root)) {
            // Try the other root.
            root = (h + sqrt_discriminant) / a;
            if (!t.surrounds(root)) {
                // Both roots are outside the range.
                return false;
            }
        }

        result.t = root;
        result.point = r.at(root);
        result.setNormal(r, (result.point - centre_) / radius_);
        result.material = material_.get();
        return true;
    }

    AABB boundingBox() const override { return bbox_; }

private:
    Vec3 centre_;
    double radius_;
    std::unique_ptr<Material> material_;

    AABB bbox_;

};

class Scene : public Hittable {
public:
    void add(std::unique_ptr<Hittable> hittable) {
        bbox_ = AABB(bbox_, hittable->boundingBox());
        hittables_.emplace_back(std::move(hittable));
    }

    bool hit(const Ray& r, Interval t, HitResult& result) const override {
        bool hitAnything = false;
        double tmax = t.max;

        for (const auto& hittable : hittables_) {
            HitResult tempResult;
            if (hittable->hit(r, Interval(t.min, tmax), tempResult)) {
                hitAnything = true;
                result = tempResult;

                // The maximum the ray can travel now, is to where this object was hit.
                tmax = tempResult.t;
            }
        }
        return hitAnything;
    }

    AABB boundingBox() const override { return bbox_; }

    const std::vector<std::unique_ptr<Hittable>>& hittables() const { return hittables_; }

private:
    std::vector<std::unique_ptr<Hittable>> hittables_;
    AABB bbox_;
};
