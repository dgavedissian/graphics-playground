#pragma once

#include <glm.hpp>
#include <gtc/epsilon.hpp>

using Vec3 = glm::dvec3;

class Ray {
public:
    Ray() = default;
    Ray(const Vec3& origin, const Vec3& direction) :
        origin_(origin),
        direction_(direction),
        invDirection_(1.0 / direction),
        originMulInvDir_(origin / direction)
    {
    }

    const Vec3& origin() const { return origin_; }
    const Vec3& direction() const { return direction_; }
    const Vec3& invDirection() const { return invDirection_; }
    const Vec3& originMulInvDir() const { return originMulInvDir_; }

    Vec3 at(double t) const {
        return origin_ + t * direction_;
    } 

private:
    Vec3 origin_;
    Vec3 direction_;
    Vec3 invDirection_;
    Vec3 originMulInvDir_;

};

struct Interval {
    double min;
    double max;

    Interval() : min(std::numeric_limits<double>::min()), max(std::numeric_limits<double>::max()) {}

    Interval(double minVal, double maxVal) : min(minVal), max(maxVal) {}

    Interval(const Interval& a, const Interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    double length() const {
        return max - min;
    }

    bool contains(double value) const {
        return value >= min && value <= max;
    }
    
    bool surrounds(double value) const {
        return value > min && value < max;
    }

    double clamp(double x) const {
        return (x < min) ? min : (x > max) ? max : x;
    }
        
    Interval expand(double delta) const {
        auto padding = delta / 2;
        return Interval(min - padding, max + padding);
    }

    static const Interval empty;
};

const Interval Interval::empty = Interval(std::numeric_limits<double>::max(), std::numeric_limits<double>::min());

class AABB {
public:
    Vec3 min, max;

    AABB() = default;

    AABB(const Interval& x, const Interval& y, const Interval& z)
      : min(x.min, y.min, z.min), max(x.max, y.max, z.max) {}

    AABB(const Vec3& a, const Vec3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.
        min = glm::min(a, b);
        max = glm::max(a, b);
    }

    AABB(const AABB& box0, const AABB& box1) {
        min = glm::min(box0.min, box1.min);
        max = glm::max(box0.max, box1.max);
    }

    bool hit(const Ray& r, Interval rayT) const {
        const Vec3& invRayDir = r.invDirection();
        const Vec3& rayOrigMulInv = r.originMulInvDir();

        for (int axis = 0; axis < 3; axis++) {
            auto t0 = std::fma(min[axis], invRayDir[axis], -rayOrigMulInv[axis]);
            auto t1 = std::fma(max[axis], invRayDir[axis], -rayOrigMulInv[axis]);

            auto tentry = std::fmin(t0, t1);
            auto texit = std::fmax(t0, t1);

            rayT.min = std::fmax(tentry, rayT.min);
            rayT.max = std::fmin(texit, rayT.max);
            if (rayT.max <= rayT.min) {
                return false;
            }
        }
        return true;
    }

    // Returns the index of the longest axis of the bounding box.
    int longestAxis() const {
        float dx = max.x - min.x;
        float dy = max.y - min.y;
        float dz = max.z - min.z;

        if (dx > dy && dx > dz) {
            return 0;
        } else if (dy > dz) {
            return 1;
        } else {
            return 2;
        }
    }

    static const AABB empty;
};

const AABB AABB::empty{};

inline double randomDouble() {
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    thread_local std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    return min + (max - min) * randomDouble();
}

static Vec3 randomVec3() {
    return Vec3(randomDouble(), randomDouble(), randomDouble());
}

static Vec3 randomVec3(double min, double max) {
    return Vec3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
}

inline int randomInt(int min, int max) {
    return int(randomDouble(double(min), double(max)));
}

inline Vec3 randomUnitVector() {
    while (true) {
        auto p = Vec3{randomDouble(-1, 1), randomDouble(-1, 1), randomDouble(-1, 1)};
        auto lengthSq = glm::dot(p, p);
        if (1e-160 < lengthSq) {
            return p / sqrt(lengthSq);
        }
    }
}
