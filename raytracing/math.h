#pragma once

#include <glm.hpp>
#include <gtc/epsilon.hpp>

using color = glm::dvec3;
using vec3 = glm::dvec3;

class Ray {
public:
    Ray(const vec3& origin, const vec3& direction) : origin_(origin), direction_(direction) {}

    const vec3& origin() const { return origin_; }
    const vec3& direction() const { return direction_; }

    vec3 at(double t) const {
        return origin_ + t * direction_;
    } 

private:
    vec3 origin_;
    vec3 direction_;
};

struct interval {
    double min;
    double max;

    interval() : min(std::numeric_limits<double>::min()), max(std::numeric_limits<double>::max()) {}

    interval(double minVal, double maxVal) : min(minVal), max(maxVal) {}

    interval(const interval& a, const interval& b) {
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
        
    interval expand(double delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    static const interval empty;
};

const interval interval::empty = interval(std::numeric_limits<double>::max(), std::numeric_limits<double>::min());

class AABB {
  public:
    interval x, y, z;

    AABB() {}

    AABB(const interval& x, const interval& y, const interval& z)
      : x(x), y(y), z(z) {}

    AABB(const vec3& a, const vec3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
    }

    AABB(const AABB& box0, const AABB& box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }

    const interval& axisInterval(int n) const {
        switch (n) {
        case 1:
            return y;
        case 2:
            return z;
        default:
            return x;
        }
    }

    bool hit(const Ray& r, interval rayT) const {
        const vec3& rayOrigin = r.origin();
        const vec3& rayDir = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const interval& ax = axisInterval(axis);
            const double adinv = 1.0 / rayDir[axis];

            auto t0 = (ax.min - rayOrigin[axis]) * adinv;
            auto t1 = (ax.max - rayOrigin[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > rayT.min) { 
                    rayT.min = t0;
                }
                if (t1 < rayT.max) {
                    rayT.max = t1;
                }
            } else {
                if (t1 > rayT.min) {
                    rayT.min = t1;
                }
                if (t0 < rayT.max) {
                    rayT.max = t0;
                }
            }

            if (rayT.max <= rayT.min) {
                return false;
            }
        }
        return true;
    }

    // Returns the index of the longest axis of the bounding box.
    int longestAxis() const {
        if (x.length() > y.length()) {
            return x.length() > z.length() ? 0 : 2;
        } else {
            return y.length() > z.length() ? 1 : 2;
        }
    }

    static const AABB empty;
};

const AABB AABB::empty = AABB(interval::empty, interval::empty, interval::empty);

inline double randomDouble() {
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    thread_local std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    return min + (max-min) * randomDouble();
}

static vec3 randomVec3() {
    return vec3(randomDouble(), randomDouble(), randomDouble());
}

static vec3 randomVec3(double min, double max) {
    return vec3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
}

inline int randomInt(int min, int max) {
    return int(randomDouble(double(min), double(max)));
}

inline vec3 randomUnitVector() {
    while (true) {
        auto p = vec3{randomDouble(-1, 1), randomDouble(-1, 1), randomDouble(-1, 1)};
        auto lengthSq = glm::dot(p, p);
        if (1e-160 < lengthSq) {
            return p / sqrt(lengthSq);
        }
    }
}
