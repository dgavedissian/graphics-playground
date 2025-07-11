#pragma once

#include <glm.hpp>
#include <gtc/epsilon.hpp>

using Vec3 = glm::dvec3;

class Ray {
public:
    Ray(const Vec3& origin, const Vec3& direction) : origin_(origin), direction_(direction) {}

    const Vec3& origin() const { return origin_; }
    const Vec3& direction() const { return direction_; }

    Vec3 at(double t) const {
        return origin_ + t * direction_;
    } 

private:
    Vec3 origin_;
    Vec3 direction_;
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
    Interval x, y, z;

    AABB() {}

    AABB(const Interval& x, const Interval& y, const Interval& z)
      : x(x), y(y), z(z) {}

    AABB(const Vec3& a, const Vec3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
    }

    AABB(const AABB& box0, const AABB& box1) {
        x = Interval(box0.x, box1.x);
        y = Interval(box0.y, box1.y);
        z = Interval(box0.z, box1.z);
    }

    const Interval& axisInterval(int n) const {
        switch (n) {
        case 1:
            return y;
        case 2:
            return z;
        default:
            return x;
        }
    }

    bool hit(const Ray& r, Interval rayT) const {
        const Vec3& rayOrigin = r.origin();
        const Vec3& rayDir = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const Interval& ax = axisInterval(axis);
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

const AABB AABB::empty = AABB(Interval::empty, Interval::empty, Interval::empty);

inline double randomDouble() {
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    thread_local std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    return min + (max-min) * randomDouble();
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
