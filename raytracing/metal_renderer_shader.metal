#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    float2 pos[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    float2 uv = (pos[vertexID] + 1.0) * 0.5;
    
    VertexOut out;
    out.position = float4(pos[vertexID], 0.0, 1.0);
    out.uv = uv;
    return out;
}

enum MaterialType {
    MAT_LAMBERTIAN = 1,
    MAT_METAL,
    MAT_GLASS,
    MAT_LIGHT
};
    
int xorshift(int value) {
    // Xorshift*32
    // Based on George Marsaglia's work: http://www.jstatsoft.org/v08/i14/paper
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}
    
uint hashFloat2(float2 p) {
    uint2 i = uint2(p.x * 1e4, p.y * 1e4);
    i = (i ^ (i.yx >> 1u)) * 0x27d4eb2du;
    i ^= (i >> 15u);
    return i.x ^ i.y;
}
    
class RNG {
public:
    RNG(float2 uv) : state_(hashFloat2(uv)) {}
    
    float nextFloat() {
        state_ = xorshift(state_);
        return abs(fract(float(state_) / 3141.592653));
    }
        
    float nextFloat(float min, float max) {
        return min + (max - min) * nextFloat();
    }

    float3 nextUnitVector() {
        while (true) {
            float3 p(nextFloat(-1, 1), nextFloat(-1, 1), nextFloat(-1, 1));
            auto lengthSq = dot(p, p);
            if (1e-18 < lengthSq) {
                return p / sqrt(lengthSq);
            }
        }
    }
    
private:
    int state_;
};

struct Interval {
    float min;
    float max;

    Interval() : min(FLT_MIN), max(FLT_MAX) {}

    Interval(float minVal, float maxVal) : min(minVal), max(maxVal) {}

    Interval(thread const Interval& a, thread const Interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    float length() const {
        return max - min;
    }

    bool contains(float value) const {
        return value >= min && value <= max;
    }
    
    bool surrounds(float value) const {
        return value > min && value < max;
    }

    float clamp(float x) const {
        return (x < min) ? min : (x > max) ? max : x;
    }
        
    Interval expand(float delta) const {
        auto padding = delta / 2;
        return Interval(min - padding, max + padding);
    }
};

class Ray {
public:
    Ray() = default;
    Ray(float3 origin, float3 direction) :
        origin_(origin),
        direction_(direction),
        invDirection_(1.0 / direction),
        originMulInvDir_(origin / direction)
    {
    }

    float3 origin() const { return origin_; }
    float3 direction() const { return direction_; }
    float3 invDirection() const { return invDirection_; }
    float3 originMulInvDir() const { return originMulInvDir_; }

    float3 at(float t) const {
        return origin_ + t * direction_;
    }

private:
    float3 origin_;
    float3 direction_;
    float3 invDirection_;
    float3 originMulInvDir_;

};

struct HitResult {
    float3 point;
    float3 normal;
    float t;
    bool frontFace;
    uint materialIndex;
    
    void setNormal(thread const Ray& r, float3 outwardNormal) {
        frontFace = dot(r.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

struct Object {
    packed_float3 centre;
    float radius;
    uint materialIndex;
};

class ObjectList {
public:
    ObjectList(constant Object* objects, uint objectCount) : objects_(objects), objectCount_(objectCount) {}
    
    // r.direction() must be normalised.
    bool hit(thread const Ray& r, Interval t, thread HitResult& finalResult) const {
        float minT = FLT_MAX;
        bool hasHitObject = false;
        for (uint i = 0; i < objectCount_; ++i) {
            HitResult result;
            if (hitObject(i, r, t, result)) {
                if (result.t < minT) {
                    minT = result.t;
                    finalResult = result;
                    hasHitObject = true;
                }
            }
        }
        return hasHitObject;
    }
    
private:
    constant Object* objects_;
    uint objectCount_;
    
    bool hitObject(uint objectIdx, thread const Ray& r, Interval t, thread HitResult& result) const {
        constant auto& object = objects_[objectIdx];
        
        float3 oc = object.centre - r.origin();
        auto h = dot(r.direction(), oc);
        auto c = dot(oc, oc) - object.radius * object.radius;
        auto discriminant = h * h - c;
        
        if (discriminant < 0) {
            return false;
        }

        auto sqrtDiscriminant = sqrt(discriminant);
        auto root = h - sqrtDiscriminant;
        if (!t.surrounds(root)) {
            // Try the other root.
            root = h + sqrtDiscriminant;
            if (!t.surrounds(root)) {
                // Both roots are outside the range.
                return false;
            }
        }

        result.t = root;
        result.point = r.at(root);
        result.setNormal(r, (result.point - object.centre) / object.radius);
        result.materialIndex = object.materialIndex;
        return true;
    }
    
};

struct Material {
    uint type;
    packed_float3 albedo;
    packed_float3 emit;
    float fuzz;
    float refractionIndex;
};

float reflectance(float cosine, float refractionIndex) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
    r0 *= r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

bool scatter(thread RNG& rng, constant Material& mat, Ray ray, HitResult result, thread float3& attenuation, thread Ray& scattered) {
//    switch (mat.type) {
//    case MAT_LAMBERTIAN: {
        float3 direction = result.normal + rng.nextUnitVector();

        // Prevent degenerate rays.
        if (all(abs(direction) < float3(1e-8f))) {
            direction = result.normal;
        }

        scattered = Ray{result.point, normalize(direction)};
        attenuation = mat.albedo;
        return true;
//    }
//    case MAT_METAL: {
//        float3 reflected = reflect(ray.direction(), result.normal);
//        reflected = normalize(reflected) + mat.fuzz * rng.nextUnitVector();
//        scattered = Ray{result.point, normalize(reflected)};
//        attenuation = mat.albedo;
//        return dot(reflected, result.normal) > 0;
//    }
//    case MAT_GLASS: {
//        attenuation = float3(1.0);
//
//        float ri = result.frontFace ? (1.0 / mat.refractionIndex) : mat.refractionIndex;
//
//        float3 unitDirection = normalize(ray.direction());
//
//        float cosTheta = fmin(dot(-unitDirection, result.normal), 1.0);
//        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
//
//        bool cannotRefract = ri * sinTheta > 1.0;
//
//        float3 direction;
//        if (cannotRefract || reflectance(cosTheta, ri) > rng.nextFloat()) {
//            direction = reflect(unitDirection, result.normal);
//        } else {
//            direction = refract(unitDirection, result.normal, ri);
//        }
//
//        scattered = Ray{result.point, normalize(direction)};
//        return true;
//    }
//    default:
//        return false;
//    }
}

float3 rayColour(thread RNG& rng, thread const ObjectList& objects, constant Material* materials, Ray r, int depth) {
    float3 colour = float3(0.0);
    float3 totalAttenuation = float3(1.0);
    for (int i = 0; i < depth; ++i) {
        // Perform ray test.
        HitResult result;
        bool hit = objects.hit(r, Interval(0.001, 10000000000.0), result);
        if (!hit) {
            colour += totalAttenuation * float3(0.4, 0.6, 0.9);
            break;
        }

        Ray scattered;
        float3 attenuation;
        float3 emission = materials[result.materialIndex].emit;
        bool didScatter = scatter(rng, materials[result.materialIndex], r, result, attenuation, scattered);
        
        colour += totalAttenuation * emission;
        if (!didScatter) {
            break;
        }
        
        // If we did scatter a ray, then go to the next iteration.
        totalAttenuation *= attenuation;
        r = scattered;
    }
    return colour;
}
    
struct Camera {
    packed_float3 origin;
    packed_float3 lowerLeftCorner;
    packed_float3 horizontal;
    packed_float3 vertical;
};

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant Object* objects [[buffer(0)]],
                               constant uint& objectCount [[buffer(1)]],
                               constant Material* materials [[buffer(2)]],
                               constant Camera& camera [[buffer(3)]])
{
    RNG rng(in.uv);
    
    float2 uv = in.uv + float2(rng.nextFloat(), rng.nextFloat()) / float2(1000);
    float3 dir = normalize(camera.lowerLeftCorner + uv.x * camera.horizontal + uv.y * camera.vertical - camera.origin);

    float3 samples(0.0);
    const int numSamples = 10;
    
    ObjectList objectList{objects, objectCount};
    Ray ray{float3(camera.origin), dir};
    for (int i = 0; i < numSamples; ++i) {
        samples += rayColour(rng, objectList, materials, ray, 10) / numSamples;
    }
    return float4(samples, 1.0);
}
