#pragma once

#include "hittable.h"

using HittableList = std::vector<Hittable*>;

struct BVHNode {
    // hittables is non-empty if it's a leaf node.
    HittableList hittables;

    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    AABB bbox;
};

std::unique_ptr<BVHNode> createNode(HittableList& objects, size_t start, size_t end) {
    // Build the bounding box of the span of source objects.
    auto bbox = objects[start]->boundingBox();
    for (size_t i = start + 1; i < end; i++) {
        bbox = AABB(bbox, objects[i]->boundingBox());
    }

    int axis = bbox.longestAxis();
    auto comparator = [axis](const Hittable* a, const Hittable* b) {
        return a->boundingBox().min[axis] < b->boundingBox().min[axis];
    };
    
    size_t objectSpan = end - start;

    if (objectSpan == 1) {
        return std::make_unique<BVHNode>(HittableList{objects[start]}, nullptr, nullptr, bbox);
    } else if (objectSpan == 2) {
        return std::make_unique<BVHNode>(HittableList{objects[start], objects[start + 1]}, nullptr, nullptr, bbox);
    } else {
        std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

        auto mid = start + objectSpan / 2;
        return std::make_unique<BVHNode>(
            HittableList{},
            createNode(objects, start, mid), 
            createNode(objects, mid, end),
            bbox
        );
    }
}

class BVHTree : public Hittable {
public:
    BVHTree(std::unique_ptr<BVHNode> root) :
        root_(std::move(root))
    {
    }

    bool hit(const Ray& r, Interval t, HitResult& result) const override {
        BVHNode* stack[32];
        int stackLen = 0;
        double closestSoFar = t.max;
        bool hitAnything = false;

        stack[stackLen++] = root_->right.get();
        stack[stackLen++] = root_->left.get();

        while (stackLen > 0) {
            BVHNode* n = stack[stackLen - 1];
            stackLen--;

            // We should only traverse this node if the ray intersects with the AABB.
            if (!n->bbox.hit(r, t)) {
                continue;
            }

            // If this is a leaf node.
            if (n->hittables.size() > 0) {
                for (Hittable* h : n->hittables) {
                    HitResult tempResult;
                    if (h->hit(r, Interval{t.min, closestSoFar}, tempResult)) {
                        closestSoFar = tempResult.t;
                        result = tempResult;
                        hitAnything = true;
                    }
                }
            } else {
                if (stackLen >= 31) {
                    // stack overflow.
                    continue;
                }
                stack[stackLen++] = n->left.get();
                if (n->right) {
                    stack[stackLen++] = n->right.get();
                }
            }
        }

        return hitAnything;
    }

    AABB boundingBox() const override {
        return root_->bbox;
    }

private:
    std::unique_ptr<BVHNode> root_;

};

std::unique_ptr<BVHTree> generateBVHTree(const Scene& scene) {
    HittableList hittables;
    std::transform(
        scene.hittables().begin(), scene.hittables().end(),
        std::back_inserter(hittables),
        [](const auto& p) { return p.get(); }
    );

    return std::make_unique<BVHTree>(createNode(hittables, 0, hittables.size()));
}
