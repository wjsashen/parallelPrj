// raycast_ssr_cuda_bvh.cu
// CUDA ray tracer with Screen-Space Reflections (SSR) and BVH acceleration for triangles.
// - Spheres are still intersected with a linear loop.
// - Triangles are intersected using a BVH both for primary rays and shadow rays.

#include "raycast_cuda.h"
#include "parse.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

std::vector<Vec3d> Objects::vertices;

// ---------------------------
// CUDA error checking macro
// ---------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------------------------
// Device clamp helpers
// ---------------------------
__device__ __host__ inline float clamp_cuda(float value, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(value, maxVal));
}

__device__ __host__ inline int clamp_cuda(int value, int minVal, int maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

// ---------------------------
// Device-side scene structs
// ---------------------------
struct DeviceSphere {
    Vec3d center;
    float radius;
    MaterialColor material;
    bool hasTexture;
};

struct DeviceTriangle {
    Vec3d v0, v1, v2;
    Vec3d n0, n1, n2;
    Vec2d vt0, vt1, vt2;
    MaterialColor material;
    bool isSmooth;
    bool hasTexture;
};

struct DeviceLight {
    Vec3d positionOrdir;
    float intensity;
    bool isPoint;
};

struct DeviceTexture {
    Color* pixels;
    int width;
    int height;
};

// ---------------------------
// BVH data structures
// ---------------------------

// Axis-aligned bounding box for triangles
struct DeviceAABB {
    Vec3d min_pt;
    Vec3d max_pt;
};

// BVH node (shared layout on host and device)
struct DeviceBVHNode {
    DeviceAABB bounds;  // bounding box for this node
    int left;           // index of left child, -1 if none
    int right;          // index of right child, -1 if none
    int start;          // start index in the triangle index array (for leaves)
    int count;          // number of triangles in this leaf
    int isLeaf;         // 1 if leaf, 0 if internal node
};

// Helper struct used only on the host for BVH building
struct BVHBuildTriangle {
    int triIndex;       // index into the DeviceTriangle array
    Vec3d min;
    Vec3d max;
    Vec3d centroid;
};

// ---------------------------
// Device helper functions
// ---------------------------

__device__ Vec2d calculateSphereUV_device(const Vec3d &normal) {
    float f = acosf(normal.z);
    float q = atan2f(normal.y, normal.x);
    float u = (q >= 0) ? (q / (2 * M_PI)) : ((q + 2 * M_PI) / (2 * M_PI));
    float v = f / M_PI;
    return Vec2d(u, v);
}

__device__ bool intersectRaySphere_device(const Ray &ray, const DeviceSphere &sphere,
                                          float &t, Vec2d &texCoordOut) {
    float cx = sphere.center.x, cy = sphere.center.y, cz = sphere.center.z;
    float r = sphere.radius;

    float a = ray.dx * ray.dx + ray.dy * ray.dy + ray.dz * ray.dz;
    float b = 2.0f * (ray.dx * (ray.x - cx) +
                      ray.dy * (ray.y - cy) +
                      ray.dz * (ray.z - cz));
    float c = (ray.x - cx) * (ray.x - cx) +
              (ray.y - cy) * (ray.y - cy) +
              (ray.z - cz) * (ray.z - cz) - r * r;

    float d = b * b - 4.0f * a * c;
    if (d < 0.0f) return false;

    float t1 = (-b - sqrtf(d)) / (2.0f * a);
    float t2 = (-b + sqrtf(d)) / (2.0f * a);

    if (t1 > 0.0f) {
        t = t1;
    } else if (t2 > 0.0f) {
        t = t2;
    } else {
        return false;
    }

    Vec3d intersectionPoint = ray.getOrigin() + ray.getDirection() * t;
    Vec3d normal = (intersectionPoint - sphere.center).norm();
    texCoordOut = calculateSphereUV_device(normal);
    return true;
}

__device__ bool intersectRayTriangle_device(const Ray &ray,
                                            float &t,
                                            Vec3d &normalOut,
                                            Vec2d &texCoordOut,
                                            const DeviceTriangle &triangle) {
    Vec3d edge1 = triangle.v1 - triangle.v0;
    Vec3d edge2 = triangle.v2 - triangle.v0;
    Vec3d cr = edge1.cross(edge2);
    Vec3d normal = cr.norm();

    float denominator = cr.dot(ray.getDirection());
    if (fabsf(denominator) < 1e-6f) return false;

    float d = -cr.dot(triangle.v0);
    t = -(cr.dot(ray.getOrigin()) + d) / denominator;
    if (t < 0.0f) return false;

    Vec3d P = ray.getOrigin() + ray.getDirection() * t;
    Vec3d vP = P - triangle.v0;

    float e1e1 = edge1.dot(edge1);
    float e2e2 = edge2.dot(edge2);
    float e1e2 = edge1.dot(edge2);
    float det = e1e1 * e2e2 - e1e2 * e1e2;

    float vPe1 = vP.dot(edge1);
    float vPe2 = vP.dot(edge2);

    float beta  = (vPe1 * e2e2 - vPe2 * e1e2) / det;
    float gamma = (e1e1 * vPe2 - e1e2 * vPe1) / det;
    float alpha = 1.0f - beta - gamma;

    if (alpha >= 0.0f && beta >= 0.0f && gamma >= 0.0f) {
        if (triangle.isSmooth) {
            normalOut = (triangle.n0 * alpha) +
                        (triangle.n1 * beta) +
                        (triangle.n2 * gamma);
            normalOut = normalOut.norm();
        } else {
            normalOut = normal;
        }

        if (triangle.hasTexture) {
            texCoordOut = (triangle.vt0 * alpha) +
                          (triangle.vt1 * beta) +
                          (triangle.vt2 * gamma);
        }
        return true;
    }
    return false;
}

__device__ Color getTextureColor_device(const DeviceTexture &texture, float u, float v) {
    if (!texture.pixels || texture.width == 0 || texture.height == 0) {
        // Fallback to white when texture is not available
        return Color(1.0f, 1.0f, 1.0f);
    }

    u = fmodf(fmodf(u, 1.0f) + 1.0f, 1.0f);
    v = fmodf(fmodf(v, 1.0f) + 1.0f, 1.0f);

    float x = u * texture.width - 0.5f;
    float y = v * texture.height - 0.5f;

    int x0 = clamp_cuda((int)floorf(x), 0, texture.width - 1);
    int x1 = clamp_cuda(x0 + 1, 0, texture.width - 1);
    int y0 = clamp_cuda((int)floorf(y), 0, texture.height - 1);
    int y1 = clamp_cuda(y0 + 1, 0, texture.height - 1);

    float tx = x - (float)x0;
    float ty = y - (float)y0;

    Color c00 = texture.pixels[y0 * texture.width + x0];
    Color c10 = texture.pixels[y0 * texture.width + x1];
    Color c01 = texture.pixels[y1 * texture.width + x0];
    Color c11 = texture.pixels[y1 * texture.width + x1];

    Color cx0 = c00 * (1.0f - tx) + c10 * tx;
    Color cx1 = c01 * (1.0f - tx) + c11 * tx;
    return cx0 * (1.0f - ty) + cx1 * ty;
}

__device__ bool refract_device(const Vec3d &I, const Vec3d &N, float ior, Vec3d &refractedDir) {
    float cosi = clamp_cuda(I.dot(N), -1.0f, 1.0f);
    float etai = 1.0f;
    float etat = ior;
    Vec3d n = N;

    if (cosi < 0.0f) {
        cosi = -cosi;
    } else {
        float tmp = etai;
        etai = etat;
        etat = tmp;
        n = N * -1.0;
    }

    float eta = etai / etat;
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f) return false;

    refractedDir = (I * eta + n * (eta * cosi - sqrtf(k))).norm();
    return true;
}

__device__ float fresnelSchlick_device(const Vec3d &I, const Vec3d &N, float ior) {
    float cosi = clamp_cuda(I.dot(N), -1.0f, 1.0f);
    float etai = 1.0f;
    float etat = ior;

    if (cosi > 0.0f) {
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }

    float r0 = (etai - etat) / (etai + etat);
    float f0 = r0 * r0;
    return f0 + (1.0f - f0) * powf(1.0f - fabsf(cosi), 5.0f);
}

// ---------------------------
// AABB intersection
// ---------------------------
__device__ bool intersectAABB_device(const Ray &ray,
                                     const DeviceAABB &box,
                                     float tMin,
                                     float tMax) {
    Vec3d orig = ray.getOrigin();
    Vec3d dir  = ray.getDirection();

    // X axis
    if (fabsf((float)dir.x) < 1e-8f) {
        if (orig.x < box.min_pt.x || orig.x > box.max_pt.x) return false;
    } else {
        float invD = 1.0f / (float)dir.x;
        float t0 = (box.min_pt.x - orig.x) * invD;
        float t1 = (box.max_pt.x - orig.x) * invD;
        if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        tMin = t0 > tMin ? t0 : tMin;
        tMax = t1 < tMax ? t1 : tMax;
        if (tMax <= tMin) return false;
    }

    // Y axis
    if (fabsf((float)dir.y) < 1e-8f) {
        if (orig.y < box.min_pt.y || orig.y > box.max_pt.y) return false;
    } else {
        float invD = 1.0f / (float)dir.y;
        float t0 = (box.min_pt.y - orig.y) * invD;
        float t1 = (box.max_pt.y - orig.y) * invD;
        if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        tMin = t0 > tMin ? t0 : tMin;
        tMax = t1 < tMax ? t1 : tMax;
        if (tMax <= tMin) return false;
    }

    // Z axis
    if (fabsf((float)dir.z) < 1e-8f) {
        if (orig.z < box.min_pt.z || orig.z > box.max_pt.z) return false;
    } else {
        float invD = 1.0f / (float)dir.z;
        float t0 = (box.min_pt.z - orig.z) * invD;
        float t1 = (box.max_pt.z - orig.z) * invD;
        if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        tMin = t0 > tMin ? t0 : tMin;
        tMax = t1 < tMax ? t1 : tMax;
        if (tMax <= tMin) return false;
    }

    return true;
}

// ---------------------------
// BVH traversal (closest hit)
// ---------------------------
__device__ bool intersectBVHClosest_device(const Ray &ray,
                                           const DeviceBVHNode* nodes,
                                           int numNodes,
                                           const int* triIndices,
                                           const DeviceTriangle* triangles,
                                           float &closestT,
                                           int &hitTriIndex,
                                           Vec3d &hitNormal,
                                           Vec2d &hitTexCoord) {
    if (!nodes || numNodes <= 0) return false;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // root index = 0

    bool hit = false;
    closestT = 1e30f;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const DeviceBVHNode &node = nodes[nodeIdx];

        if (!intersectAABB_device(ray, node.bounds, 0.001f, closestT))
            continue;

        if (node.isLeaf) {
            for (int i = 0; i < node.count; ++i) {
                int triIdx = triIndices[node.start + i];
                float t;
                Vec3d n;
                Vec2d tc;
                if (intersectRayTriangle_device(ray, t, n, tc, triangles[triIdx]) &&
                    t > 0.001f && t < closestT) {
                    closestT = t;
                    hitTriIndex = triIdx;
                    hitNormal = n;
                    hitTexCoord = tc;
                    hit = true;
                }
            }
        } else {
            if (node.left >= 0)  stack[stackPtr++] = node.left;
            if (node.right >= 0) stack[stackPtr++] = node.right;
        }
    }

    return hit;
}

// ---------------------------
// BVH traversal (shadow test)
// ---------------------------
__device__ bool intersectBVHShadow_device(const Ray &ray,
                                          const DeviceBVHNode* nodes,
                                          int numNodes,
                                          const int* triIndices,
                                          const DeviceTriangle* triangles,
                                          float maxDist) {
    if (!nodes || numNodes <= 0) return false;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // root

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const DeviceBVHNode &node = nodes[nodeIdx];

        if (!intersectAABB_device(ray, node.bounds, 0.001f, maxDist))
            continue;

        if (node.isLeaf) {
            for (int i = 0; i < node.count; ++i) {
                int triIdx = triIndices[node.start + i];
                float t;
                Vec3d n;
                Vec2d tc;
                if (intersectRayTriangle_device(ray, t, n, tc, triangles[triIdx]) &&
                    t > 0.001f && t < maxDist) {
                    // Any triangle hit before the light blocks it
                    return true;
                }
            }
        } else {
            if (node.left >= 0)  stack[stackPtr++] = node.left;
            if (node.right >= 0) stack[stackPtr++] = node.right;
        }
    }

    return false;
}

// ---------------------------
// Shadow factor with spheres + BVH triangles
// ---------------------------
__device__ float computeShadowFactor_device(const Vec3d &point,
                                            const Vec3d &lightDir,
                                            float maxDist,
                                            const DeviceSphere* spheres,
                                            int numSpheres,
                                            const DeviceBVHNode* bvhNodes,
                                            int numBVHNodes,
                                            const int* bvhTriIndices,
                                            const DeviceTriangle* triangles) {
    Ray shadowRay(point + lightDir * 1e-4f, lightDir);
    float transparency = 1.0f;

    // First, check spheres (supporting semi-transparent materials)
    for (int i = 0; i < numSpheres; ++i) {
        float t;
        Vec2d texCoord;
        if (intersectRaySphere_device(shadowRay, spheres[i], t, texCoord) &&
            t > 0.001f && t < maxDist) {
            transparency *= (1.0f - spheres[i].material.alpha);
            if (transparency <= 0.01f)
                return 0.0f; // Almost fully shadowed
        }
    }

    // Then, check triangles using BVH (opaque)
    if (bvhNodes && numBVHNodes > 0) {
        if (intersectBVHShadow_device(shadowRay, bvhNodes, numBVHNodes,
                                      bvhTriIndices, triangles, maxDist)) {
            // Any triangle hit blocks the light completely
            return 0.0f;
        }
    }

    return transparency;
}

// ---------------------------
// Projection + SSR helpers
// ---------------------------

// Project a 3D point to screen space pixel coordinates
__device__ bool projectToScreen_device(const Vec3d &point, const Vec3d &eyePos,
                                       const Vec3d &viewDir, const Vec3d &upDir,
                                       float vfov, int width, int height,
                                       int &screenX, int &screenY) {
    Vec3d right = viewDir.cross(upDir).norm();
    Vec3d up    = right.cross(viewDir).norm();

    Vec3d toPoint = point - eyePos;
    float distance = toPoint.dot(viewDir);

    if (distance <= 0.0f) return false;

    double ar = (double)width / (double)height;
    double vh = 2.0 * tanf(vfov / 2.0f);
    double vw = vh * ar;

    float u = toPoint.dot(right) / distance;
    float v = -toPoint.dot(up    ) / distance;

    screenX = (int)((u / vw + 0.5) * width);
    screenY = (int)((v / vh + 0.5) * height);

    return (screenX >= 0 && screenX < width &&
            screenY >= 0 && screenY < height);
}

// Screen-space reflection lookup
__device__ Color getSSRColor_device(const Vec3d &reflectDir, const Vec3d &hitPoint,
                                    const Color* imageBuffer, int width, int height,
                                    const Vec3d &eyePos, const Vec3d &viewDir,
                                    const Vec3d &upDir, float vfov,
                                    const Color &bkgcolor) {
    const int   MAX_STEPS = 50;
    const float STEP_SIZE = 0.1f;

    Vec3d currentPos = hitPoint;

    for (int i = 0; i < MAX_STEPS; ++i) {
        currentPos = currentPos + reflectDir * STEP_SIZE;

        int screenX, screenY;
        if (projectToScreen_device(currentPos, eyePos, viewDir, upDir, vfov,
                                   width, height, screenX, screenY)) {
            if (screenX >= 0 && screenX < width &&
                screenY >= 0 && screenY < height) {
                return imageBuffer[screenY * width + screenX];
            }
        } else {
            // Ray went off screen or behind camera
            break;
        }
    }

    // Fallback to background if no valid reflection sample
    return bkgcolor;
}

// ---------------------------
// Shading
// ---------------------------
__device__ Color shade_device(const Vec3d &point,
                              const Vec3d &rayDir,
                              const Vec3d &normal,
                              const Vec2d &texCoord,
                              const MaterialColor &material,
                              const DeviceLight* lights,
                              int numLights,
                              const DeviceSphere* spheres,
                              int numSpheres,
                              const DeviceBVHNode* bvhNodes,
                              int numBVHNodes,
                              const int* bvhTriIndices,
                              const DeviceTriangle* triangles,
                              const DeviceTexture &texture,
                              bool hasTexture,
                              const Color &bkgcolor) {
    Color texturec = material.color;

    if (hasTexture && texture.pixels) {
        texturec = getTextureColor_device(texture, texCoord.x, texCoord.y);
    }

    Vec3d viewDir = (rayDir * -1.0).norm();

    Color ambient(0, 0, 0);
    Color totalDiffuse(0, 0, 0);
    Color totalSpecular(0, 0, 0);

    ambient = texturec * material.ka;

    for (int i = 0; i < numLights; ++i) {
        // Direction to light
        Vec3d L = lights[i].isPoint
                    ? (lights[i].positionOrdir - point).norm()
                    : lights[i].positionOrdir.norm() * -1.0;
        Vec3d H = (L + viewDir).norm();

        float df = fmaxf(normal.dot(L), 0.0f);
        Color diffuse = texturec * material.kd * df;

        float sf = powf(fmaxf(normal.dot(H), 0.0f), material.shininess);
        Color specular = material.specular * material.ks * sf;

        // Distance to point light (for shadow ray maximum distance)
        float maxDist = 1e30f;
        if (lights[i].isPoint) {
            Vec3d diff = lights[i].positionOrdir - point;
            maxDist = (float)diff.length();
        }

        // Shadow factor: includes spheres and triangles via BVH
        float shadowFactor = computeShadowFactor_device(
                                 point, L, maxDist,
                                 spheres, numSpheres,
                                 bvhNodes, numBVHNodes,
                                 bvhTriIndices, triangles);

        totalDiffuse  = totalDiffuse  + diffuse  * lights[i].intensity * shadowFactor;
        totalSpecular = totalSpecular + specular * lights[i].intensity * shadowFactor;
    }

    return ambient + totalDiffuse + totalSpecular;
}

// ---------------------------
// Ray tracing kernel
// ---------------------------
__global__ void raytrace_kernel(Color* output,
                                const Color* imageBuffer,
                                const DeviceSphere* spheres, int numSpheres,
                                const DeviceTriangle* triangles, int numTriangles,
                                const DeviceLight* lights, int numLights,
                                const DeviceTexture* textures,
                                const DeviceBVHNode* bvhNodes, int numBVHNodes,
                                const int* bvhTriIndices,
                                Vec3d eyePos, Vec3d viewDir, Vec3d upDir, float vfov,
                                Vec3d ul, Vec3d delta_h, Vec3d delta_v,
                                int width, int height,
                                Color bkgcolor,
                                bool useSSR,
                                int maxDepth /* currently unused, kept for API symmetry */) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Build primary ray
    Vec3d vwPosition = ul + delta_h * (float)x + delta_v * (float)y;
    Vec3d rayDir = (vwPosition - eyePos).norm();
    Ray ray(eyePos, rayDir);

    float tClosestSphere = 1e30f;
    int   sphereHitIndex = -1;
    Vec2d sphereTexCoord;
    Vec3d sphereNormal;

    // --------- spheres: still linear traversal ---------
    for (int i = 0; i < numSpheres; ++i) {
        float t;
        Vec2d tc;
        if (intersectRaySphere_device(ray, spheres[i], t, tc) &&
            t > 0.001f && t < tClosestSphere) {
            tClosestSphere = t;
            sphereHitIndex = i;
            sphereTexCoord = tc;
            Vec3d hitPoint = ray.getOrigin() + ray.getDirection() * t;
            sphereNormal = (hitPoint - spheres[i].center).norm();
        }
    }

    // --------- triangles: accelerated via BVH ---------
    float tClosestTri = 1e30f;
    int   triHitIndex = -1;
    Vec3d triNormal;
    Vec2d triTexCoord;

    bool triHit = intersectBVHClosest_device(
                      ray,
                      bvhNodes, numBVHNodes,
                      bvhTriIndices,
                      triangles,
                      tClosestTri,
                      triHitIndex,
                      triNormal,
                      triTexCoord);

    // Choose the closest among sphere and triangle hits
    float t_min = 1e30f;
    int   hitType   = -1; // -1: none, 0: sphere, 1: triangle
    int   hitIndex  = -1;
    Vec3d hitNormal;
    Vec2d hitTexCoord;

    if (sphereHitIndex >= 0 && tClosestSphere < t_min) {
        t_min      = tClosestSphere;
        hitType    = 0;
        hitIndex   = sphereHitIndex;
        hitNormal  = sphereNormal;
        hitTexCoord = sphereTexCoord;
    }

    if (triHit && tClosestTri < t_min) {
        t_min      = tClosestTri;
        hitType    = 1;
        hitIndex   = triHitIndex;
        hitNormal  = triNormal;
        hitTexCoord = triTexCoord;
    }

    Color finalColor = bkgcolor;

    if (hitType >= 0) {
        Vec3d hitPoint = eyePos + rayDir * t_min;

        MaterialColor material;
        DeviceTexture tex;
        bool hasTexture = false;

        if (hitType == 0) {
            // Sphere
            material   = spheres[hitIndex].material;
            hasTexture = spheres[hitIndex].hasTexture;
            tex        = textures[hitIndex]; // first numSpheres entries
        } else {
            // Triangle
            material   = triangles[hitIndex].material;
            hasTexture = triangles[hitIndex].hasTexture;
            tex        = textures[numSpheres + hitIndex]; // triangles follow spheres
        }

        // Local shading with spheres + BVH triangles in shadows
        Color localColor = shade_device(
                               hitPoint, rayDir, hitNormal, hitTexCoord, material,
                               lights, numLights,
                               spheres, numSpheres,
                               bvhNodes, numBVHNodes,
                               bvhTriIndices, triangles,
                               tex, hasTexture,
                               bkgcolor);

        // Reflection (SSR)
        Color reflectedColor(0, 0, 0);
        if (material.ks > 0.0f && useSSR && imageBuffer) {
            Vec3d reflectDir = (rayDir - hitNormal * 2.0 * (rayDir.dot(hitNormal))).norm();
            reflectedColor = getSSRColor_device(reflectDir, hitPoint,
                                                imageBuffer,
                                                width, height,
                                                eyePos, viewDir, upDir,
                                                vfov,
                                                bkgcolor);
        }

        // Simple refraction placeholder (no recursive refraction)
        Color refractedColor(0, 0, 0);
        if (material.alpha < 1.0f) {
            Vec3d refractDir;
            if (refract_device(rayDir, hitNormal, material.ior, refractDir)) {
                // For now, we do not trace a separate refraction ray on GPU.
                refractedColor = bkgcolor;
            }
        }

        float Fr = fresnelSchlick_device(rayDir, hitNormal, material.ior);
        float Ft = 1.0f - Fr;

        finalColor = localColor + reflectedColor * Fr
                               + refractedColor * (1.0f - material.alpha) * Ft;
    }

    output[idx] = finalColor;
}

// ---------------------------
// BVH building on host
// ---------------------------

static int buildBVHRecursive(std::vector<BVHBuildTriangle> &buildTris,
                             int start, int end,
                             std::vector<DeviceBVHNode> &nodes,
                             std::vector<int> &triIndices) {
    int nodeIndex = (int)nodes.size();
    nodes.push_back(DeviceBVHNode());
    DeviceBVHNode &node = nodes.back();

    node.left   = -1;
    node.right  = -1;
    node.start  = -1;
    node.count  = 0;
    node.isLeaf = 0;

    // Compute bounding box and centroid bounds for this node
    DeviceAABB bounds;
    bounds.min_pt = Vec3d( FLT_MAX,  FLT_MAX,  FLT_MAX);
    bounds.max_pt = Vec3d(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    Vec3d cmin( FLT_MAX,  FLT_MAX,  FLT_MAX);
    Vec3d cmax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = start; i < end; ++i) {
        const BVHBuildTriangle &bt = buildTris[i];

        // Triangle bounds
        bounds.min_pt.x = std::min(bounds.min_pt.x, bt.min.x);
        bounds.min_pt.y = std::min(bounds.min_pt.y, bt.min.y);
        bounds.min_pt.z = std::min(bounds.min_pt.z, bt.min.z);

        bounds.max_pt.x = std::max(bounds.max_pt.x, bt.max.x);
        bounds.max_pt.y = std::max(bounds.max_pt.y, bt.max.y);
        bounds.max_pt.z = std::max(bounds.max_pt.z, bt.max.z);

        // Centroid bounds
        cmin.x = std::min(cmin.x, bt.centroid.x);
        cmin.y = std::min(cmin.y, bt.centroid.y);
        cmin.z = std::min(cmin.z, bt.centroid.z);

        cmax.x = std::max(cmax.x, bt.centroid.x);
        cmax.y = std::max(cmax.y, bt.centroid.y);
        cmax.z = std::max(cmax.z, bt.centroid.z);
    }

    node.bounds = bounds;

    int n = end - start;
    const int LEAF_SIZE = 4;

    // Leaf node condition
    if (n <= LEAF_SIZE) {
        node.start = (int)triIndices.size();
        node.count = n;
        node.isLeaf = 1;

        for (int i = start; i < end; ++i) {
            triIndices.push_back(buildTris[i].triIndex);
        }
        return nodeIndex;
    }

    // Choose split axis by largest centroid extent
    Vec3d extent = cmax - cmin;
    int axis = 0;
    if (extent.y > extent.x && extent.y >= extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z >= extent.y) axis = 2;

    int mid = (start + end) / 2;

    // If the extent is nearly zero, just split in the middle without sorting
    if (extent.x < 1e-5 && extent.y < 1e-5 && extent.z < 1e-5) {
        // Do nothing, mid already set
    } else {
        if (axis == 0) {
            std::sort(buildTris.begin() + start, buildTris.begin() + end,
                      [](const BVHBuildTriangle &a, const BVHBuildTriangle &b) {
                          return a.centroid.x < b.centroid.x;
                      });
        } else if (axis == 1) {
            std::sort(buildTris.begin() + start, buildTris.begin() + end,
                      [](const BVHBuildTriangle &a, const BVHBuildTriangle &b) {
                          return a.centroid.y < b.centroid.y;
                      });
        } else {
            std::sort(buildTris.begin() + start, buildTris.begin() + end,
                      [](const BVHBuildTriangle &a, const BVHBuildTriangle &b) {
                          return a.centroid.z < b.centroid.z;
                      });
        }
    }

    // Recursively build children
    int leftIndex  = buildBVHRecursive(buildTris, start, mid, nodes, triIndices);
    int rightIndex = buildBVHRecursive(buildTris, mid,   end, nodes, triIndices);

    node.left  = leftIndex;
    node.right = rightIndex;

    return nodeIndex;
}

// ---------------------------
// Host main()
// ---------------------------
int main(int argc, char* argv[]) {
    Scene scene;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename.txt>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Error: Could not open file '" << filename << "'." << std::endl;
        return 1;
    }

    std::string basename = filename;
    size_t lastdot = basename.find_last_of(".");
    if (lastdot != std::string::npos) {
        basename = basename.substr(0, lastdot);
    }

    // Parse scene from text file
    parse(filename, scene);

    int width  = (int)scene.camera.w;
    int height = (int)scene.camera.h;

    std::cout << "Rendering " << width << "x" << height << " image with BVH..." << std::endl;

    // Host output buffers
    Color* h_output      = new Color[width * height];
    Color* h_imageBuffer = new Color[width * height]; // not used directly, but kept for clarity

    // Host-side device data
    std::vector<DeviceSphere>   h_spheres;
    std::vector<DeviceTriangle> h_triangles;
    std::vector<DeviceLight>    h_lights;
    std::vector<DeviceTexture>  h_textures; // textures for spheres + triangles

    // Convert scene objects to device-friendly representation
    for (const auto &obj : scene.objects) {
        if (auto* sphere = dynamic_cast<Sphere*>(obj.get())) {
            DeviceSphere ds;
            ds.center    = sphere->center;
            ds.radius    = sphere->radius;
            ds.material  = sphere->getColor();
            ds.hasTexture = sphere->hasTexture;
            h_spheres.push_back(ds);

            // Placeholder texture record for sphere (no GPU upload yet)
            DeviceTexture dt;
            dt.pixels = nullptr;
            dt.width  = 0;
            dt.height = 0;
            h_textures.push_back(dt);
        } else if (auto* triangle = dynamic_cast<Triangle*>(obj.get())) {
            DeviceTriangle dt;
            dt.v0 = triangle->v0;
            dt.v1 = triangle->v1;
            dt.v2 = triangle->v2;
            dt.n0 = triangle->n0;
            dt.n1 = triangle->n1;
            dt.n2 = triangle->n2;
            dt.vt0 = triangle->vt0;
            dt.vt1 = triangle->vt1;
            dt.vt2 = triangle->vt2;
            dt.material  = triangle->getColor();
            dt.isSmooth  = triangle->isSmooth;
            dt.hasTexture = triangle->hasTexture;
            h_triangles.push_back(dt);

            // Placeholder texture record for triangle (no GPU upload yet)
            DeviceTexture tex;
            tex.pixels = nullptr;
            tex.width  = 0;
            tex.height = 0;
            h_textures.push_back(tex);
        }
    }

    for (const auto &light : scene.lights) {
        DeviceLight dl;
        dl.positionOrdir = light->positionOrdir;
        dl.intensity     = light->intensity;
        dl.isPoint       = light->isPoint;
        h_lights.push_back(dl);
    }

    int numSpheres   = (int)h_spheres.size();
    int numTriangles = (int)h_triangles.size();

    // ---------------------------
    // Build BVH for triangles
    // ---------------------------
    std::vector<DeviceBVHNode> bvhNodes;
    std::vector<int>           bvhTriIndices;

    if (numTriangles > 0) {
        std::vector<BVHBuildTriangle> buildTris;
        buildTris.reserve(numTriangles);

        for (int i = 0; i < numTriangles; ++i) {
            BVHBuildTriangle bt;
            bt.triIndex = i;

            const Vec3d &v0 = h_triangles[i].v0;
            const Vec3d &v1 = h_triangles[i].v1;
            const Vec3d &v2 = h_triangles[i].v2;

            bt.min.x = std::min(std::min(v0.x, v1.x), v2.x);
            bt.min.y = std::min(std::min(v0.y, v1.y), v2.y);
            bt.min.z = std::min(std::min(v0.z, v1.z), v2.z);

            bt.max.x = std::max(std::max(v0.x, v1.x), v2.x);
            bt.max.y = std::max(std::max(v0.y, v1.y), v2.y);
            bt.max.z = std::max(std::max(v0.z, v1.z), v2.z);

            bt.centroid = (v0 + v1 + v2) * (1.0 / 3.0);

            buildTris.push_back(bt);
        }

        bvhNodes.reserve(numTriangles * 2);
        // Build root node recursively (root index will be 0)
        buildBVHRecursive(buildTris, 0, (int)buildTris.size(),
                          bvhNodes, bvhTriIndices);
    }

    int numBVHNodes = (int)bvhNodes.size();

    // ---------------------------
    // Compute camera / view window
    // ---------------------------
    double ar = (double)width / (double)height;
    double vh = 2.0 * tan(scene.camera.vfov_rad() / 2.0);
    double vw = vh * ar;

    Vec3d viewDir = scene.camera.viewDir.norm();
    Vec3d right   = viewDir.cross(scene.camera.upDir).norm();
    Vec3d up      = right.cross(viewDir).norm();

    Vec3d center = scene.camera.eye + viewDir;
    Vec3d ul = center - right * (vw * 0.5) + up * (vh * 0.5);
    Vec3d ur = center + right * (vw * 0.5) + up * (vh * 0.5);
    Vec3d ll = center - right * (vw * 0.5) - up * (vh * 0.5);

    Vec3d delta_h = (ur - ul) / (double)width;
    Vec3d delta_v = (ll - ul) / (double)height;

    // ---------------------------
    // Allocate device memory
    // ---------------------------

    Color*         d_output      = nullptr;
    Color*         d_imageBuffer = nullptr;
    DeviceSphere*  d_spheres     = nullptr;
    DeviceTriangle* d_triangles  = nullptr;
    DeviceLight*   d_lights      = nullptr;
    DeviceTexture* d_textures    = nullptr;
    DeviceBVHNode* d_bvhNodes    = nullptr;
    int*           d_bvhTriIndices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_output,      width * height * sizeof(Color)));
    CUDA_CHECK(cudaMalloc(&d_imageBuffer, width * height * sizeof(Color)));

    if (numSpheres > 0) {
        CUDA_CHECK(cudaMalloc(&d_spheres, numSpheres * sizeof(DeviceSphere)));
        CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(),
                              numSpheres * sizeof(DeviceSphere),
                              cudaMemcpyHostToDevice));
    }

    if (numTriangles > 0) {
        CUDA_CHECK(cudaMalloc(&d_triangles, numTriangles * sizeof(DeviceTriangle)));
        CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(),
                              numTriangles * sizeof(DeviceTriangle),
                              cudaMemcpyHostToDevice));
    }

    if (!h_lights.empty()) {
        CUDA_CHECK(cudaMalloc(&d_lights, h_lights.size() * sizeof(DeviceLight)));
        CUDA_CHECK(cudaMemcpy(d_lights, h_lights.data(),
                              h_lights.size() * sizeof(DeviceLight),
                              cudaMemcpyHostToDevice));
    }

    if (!h_textures.empty()) {
        CUDA_CHECK(cudaMalloc(&d_textures, h_textures.size() * sizeof(DeviceTexture)));
        CUDA_CHECK(cudaMemcpy(d_textures, h_textures.data(),
                              h_textures.size() * sizeof(DeviceTexture),
                              cudaMemcpyHostToDevice));
    }

    if (numBVHNodes > 0) {
        CUDA_CHECK(cudaMalloc(&d_bvhNodes, numBVHNodes * sizeof(DeviceBVHNode)));
        CUDA_CHECK(cudaMemcpy(d_bvhNodes, bvhNodes.data(),
                              numBVHNodes * sizeof(DeviceBVHNode),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_bvhTriIndices, bvhTriIndices.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bvhTriIndices, bvhTriIndices.data(),
                              bvhTriIndices.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // ---------------------------
    // Launch configuration
    // ---------------------------
    dim3 blockSize(16, 16);
    dim3 gridSize((width  + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Rendering " << width << "x" << height
          << " image on GPU (BVH + SSR, 2 passes)..." << std::endl;

    // ---- GPU timing ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // ---------------------------
    // Pass 1: Render base image into d_imageBuffer (no SSR)
    // ---------------------------
    std::cout << "Pass 1: Rendering base image (no SSR)..." << std::endl;
    raytrace_kernel<<<gridSize, blockSize>>>(
        d_imageBuffer,   // output
        nullptr,         // imageBuffer for SSR (not used)
        d_spheres, numSpheres,
        d_triangles, numTriangles,
        d_lights, (int)h_lights.size(),
        d_textures,
        d_bvhNodes, numBVHNodes,
        d_bvhTriIndices,
        scene.camera.eye, viewDir, scene.camera.upDir, scene.camera.vfov_rad(),
        ul, delta_h, delta_v,
        width, height,
        scene.bkgcolor,
        false,  // useSSR
        10      // maxDepth (not used in this kernel)
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------------------
    // Pass 2: Render final image with SSR, using d_imageBuffer as SSR source
    // ---------------------------
    std::cout << "Pass 2: Rendering with SSR..." << std::endl;
    raytrace_kernel<<<gridSize, blockSize>>>(
        d_output,        // output
        d_imageBuffer,   // imageBuffer used by SSR
        d_spheres, numSpheres,
        d_triangles, numTriangles,
        d_lights, (int)h_lights.size(),
        d_textures,
        d_bvhNodes, numBVHNodes,
        d_bvhTriIndices,
        scene.camera.eye, viewDir, scene.camera.upDir, scene.camera.vfov_rad(),
        ul, delta_h, delta_v,
        width, height,
        scene.bkgcolor,
        true,   // useSSR
        10
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Rendering with BVH (2 passes) took "
            << (milliseconds / 1000.0f) << " seconds" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output,
                          width * height * sizeof(Color),
                          cudaMemcpyDeviceToHost));

    // ---------------------------
    // Write PPM output
    // ---------------------------
    std::string perspective_filename = basename + "_perspective.ppm";
    std::ofstream ofs(perspective_filename);
    ofs << "P3\n" << width << " " << height << "\n255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Color &c = h_output[y * width + x];
            int ir = (int)(clamp_cuda(c.r, 0.0f, 1.0f) * 255.0f);
            int ig = (int)(clamp_cuda(c.g, 0.0f, 1.0f) * 255.0f);
            int ib = (int)(clamp_cuda(c.b, 0.0f, 1.0f) * 255.0f);
            ofs << ir << " " << ig << " " << ib << " ";
        }
        ofs << "\n";
    }
    ofs.close();

    // ---------------------------
    // Cleanup
    // ---------------------------
    delete[] h_output;
    delete[] h_imageBuffer;

    if (d_output)        CUDA_CHECK(cudaFree(d_output));
    if (d_imageBuffer)   CUDA_CHECK(cudaFree(d_imageBuffer));
    if (d_spheres)       CUDA_CHECK(cudaFree(d_spheres));
    if (d_triangles)     CUDA_CHECK(cudaFree(d_triangles));
    if (d_lights)        CUDA_CHECK(cudaFree(d_lights));
    if (d_textures)      CUDA_CHECK(cudaFree(d_textures));
    if (d_bvhNodes)      CUDA_CHECK(cudaFree(d_bvhNodes));
    if (d_bvhTriIndices) CUDA_CHECK(cudaFree(d_bvhTriIndices));

    std::cout << "Rendering complete. Image saved as '" << perspective_filename << "'." << std::endl;
    return 0;
}
