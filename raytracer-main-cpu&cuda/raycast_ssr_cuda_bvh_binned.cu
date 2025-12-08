// raycast_ssr_cuda_bvh_binned.cu
// Ray binning optimization - sorts rays by direction for better coherence
// Simpler version without warp-sync (which can cause hangs)

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
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<Vec3d> Objects::vertices;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            FILE* f = fopen("cuda_error.log", "w"); \
            if (f) { fprintf(f, "CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); fclose(f); } \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ __host__ inline float clamp_cuda(float value, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(value, maxVal));
}

__device__ __host__ inline int clamp_cuda(int value, int minVal, int maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

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

struct DeviceAABB {
    Vec3d min_pt;
    Vec3d max_pt;
};

struct DeviceBVHNode {
    DeviceAABB bounds;
    int left;
    int right;
    int start;
    int count;
    int isLeaf;
    int splitAxis;
    float splitPos;
};

struct BVHBuildTriangle {
    int triIndex;
    Vec3d min;
    Vec3d max;
    Vec3d centroid;
};

struct CompactRay {
    float ox, oy, oz;
    float dx, dy, dz;
    float invDx, invDy, invDz;
};

__device__ CompactRay makeRay(const Vec3d& origin, const Vec3d& dir) {
    CompactRay r;
    r.ox = origin.x; r.oy = origin.y; r.oz = origin.z;
    r.dx = dir.x; r.dy = dir.y; r.dz = dir.z;
    r.invDx = (fabsf(dir.x) > 1e-8f) ? (1.0f / dir.x) : ((dir.x >= 0) ? 1e8f : -1e8f);
    r.invDy = (fabsf(dir.y) > 1e-8f) ? (1.0f / dir.y) : ((dir.y >= 0) ? 1e8f : -1e8f);
    r.invDz = (fabsf(dir.z) > 1e-8f) ? (1.0f / dir.z) : ((dir.z >= 0) ? 1e8f : -1e8f);
    return r;
}

// Morton code for ray direction
__host__ __device__ uint32_t directionToMorton(float dx, float dy, float dz) {
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len < 1e-8f) return 0;
    dx /= len; dy /= len; dz /= len;
    
    float l = fabsf(dx) + fabsf(dy) + fabsf(dz);
    float ox = dx / l;
    float oy = dy / l;
    
    if (dz < 0.0f) {
        float newX = (1.0f - fabsf(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        float newY = (1.0f - fabsf(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = newX;
        oy = newY;
    }
    
    uint32_t x = (uint32_t)((ox * 0.5f + 0.5f) * 1023.0f);
    uint32_t y = (uint32_t)((oy * 0.5f + 0.5f) * 1023.0f);
    
    x = x > 1023 ? 1023 : x;
    y = y > 1023 ? 1023 : y;
    
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    
    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8))  & 0x0300F00F;
    y = (y | (y << 4))  & 0x030C30C3;
    y = (y | (y << 2))  & 0x09249249;
    
    return x | (y << 1);
}

__host__ uint32_t rayToMorton(float dx, float dy, float dz, int pixelX, int pixelY, int width, int tileSize = 8) {
    uint32_t dirHash = directionToMorton(dx, dy, dz) >> 4;
    int tileX = pixelX / tileSize;
    int tileY = pixelY / tileSize;
    int tilesPerRow = (width + tileSize - 1) / tileSize;
    uint32_t tileIdx = tileY * tilesPerRow + tileX;
    return (dirHash << 16) | (tileIdx & 0xFFFF);
}

__device__ Vec2d calculateSphereUV_device(const Vec3d &normal) {
    float f = acosf(clamp_cuda((float)normal.z, -1.0f, 1.0f));
    float q = atan2f(normal.y, normal.x);
    float u = (q >= 0) ? (q / (2.0f * M_PI)) : ((q + 2.0f * M_PI) / (2.0f * M_PI));
    float v = f / M_PI;
    return Vec2d(u, v);
}

__device__ bool intersectRaySphere_device(const CompactRay &ray, const DeviceSphere &sphere,
                                          float &t, Vec2d &texCoordOut) {
    float cx = sphere.center.x, cy = sphere.center.y, cz = sphere.center.z;
    float r = sphere.radius;

    float a = ray.dx * ray.dx + ray.dy * ray.dy + ray.dz * ray.dz;
    float b = 2.0f * (ray.dx * (ray.ox - cx) + ray.dy * (ray.oy - cy) + ray.dz * (ray.oz - cz));
    float c = (ray.ox - cx) * (ray.ox - cx) + (ray.oy - cy) * (ray.oy - cy) + 
              (ray.oz - cz) * (ray.oz - cz) - r * r;

    float d = b * b - 4.0f * a * c;
    if (d < 0.0f) return false;

    float t1 = (-b - sqrtf(d)) / (2.0f * a);
    float t2 = (-b + sqrtf(d)) / (2.0f * a);

    if (t1 > 0.0f) t = t1;
    else if (t2 > 0.0f) t = t2;
    else return false;

    Vec3d hitPt = Vec3d(ray.ox + ray.dx * t, ray.oy + ray.dy * t, ray.oz + ray.dz * t);
    Vec3d normal = (hitPt - sphere.center).norm();
    texCoordOut = calculateSphereUV_device(normal);
    return true;
}

__device__ bool intersectRayTriangle_device(const CompactRay &ray, float &t,
                                            Vec3d &normalOut, Vec2d &texCoordOut,
                                            const DeviceTriangle &triangle) {
    Vec3d rayOrig(ray.ox, ray.oy, ray.oz);
    Vec3d rayDir(ray.dx, ray.dy, ray.dz);
    
    Vec3d edge1 = triangle.v1 - triangle.v0;
    Vec3d edge2 = triangle.v2 - triangle.v0;
    Vec3d cr = edge1.cross(edge2);
    Vec3d normal = cr.norm();

    float denominator = cr.dot(rayDir);
    if (fabsf(denominator) < 1e-6f) return false;

    float d = -cr.dot(triangle.v0);
    t = -(cr.dot(rayOrig) + d) / denominator;
    if (t < 0.0f) return false;

    Vec3d P = rayOrig + rayDir * t;
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
            normalOut = (triangle.n0 * alpha + triangle.n1 * beta + triangle.n2 * gamma).norm();
        } else {
            normalOut = normal;
        }
        if (triangle.hasTexture) {
            texCoordOut = triangle.vt0 * alpha + triangle.vt1 * beta + triangle.vt2 * gamma;
        }
        return true;
    }
    return false;
}

__device__ Color getTextureColor_device(const DeviceTexture &texture, float u, float v) {
    if (!texture.pixels || texture.width == 0 || texture.height == 0) {
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
    float etai = 1.0f, etat = ior;
    Vec3d n = N;
    if (cosi < 0.0f) cosi = -cosi;
    else { float tmp = etai; etai = etat; etat = tmp; n = N * -1.0; }
    float eta = etai / etat;
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f) return false;
    refractedDir = (I * eta + n * (eta * cosi - sqrtf(k))).norm();
    return true;
}

__device__ float fresnelSchlick_device(const Vec3d &I, const Vec3d &N, float ior) {
    float cosi = clamp_cuda(I.dot(N), -1.0f, 1.0f);
    float etai = 1.0f, etat = ior;
    if (cosi > 0.0f) { float tmp = etai; etai = etat; etat = tmp; }
    float r0 = (etai - etat) / (etai + etat);
    float f0 = r0 * r0;
    return f0 + (1.0f - f0) * powf(1.0f - fabsf(cosi), 5.0f);
}

__device__ __forceinline__ bool intersectAABB(const CompactRay &ray, const DeviceAABB &box,
                                               float tMin, float tMax) {
    float t0x = (box.min_pt.x - ray.ox) * ray.invDx;
    float t1x = (box.max_pt.x - ray.ox) * ray.invDx;
    float t0y = (box.min_pt.y - ray.oy) * ray.invDy;
    float t1y = (box.max_pt.y - ray.oy) * ray.invDy;
    float t0z = (box.min_pt.z - ray.oz) * ray.invDz;
    float t1z = (box.max_pt.z - ray.oz) * ray.invDz;
    
    tMin = fmaxf(tMin, fmaxf(fminf(t0x, t1x), fmaxf(fminf(t0y, t1y), fminf(t0z, t1z))));
    tMax = fminf(tMax, fminf(fmaxf(t0x, t1x), fminf(fmaxf(t0y, t1y), fmaxf(t0z, t1z))));
    
    return tMin <= tMax;
}

// Simple BVH traversal (no warp sync - more reliable)
__device__ bool intersectBVH(const CompactRay &ray,
                             const DeviceBVHNode* __restrict__ nodes,
                             int numNodes,
                             const int* __restrict__ triIndices,
                             const DeviceTriangle* __restrict__ triangles,
                             float &closestT,
                             int &hitTriIndex,
                             Vec3d &hitNormal,
                             Vec2d &hitTexCoord) {
    if (!nodes || numNodes <= 0) return false;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    bool hit = false;
    closestT = 1e30f;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const DeviceBVHNode &node = nodes[nodeIdx];

        if (!intersectAABB(ray, node.bounds, 0.001f, closestT))
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
            // Ordered traversal
            bool leftFirst = true;
            if (node.splitAxis == 0) leftFirst = ray.dx > 0;
            else if (node.splitAxis == 1) leftFirst = ray.dy > 0;
            else leftFirst = ray.dz > 0;
            
            if (leftFirst) {
                if (node.right >= 0) stack[stackPtr++] = node.right;
                if (node.left >= 0) stack[stackPtr++] = node.left;
            } else {
                if (node.left >= 0) stack[stackPtr++] = node.left;
                if (node.right >= 0) stack[stackPtr++] = node.right;
            }
        }
    }

    return hit;
}

__device__ bool intersectBVH_shadow(const CompactRay &ray,
                                    const DeviceBVHNode* __restrict__ nodes,
                                    int numNodes,
                                    const int* __restrict__ triIndices,
                                    const DeviceTriangle* __restrict__ triangles,
                                    float maxDist) {
    if (!nodes || numNodes <= 0) return false;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const DeviceBVHNode &node = nodes[nodeIdx];

        if (!intersectAABB(ray, node.bounds, 0.001f, maxDist))
            continue;

        if (node.isLeaf) {
            for (int i = 0; i < node.count; ++i) {
                int triIdx = triIndices[node.start + i];
                float t;
                Vec3d n;
                Vec2d tc;
                if (intersectRayTriangle_device(ray, t, n, tc, triangles[triIdx]) &&
                    t > 0.001f && t < maxDist) {
                    return true;
                }
            }
        } else {
            if (node.left >= 0) stack[stackPtr++] = node.left;
            if (node.right >= 0) stack[stackPtr++] = node.right;
        }
    }
    return false;
}

__device__ float computeShadowFactor(const Vec3d &point, const Vec3d &lightDir, float maxDist,
                                     const DeviceSphere* spheres, int numSpheres,
                                     const DeviceBVHNode* bvhNodes, int numBVHNodes,
                                     const int* bvhTriIndices, const DeviceTriangle* triangles) {
    Vec3d shadowOrig = point + lightDir * 1e-4f;
    CompactRay shadowRay = makeRay(shadowOrig, lightDir);
    float transparency = 1.0f;

    for (int i = 0; i < numSpheres; ++i) {
        float t;
        Vec2d texCoord;
        if (intersectRaySphere_device(shadowRay, spheres[i], t, texCoord) &&
            t > 0.001f && t < maxDist) {
            transparency *= (1.0f - spheres[i].material.alpha);
            if (transparency <= 0.01f) return 0.0f;
        }
    }

    if (bvhNodes && numBVHNodes > 0) {
        if (intersectBVH_shadow(shadowRay, bvhNodes, numBVHNodes, bvhTriIndices, triangles, maxDist)) {
            return 0.0f;
        }
    }
    return transparency;
}

__device__ bool projectToScreen(const Vec3d &point, const Vec3d &eyePos,
                                const Vec3d &viewDir, const Vec3d &upDir,
                                float vfov, int width, int height,
                                int &screenX, int &screenY) {
    Vec3d right = viewDir.cross(upDir).norm();
    Vec3d up = right.cross(viewDir).norm();
    Vec3d toPoint = point - eyePos;
    float distance = toPoint.dot(viewDir);
    if (distance <= 0.0f) return false;

    double ar = (double)width / (double)height;
    double vh = 2.0 * tanf(vfov / 2.0f);
    double vw = vh * ar;

    float u = toPoint.dot(right) / distance;
    float v = -toPoint.dot(up) / distance;

    screenX = (int)((u / vw + 0.5) * width);
    screenY = (int)((v / vh + 0.5) * height);
    return (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height);
}

__device__ Color getSSRColor(const Vec3d &reflectDir, const Vec3d &hitPoint,
                             const Color* imageBuffer, int width, int height,
                             const Vec3d &eyePos, const Vec3d &viewDir,
                             const Vec3d &upDir, float vfov, const Color &bkgcolor) {
    const int MAX_STEPS = 32;
    const float STEP_SIZE = 0.15f;
    Vec3d currentPos = hitPoint;

    for (int i = 0; i < MAX_STEPS; ++i) {
        currentPos = currentPos + reflectDir * STEP_SIZE;
        int screenX, screenY;
        if (projectToScreen(currentPos, eyePos, viewDir, upDir, vfov, width, height, screenX, screenY)) {
            return imageBuffer[screenY * width + screenX];
        }
    }
    return bkgcolor;
}

__device__ Color shade(const Vec3d &point, const Vec3d &rayDir, const Vec3d &normal,
                       const Vec2d &texCoord, const MaterialColor &material,
                       const DeviceLight* lights, int numLights,
                       const DeviceSphere* spheres, int numSpheres,
                       const DeviceBVHNode* bvhNodes, int numBVHNodes,
                       const int* bvhTriIndices, const DeviceTriangle* triangles,
                       const DeviceTexture &texture, bool hasTexture, const Color &bkgcolor) {
    Color texturec = material.color;
    if (hasTexture && texture.pixels) {
        texturec = getTextureColor_device(texture, texCoord.x, texCoord.y);
    }

    Vec3d viewDir = (rayDir * -1.0).norm();
    Color ambient = texturec * material.ka;
    Color totalDiffuse(0, 0, 0);
    Color totalSpecular(0, 0, 0);

    for (int i = 0; i < numLights; ++i) {
        Vec3d L = lights[i].isPoint ? (lights[i].positionOrdir - point).norm()
                                    : lights[i].positionOrdir.norm() * -1.0;
        Vec3d H = (L + viewDir).norm();

        float df = fmaxf(normal.dot(L), 0.0f);
        Color diffuse = texturec * material.kd * df;

        float sf = powf(fmaxf(normal.dot(H), 0.0f), material.shininess);
        Color specular = material.specular * material.ks * sf;

        float maxDist = 1e30f;
        if (lights[i].isPoint) {
            maxDist = (lights[i].positionOrdir - point).length();
        }

        float shadowFactor = computeShadowFactor(point, L, maxDist, spheres, numSpheres,
                                                  bvhNodes, numBVHNodes, bvhTriIndices, triangles);

        totalDiffuse = totalDiffuse + diffuse * lights[i].intensity * shadowFactor;
        totalSpecular = totalSpecular + specular * lights[i].intensity * shadowFactor;
    }

    return ambient + totalDiffuse + totalSpecular;
}

// Binned kernel - processes rays in sorted order
__global__ void raytrace_kernel_binned(
    Color* __restrict__ output,
    const Color* __restrict__ imageBuffer,
    const int* __restrict__ sortedPixelIndices,
    const DeviceSphere* __restrict__ spheres, int numSpheres,
    const DeviceTriangle* __restrict__ triangles, int numTriangles,
    const DeviceLight* __restrict__ lights, int numLights,
    const DeviceTexture* __restrict__ textures,
    const DeviceBVHNode* __restrict__ bvhNodes, int numBVHNodes,
    const int* __restrict__ bvhTriIndices,
    Vec3d eyePos, Vec3d viewDir, Vec3d upDir, float vfov,
    Vec3d ul, Vec3d delta_h, Vec3d delta_v,
    int width, int height,
    Color bkgcolor,
    bool useSSR,
    int totalPixels)
{
    int sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sortedIdx >= totalPixels) return;
    
    int pixelIdx = sortedPixelIndices[sortedIdx];
    int x = pixelIdx % width;
    int y = pixelIdx / width;

    Vec3d vwPosition = ul + delta_h * (float)x + delta_v * (float)y;
    Vec3d rayDir = (vwPosition - eyePos).norm();
    CompactRay ray = makeRay(eyePos, rayDir);

    float tClosestSphere = 1e30f;
    int sphereHitIndex = -1;
    Vec2d sphereTexCoord;
    Vec3d sphereNormal;

    for (int i = 0; i < numSpheres; ++i) {
        float t;
        Vec2d tc;
        if (intersectRaySphere_device(ray, spheres[i], t, tc) && t > 0.001f && t < tClosestSphere) {
            tClosestSphere = t;
            sphereHitIndex = i;
            sphereTexCoord = tc;
            Vec3d hitPt(ray.ox + ray.dx * t, ray.oy + ray.dy * t, ray.oz + ray.dz * t);
            sphereNormal = (hitPt - spheres[i].center).norm();
        }
    }

    float tClosestTri = 1e30f;
    int triHitIndex = -1;
    Vec3d triNormal;
    Vec2d triTexCoord;

    bool triHit = intersectBVH(ray, bvhNodes, numBVHNodes, bvhTriIndices, triangles,
                               tClosestTri, triHitIndex, triNormal, triTexCoord);

    float t_min = 1e30f;
    int hitType = -1;
    int hitIndex = -1;
    Vec3d hitNormal;
    Vec2d hitTexCoord;

    if (sphereHitIndex >= 0 && tClosestSphere < t_min) {
        t_min = tClosestSphere;
        hitType = 0;
        hitIndex = sphereHitIndex;
        hitNormal = sphereNormal;
        hitTexCoord = sphereTexCoord;
    }

    if (triHit && tClosestTri < t_min) {
        t_min = tClosestTri;
        hitType = 1;
        hitIndex = triHitIndex;
        hitNormal = triNormal;
        hitTexCoord = triTexCoord;
    }

    Color finalColor = bkgcolor;

    if (hitType >= 0) {
        Vec3d hitPoint = eyePos + rayDir * t_min;

        MaterialColor material;
        DeviceTexture tex;
        bool hasTexture = false;

        if (hitType == 0) {
            material = spheres[hitIndex].material;
            hasTexture = spheres[hitIndex].hasTexture;
            tex = textures[hitIndex];
        } else {
            material = triangles[hitIndex].material;
            hasTexture = triangles[hitIndex].hasTexture;
            tex = textures[numSpheres + hitIndex];
        }

        Color localColor = shade(hitPoint, rayDir, hitNormal, hitTexCoord, material,
                                 lights, numLights, spheres, numSpheres,
                                 bvhNodes, numBVHNodes, bvhTriIndices, triangles,
                                 tex, hasTexture, bkgcolor);

        Color reflectedColor(0, 0, 0);
        if (material.ks > 0.0f && useSSR && imageBuffer) {
            Vec3d reflDir = (rayDir - hitNormal * 2.0 * (rayDir.dot(hitNormal))).norm();
            reflectedColor = getSSRColor(reflDir, hitPoint, imageBuffer, width, height,
                                         eyePos, viewDir, upDir, vfov, bkgcolor);
        }

        Color refractedColor(0, 0, 0);
        if (material.alpha < 1.0f) {
            Vec3d refractDir;
            if (refract_device(rayDir, hitNormal, material.ior, refractDir)) {
                refractedColor = bkgcolor;
            }
        }

        float Fr = fresnelSchlick_device(rayDir, hitNormal, material.ior);
        finalColor = localColor + reflectedColor * Fr + refractedColor * (1.0f - material.alpha) * (1.0f - Fr);
    }

    output[pixelIdx] = finalColor;
}

// BVH Building
static int buildBVHRecursive(std::vector<BVHBuildTriangle> &buildTris,
                             int start, int end,
                             std::vector<DeviceBVHNode> &nodes,
                             std::vector<int> &triIndices) {
    int nodeIndex = (int)nodes.size();
    nodes.push_back(DeviceBVHNode());

    DeviceAABB bounds;
    bounds.min_pt = Vec3d(FLT_MAX, FLT_MAX, FLT_MAX);
    bounds.max_pt = Vec3d(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    Vec3d cmin(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3d cmax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = start; i < end; ++i) {
        const BVHBuildTriangle &bt = buildTris[i];
        bounds.min_pt.x = std::min(bounds.min_pt.x, bt.min.x);
        bounds.min_pt.y = std::min(bounds.min_pt.y, bt.min.y);
        bounds.min_pt.z = std::min(bounds.min_pt.z, bt.min.z);
        bounds.max_pt.x = std::max(bounds.max_pt.x, bt.max.x);
        bounds.max_pt.y = std::max(bounds.max_pt.y, bt.max.y);
        bounds.max_pt.z = std::max(bounds.max_pt.z, bt.max.z);
        cmin.x = std::min(cmin.x, bt.centroid.x);
        cmin.y = std::min(cmin.y, bt.centroid.y);
        cmin.z = std::min(cmin.z, bt.centroid.z);
        cmax.x = std::max(cmax.x, bt.centroid.x);
        cmax.y = std::max(cmax.y, bt.centroid.y);
        cmax.z = std::max(cmax.z, bt.centroid.z);
    }

    nodes[nodeIndex].bounds = bounds;
    nodes[nodeIndex].left = -1;
    nodes[nodeIndex].right = -1;
    nodes[nodeIndex].start = -1;
    nodes[nodeIndex].count = 0;
    nodes[nodeIndex].isLeaf = 0;
    nodes[nodeIndex].splitAxis = 0;
    nodes[nodeIndex].splitPos = 0.0f;

    int n = end - start;
    const int LEAF_SIZE = 4;

    if (n <= LEAF_SIZE) {
        nodes[nodeIndex].start = (int)triIndices.size();
        nodes[nodeIndex].count = n;
        nodes[nodeIndex].isLeaf = 1;
        for (int i = start; i < end; ++i) {
            triIndices.push_back(buildTris[i].triIndex);
        }
        return nodeIndex;
    }

    Vec3d extent = cmax - cmin;
    int axis = 0;
    if (extent.y > extent.x && extent.y >= extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z >= extent.y) axis = 2;

    int mid = (start + end) / 2;

    if (!(extent.x < 1e-5 && extent.y < 1e-5 && extent.z < 1e-5)) {
        if (axis == 0) {
            std::sort(buildTris.begin() + start, buildTris.begin() + end,
                      [](const BVHBuildTriangle &a, const BVHBuildTriangle &b) { return a.centroid.x < b.centroid.x; });
        } else if (axis == 1) {
            std::sort(buildTris.begin() + start, buildTris.begin() + end,
                      [](const BVHBuildTriangle &a, const BVHBuildTriangle &b) { return a.centroid.y < b.centroid.y; });
        } else {
            std::sort(buildTris.begin() + start, buildTris.begin() + end,
                      [](const BVHBuildTriangle &a, const BVHBuildTriangle &b) { return a.centroid.z < b.centroid.z; });
        }
    }

    nodes[nodeIndex].splitAxis = axis;
    if (axis == 0) nodes[nodeIndex].splitPos = buildTris[mid].centroid.x;
    else if (axis == 1) nodes[nodeIndex].splitPos = buildTris[mid].centroid.y;
    else nodes[nodeIndex].splitPos = buildTris[mid].centroid.z;

    int leftIndex = buildBVHRecursive(buildTris, start, mid, nodes, triIndices);
    int rightIndex = buildBVHRecursive(buildTris, mid, end, nodes, triIndices);

    nodes[nodeIndex].left = leftIndex;
    nodes[nodeIndex].right = rightIndex;

    return nodeIndex;
}

// Ray sorting on CPU
struct RaySortEntry {
    uint32_t mortonCode;
    int pixelIndex;
};

void sortRaysByMorton(std::vector<int>& sortedIndices, 
                      int width, int height,
                      const Vec3d& eyePos, const Vec3d& ul,
                      const Vec3d& delta_h, const Vec3d& delta_v) {
    int totalPixels = width * height;
    std::vector<RaySortEntry> entries(totalPixels);
    
    std::cout << "Computing Morton codes..." << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            Vec3d vwPosition = ul + delta_h * (float)x + delta_v * (float)y;
            Vec3d rayDir = (vwPosition - eyePos).norm();
            
            entries[idx].mortonCode = rayToMorton(rayDir.x, rayDir.y, rayDir.z, x, y, width);
            entries[idx].pixelIndex = idx;
        }
    }
    
    std::cout << "Sorting " << totalPixels << " rays..." << std::endl;
    std::sort(entries.begin(), entries.end(), 
              [](const RaySortEntry& a, const RaySortEntry& b) {
                  return a.mortonCode < b.mortonCode;
              });
    
    sortedIndices.resize(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
        sortedIndices[i] = entries[i].pixelIndex;
    }
    std::cout << "Sorting complete." << std::endl;
}

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
    if (lastdot != std::string::npos) basename = basename.substr(0, lastdot);

    parse(filename, scene);

    int width = (int)scene.camera.w;
    int height = (int)scene.camera.h;
    int totalPixels = width * height;

    std::cout << "Rendering " << width << "x" << height << " with ray binning + BVH..." << std::endl;

    Color* h_output = new Color[totalPixels];

    std::vector<DeviceSphere> h_spheres;
    std::vector<DeviceTriangle> h_triangles;
    std::vector<DeviceLight> h_lights;
    std::vector<DeviceTexture> h_textures;

    for (const auto &obj : scene.objects) {
        if (auto* sphere = dynamic_cast<Sphere*>(obj.get())) {
            DeviceSphere ds;
            ds.center = sphere->center;
            ds.radius = sphere->radius;
            ds.material = sphere->getColor();
            ds.hasTexture = sphere->hasTexture;
            h_spheres.push_back(ds);
            DeviceTexture dt = {nullptr, 0, 0};
            h_textures.push_back(dt);
        } else if (auto* triangle = dynamic_cast<Triangle*>(obj.get())) {
            DeviceTriangle dt;
            dt.v0 = triangle->v0; dt.v1 = triangle->v1; dt.v2 = triangle->v2;
            dt.n0 = triangle->n0; dt.n1 = triangle->n1; dt.n2 = triangle->n2;
            dt.vt0 = triangle->vt0; dt.vt1 = triangle->vt1; dt.vt2 = triangle->vt2;
            dt.material = triangle->getColor();
            dt.isSmooth = triangle->isSmooth;
            dt.hasTexture = triangle->hasTexture;
            h_triangles.push_back(dt);
            DeviceTexture tex = {nullptr, 0, 0};
            h_textures.push_back(tex);
        }
    }

    for (const auto &light : scene.lights) {
        DeviceLight dl;
        dl.positionOrdir = light->positionOrdir;
        dl.intensity = light->intensity;
        dl.isPoint = light->isPoint;
        h_lights.push_back(dl);
    }

    int numSpheres = (int)h_spheres.size();
    int numTriangles = (int)h_triangles.size();

    // Build BVH
    std::vector<DeviceBVHNode> bvhNodes;
    std::vector<int> bvhTriIndices;

    if (numTriangles > 0) {
        std::vector<BVHBuildTriangle> buildTris;
        buildTris.reserve(numTriangles);
        for (int i = 0; i < numTriangles; ++i) {
            BVHBuildTriangle bt;
            bt.triIndex = i;
            const Vec3d &v0 = h_triangles[i].v0;
            const Vec3d &v1 = h_triangles[i].v1;
            const Vec3d &v2 = h_triangles[i].v2;
            bt.min.x = std::min({v0.x, v1.x, v2.x});
            bt.min.y = std::min({v0.y, v1.y, v2.y});
            bt.min.z = std::min({v0.z, v1.z, v2.z});
            bt.max.x = std::max({v0.x, v1.x, v2.x});
            bt.max.y = std::max({v0.y, v1.y, v2.y});
            bt.max.z = std::max({v0.z, v1.z, v2.z});
            bt.centroid = (v0 + v1 + v2) * (1.0 / 3.0);
            buildTris.push_back(bt);
        }
        bvhNodes.reserve(numTriangles * 2);
        buildBVHRecursive(buildTris, 0, (int)buildTris.size(), bvhNodes, bvhTriIndices);
    }

    int numBVHNodes = (int)bvhNodes.size();
    std::cout << "BVH: " << numBVHNodes << " nodes for " << numTriangles << " triangles" << std::endl;

    // Camera
    double ar = (double)width / (double)height;
    double vh = 2.0 * tan(scene.camera.vfov_rad() / 2.0);
    double vw = vh * ar;

    Vec3d viewDir = scene.camera.viewDir.norm();
    Vec3d right = viewDir.cross(scene.camera.upDir).norm();
    Vec3d up = right.cross(viewDir).norm();

    Vec3d center = scene.camera.eye + viewDir;
    Vec3d ul = center - right * (vw * 0.5) + up * (vh * 0.5);
    Vec3d ur = center + right * (vw * 0.5) + up * (vh * 0.5);
    Vec3d ll = center - right * (vw * 0.5) - up * (vh * 0.5);

    Vec3d delta_h = (ur - ul) / (double)width;
    Vec3d delta_v = (ll - ul) / (double)height;

    // Sort rays
    std::cout << "Sorting rays by direction (Morton code)..." << std::endl;
    std::vector<int> sortedPixelIndices;
    sortRaysByMorton(sortedPixelIndices, width, height, scene.camera.eye, ul, delta_h, delta_v);

    // Allocate device memory
    Color* d_output = nullptr;
    Color* d_imageBuffer = nullptr;
    int* d_sortedPixelIndices = nullptr;
    DeviceSphere* d_spheres = nullptr;
    DeviceTriangle* d_triangles = nullptr;
    DeviceLight* d_lights = nullptr;
    DeviceTexture* d_textures = nullptr;
    DeviceBVHNode* d_bvhNodes = nullptr;
    int* d_bvhTriIndices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_output, totalPixels * sizeof(Color)));
    CUDA_CHECK(cudaMalloc(&d_imageBuffer, totalPixels * sizeof(Color)));
    CUDA_CHECK(cudaMalloc(&d_sortedPixelIndices, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sortedPixelIndices, sortedPixelIndices.data(), 
                          totalPixels * sizeof(int), cudaMemcpyHostToDevice));

    if (numSpheres > 0) {
        CUDA_CHECK(cudaMalloc(&d_spheres, numSpheres * sizeof(DeviceSphere)));
        CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(), numSpheres * sizeof(DeviceSphere), cudaMemcpyHostToDevice));
    }
    if (numTriangles > 0) {
        CUDA_CHECK(cudaMalloc(&d_triangles, numTriangles * sizeof(DeviceTriangle)));
        CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), numTriangles * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    }
    if (!h_lights.empty()) {
        CUDA_CHECK(cudaMalloc(&d_lights, h_lights.size() * sizeof(DeviceLight)));
        CUDA_CHECK(cudaMemcpy(d_lights, h_lights.data(), h_lights.size() * sizeof(DeviceLight), cudaMemcpyHostToDevice));
    }
    if (!h_textures.empty()) {
        CUDA_CHECK(cudaMalloc(&d_textures, h_textures.size() * sizeof(DeviceTexture)));
        CUDA_CHECK(cudaMemcpy(d_textures, h_textures.data(), h_textures.size() * sizeof(DeviceTexture), cudaMemcpyHostToDevice));
    }
    if (numBVHNodes > 0) {
        CUDA_CHECK(cudaMalloc(&d_bvhNodes, numBVHNodes * sizeof(DeviceBVHNode)));
        CUDA_CHECK(cudaMemcpy(d_bvhNodes, bvhNodes.data(), numBVHNodes * sizeof(DeviceBVHNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bvhTriIndices, bvhTriIndices.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bvhTriIndices, bvhTriIndices.data(), bvhTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    std::cout << "Launch: " << numBlocks << " blocks x " << blockSize << " threads" << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Pass 1
    std::cout << "Pass 1: Base image (ray binning)..." << std::endl;
    raytrace_kernel_binned<<<numBlocks, blockSize>>>(
        d_imageBuffer, nullptr,
        d_sortedPixelIndices,
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
        false,
        totalPixels
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 2
    std::cout << "Pass 2: With SSR (ray binning)..." << std::endl;
    raytrace_kernel_binned<<<numBlocks, blockSize>>>(
        d_output, d_imageBuffer,
        d_sortedPixelIndices,
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
        true,
        totalPixels
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Rendering (ray binning, 2 passes) took " << (milliseconds / 1000.0f) << " seconds" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, totalPixels * sizeof(Color), cudaMemcpyDeviceToHost));

    // Write output
    std::string out_filename = basename + "_perspective.ppm";
    std::ofstream ofs(out_filename);
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

    // Cleanup
    delete[] h_output;
    if (d_output) CUDA_CHECK(cudaFree(d_output));
    if (d_imageBuffer) CUDA_CHECK(cudaFree(d_imageBuffer));
    if (d_sortedPixelIndices) CUDA_CHECK(cudaFree(d_sortedPixelIndices));
    if (d_spheres) CUDA_CHECK(cudaFree(d_spheres));
    if (d_triangles) CUDA_CHECK(cudaFree(d_triangles));
    if (d_lights) CUDA_CHECK(cudaFree(d_lights));
    if (d_textures) CUDA_CHECK(cudaFree(d_textures));
    if (d_bvhNodes) CUDA_CHECK(cudaFree(d_bvhNodes));
    if (d_bvhTriIndices) CUDA_CHECK(cudaFree(d_bvhTriIndices));

    std::cout << "Done. Saved to '" << out_filename << "'" << std::endl;
    return 0;
}