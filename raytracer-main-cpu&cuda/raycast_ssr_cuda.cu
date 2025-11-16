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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device clamp function
__device__ __host__ inline float clamp_cuda(float value, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(value, maxVal));
}

__device__ __host__ inline int clamp_cuda(int value, int minVal, int maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

// Device structures for GPU
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

// Device functions
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
    float b = 2 * (ray.dx * (ray.x - cx) + ray.dy * (ray.y - cy) + ray.dz * (ray.z - cz));
    float c = (ray.x - cx) * (ray.x - cx) + (ray.y - cy) * (ray.y - cy) + 
              (ray.z - cz) * (ray.z - cz) - r * r;

    float d = b * b - 4 * a * c;
    if (d < 0) return false;

    float t1 = (-b - sqrtf(d)) / (2 * a);
    float t2 = (-b + sqrtf(d)) / (2 * a);

    if (t1 > 0) {
        t = t1;
    } else if (t2 > 0) {
        t = t2;
    } else {
        return false;
    }

    Vec3d intersectionPoint = ray.getOrigin() + ray.getDirection() * t;
    Vec3d normal = (intersectionPoint - sphere.center).norm();
    texCoordOut = calculateSphereUV_device(normal);
    return true;
}

__device__ bool intersectRayTriangle_device(const Ray &ray, float &t, Vec3d &normalOut, 
                                            Vec2d &texCoordOut, const DeviceTriangle &triangle) {
    Vec3d edge1 = triangle.v1 - triangle.v0;
    Vec3d edge2 = triangle.v2 - triangle.v0;
    Vec3d cr = edge1.cross(edge2);
    Vec3d normal = cr.norm();

    float denominator = cr.dot(ray.getDirection());
    if (fabsf(denominator) < 1e-6f) return false;

    float d = -cr.dot(triangle.v0);
    t = -(cr.dot(ray.getOrigin()) + d) / denominator;
    if (t < 0) return false;

    Vec3d P = ray.getOrigin() + ray.getDirection() * t;
    Vec3d vP = P - triangle.v0;
    float det = edge1.dot(edge1) * edge2.dot(edge2) - edge1.dot(edge2) * edge1.dot(edge2);
    float beta = (vP.dot(edge1) * edge2.dot(edge2) - vP.dot(edge2) * edge1.dot(edge2)) / det;
    float gamma = (edge1.dot(edge1) * vP.dot(edge2) - edge1.dot(edge2) * vP.dot(edge1)) / det;
    float alpha = 1.0f - beta - gamma;

    if (alpha >= 0 && beta >= 0 && gamma >= 0) {
        if (triangle.isSmooth) {
            normalOut = (triangle.n0 * alpha) + (triangle.n1 * beta) + (triangle.n2 * gamma);
            normalOut = normalOut.norm();
        } else {
            normalOut = normal;
        }

        if (triangle.hasTexture) {
            texCoordOut = (triangle.vt0 * alpha) + (triangle.vt1 * beta) + (triangle.vt2 * gamma);
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

    float tx = x - x0;
    float ty = y - y0;

    Color c00 = texture.pixels[y0 * texture.width + x0];
    Color c10 = texture.pixels[y0 * texture.width + x1];
    Color c01 = texture.pixels[y1 * texture.width + x0];
    Color c11 = texture.pixels[y1 * texture.width + x1];

    Color cx0 = c00 * (1 - tx) + c10 * tx;
    Color cx1 = c01 * (1 - tx) + c11 * tx;
    return cx0 * (1 - ty) + cx1 * ty;
}

__device__ bool refract_device(const Vec3d &I, const Vec3d &N, float ior, Vec3d &refractedDir) {
    float cosi = clamp_cuda(I.dot(N), -1.0f, 1.0f);
    float etai = 1.0f;
    float etat = ior;
    Vec3d n = N;
    
    if (cosi < 0) {
        cosi = -cosi;
    } else {
        float temp = etai;
        etai = etat;
        etat = temp;
        n = N * -1;
    }

    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    if (k < 0) return false;
    
    refractedDir = (I * eta + n * (eta * cosi - sqrtf(k))).norm();
    return true;
}

__device__ float fresnelSchlick_device(const Vec3d &I, const Vec3d &N, float ior) {
    float cosi = clamp_cuda(I.dot(N), -1.0f, 1.0f);
    float etai = 1.0f;
    float etat = ior;
    
    if (cosi > 0) {
        float temp = etai;
        etai = etat;
        etat = temp;
    }

    float r0 = (etai - etat) / (etai + etat);
    float f0 = r0 * r0;
    return f0 + (1 - f0) * powf(1 - fabsf(cosi), 5);
}

__device__ float computeShadowFactor_device(const Vec3d &point, const Vec3d &lightDir, 
                                           DeviceSphere* spheres, int numSpheres) {
    Ray shadowRay(point + lightDir * 1e-4f, lightDir);
    float transparency = 1.0f;

    for (int i = 0; i < numSpheres; i++) {
        float t;
        Vec2d texCoord;
        if (intersectRaySphere_device(shadowRay, spheres[i], t, texCoord)) {
            transparency *= (1.0f - spheres[i].material.alpha);
            if (transparency <= 0.01f) return 0.0f;
        }
    }
    return transparency;
}

__device__ bool projectToScreen_device(const Vec3d &point, const Vec3d &eyePos,
                                       const Vec3d &viewDir, const Vec3d &upDir,
                                       float vfov, int width, int height,
                                       int &screenX, int &screenY) {
    Vec3d right = viewDir.cross(upDir).norm();
    Vec3d up = right.cross(viewDir).norm();
    
    Vec3d toPoint = point - eyePos;
    float distance = toPoint.dot(viewDir);
    
    if (distance <= 0) return false;
    
    double ar = (double)width / height;
    double vh = 2.0 * tanf(vfov / 2);
    double vw = vh * ar;
    
    float u = toPoint.dot(right) / distance;
    float v = -toPoint.dot(up) / distance;
    
    screenX = (int)((u / vw + 0.5) * width);
    screenY = (int)((v / vh + 0.5) * height);
    
    return (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height);
}

__device__ Color getSSRColor_device(const Vec3d &reflectDir, const Vec3d &hitPoint,
                                    Color* imageBuffer, int width, int height,
                                    const Vec3d &eyePos, const Vec3d &viewDir, 
                                    const Vec3d &upDir, float vfov,
                                    const Color &bkgcolor) {
    const int MAX_STEPS = 50;
    const float STEP_SIZE = 0.1f;
    
    Vec3d currentPos = hitPoint;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        currentPos = currentPos + reflectDir * STEP_SIZE;
        
        int screenX, screenY;
        if (projectToScreen_device(currentPos, eyePos, viewDir, upDir, vfov, 
                                   width, height, screenX, screenY)) {
            if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
                return imageBuffer[screenY * width + screenX];
            }
        } else {
            break;
        }
    }
    
    return bkgcolor;
}

__device__ Color shade_device(const Vec3d &point, const Vec3d &rayDir, const Vec3d &normal,
                              const Vec2d &texCoord, const MaterialColor &material,
                              DeviceLight* lights, int numLights,
                              DeviceSphere* spheres, int numSpheres,
                              const DeviceTexture &texture, bool hasTexture,
                              const Color &bkgcolor) {
    Color texturec = material.color;
    
    if (hasTexture && texture.pixels) {
        texturec = getTextureColor_device(texture, texCoord.x, texCoord.y);
    }

    Vec3d viewDir = (rayDir * -1).norm();
    Color ambient = texturec * material.ka;
    Color totalDiffuse(0, 0, 0);
    Color totalSpecular(0, 0, 0);

    for (int i = 0; i < numLights; i++) {
        Vec3d L = lights[i].isPoint ? 
                  (lights[i].positionOrdir - point).norm() : 
                  lights[i].positionOrdir.norm() * -1;
        
        Vec3d H = (L + viewDir).norm();
        float df = fmaxf(normal.dot(L), 0.0f);
        
        Color diffuse = texturec * material.kd * df;
        float sf = powf(fmaxf(normal.dot(H), 0.0f), material.shininess);
        Color specular = material.specular * material.ks * sf;
        
        float shadowFactor = computeShadowFactor_device(point, L, spheres, numSpheres);
        
        totalDiffuse = totalDiffuse + diffuse * lights[i].intensity * shadowFactor;
        totalSpecular = totalSpecular + specular * lights[i].intensity * shadowFactor;
    }

    return ambient + totalDiffuse + totalSpecular;
}

__global__ void raytrace_kernel(Color* output, Color* imageBuffer,
                                DeviceSphere* spheres, int numSpheres,
                                DeviceTriangle* triangles, int numTriangles,
                                DeviceLight* lights, int numLights,
                                DeviceTexture* textures,
                                Vec3d eyePos, Vec3d viewDir, Vec3d upDir, float vfov,
                                Vec3d ul, Vec3d delta_h, Vec3d delta_v,
                                int width, int height, Color bkgcolor,
                                bool useSSR, int maxDepth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Calculate ray
    Vec3d vwPosition = ul + delta_h * (float)x + delta_v * (float)y;
    Vec3d rayDir = (vwPosition - eyePos).norm();
    Ray ray(eyePos, rayDir);
    
    // Find closest intersection
    float t_min = 1e30f; // Use large constant instead of FLT_MAX
    int hitType = -1; // -1: none, 0: sphere, 1: triangle
    int hitIndex = -1;
    Vec3d hitNormal;
    Vec2d texCoord;
    
    // Check spheres
    for (int i = 0; i < numSpheres; i++) {
        float t;
        Vec2d tc;
        if (intersectRaySphere_device(ray, spheres[i], t, tc) && t < t_min) {
            t_min = t;
            hitType = 0;
            hitIndex = i;
            texCoord = tc;
            Vec3d hitPoint = ray.getOrigin() + ray.getDirection() * t;
            hitNormal = (hitPoint - spheres[i].center).norm();
        }
    }
    
    // Check triangles
    for (int i = 0; i < numTriangles; i++) {
        float t;
        Vec3d normal;
        Vec2d tc;
        if (intersectRayTriangle_device(ray, t, normal, tc, triangles[i]) && t < t_min) {
            t_min = t;
            hitType = 1;
            hitIndex = i;
            hitNormal = normal;
            texCoord = tc;
        }
    }
    
    // Default to background
    Color finalColor = bkgcolor;
    
    if (hitType >= 0) {
        Vec3d hitPoint = eyePos + rayDir * t_min;
        MaterialColor material;
        DeviceTexture texture;
        bool hasTexture = false;
        
        if (hitType == 0) {
            material = spheres[hitIndex].material;
            hasTexture = spheres[hitIndex].hasTexture;
            if (hasTexture) texture = textures[hitIndex];
        } else {
            material = triangles[hitIndex].material;
            hasTexture = triangles[hitIndex].hasTexture;
            if (hasTexture) texture = textures[numSpheres + hitIndex];
        }
        
        // Local shading
        Color localColor = shade_device(hitPoint, rayDir, hitNormal, texCoord, material,
                                       lights, numLights, spheres, numSpheres,
                                       texture, hasTexture, bkgcolor);
        
        // Reflection
        Color reflectedColor(0, 0, 0);
        if (material.ks > 0 && useSSR) {
            Vec3d reflectDir = (rayDir - hitNormal * 2 * (rayDir.dot(hitNormal))).norm();
            reflectedColor = getSSRColor_device(reflectDir, hitPoint, imageBuffer, 
                                               width, height, eyePos, viewDir, upDir, 
                                               vfov, bkgcolor);
        }
        
        // Simple refraction (single level)
        Color refractedColor(0, 0, 0);
        if (material.alpha < 1.0f) {
            Vec3d refractDir;
            if (refract_device(rayDir, hitNormal, material.ior, refractDir)) {
                // Simple refraction without full recursion
                refractedColor = bkgcolor; // Simplified
            }
        }
        
        float Fr = fresnelSchlick_device(rayDir, hitNormal, material.ior);
        float Ft = 1 - Fr;
        
        finalColor = localColor + reflectedColor * Fr + refractedColor * (1 - material.alpha) * Ft;
    }
    
    output[idx] = finalColor;
}

// Host code
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

    parse(filename, scene);
    int width = (int)scene.camera.w;
    int height = (int)scene.camera.h;
    
    std::cout << "Rendering " << width << "x" << height << " image..." << std::endl;

    // Allocate host memory
    Color* h_output = new Color[width * height];
    Color* h_imageBuffer = new Color[width * height];
    
    // Prepare device data
    std::vector<DeviceSphere> h_spheres;
    std::vector<DeviceTriangle> h_triangles;
    std::vector<DeviceLight> h_lights;
    std::vector<DeviceTexture> h_textures;
    
    // Convert scene objects to device format
    for (const auto& obj : scene.objects) {
        if (auto* sphere = dynamic_cast<Sphere*>(obj.get())) {
            DeviceSphere ds;
            ds.center = sphere->center;
            ds.radius = sphere->radius;
            ds.material = sphere->getColor();
            ds.hasTexture = sphere->hasTexture;
            h_spheres.push_back(ds);
            
            // Handle texture (simplified - would need proper GPU texture upload)
            DeviceTexture dt;
            dt.pixels = nullptr;
            dt.width = 0;
            dt.height = 0;
            h_textures.push_back(dt);
        }
        else if (auto* triangle = dynamic_cast<Triangle*>(obj.get())) {
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
            dt.material = triangle->getColor();
            dt.isSmooth = triangle->isSmooth;
            dt.hasTexture = triangle->hasTexture;
            h_triangles.push_back(dt);
        }
    }
    
    for (const auto& light : scene.lights) {
        DeviceLight dl;
        dl.positionOrdir = light->positionOrdir;
        dl.intensity = light->intensity;
        dl.isPoint = light->isPoint;
        h_lights.push_back(dl);
    }

    // Calculate view parameters
    double ar = (double)width / height;
    double vh = 2.0 * tan(scene.camera.vfov_rad() / 2);
    double vw = vh * ar;
    Vec3d viewDir = scene.camera.viewDir.norm();
    Vec3d right = viewDir.cross(scene.camera.upDir).norm();
    Vec3d up = right.cross(viewDir).norm();
    Vec3d center = scene.camera.eye + viewDir;
    Vec3d ul = center - right * (vw * 0.5) + up * (vh * 0.5);
    Vec3d ur = center + right * (vw * 0.5) + up * (vh * 0.5);
    Vec3d ll = center - right * (vw * 0.5) - up * (vh * 0.5);
    Vec3d delta_h = (ur - ul) / width;
    Vec3d delta_v = (ll - ul) / height;

    // Allocate device memory
    Color *d_output, *d_imageBuffer;
    DeviceSphere* d_spheres;
    DeviceTriangle* d_triangles;
    DeviceLight* d_lights;
    DeviceTexture* d_textures;
    
    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(Color)));
    CUDA_CHECK(cudaMalloc(&d_imageBuffer, width * height * sizeof(Color)));
    CUDA_CHECK(cudaMalloc(&d_spheres, h_spheres.size() * sizeof(DeviceSphere)));
    CUDA_CHECK(cudaMalloc(&d_triangles, h_triangles.size() * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMalloc(&d_lights, h_lights.size() * sizeof(DeviceLight)));
    CUDA_CHECK(cudaMalloc(&d_textures, h_textures.size() * sizeof(DeviceTexture)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(), 
                         h_spheres.size() * sizeof(DeviceSphere), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), 
                         h_triangles.size() * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lights, h_lights.data(), 
                         h_lights.size() * sizeof(DeviceLight), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_textures, h_textures.data(), 
                         h_textures.size() * sizeof(DeviceTexture), cudaMemcpyHostToDevice));

    // Launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Pass 1: Rendering base image..." << std::endl;
    raytrace_kernel<<<gridSize, blockSize>>>(
        d_imageBuffer, nullptr,
        d_spheres, h_spheres.size(),
        d_triangles, h_triangles.size(),
        d_lights, h_lights.size(),
        d_textures,
        scene.camera.eye, viewDir, scene.camera.upDir, scene.camera.vfov_rad(),
        ul, delta_h, delta_v,
        width, height, scene.bkgcolor,
        false, 10
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Pass 2: Rendering with SSR..." << std::endl;
    raytrace_kernel<<<gridSize, blockSize>>>(
        d_output, d_imageBuffer,
        d_spheres, h_spheres.size(),
        d_triangles, h_triangles.size(),
        d_lights, h_lights.size(),
        d_textures,
        scene.camera.eye, viewDir, scene.camera.upDir, scene.camera.vfov_rad(),
        ul, delta_h, delta_v,
        width, height, scene.bkgcolor,
        true, 10
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, width * height * sizeof(Color), 
                         cudaMemcpyDeviceToHost));

    // Write output
    std::string perspective_filename = basename + "_perspective.ppm";
    std::ofstream output(perspective_filename);
    output << "P3\n" << width << " " << height << "\n255\n";
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color c = h_output[y * width + x];
            output << (int)(c.r * 255) << " " 
                   << (int)(c.g * 255) << " " 
                   << (int)(c.b * 255) << " ";
        }
        output << "\n";
    }
    output.close();

    // Cleanup
    delete[] h_output;
    delete[] h_imageBuffer;
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_imageBuffer));
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_lights));
    CUDA_CHECK(cudaFree(d_textures));

    std::cout << "Rendering complete. Image saved as '" << perspective_filename << "'." << std::endl;
    return 0;
}
