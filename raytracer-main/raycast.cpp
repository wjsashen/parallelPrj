#include "raycast.h"
#include "parse.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::vector<Vec3d> Objects::vertices;

// Device-compatible structures
struct DeviceLight {
    Vec3d positionOrdir;
    float intensity;
    bool isPoint;
};

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

struct DeviceTexture {
    Color* pixels;
    int width;
    int height;
};

struct DeviceScene {
    DeviceLight* lights;
    int numLights;
    DeviceSphere* spheres;
    int numSpheres;
    DeviceTriangle* triangles;
    int numTriangles;
    Vec3d eyePos;
    Color bkgcolor;
};

// Device functions
__device__ Vec2d calculateSphereUV(const Vec3d &normal) {
    float f = acosf(normal.z);
    float q = atan2f(normal.y, normal.x);
    float u = (q >= 0) ? (q / (2 * M_PI)) : ((q + 2 * M_PI) / (2 * M_PI));
    float v = f / M_PI;
    return Vec2d(u, v);
}

__device__ bool intersectRaySphere(const Ray &ray, const DeviceSphere &sphere, 
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
    texCoordOut = calculateSphereUV(normal);
    return true;
}

__device__ bool intersectRayTriangle(const Ray &ray, float &t, Vec3d &normalOut, 
                                     Vec2d &texCoordOut, const DeviceTriangle &triangle) {
    Vec3d edge1 = triangle.v1 - triangle.v0;
    Vec3d edge2 = triangle.v2 - triangle.v0;
    Vec3d cr = edge1.cross(edge2);
    Vec3d normal = cr.norm();

    float denominator = cr.dot(ray.getDirection());
    if (fabsf(denominator) < 1e-8f) return false;

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

__device__ float computeShadowFactor(const DeviceScene &scene, const Vec3d &point, 
                                     const Vec3d &lightDir, const Vec3d &normal) {
    Ray shadowRay(point + lightDir * 1e-4f, lightDir);
    float transparency = 1.0f;

    for (int i = 0; i < scene.numSpheres; i++) {
        float t;
        Vec2d texCoord;
        if (intersectRaySphere(shadowRay, scene.spheres[i], t, texCoord)) {
            transparency *= (1.0f - scene.spheres[i].material.alpha);
            if (transparency <= 0.01f) return 0.0f;
        }
    }

    return transparency;
}

__device__ Color shade(const DeviceScene &scene, float t_min, Vec3d rayDir, 
                       float t, const MaterialColor &mt, const Vec3d &normal, 
                       const Vec2d &texCoord, Color texturec) {
    Vec3d point = scene.eyePos + rayDir * t;
    Vec3d N = normal;
    Vec3d viewDir = (rayDir * -1.0f).norm();
    
    Color ambient = texturec * mt.ka;
    Color totalDiffuse(0, 0, 0);
    Color totalSpecular(0, 0, 0);

    for (int i = 0; i < scene.numLights; i++) {
        const DeviceLight &light = scene.lights[i];
        Vec3d L = light.isPoint ? (light.positionOrdir - point).norm() 
                                : light.positionOrdir.norm() * -1.0f;
        
        float fatt = 1.0f;
        Vec3d H = (L + viewDir).norm();
        float df = fmaxf(N.dot(L), 0.0f);

        Color diffuse = texturec * mt.kd * df;
        float sf = powf(fmaxf(N.dot(H), 0.0f), mt.shininess);
        Color specular = mt.specular * mt.ks * sf;

        float shadowFactor = computeShadowFactor(scene, point, L, N);

        totalDiffuse = totalDiffuse + diffuse * light.intensity * fatt * shadowFactor;
        totalSpecular = totalSpecular + specular * light.intensity * fatt * shadowFactor;
    }

    return ambient + totalDiffuse + totalSpecular;
}

__device__ bool refract(const Vec3d &I, const Vec3d &N, float ior, Vec3d &refractedDir) {
    float cosi = fminf(fmaxf(I.dot(N), -1.0f), 1.0f);
    float float1 = 1.0f;
    Vec3d n = N;
    
    if (cosi < 0) {
        cosi = -cosi;
    } else {
        float temp = float1;
        float1 = ior;
        ior = temp;
        n = N * -1.0f;
    }
    
    float eta = float1 / ior;
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0) return false;
    
    refractedDir = (I * eta + n * (eta * cosi - sqrtf(k))).norm();
    return true;
}

__device__ float frCal(const Vec3d &I, const Vec3d &N, float ior) {
    float cosi = fminf(fmaxf(I.dot(N), -1.0f), 1.0f);
    float float1 = 1.0f;
    
    if (cosi > 0) {
        float temp = float1;
        float1 = ior;
        ior = temp;
    }

    float r0 = (1.0f - ior) / (1.0f + ior);
    float f0 = r0 * r0;
    return f0 + (1.0f - f0) * powf(1.0f - fabsf(cosi), 5.0f);
}

__device__ Color trace_recursive(const DeviceScene &scene, const Ray &ray, int depth) {
    if (depth <= 0) return scene.bkgcolor;

    float t_min = INFINITY;
    int hitSphere = -1;
    int hitTriangle = -1;
    Vec3d hitNormal;
    Vec2d texCoord;
    float t_hit;

    // Find closest sphere intersection
    for (int i = 0; i < scene.numSpheres; i++) {
        float t;
        Vec2d tc;
        if (intersectRaySphere(ray, scene.spheres[i], t, tc) && t < t_min) {
            t_min = t;
            hitSphere = i;
            hitTriangle = -1;
            texCoord = tc;
            Vec3d intersectionPoint = ray.getOrigin() + ray.getDirection() * t;
            hitNormal = (intersectionPoint - scene.spheres[i].center).norm();
            t_hit = t;
        }
    }

    // Find closest triangle intersection
    for (int i = 0; i < scene.numTriangles; i++) {
        float t;
        Vec3d normal;
        Vec2d tc;
        if (intersectRayTriangle(ray, t, normal, tc, scene.triangles[i]) && t < t_min) {
            t_min = t;
            hitTriangle = i;
            hitSphere = -1;
            texCoord = tc;
            hitNormal = normal;
            t_hit = t;
        }
    }

    if (hitSphere == -1 && hitTriangle == -1) return scene.bkgcolor;

    MaterialColor mt;
    Color texturec;
    
    if (hitSphere >= 0) {
        mt = scene.spheres[hitSphere].material;
        texturec = mt.color;
    } else {
        mt = scene.triangles[hitTriangle].material;
        texturec = mt.color;
    }

    Color localColor = shade(scene, t_min, ray.getDirection(), t_hit, mt, hitNormal, texCoord, texturec);

    if (depth <= 1) return localColor;

    Color reflectedColor(0, 0, 0);
    Color refractedColor(0, 0, 0);
    Vec3d hitPoint = ray.getOrigin() + ray.getDirection() * t_hit;
    float epsilon = 1e-4f * t_hit;

    // Reflection
    if (mt.ks > 0) {
        Vec3d I = ray.getDirection();
        Vec3d reflectDir = (I - hitNormal * 2.0f * (I.dot(hitNormal))).norm();
        Ray reflectedRay(hitPoint + reflectDir * epsilon, reflectDir);
        reflectedColor = trace_recursive(scene, reflectedRay, depth - 1);
    }

    // Refraction
    if (mt.alpha < 1.0f) {
        Vec3d refractDir;
        if (refract(ray.getDirection(), hitNormal, mt.ior, refractDir)) {
            Ray refractedRay(hitPoint + refractDir * epsilon, refractDir);
            refractedColor = trace_recursive(scene, refractedRay, depth - 1);
        }
    }

    float Fr = frCal(ray.getDirection(), hitNormal, mt.ior);
    float Ft = 1.0f - Fr;

    return localColor + reflectedColor * Fr + refractedColor * (1.0f - mt.alpha) * Ft;
}

// CUDA kernel for parallel rendering
__global__ void renderKernel(Color* output, int width, int height, 
                             Vec3d ul, Vec3d delta_h, Vec3d delta_v, 
                             DeviceScene scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    Vec3d vwPosition = ul + delta_h * (float)x + delta_v * (float)y;
    Vec3d rayDir = (vwPosition - scene.eyePos).norm();
    Ray ray(scene.eyePos, rayDir);
    
    Color pixel = trace_recursive(scene, ray, 10);
    output[y * width + x] = pixel;
}

bool isValidTxtFile(const std::string &filename) {
    return filename.size() >= 4 && filename.substr(filename.size() - 4) == ".txt";
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
    } else if (!isValidTxtFile(filename)) {
        std::cerr << "Error: Invalid file format. Please provide a .txt file." << std::endl;
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
    double ar = (double)width / height;

    // Setup view window
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

    // Prepare device scene
    DeviceScene d_scene;
    d_scene.eyePos = scene.camera.eye;
    d_scene.bkgcolor = scene.bkgcolor;

    // Copy lights
    std::vector<DeviceLight> h_lights;
    for (const auto &light : scene.lights) {
        DeviceLight dl;
        dl.positionOrdir = light->positionOrdir;
        dl.intensity = light->intensity;
        dl.isPoint = light->isPoint;
        h_lights.push_back(dl);
    }
    d_scene.numLights = h_lights.size();
    CUDA_CHECK(cudaMalloc(&d_scene.lights, h_lights.size() * sizeof(DeviceLight)));
    CUDA_CHECK(cudaMemcpy(d_scene.lights, h_lights.data(), 
                         h_lights.size() * sizeof(DeviceLight), cudaMemcpyHostToDevice));

    // Copy spheres and triangles
    std::vector<DeviceSphere> h_spheres;
    std::vector<DeviceTriangle> h_triangles;
    
    for (const auto &obj : scene.objects) {
        if (auto *sphere = dynamic_cast<Sphere*>(obj.get())) {
            DeviceSphere ds;
            ds.center = sphere->center;
            ds.radius = sphere->radius;
            ds.material = sphere->getColor();
            ds.hasTexture = sphere->hasTexture;
            h_spheres.push_back(ds);
        } else if (auto *triangle = dynamic_cast<Triangle*>(obj.get())) {
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

    d_scene.numSpheres = h_spheres.size();
    d_scene.numTriangles = h_triangles.size();
    
    CUDA_CHECK(cudaMalloc(&d_scene.spheres, h_spheres.size() * sizeof(DeviceSphere)));
    CUDA_CHECK(cudaMemcpy(d_scene.spheres, h_spheres.data(), 
                         h_spheres.size() * sizeof(DeviceSphere), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&d_scene.triangles, h_triangles.size() * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMemcpy(d_scene.triangles, h_triangles.data(), 
                         h_triangles.size() * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));

    // Allocate output buffer
    Color* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(Color)));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Rendering " << width << "x" << height << " image..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    renderKernel<<<gridSize, blockSize>>>(d_output, width, height, ul, delta_h, delta_v, d_scene);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Rendering took " << milliseconds / 1000.0f << " seconds" << std::endl;

    CUDA_CHECK(cudaGetLastError());

    // Copy results back
    std::vector<Color> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 
                         width * height * sizeof(Color), cudaMemcpyDeviceToHost));

    // Write output
    std::string perspective_filename = basename + "_perspective.ppm";
    std::ofstream output(perspective_filename);
    output << "P3\n" << width << " " << height << "\n255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color &pixel = h_output[y * width + x];
            output << (int)(pixel.r * 255) << " " 
                   << (int)(pixel.g * 255) << " " 
                   << (int)(pixel.b * 255) << " ";
        }
        output << "\n";
    }
    output.close();

    // Cleanup
    cudaFree(d_output);
    cudaFree(d_scene.lights);
    cudaFree(d_scene.spheres);
    cudaFree(d_scene.triangles);

    std::cout << "Rendering complete. Image saved as '" << perspective_filename << "'." << std::endl;
    return 0;
}