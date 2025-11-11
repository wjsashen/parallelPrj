// raycast_cuda.cu - CUDA-accelerated raytracer
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <memory>

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

// Forward declarations for host structures
struct Vec3d;
struct Vec2d;
struct Color;
struct Ray;
struct MaterialColor;
struct Light;
struct Camera;
struct Sphere;
struct Triangle;
struct Objects;
struct Scene;

// Include the parse function declaration
void parse(const std::string& filename, Scene& scene);

// Device-compatible versions of our structures
struct DeviceVec3d {
    float x, y, z;
    __device__ __host__ DeviceVec3d() : x(0), y(0), z(0) {}
    __device__ __host__ DeviceVec3d(float x, float y, float z) : x(x), y(y), z(z) {}
    __device__ DeviceVec3d operator+(const DeviceVec3d& v) const { return DeviceVec3d(x + v.x, y + v.y, z + v.z); }
    __device__ DeviceVec3d operator-(const DeviceVec3d& v) const { return DeviceVec3d(x - v.x, y - v.y, z - v.z); }
    __device__ DeviceVec3d operator*(float s) const { return DeviceVec3d(x * s, y * s, z * s); }
    __device__ float dot(const DeviceVec3d& v) const { return x * v.x + y * v.y + z * v.z; }
    __device__ DeviceVec3d cross(const DeviceVec3d& v) const {
        return DeviceVec3d(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    __device__ DeviceVec3d norm() const {
        float len = length();
        return (len > 0) ? DeviceVec3d(x / len, y / len, z / len) : DeviceVec3d(0, 0, 0);
    }
};

struct DeviceVec2d {
    float x, y;
    __device__ __host__ DeviceVec2d() : x(0), y(0) {}
    __device__ __host__ DeviceVec2d(float x, float y) : x(x), y(y) {}
    __device__ DeviceVec2d operator+(const DeviceVec2d& v) const { return DeviceVec2d(x + v.x, y + v.y); }
    __device__ DeviceVec2d operator*(float s) const { return DeviceVec2d(x * s, y * s); }
};

struct DeviceColor {
    float r, g, b;
    __device__ __host__ DeviceColor() : r(0), g(0), b(0) {}
    __device__ __host__ DeviceColor(float r, float g, float b) : r(r), g(g), b(b) {}
    __device__ DeviceColor operator+(const DeviceColor& c) const { return DeviceColor(r + c.r, g + c.g, b + c.b); }
    __device__ DeviceColor operator*(float s) const { return DeviceColor(r * s, g * s, b * s); }
};

struct DeviceRay {
    float x, y, z, dx, dy, dz;
    __device__ DeviceRay() : x(0), y(0), z(0), dx(0), dy(0), dz(1) {}
    __device__ DeviceRay(DeviceVec3d origin, DeviceVec3d direction) 
        : x(origin.x), y(origin.y), z(origin.z),
          dx(direction.x), dy(direction.y), dz(direction.z) {}
    __device__ DeviceVec3d getOrigin() const { return DeviceVec3d(x, y, z); }
    __device__ DeviceVec3d getDirection() const { return DeviceVec3d(dx, dy, dz); }
};

struct DeviceMaterial {
    DeviceColor color, specular;
    float ka, kd, ks, shininess, alpha, ior;
    __device__ __host__ DeviceMaterial() 
        : color(1, 1, 1), specular(1, 1, 1),
          ka(0.1f), kd(0.7f), ks(0.3f), shininess(32.0f), alpha(1.0f), ior(1.5f) {}
};

struct DeviceLight {
    DeviceVec3d positionOrdir;
    float intensity;
    bool isPoint;
};

struct DeviceSphere {
    DeviceVec3d center;
    float radius;
    DeviceMaterial material;
};

struct DeviceTriangle {
    DeviceVec3d v0, v1, v2;
    DeviceVec3d n0, n1, n2;
    DeviceVec2d vt0, vt1, vt2;
    DeviceMaterial material;
    bool isSmooth;
};

struct DeviceScene {
    DeviceLight* lights;
    int numLights;
    DeviceSphere* spheres;
    int numSpheres;
    DeviceTriangle* triangles;
    int numTriangles;
    DeviceVec3d eyePos;
    DeviceColor bkgcolor;
};

// Now include the actual header with host structures
#include "raycast.h"
#include "parse.h"

// Conversion helpers
DeviceVec3d toDevice(const Vec3d& v) {
    return DeviceVec3d(v.x, v.y, v.z);
}

DeviceVec2d toDevice(const Vec2d& v) {
    return DeviceVec2d(v.x, v.y);
}

DeviceColor toDevice(const Color& c) {
    return DeviceColor(c.r, c.g, c.b);
}

DeviceMaterial toDevice(const MaterialColor& m) {
    DeviceMaterial dm;
    dm.color = toDevice(m.color);
    dm.specular = toDevice(m.specular);
    dm.ka = m.ka;
    dm.kd = m.kd;
    dm.ks = m.ks;
    dm.shininess = m.shininess;
    dm.alpha = m.alpha;
    dm.ior = m.ior;
    return dm;
}

// Device intersection functions
__device__ DeviceVec2d calculateSphereUV(const DeviceVec3d &normal) {
    float f = acosf(normal.z);
    float q = atan2f(normal.y, normal.x);
    float u = (q >= 0) ? (q / (2 * M_PI)) : ((q + 2 * M_PI) / (2 * M_PI));
    float v = f / M_PI;
    return DeviceVec2d(u, v);
}

__device__ bool intersectRaySphere(const DeviceRay &ray, const DeviceSphere &sphere, 
                                   float &t, DeviceVec2d &texCoordOut) {
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
    
    DeviceVec3d intersectionPoint = ray.getOrigin() + ray.getDirection() * t;
    DeviceVec3d normal = (intersectionPoint - sphere.center).norm();
    texCoordOut = calculateSphereUV(normal);
    return true;
}

__device__ bool intersectRayTriangle(const DeviceRay &ray, float &t, DeviceVec3d &normalOut, 
                                     DeviceVec2d &texCoordOut, const DeviceTriangle &triangle) {
    DeviceVec3d edge1 = triangle.v1 - triangle.v0;
    DeviceVec3d edge2 = triangle.v2 - triangle.v0;
    DeviceVec3d cr = edge1.cross(edge2);
    DeviceVec3d normal = cr.norm();

    float denominator = cr.dot(ray.getDirection());
    if (fabsf(denominator) < 1e-8f) return false;

    float d = -cr.dot(triangle.v0);
    t = -(cr.dot(ray.getOrigin()) + d) / denominator;
    if (t < 0) return false;

    DeviceVec3d P = ray.getOrigin() + ray.getDirection() * t;
    DeviceVec3d vP = P - triangle.v0;
    
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
        texCoordOut = (triangle.vt0 * alpha) + (triangle.vt1 * beta) + (triangle.vt2 * gamma);
        return true;
    }
    return false;
}

__device__ float computeShadowFactor(const DeviceScene &scene, const DeviceVec3d &point, 
                                     const DeviceVec3d &lightDir, const DeviceVec3d &normal) {
    DeviceRay shadowRay(point + lightDir * 1e-4f, lightDir);
    float transparency = 1.0f;

    for (int i = 0; i < scene.numSpheres; i++) {
        float t;
        DeviceVec2d texCoord;
        if (intersectRaySphere(shadowRay, scene.spheres[i], t, texCoord)) {
            transparency *= (1.0f - scene.spheres[i].material.alpha);
            if (transparency <= 0.01f) return 0.0f;
        }
    }
    return transparency;
}

__device__ DeviceColor shade(const DeviceScene &scene, float t_min, DeviceVec3d rayDir, 
                             float t, const DeviceMaterial &mt, const DeviceVec3d &normal, 
                             const DeviceVec2d &texCoord, DeviceColor texturec) {
    DeviceVec3d point = scene.eyePos + rayDir * t;
    DeviceVec3d N = normal;
    DeviceVec3d viewDir = (rayDir * -1.0f).norm();
    
    DeviceColor ambient = texturec * mt.ka;
    DeviceColor totalDiffuse(0, 0, 0);
    DeviceColor totalSpecular(0, 0, 0);

    for (int i = 0; i < scene.numLights; i++) {
        const DeviceLight &light = scene.lights[i];
        DeviceVec3d L = light.isPoint ? (light.positionOrdir - point).norm() 
                                      : light.positionOrdir.norm() * -1.0f;
        
        float fatt = 1.0f;
        DeviceVec3d H = (L + viewDir).norm();
        float df = fmaxf(N.dot(L), 0.0f);

        DeviceColor diffuse = texturec * mt.kd * df;
        float sf = powf(fmaxf(N.dot(H), 0.0f), mt.shininess);
        DeviceColor specular = mt.specular * mt.ks * sf;

        float shadowFactor = computeShadowFactor(scene, point, L, N);

        totalDiffuse = totalDiffuse + diffuse * light.intensity * fatt * shadowFactor;
        totalSpecular = totalSpecular + specular * light.intensity * fatt * shadowFactor;
    }

    return ambient + totalDiffuse + totalSpecular;
}

__device__ bool refract(const DeviceVec3d &I, const DeviceVec3d &N, float ior, DeviceVec3d &refractedDir) {
    float cosi = fminf(fmaxf(I.dot(N), -1.0f), 1.0f);
    float float1 = 1.0f;
    DeviceVec3d n = N;
    
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

__device__ float frCal(const DeviceVec3d &I, const DeviceVec3d &N, float ior) {
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

__device__ DeviceColor trace_recursive(const DeviceScene &scene, const DeviceRay &ray, int depth) {
    if (depth <= 0) return scene.bkgcolor;

    float t_min = INFINITY;
    int hitSphere = -1;
    int hitTriangle = -1;
    DeviceVec3d hitNormal;
    DeviceVec2d texCoord;
    float t_hit;

    // Find closest sphere
    for (int i = 0; i < scene.numSpheres; i++) {
        float t;
        DeviceVec2d tc;
        if (intersectRaySphere(ray, scene.spheres[i], t, tc) && t > 0.001f && t < t_min) {
            t_min = t;
            hitSphere = i;
            hitTriangle = -1;
            texCoord = tc;
            DeviceVec3d intersectionPoint = ray.getOrigin() + ray.getDirection() * t;
            hitNormal = (intersectionPoint - scene.spheres[i].center).norm();
            t_hit = t;
        }
    }

    // Find closest triangle
    for (int i = 0; i < scene.numTriangles; i++) {
        float t;
        DeviceVec3d normal;
        DeviceVec2d tc;
        if (intersectRayTriangle(ray, t, normal, tc, scene.triangles[i]) && t > 0.001f && t < t_min) {
            t_min = t;
            hitTriangle = i;
            hitSphere = -1;
            texCoord = tc;
            hitNormal = normal;
            t_hit = t;
        }
    }

    if (hitSphere == -1 && hitTriangle == -1) return scene.bkgcolor;

    DeviceMaterial mt;
    DeviceColor texturec;
    
    if (hitSphere >= 0) {
        mt = scene.spheres[hitSphere].material;
        texturec = mt.color;
    } else {
        mt = scene.triangles[hitTriangle].material;
        texturec = mt.color;
    }

    DeviceColor localColor = shade(scene, t_min, ray.getDirection(), t_hit, mt, hitNormal, texCoord, texturec);

    // Stop recursion at depth 1 to avoid stack overflow
    if (depth <= 1) return localColor;

    DeviceColor reflectedColor(0, 0, 0);
    DeviceColor refractedColor(0, 0, 0);
    DeviceVec3d hitPoint = ray.getOrigin() + ray.getDirection() * t_hit;
    float epsilon = 1e-3f;  // Increased epsilon

    // Reflection - only if significant
    if (mt.ks > 0.01f && depth > 1) {
        DeviceVec3d I = ray.getDirection();
        DeviceVec3d reflectDir = (I - hitNormal * 2.0f * (I.dot(hitNormal))).norm();
        DeviceRay reflectedRay(hitPoint + reflectDir * epsilon, reflectDir);
        reflectedColor = trace_recursive(scene, reflectedRay, depth - 1);
    }

    // Refraction - only if significant transparency
    if (mt.alpha < 0.99f && depth > 1) {
        DeviceVec3d refractDir;
        if (refract(ray.getDirection(), hitNormal, mt.ior, refractDir)) {
            DeviceRay refractedRay(hitPoint + refractDir * epsilon, refractDir);
            refractedColor = trace_recursive(scene, refractedRay, depth - 1);
        }
    }

    float Fr = frCal(ray.getDirection(), hitNormal, mt.ior);
    float Ft = 1.0f - Fr;

    return localColor + reflectedColor * Fr + refractedColor * (1.0f - mt.alpha) * Ft;
}

// CUDA kernel
__global__ void renderKernel(DeviceColor* output, int width, int height, 
                             DeviceVec3d ul, DeviceVec3d delta_h, DeviceVec3d delta_v, 
                             DeviceScene scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    DeviceVec3d vwPosition = ul + delta_h * (float)x + delta_v * (float)y;
    DeviceVec3d rayDir = (vwPosition - scene.eyePos).norm();
    DeviceRay ray(scene.eyePos, rayDir);
    
    DeviceColor pixel = trace_recursive(scene, ray, 3);  // Reduced depth from 10 to 3
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
    const char* cfilename = filename.c_str();
    std::ifstream file(cfilename);

    if (!file) {
        std::cerr << "Error: Could not open file '" << filename << "'." << std::endl;
        return 1;
    } else if (!isValidTxtFile(filename)) {
        std::cerr << "Error: Invalid file format." << std::endl;
        return 1;
    }
    file.close();

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
    Vec3d viewDir_h = scene.camera.viewDir.norm();
    Vec3d right_h = viewDir_h.cross(scene.camera.upDir).norm();
    Vec3d up_h = right_h.cross(viewDir_h).norm();
    Vec3d center_h = scene.camera.eye + viewDir_h;
    Vec3d ul_h = center_h - right_h * (vw * 0.5) + up_h * (vh * 0.5);
    Vec3d ur_h = center_h + right_h * (vw * 0.5) + up_h * (vh * 0.5);
    Vec3d ll_h = center_h - right_h * (vw * 0.5) - up_h * (vh * 0.5);
    Vec3d delta_h_h = (ur_h - ul_h) / width;
    Vec3d delta_v_h = (ll_h - ul_h) / height;

    DeviceVec3d ul = toDevice(ul_h);
    DeviceVec3d delta_h = toDevice(delta_h_h);
    DeviceVec3d delta_v = toDevice(delta_v_h);

    // Prepare device scene
    DeviceScene d_scene;
    d_scene.eyePos = toDevice(scene.camera.eye);
    d_scene.bkgcolor = toDevice(scene.bkgcolor);

    // Copy lights
    std::vector<DeviceLight> h_lights;
    for (size_t i = 0; i < scene.lights.size(); i++) {
        DeviceLight dl;
        dl.positionOrdir = toDevice(scene.lights[i]->positionOrdir);
        dl.intensity = scene.lights[i]->intensity;
        dl.isPoint = scene.lights[i]->isPoint;
        h_lights.push_back(dl);
    }
    d_scene.numLights = h_lights.size();
    CUDA_CHECK(cudaMalloc(&d_scene.lights, h_lights.size() * sizeof(DeviceLight)));
    CUDA_CHECK(cudaMemcpy(d_scene.lights, h_lights.data(), 
                         h_lights.size() * sizeof(DeviceLight), cudaMemcpyHostToDevice));

    // Copy spheres and triangles
    std::vector<DeviceSphere> h_spheres;
    std::vector<DeviceTriangle> h_triangles;
    
    for (size_t i = 0; i < scene.objects.size(); i++) {
        Sphere* sphere = dynamic_cast<Sphere*>(scene.objects[i].get());
        if (sphere) {
            DeviceSphere ds;
            ds.center = toDevice(sphere->center);
            ds.radius = sphere->radius;
            ds.material = toDevice(sphere->material);
            h_spheres.push_back(ds);
        } else {
            Triangle* triangle = dynamic_cast<Triangle*>(scene.objects[i].get());
            if (triangle) {
                DeviceTriangle dt;
                dt.v0 = toDevice(triangle->v0);
                dt.v1 = toDevice(triangle->v1);
                dt.v2 = toDevice(triangle->v2);
                dt.n0 = toDevice(triangle->n0);
                dt.n1 = toDevice(triangle->n1);
                dt.n2 = toDevice(triangle->n2);
                dt.vt0 = toDevice(triangle->vt0);
                dt.vt1 = toDevice(triangle->vt1);
                dt.vt2 = toDevice(triangle->vt2);
                dt.material = toDevice(triangle->material);
                dt.isSmooth = triangle->isSmooth;
                h_triangles.push_back(dt);
            }
        }
    }

    d_scene.numSpheres = h_spheres.size();
    d_scene.numTriangles = h_triangles.size();
    
    std::cout << "Scene info: " << h_spheres.size() << " spheres, " 
              << h_triangles.size() << " triangles, " 
              << h_lights.size() << " lights" << std::endl;
    
    if (h_lights.size() > 0) {
        std::cout << "Light 0: pos(" << h_lights[0].positionOrdir.x << ", " 
                  << h_lights[0].positionOrdir.y << ", " << h_lights[0].positionOrdir.z 
                  << "), intensity=" << h_lights[0].intensity 
                  << ", isPoint=" << h_lights[0].isPoint << std::endl;
    }
    
    if (h_spheres.size() > 0) {
        std::cout << "Sphere 0: center(" << h_spheres[0].center.x << ", "
                  << h_spheres[0].center.y << ", " << h_spheres[0].center.z
                  << "), radius=" << h_spheres[0].radius << std::endl;
        std::cout << "Material: color(" << h_spheres[0].material.color.r << ", "
                  << h_spheres[0].material.color.g << ", " << h_spheres[0].material.color.b
                  << "), ka=" << h_spheres[0].material.ka 
                  << ", kd=" << h_spheres[0].material.kd << std::endl;
    }
    
    CUDA_CHECK(cudaMalloc(&d_scene.spheres, h_spheres.size() * sizeof(DeviceSphere)));
    CUDA_CHECK(cudaMemcpy(d_scene.spheres, h_spheres.data(), 
                         h_spheres.size() * sizeof(DeviceSphere), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&d_scene.triangles, h_triangles.size() * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMemcpy(d_scene.triangles, h_triangles.data(), 
                         h_triangles.size() * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));

    // Allocate output
    DeviceColor* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(DeviceColor)));

    // Increase stack size for recursive raytracing
    size_t stackSize = 8192;  // 8KB per thread
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Rendering " << width << "x" << height << " image on GPU..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    renderKernel<<<gridSize, blockSize>>>(d_output, width, height, ul, delta_h, delta_v, d_scene);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Rendering took " << milliseconds / 1000.0f << " seconds" << std::endl;

    CUDA_CHECK(cudaGetLastError());

    // Copy results
    std::vector<DeviceColor> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 
                         width * height * sizeof(DeviceColor), cudaMemcpyDeviceToHost));

    // Write output
    std::string perspective_filename = basename + "_perspective.ppm";
    const char* outfile = perspective_filename.c_str();
    std::ofstream output(outfile);
    output << "P3\n" << width << " " << height << "\n255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            DeviceColor &pixel = h_output[y * width + x];
            // Clamp values to [0, 1]
            float r = fminf(fmaxf(pixel.r, 0.0f), 1.0f);
            float g = fminf(fmaxf(pixel.g, 0.0f), 1.0f);
            float b = fminf(fmaxf(pixel.b, 0.0f), 1.0f);
            output << (int)(r * 255) << " " 
                   << (int)(g * 255) << " " 
                   << (int)(b * 255) << " ";
        }
        output << "\n";
    }
    output.close();

    // Cleanup
    cudaFree(d_output);
    cudaFree(d_scene.lights);
    cudaFree(d_scene.spheres);
    cudaFree(d_scene.triangles);

    std::cout << "Complete! Image saved as '" << perspective_filename << "'." << std::endl;
    return 0;
}