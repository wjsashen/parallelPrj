#include "raycast.h"
#include "parse.h"
#include "raycast.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

std::vector<Vec3d> Objects::vertices;

// Custom clamp function for C++11 compatibility
template<typename T>
T clamp(T value, T min, T max) {
    return std::max(min, std::min(value, max));
}

// Global image buffer for SSR
std::vector<std::vector<Color>> g_imageBuffer;
int g_width = 0;
int g_height = 0;
Camera g_camera;

Vec2d calculateSphereUV(const Vec3d &normal) {
  float f = acos(normal.z);  // latitude angle
  float q = atan2(normal.y, normal.x); // longitude angle
  float u = (q >= 0) ? (q / (2 * M_PI)) : ((q + 2 * M_PI) / (2 * M_PI));
  float v = f / M_PI; // Normalize latitude to [0, 1]

  return Vec2d(u, v);
}

bool intersectRaySphere(const Ray &ray, const Sphere &sphere, float &t,
                        Vec2d &texCoordOut) {
  float cx = sphere.center.x, cy = sphere.center.y, cz = sphere.center.z;
  float r = sphere.radius;

  float a = ray.dx * ray.dx + ray.dy * ray.dy + ray.dz * ray.dz;
  float b = 2 * (ray.dx * (ray.x - cx) + ray.dy * (ray.y - cy) +
                 ray.dz * (ray.z - cz));
  float c = (ray.x - cx) * (ray.x - cx) + (ray.y - cy) * (ray.y - cy) +
            (ray.z - cz) * (ray.z - cz) - r * r;

  float d = b * b - 4 * a * c;
  if (d < 0)
    return false;

  float t1 = (-b - sqrt(d)) / (2 * a);
  float t2 = (-b + sqrt(d)) / (2 * a);

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

Color getTextureColor1(const Texture &texture, float u, float v) {
  if (texture.pixels.empty()) {
    std::cout << "Texture is empty" << std::endl;
    return Color(1.0f, 1.0f, 1.0f);
  }

  u = std::fmod(std::fmod(u, 1.0f) + 1.0f, 1.0f);
  v = std::fmod(std::fmod(v, 1.0f) + 1.0f, 1.0f);

  int x = clamp(int(std::round(u * (texture.width - 1))), 0, texture.width - 1);
  int y = clamp(int(std::round((1.0f - v) * (texture.height - 1))), 0, texture.height - 1);

  return texture.getPixel(x, y); 
}

Color getTextureColorBilinear(const Texture &texture, float u, float v) {
  if (texture.pixels.empty() || texture.width == 0 || texture.height == 0) {
      return Color(1.0f, 1.0f, 1.0f);
  }

  u = std::fmod(std::fmod(u, 1.0f) + 1.0f, 1.0f);
  v = std::fmod(std::fmod(v, 1.0f) + 1.0f, 1.0f);

  if (texture.width <= 0 || texture.height <= 0) {
      return Color(1.0f, 1.0f, 1.0f);
  }

  float x = u * texture.width - 0.5f;
  float y = v * texture.height - 0.5f;

  int x0 = clamp(int(floor(x)), 0, texture.width - 1);
  int x1 = clamp(x0 + 1, 0, texture.width - 1);
  int y0 = clamp(int(floor(y)), 0, texture.height - 1);
  int y1 = clamp(y0 + 1, 0, texture.height - 1);

  float tx = x - x0;
  float ty = y - y0;

  if (x0 < 0 || x1 < 0 || y0 < 0 || y1 < 0 ||
      x0 >= texture.width || x1 >= texture.width ||
      y0 >= texture.height || y1 >= texture.height) {
      return Color(1.0f, 1.0f, 1.0f);
  }

  Color c00 = texture.getPixel(x0, y0);
  Color c10 = texture.getPixel(x1, y0);
  Color c01 = texture.getPixel(x0, y1);
  Color c11 = texture.getPixel(x1, y1);

  Color cx0 = c00 * (1 - tx) + c10 * tx;
  Color cx1 = c01 * (1 - tx) + c11 * tx;
  return cx0 * (1 - ty) + cx1 * ty;
}

bool intersectRayTriangle(const Ray &ray, float &t,
                          Vec3d &normalOut, Vec2d &texCoordOut,
                          const Triangle &triangle) {
  Vec3d edge1 = triangle.v1 - triangle.v0;
  Vec3d edge2 = triangle.v2 - triangle.v0;
  Vec3d cr = edge1.cross(edge2);
  Vec3d normal = cr.norm();

  float denominator = cr.dot(ray.getDirection());
  if (fabs(denominator) < 0)
    return false;

  float d = -cr.dot(triangle.v0);
  t = -(cr.dot(ray.getOrigin()) + d) / denominator;
  if (t < 0)
    return false;

  Vec3d P = ray.getOrigin() + ray.getDirection() * t;

  Vec3d vP = P - triangle.v0;
  float det = edge1.dot(edge1) * edge2.dot(edge2) - edge1.dot(edge2) * edge1.dot(edge2);
  float beta =(vP.dot(edge1) * edge2.dot(edge2) - vP.dot(edge2) * edge1.dot(edge2)) /det;
  float gamma =
      (edge1.dot(edge1) * vP.dot(edge2) - edge1.dot(edge2) * vP.dot(edge1)) / det;
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

bool intersectRaySphereSh(const Ray& ray, const Sphere& sphere, float& t) {
      float cx = sphere.center.x, cy = sphere.center.y, cz = sphere.center.z;
      float r = sphere.radius;
      float a = ray.dx * ray.dx + ray.dy * ray.dy + ray.dz * ray.dz;
      float b = 2 * (ray.dx * (ray.x - cx) + ray.dy * (ray.y - cy) + ray.dz * (ray.z - cz));
      float c = (ray.x - cx) * (ray.x - cx) + (ray.y - cy) * (ray.y - cy) + (ray.z - cz) * (ray.z - cz) - r * r;
  
      float d = b * b - 4 * a * c;
      if (d < 0) return false; 
  
      float t1 = (-b - sqrt(d)) / (2 * a);
      float t2 = (-b + sqrt(d)) / (2 * a);
  
      if (t1 > 0) {
          t = t1;
      } else if (t2 > 0) {
          t = t2;
      } else {
          return false; 
      }
      return true;
}

bool isInShadow(const Vec3d& point, const Vec3d& L, const Scene& scene, Vec3d N) {
      Ray shadowRay(point + L * 0.001, L);
  
      for (const auto& obj : scene.objects) {
          Sphere* sphere = dynamic_cast<Sphere*>(obj.get());
          if (sphere) {
              float t;
              if (intersectRaySphereSh(shadowRay, *sphere, t) && t > 0.001) {
                  return true;
              }
          }
          else if (Triangle* triangle = dynamic_cast<Triangle*>(obj.get())) {
            float t;
            Vec3d N;
            Vec2d dummyTexCoord;
            if (intersectRayTriangle(shadowRay, t,
              N, dummyTexCoord,
              *triangle) ) {
          return true;
      }
        }
    }
      
      return false;
}

bool refract(const Vec3d &I, const Vec3d &N, float ior, Vec3d &refractedDir) {
    float cosi = clamp(I.dot(N), -1.0f, 1.0f);
    float float1 = 1;
    Vec3d n = N;
    if (cosi < 0) {
        cosi = -cosi;
    } else {
        std::swap(float1, ior);
        n = N*-1;
    }
    float eta = float1 / ior;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    if (k < 0) return false;
    refractedDir = (I * eta + n * (eta * cosi - sqrtf(k))).norm();
    return true;
}

float frCal(const Vec3d &I, const Vec3d &N, float ior) {
  float cosi = clamp(I.dot(N), -1.0f, 1.0f);
  float float1 = 1.0f;
  if (cosi > 0) std::swap(float1, ior);

  float r0 = (1.0f - ior) / (1.0f + ior);
  float f0 = r0 * r0;
  return f0 + (1 - f0) * pow(1 - std::abs(cosi), 5);
}

float computeShadowFactor(const Scene &scene, const Vec3d &point, const Vec3d &lightDir, const Vec3d &normal) {
  Ray shadowRay(point + lightDir * 1e-4, lightDir);
  float transparency = 1.0;

  for (const auto &obj : scene.objects) {
      float t;
      Vec2d texCoord;

      if (auto *sphere = dynamic_cast<const Sphere*>(obj.get())) {
          if (intersectRaySphere(shadowRay, *sphere, t, texCoord)) {
              MaterialColor material = sphere->getColor();
              transparency *= (1.0 - material.alpha);
              if (transparency <= 0.01)
                  return 0.0;
          }
      }
  }

  return transparency;
}

// Project a 3D point to screen space coordinates
bool projectToScreen(const Vec3d &point, const Scene &scene, int &screenX, int &screenY) {
    Vec3d viewDir = scene.camera.viewDir.norm();
    Vec3d right = viewDir.cross(scene.camera.upDir).norm();
    Vec3d up = right.cross(viewDir).norm();
    
    Vec3d toPoint = point - scene.camera.eye;
    float distance = toPoint.dot(viewDir);
    
    // Point is behind camera
    if (distance <= 0) return false;
    
    // Calculate the projection onto the view plane
    float height = (int)scene.camera.h;
    float width = (int)scene.camera.w;
    double ar = (double)width / height;
    double vh = 2.0 * tan(scene.camera.vfov_rad() / 2);
    double vw = vh * ar;
    
    // Project point onto screen
    float u = toPoint.dot(right) / distance;
    float v = -toPoint.dot(up) / distance;
    
    // Convert to pixel coordinates
    screenX = (int)((u / vw + 0.5) * width);
    screenY = (int)((v / vh + 0.5) * height);
    
    // Check if within screen bounds
    return (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height);
}

// Screen Space Reflection lookup
Color getSSRColor(const Vec3d &reflectDir, const Vec3d &hitPoint, const Scene &scene) {
    const int MAX_STEPS = 50;
    const float STEP_SIZE = 0.1f;
    
    Vec3d currentPos = hitPoint;
    
    // March along reflection ray
    for (int i = 0; i < MAX_STEPS; i++) {
        currentPos = currentPos + reflectDir * STEP_SIZE;
        
        int screenX, screenY;
        if (projectToScreen(currentPos, scene, screenX, screenY)) {
            // Successfully projected to screen space
            if (screenX >= 0 && screenX < g_width && screenY >= 0 && screenY < g_height) {
                // Check if we hit something by comparing depths
                // For simplicity, we'll use the color from the buffer
                // A more sophisticated implementation would use a depth buffer
                return g_imageBuffer[screenY][screenX];
            }
        } else {
            // Ray went off screen or behind camera
            break;
        }
    }
    
    // Fallback to background color if no valid reflection found
    return scene.bkgcolor;
}

Color shade(const Scene &scene, float t_min, Vec3d rayDir, float t,
            const Objects &obj, const Vec3d &normal, const Vec2d &texCoord,
            int depth) {
  if (depth <= 0)
    return Color(0, 0, 0);

  Color finalColor = scene.bkgcolor;
  Vec3d point = scene.camera.eye + rayDir * t;
  MaterialColor mt = obj.getColor();
  Color texturec = mt.color;

  if (auto *sphere = dynamic_cast<const Sphere *>(&obj)) {
    if (sphere->hasTexture) {
      texturec =
          getTextureColorBilinear(sphere->texture, texCoord.x, texCoord.y);
    }
  }

  if (auto *triangle = dynamic_cast<const Triangle *>(&obj)) {
    if (triangle->hasTexture) {
      texturec =
          getTextureColorBilinear(triangle->texture, texCoord.x, texCoord.y);
    }
  }

  Vec3d N = normal;

  if (t_min < std::numeric_limits<float>::max()) {
    Vec3d viewDir = (rayDir * -1).norm();
    Color ambient = texturec * mt.ka;

    Color totalDiffuse(0, 0, 0);
    Color totalSpecular(0, 0, 0);

    for (const auto &light : scene.lights) {
      Vec3d L = light->isPoint ? (light->positionOrdir - point).norm()
                               : light->positionOrdir.norm() * -1;
      float fatt = 1.0f;
      Vec3d H = (L + viewDir).norm();
      float df = std::max(N.dot(L), 0.0f);

      Color diffuse = texturec * mt.kd * df;
      float sf = std::pow(std::max(N.dot(H), 0.0f), mt.shininess);
      Color specular = mt.specular * mt.ks * sf;

      float shadowFactor = computeShadowFactor(scene, point, L, N);

      totalDiffuse =
          totalDiffuse + diffuse * light->intensity * fatt * shadowFactor;
      totalSpecular =
          totalSpecular + specular * light->intensity * fatt * shadowFactor;
    }

    finalColor = ambient + totalDiffuse + totalSpecular;
  }
  return finalColor;
}

Color trace_recursive(const Scene &scene, const Ray &ray, int depth) {
  if (depth <= 0) return scene.bkgcolor;

  float t_min = std::numeric_limits<float>::max();
  const Objects* hitObject = nullptr;
  Vec3d hitNormal;
  Vec2d texCoord;
  float t_hit;

  for (const auto &obj : scene.objects) {
      float t;
      Vec3d normal;
      Vec2d objectTexCoord;

      if (auto *sphere = dynamic_cast<Sphere*>(obj.get())) {
          if (intersectRaySphere(ray, *sphere, t, objectTexCoord) && t < t_min) {
              t_min = t;
              hitObject = sphere;
              texCoord = objectTexCoord;
              hitNormal = (ray.getOrigin() + ray.getDirection() * t - sphere->center).norm();
              t_hit = t;
          }
      }

      if (auto *triangle = dynamic_cast<Triangle*>(obj.get())) {
          if (intersectRayTriangle(ray, t, normal, objectTexCoord, *triangle) && t < t_min) {
              t_min = t;
              hitObject = triangle;
              texCoord = objectTexCoord;
              hitNormal = normal;
              t_hit = t;
          }
      }
  }

  if (!hitObject) return scene.bkgcolor;

  Color localColor = shade(scene, t_min, ray.getDirection(), t_hit, *hitObject, hitNormal, texCoord, depth);

  MaterialColor mt = hitObject->getColor();
  float ks = mt.ks;
  float alpha = mt.alpha;
  Color reflectedColor(0, 0, 0);
  Color refractedColor(0, 0, 0);

  Vec3d hitPoint = ray.getOrigin() + ray.getDirection() * t_hit;
  float epsilon = 1e-4 * t_hit;

  // SSR Reflection (instead of recursive raytracing)
  if (ks > 0 && depth > 1) {
      Vec3d I = ray.getDirection();
      Vec3d reflectDir = (I - hitNormal * 2 * (I.dot(hitNormal))).norm();
      
      // Use SSR instead of recursive tracing
      if (!g_imageBuffer.empty()) {
          reflectedColor = getSSRColor(reflectDir, hitPoint, scene);
      } else {
          // Fallback to recursive if buffer not available (first pass)
          Ray reflectedRay(hitPoint + reflectDir * epsilon, reflectDir);
          reflectedColor = trace_recursive(scene, reflectedRay, depth - 1);
      }
  }

  // Refraction
  if (alpha < 1 && depth > 1) {
      Vec3d refractDir;
      if (refract(ray.getDirection(), hitNormal, mt.ior, refractDir)) {
          Ray refractedRay(hitPoint + refractDir * epsilon, refractDir);
          refractedColor = trace_recursive(scene, refractedRay, depth - 1);
      }
  }

  float Fr = frCal(ray.getDirection(), hitNormal, mt.ior);
  float Ft = 1 - Fr;

  return localColor + reflectedColor * Fr + refractedColor * (1 - alpha) * Ft;
}

Color trace(int x, int y, Vec3d ul, Vec3d delta_h, Vec3d delta_v, Scene &scene) {
  Vec3d vwPosition = ul + delta_h * x + delta_v * y;
  Vec3d rayDir = (vwPosition - scene.camera.eye).norm();
  Ray r(scene.camera.eye, rayDir);
  return trace_recursive(scene, r, 10);
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
  g_width = (int)scene.camera.w;
  g_height = (int)scene.camera.h;
  g_camera = scene.camera;
  
  double ar = (double)g_width / g_height;
  g_imageBuffer.resize(g_height, std::vector<Color>(g_width, Color(0, 0, 0)));
  
  std::string perspective_filename = basename + "_perspective.ppm";
  std::ofstream output(perspective_filename);
  output << "P3\n" << g_width << " " << g_height << "\n255\n";

  double vh = 2.0 * tan(scene.camera.vfov_rad() / 2);
  double vw = vh * ar;
  Vec3d viewDir = scene.camera.viewDir.norm();
  Vec3d right = viewDir.cross(scene.camera.upDir).norm();
  Vec3d up = right.cross(viewDir).norm();

  Vec3d center = scene.camera.eye + viewDir;
  Vec3d ul = center - right * (vw * 0.5) + up * (vh * 0.5);
  Vec3d ur = center + right * (vw * 0.5) + up * (vh * 0.5);
  Vec3d ll = center - right * (vw * 0.5) - up * (vh * 0.5);
  
  Vec3d delta_h = (ur - ul) / g_width;
  Vec3d delta_v = (ll - ul) / g_height;
  
  // PASS 1: Render without SSR (populate image buffer)
  std::cout << "Pass 1: Rendering base image..." << std::endl;
  for (int y = 0; y < g_height; ++y) {
    for (int x = 0; x < g_width; ++x) {
      g_imageBuffer[y][x] = trace(x, y, ul, delta_h, delta_v, scene);
    }
  }
  
  // PASS 2: Render with SSR enabled
  std::cout << "Pass 2: Rendering with SSR reflections..." << std::endl;
  for (int y = 0; y < g_height; ++y) {
    for (int x = 0; x < g_width; ++x) {
      Color finalColor = trace(x, y, ul, delta_h, delta_v, scene);
      g_imageBuffer[y][x] = finalColor;
      
      output << (int)(finalColor.r * 255) << " " 
             << (int)(finalColor.g * 255) << " " 
             << (int)(finalColor.b * 255) << " ";
    }
    output << "\n";
  }
  
  output.close();

  std::cout << "Rendering complete. Image saved as '" << perspective_filename << "'." << std::endl;
  return 0;
}