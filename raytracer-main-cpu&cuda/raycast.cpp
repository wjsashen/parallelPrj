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

// Custom clamp function for C++11 compatibility
template<typename T>
T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

std::vector<Vec3d> Objects::vertices;
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
  texCoordOut = calculateSphereUV(normal); //mapping
  return true;
}

Color getTextureColor1(const Texture &texture, float u, float v) { //nn inter
  if (texture.pixels.empty()) {
    std::cout << "Texture is empty" << std::endl;
    return Color(1.0f, 1.0f, 1.0f); // Default to white if no texture
  }

  u = std::fmod(std::fmod(u, 1.0f) + 1.0f, 1.0f);
  v = std::fmod(std::fmod(v, 1.0f) + 1.0f, 1.0f);

  int x = clamp(int(std::round(u * (texture.width - 1))), 0, texture.width - 1);
  int y = clamp(int(std::round((1.0f - v) * (texture.height - 1))), 0, texture.height - 1);

  return texture.getPixel(x, y); 
}
Color getTextureColorBilinear(const Texture &texture, float u, float v) {
  if (texture.pixels.empty() || texture.width == 0 || texture.height == 0) {
      return Color(1.0f, 1.0f, 1.0f); // Return default color if texture is empty
  }

  u = std::fmod(std::fmod(u, 1.0f) + 1.0f, 1.0f);
  v = std::fmod(std::fmod(v, 1.0f) + 1.0f, 1.0f);

  // Check texture dimensions before accessing pixels
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

  // Ensure pixel access is valid
  if (x0 < 0 || x1 < 0 || y0 < 0 || y1 < 0 ||
      x0 >= texture.width || x1 >= texture.width ||
      y0 >= texture.height || y1 >= texture.height) {
      return Color(1.0f, 1.0f, 1.0f); // Fail-safe return
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
    return false; // Parallel ray

  float d = -cr.dot(triangle.v0);
  t = -(cr.dot(ray.getOrigin()) + d) / denominator;
  if (t < 0)
    return false; // Behind the ray

  Vec3d P = ray.getOrigin() + ray.getDirection() * t;

  Vec3d vP = P - triangle.v0;
  float det = edge1.dot(edge1) * edge2.dot(edge2) - edge1.dot(edge2) * edge1.dot(edge2);
  float beta =(vP.dot(edge1) * edge2.dot(edge2) - vP.dot(edge2) * edge1.dot(edge2)) /det;
  float gamma =
      (edge1.dot(edge1) * vP.dot(edge2) - edge1.dot(edge2) * vP.dot(edge1)) / det;
  float alpha = 1.0f - beta - gamma;

  if (alpha >= 0 && beta >= 0 && gamma >= 0) {
    // Smooth shading
    if (triangle.isSmooth) {
        normalOut = (triangle.n0 * alpha) + (triangle.n1 * beta) + (triangle.n2 * gamma);
        normalOut = normalOut.norm(); // Normalize the interpolated normal
    } else {
        normalOut = normal; // Use the flat shading normal
    }

    // get texture coordinates
    if (triangle.hasTexture) {
        texCoordOut = (triangle.vt0 * alpha) + (triangle.vt1 * beta) + (triangle.vt2 * gamma);
    }

    return true; //  intersects the triangle
}
return false; // No intersection - barycentric coordinates check failed
}
  

bool intersectRaySphereSh(const Ray& ray, const Sphere& sphere, float& t) { //overload by containing sphere texture map
  //intersect func
      float cx = sphere.center.x, cy = sphere.center.y, cz = sphere.center.z;
      float r = sphere.radius;
      //fixed a issue from 1a
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
  //check if the point is in shadow area
      Ray shadowRay(point + L * 0.001, L); // offset to avoid self-intersection
  
      for (const auto& obj : scene.objects) {
          Sphere* sphere = dynamic_cast<Sphere*>(obj.get());
          if (sphere) {
              float t;
              if (intersectRaySphereSh(shadowRay, *sphere, t) && t > 0.001) {
                  return true; // Point is in shadow
              }
          }
          else if (Triangle* triangle = dynamic_cast<Triangle*>(obj.get())) {
            float t;
            Vec3d N;
            Vec2d dummyTexCoord;
            // Use the intersectRayTriangle function
            if (intersectRayTriangle(shadowRay, t,
              N, dummyTexCoord,  //HERE texture is not relevant
              *triangle) ) {
          return true; // Point is in shadow
      }
        }
    }
      
      return false; //light reaches point
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
    //equation on note p83 
    float eta = float1 / ior;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    if (k < 0) return false; // internal reflection
    refractedDir = (I * eta + n * (eta * cosi - sqrtf(k))).norm();
    return true;
}
float frCal(const Vec3d &I, const Vec3d &N, float ior) {
  float cosi = clamp(I.dot(N), -1.0f, 1.0f);
  float float1 = 1.0f;
  if (cosi > 0) std::swap(float1, ior);

  float r0 = (1.0f - ior) / (1.0f + ior);
  float f0 = r0 * r0;
  //Fr
  return f0 + (1 - f0) * pow(1 - std::abs(cosi), 5);
}



/*

Color trace(const Scene &scene, const Ray &ray, int depth) {
  if (depth <= 0) return scene.bkgcolor;

  float t_min = std::numeric_limits<float>::max();
  Color res = scene.bkgcolor;
  Vec2d texCoord;
  const Objects* hitObject = nullptr;
  Vec3d hitNormal;
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

  if (hitObject) {
      // Call shade with full context for recursive tracing
      res = shade(scene, t_min, ray.getDirection(), t_hit, *hitObject, hitNormal, texCoord, depth);
  }

  return res;
}
*/
// new shadow function considering gi, fractional value

float computeShadowFactor(const Scene &scene, const Vec3d &point, const Vec3d &lightDir, const Vec3d &normal) {
  Ray shadowRay(point + lightDir * 1e-4, lightDir); // Small offset to avoid self-intersection
  float transparency = 1.0;  // Fully lit by default

  for (const auto &obj : scene.objects) {
      float t;
      Vec2d texCoord;

      if (auto *sphere = dynamic_cast<const Sphere*>(obj.get())) {
          if (intersectRaySphere(shadowRay, *sphere, t, texCoord)) {
              MaterialColor material = sphere->getColor();
              transparency *= (1.0 - material.alpha);  // Accumulate opacity
              if (transparency <= 0.01)  // Almost fully shadowed
                  return 0.0;
          }
      }
  }

  return transparency;  // Between 0 (full shadow) and 1 (fully lit)
}

Color shade(const Scene &scene, float t_min, Vec3d rayDir, float t,
            const Objects &obj, const Vec3d &normal, const Vec2d &texCoord,
            int depth) {
  if (depth <= 0)
    return Color(0, 0, 0); // Stop recursion at depth 0

  Color finalColor = scene.bkgcolor;
  Vec3d point = scene.camera.eye + rayDir * t;
  MaterialColor mt = obj.getColor();
  Color texturec = mt.color;

  // 1c texture
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

  Vec3d N = normal; // Normal at the intersection

  // Lighting calculations
  if (t_min < std::numeric_limits<float>::max()) {
    Vec3d viewDir = (rayDir * -1).norm();
    Color ambient = texturec * mt.ka;

    Color totalDiffuse(0, 0, 0);
    Color totalSpecular(0, 0, 0);

    // Calculate lighting from each light in the scene
    for (const auto &light : scene.lights) {
      Vec3d L = light->isPoint ? (light->positionOrdir - point).norm()
                               : light->positionOrdir.norm() * -1;
      float dis = (light->positionOrdir - point).length();
      float fatt = 1.0f; // Attenuation factor
      Vec3d H = (L + viewDir).norm();
      float df = std::max(N.dot(L), 0.0f);

      Color diffuse = texturec * mt.kd * df;
      float sf = std::pow(std::max(N.dot(H), 0.0f), mt.shininess);
      Color specular = mt.specular * mt.ks * sf;

      // Shadow check
      //float Si = isInShadow(point, L, scene, N) ? 0.0f : 1.0f;

      float shadowFactor = computeShadowFactor(scene, point, L, N);

      // Use shadowFactor instead of binary Si!!! on kd ks
      totalDiffuse =
          totalDiffuse + diffuse * light->intensity * fatt * shadowFactor;
      totalSpecular =
          totalSpecular + specular * light->intensity * fatt * shadowFactor;
    }

    finalColor = ambient + totalDiffuse + totalSpecular;
  }
  return finalColor;
}

//could also use inter but recursive is simple
Color trace_recursive(const Scene &scene, const Ray &ray, int depth) {
  if (depth <= 0) return scene.bkgcolor; // base: return background color

  // Ray intersection test
  float t_min = std::numeric_limits<float>::max();
  const Objects* hitObject = nullptr;
  Vec3d hitNormal; //N, distinguish from phong shade N
  Vec2d texCoord;
  float t_hit;

  // Find the closest intersection
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

  // If no hit, return background color
  if (!hitObject) return scene.bkgcolor;

  // Compute local shading
  Color localColor = shade(scene, t_min, ray.getDirection(), t_hit, *hitObject, hitNormal, texCoord, depth);

  // Reflection and refraction
  MaterialColor mt = hitObject->getColor();
  float ks = mt.ks;  // Reflectivity coefficient
  float alpha = mt.alpha; // Transparency
  Color reflectedColor(0, 0, 0);
  Color refractedColor(0, 0, 0);

  Vec3d hitPoint = ray.getOrigin() + ray.getDirection() * t_hit;
  float epsilon = 1e-4 * t_hit; // offset self-intersection

  // reflection R
  if (ks > 0 && depth > 1) {
      Vec3d I = ray.getDirection();
      // R(specular reflection)
      //check here too, strange it's not same to lecture note
      Vec3d reflectDir = (I-hitNormal * 2 * (I.dot(hitNormal))).norm();
      Ray reflectedRay(hitPoint + reflectDir * epsilon, reflectDir);
      reflectedColor = trace_recursive(scene, reflectedRay, depth - 1);
  }

  // refraction 
  if (alpha < 1 && depth > 1) {
      Vec3d refractDir;
      if (refract(ray.getDirection(), hitNormal, mt.ior, refractDir)) {
          Ray refractedRay(hitPoint + refractDir * epsilon, refractDir);
          refractedColor = trace_recursive(scene, refractedRay, depth - 1);
      }
  }

  // Fresnel term
  float Fr = frCal(ray.getDirection(), hitNormal, mt.ior);
  float Ft = 1 - Fr; // Transmission weight

  // Blend colors
  //return localColor * (1 - ks) + reflectedColor * Fr + refractedColor * (1 - alpha) * Ft;
  return localColor  + reflectedColor * Fr + refractedColor * (1 - alpha) * Ft;

}
Color trace(int x, int y, Vec3d ul, Vec3d delta_h, Vec3d delta_v, Scene &scene) {
  Vec3d vwPosition = ul + delta_h * x + delta_v * y;
  Vec3d rayDir = (vwPosition - scene.camera.eye).norm();
  Ray r(scene.camera.eye, rayDir);
  return trace_recursive(scene, r, 10);
}


bool isValidTxtFile(const std::string &filename) { // check file format
  return filename.size() >= 3 &&
           (filename.substr(filename.size() - 4) == ".txt" ||
            filename.substr(filename.size() - 3) == ".in");
}

int main(int argc, char* argv[]) {
  Scene scene;

  // Check if the user provided a filename argument
  if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <filename.txt>" << std::endl;
      return 1;
  }

  std::string filename = argv[1]; // Get filename from command-line argument
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
  std::vector<std::vector<Color>> image(
      height, std::vector<Color>(width, Color(0, 0, 0)));
  std::string perspective_filename = basename + "_perspective.ppm";
  std::ofstream output(perspective_filename);
  output << "P3\n" << width << " " << height << "\n255\n";

  // the view window init is here,
  //  TODO: need refactor it for a sperate func but need a new struct to store
  //  return value, like a vector of Vec3d

  double vh = 2.0 * tan(scene.camera.vfov_rad() /
                        2); // view window width and height in 3d world coord
  double vw = vh * ar; // the given w&h is in pixel so it need to change to view
                       // window measurement
  Vec3d viewDir = scene.camera.viewDir.norm();
  // pay attention to the cross sequence viewD first(right hand)
  Vec3d right = viewDir.cross(scene.camera.upDir)
                    .norm(); //(u in lecture notes, the orthogonal to the plane)
  // (up is the v in lecture note, the norm of orth of u&viewD )
  Vec3d up = right.cross(viewDir).norm(); // v in lecture notes

  // four corners of view window
  // d*n in notes
  Vec3d center = scene.camera.eye +
                 viewDir; // d(focal distance) is set to arbitary 1, that's
                          // where frustum narrow to the center of vw
  Vec3d ul = center - right * (vw * 0.5) + up * (vh * 0.5); // Upper-left corner
  Vec3d ur =
      center + right * (vw * 0.5) + up * (vh * 0.5); // Upper-right corner
  Vec3d ll = center - right * (vw * 0.5) - up * (vh * 0.5); // Lower-left corner
  // step vectors
  Vec3d delta_h = (ur - ul) / width;
  Vec3d delta_v = (ll - ul) / height;
  // Trace for perspective image (first render)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image[y][x] = trace(x, y, ul, delta_h, delta_v, scene);

      // or no need for buffer here whatever
      output << (int)(image[y][x].r * 255) << " " << (int)(image[y][x].g * 255)
             << " " << (int)(image[y][x].b * 255) << " ";
    }
    output << "\n";
  }
  output.close();

  std::cout << "Rendering complete. Images saved as 'output_perspective.ppm' "
               "and 'output_parallel.ppm'."
            << std::endl;
  return 0;
}