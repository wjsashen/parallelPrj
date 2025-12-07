#include "parse.h"
#include "raycast.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <memory>
#include <cmath>
#include <limits>

bool isWhitespaceOnly(const std::string& str) { //for parse check
    return str.find_first_not_of(" \t\r\n") == std::string::npos;
}

void parse(const std::string& filename, Scene& scene) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    std::string type;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;
        if (isWhitespaceOnly(line)) {
            continue;
        }
        std::istringstream words(line);
        words >> type;

        try {
            if (type == "eye" || type == "viewdir" || type == "updir") {
                float x, y, z;
                if (!(words >> x >> y >> z)) {
                    throw std::runtime_error("Expected 3 float values for " + type);
                }
                Vec3d vec(x, y, z);
                if (type == "eye") scene.camera.eye = vec;
                else if (type == "viewdir") scene.camera.viewDir = vec;
                else scene.camera.upDir = vec;

            } else if (type == "vfov") {
                if (!(words >> scene.camera.vfov_degrees)) {
                    throw std::runtime_error("Expected 1 float value for vfov");
                }
                if (scene.camera.vfov_degrees <= 0 || scene.camera.vfov_degrees >= 180) {
                    throw std::runtime_error("vfov must be between 0 and 180 degrees");
                }

            } else if (type == "imsize") {
                if (!(words >> scene.camera.w >> scene.camera.h)) {
                    throw std::runtime_error("Expected 2 integer values for imsize");
                }
                if (scene.camera.w <= 0 || scene.camera.h <= 0) {
                    throw std::runtime_error("Image dimensions must be positive");
                }

            } else if (type == "bkgcolor") {
                double r, g, b, ior;
                if (!(words >> r >> g >> b >> ior)) {
                    throw std::runtime_error("Expected 3 float values for bkgcolor");
                }
                if (r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1) {
                    throw std::runtime_error("Color values must be between 0 and 1");
                }
                if (ior < 1.0) {
                    throw std::runtime_error("Index of refraction (ior) must be >= 1.0");
                }
                scene.bkgcolor = Color(r, g, b);
                scene.bkgior = ior;
            } 
            else if (type == "mtlcolor") {
                double dr, dg, db, sr, sg, sb, ka, kd, ks, shininess, alpha, ior;
                if (!(words >> dr >> dg >> db >> sr >> sg >> sb >> ka >> kd >> ks >> shininess >> alpha >> ior)) {
                    throw std::runtime_error("Expected 12 values for mtlcolor (dr dg db sr sg sb ka kd ks shininess alpha ior)");
                }
                if ((dr < 0 || dr > 1) || (dg < 0 || dg > 1) || (db < 0 || db > 1) ||
                    (sr < 0 || sr > 1) || (sg < 0 || sg > 1) || (sb < 0 || sb > 1)) {
                    throw std::runtime_error("Diffuse and specular color values must be between 0 and 1");
                }
                if (ka < 0 || kd < 0 || ks < 0 || shininess < 0) {
                    throw std::runtime_error("Material coefficients (ka, kd, ks) and shininess must be non-negative");
                }
                if (alpha < 0 || alpha > 1) {
                    throw std::runtime_error("Alpha (opacity) must be between 0 and 1");
                }
                if (ior < 1.0) {
                    throw std::runtime_error("Index of refraction (ior) must be >= 1.0");
                }
                scene.temp_mtlcolor = MaterialColor(
                    Color(dr, dg, db),
                    Color(sr, sg, sb),
                    ka, kd, ks, shininess, alpha, ior
                );
            }
            else if (type == "texture") {
                std::string textureFilename;
                if (!(words >> textureFilename)) {
                    throw std::runtime_error("Expected a filename for texture");
                }
                Texture texture;
                if (!texture.loadFromPPM(textureFilename)) {
                    throw std::runtime_error("Failed to load texture: " + textureFilename);
                } else {
                    scene.textures.push_back(texture);
                }
            }
            else if (type == "sphere") {
                float cx, cy, cz, radius;
                if (!(words >> cx >> cy >> cz >> radius)) {
                    throw std::runtime_error("Expected 4 float values for sphere");
                }
                if (radius <= 0) {
                    throw std::runtime_error("Sphere radius must be positive");
                }
                Texture t;
                bool hasTexture = false;
                if (!scene.textures.empty()) {
                    t = scene.textures.back();
                    hasTexture = true;
                }
                scene.objects.push_back(std::unique_ptr<Sphere>(
                    new Sphere(Vec3d(cx, cy, cz), radius, MaterialColor(scene.temp_mtlcolor), t, hasTexture)
                ));
            }
            else if (type == "v") {
                float x, y, z;
                if (!(words >> x >> y >> z))
                    throw std::runtime_error("Expected 3 float values for vertex position");
                scene.vertices.push_back(Vec3d(x, y, z));
            }
            else if (type == "vn") {
                float nx, ny, nz;
                if (!(words >> nx >> ny >> nz))
                    throw std::runtime_error("Expected 3 float values for vertex normal");
                scene.normals.push_back(Vec3d(nx, ny, nz).norm());
            }
            else if (type == "vt") {
                float u, v;
                if (!(words >> u >> v))
                    throw std::runtime_error("Expected 2 float values for texture coordinate");
                scene.textureCoords.push_back(Vec2d(u, v));
            }
            else if (type == "f") {
                std::vector<size_t> v, vn, vt;
                std::string vertex;
                bool hasNormals = false, hasTextures = false;
                while (words >> vertex) {
                    std::istringstream vertexData(vertex);
                    std::string index;
                    std::vector<int> indices;
                    int slashCount = std::count(vertex.begin(), vertex.end(), '/');
                    if (vertex.find("//") != std::string::npos) hasNormals = true;
                    else if (vertex.find("/") != std::string::npos) {
                        hasTextures = true;
                        if (slashCount >= 2) hasNormals = true;
                    }
                    while (std::getline(vertexData, index, '/')) {
                        if (!index.empty()) indices.push_back(std::stoi(index));
                        else indices.push_back(-1);
                    }
                    if (indices.size() >= 1) v.push_back(indices[0]);
                    if (indices.size() >= 2 && indices[1] != -1) vt.push_back(indices[1]);
                    if (indices.size() >= 3 && indices[2] != -1) vn.push_back(indices[2]);
                }
                for (size_t& idx : v) if (idx > 0) idx -= 1;
                for (size_t& idx : vt) if (idx > 0) idx -= 1;
                for (size_t& idx : vn) if (idx > 0) idx -= 1;

                Texture t;
                Vec3d vdefault(-1, -1, 0);
                Vec2d defaultTexCoord(0, 0);
                if (!scene.textures.empty()) t = scene.textures.back();

                if (hasNormals && hasTextures) {
                    scene.objects.push_back(std::make_unique<Triangle>(
                        scene.vertices[v[0]], scene.vertices[v[1]], scene.vertices[v[2]],
                        scene.normals[vn[0]], scene.normals[vn[1]], scene.normals[vn[2]],
                        scene.textureCoords[vt[0]], scene.textureCoords[vt[1]], scene.textureCoords[vt[2]],
                        MaterialColor(scene.temp_mtlcolor), true, hasTextures, t
                    ));
                } else if (hasNormals) {
                    scene.objects.push_back(std::make_unique<Triangle>(
                        scene.vertices[v[0]], scene.vertices[v[1]], scene.vertices[v[2]],
                        scene.normals[vn[0]], scene.normals[vn[1]], scene.normals[vn[2]],
                        defaultTexCoord, defaultTexCoord, defaultTexCoord,
                        MaterialColor(scene.temp_mtlcolor), true, hasTextures, t
                    ));
                } else if (hasTextures) {
                    scene.objects.push_back(std::make_unique<Triangle>(
                        scene.vertices[v[0]], scene.vertices[v[1]], scene.vertices[v[2]],
                        vdefault, vdefault, vdefault,
                        scene.textureCoords[vt[0]], scene.textureCoords[vt[1]], scene.textureCoords[vt[2]],
                        MaterialColor(scene.temp_mtlcolor), false, hasTextures, t
                    ));
                } else {
                    scene.objects.push_back(std::make_unique<Triangle>(
                        scene.vertices[v[0]], scene.vertices[v[1]], scene.vertices[v[2]],
                        vdefault, vdefault, vdefault,
                        defaultTexCoord, defaultTexCoord, defaultTexCoord,
                        MaterialColor(scene.temp_mtlcolor), false, hasTextures, t
                    ));
                }
            }
            else if (type == "light") {
                float x, y, z;
                int isPoint;
                float intensity;
                if (!(words >> x >> y >> z >> isPoint >> intensity)) {
                    throw std::runtime_error("Expected format: light x y z isPoint intensity");
                }
                if (intensity < 0) {
                    throw std::runtime_error("Light intensity must be non-negative");
                }
                float c1 = 1.0f, c2 = 0.0f, c3 = 0.0f;
                if (words >> c1 >> c2 >> c3) {
                    std::cout << "have att params" << std::endl;
                }
                Light light;
                light.c1 = c1;
                light.c2 = c2;
                light.c3 = c3;
                light.positionOrdir = Vec3d(x, y, z);
                light.isPoint = (isPoint != 0);
                light.intensity = intensity;
                scene.lights.push_back(std::make_unique<Light>(light));
            }

            std::string extra;
            if (words >> extra) {
                throw std::runtime_error("Unexpected extra parameters");
            }

        } catch (const std::runtime_error& e) {
            std::cerr << "Error on line " << lineNumber << ": " << e.what() << std::endl;
            std::cerr << "Line content: " << line << std::endl;
        }
    }

    if (scene.camera.w <= 0 || scene.camera.h <= 0) {
        throw std::runtime_error("Image size not set or invalid");
    }
    if (scene.camera.vfov_degrees <= 0) {
        throw std::runtime_error("Field of view not set or invalid");
    }
}
