#ifndef RAYCAST_H
#define RAYCAST_H
#include <fstream>
#include <vector>
#include <sstream>
#include <memory>
#include <cmath>
#include <limits>
#include <iostream>
#include <cmath>

#include <vector>
#include <string>
//merge point and vector into one struct for easier calculation

#ifdef __CUDACC__           // When compiling with nvcc
#define HD __host__ __device__
#else
#define HD                  // On pure CPU builds this expands to nothing
#endif

// struct Vec3d {
//     float x, y, z;

//     Vec3d(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
//     Vec3d operator*(double s) const {
//         return Vec3d(x * s, y * s, z * s);
//     }
//     Vec3d operator/(double s) const {
//         return Vec3d(x / s, y / s, z / s);
//     }

//     Vec3d operator+(const Vec3d& other) const {
//         return Vec3d(x + other.x, y + other.y, z + other.z);
//     }
//     Vec3d operator-(const Vec3d& other) const {
//         return Vec3d(x - other.x, y - other.y, z - other.z);
//     }
//     Vec3d cross(const Vec3d& b) const {
//         return Vec3d(
//             y * b.z - z * b.y,
//             z * b.x - x * b.z,
//             x * b.y - y * b.x
//         );
//     }
//     Vec3d reflect(const Vec3d& normal) const {
//         return *this - normal * (2 * this->dot(normal));
//     }
//     //for debug
//     // Overload << operator to print Vec3d objects
//     friend std::ostream& operator<<(std::ostream& os, const Vec3d& vec) {
//         os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
//         return os;
//     }
//     float dot(const Vec3d& other) const {
//         return x * other.x + y * other.y + z * other.z;
//     }
//     double lengthSquared() const {
//         return x * x + y * y + z * z;
//     }
//     // Length (magnitude) of the vector
//     float length() const {
//         return std::sqrt(x * x + y * y + z * z);
//     }

//     Vec3d norm() const {
//         double len = sqrt(x * x + y * y + z * z);
//         return (len > 0) ? Vec3d(x / len, y / len, z / len) : Vec3d(0, 0, 0);
//     }
// };

struct Vec3d {
    float x, y, z;

    // Constructors and basic operations usable on both host and device
    HD Vec3d(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    HD Vec3d operator*(double s) const {
        return Vec3d(x * s, y * s, z * s);
    }

    HD Vec3d operator/(double s) const {
        return Vec3d(x / s, y / s, z / s);
    }

    HD Vec3d operator+(const Vec3d& other) const {
        return Vec3d(x + other.x, y + other.y, z + other.z);
    }

    HD Vec3d operator-(const Vec3d& other) const {
        return Vec3d(x - other.x, y - other.y, z - other.z);
    }

    HD Vec3d cross(const Vec3d& b) const {
        return Vec3d(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }

    HD Vec3d reflect(const Vec3d& normal) const {
        return *this - normal * (2 * this->dot(normal));
    }

    HD float dot(const Vec3d& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    HD double lengthSquared() const {
        return x * x + y * y + z * z;
    }

    HD float length() const {
        // Use sqrtf for float; CUDA provides device overload
        return sqrtf(x * x + y * y + z * z);
    }

    HD Vec3d norm() const {
        float len = length();
        return (len > 0.0f) ? Vec3d(x / len, y / len, z / len) : Vec3d(0.0f, 0.0f, 0.0f);
    }

    // Host-only debug print; not marked HD so it will not be used on device
    friend std::ostream& operator<<(std::ostream& os, const Vec3d& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
        return os;
    }
};

// struct Vec2d {
//         float x, y;
        
//         Vec2d() : x(0), y(0) {}
//         Vec2d(float x, float y) : x(x), y(y) {}
        
//         Vec2d operator+(const Vec2d& v) const {
//             return Vec2d(x + v.x, y + v.y);
//         }
        
//         Vec2d operator*(float s) const {
//             return Vec2d(x * s, y * s);
//         }
//     };
struct Vec2d {
    float x, y;

    HD Vec2d() : x(0.0f), y(0.0f) {}
    HD Vec2d(float x, float y) : x(x), y(y) {}

    HD Vec2d operator+(const Vec2d& v) const {
        return Vec2d(x + v.x, y + v.y);
    }

    HD Vec2d operator*(float s) const {
        return Vec2d(x * s, y * s);
    }
};

// struct Ray {
//     float x, y, z;        // origin components
//     float dx, dy, dz;     // direction components
    
//     Ray(const Vec3d& o, const Vec3d& d)
//         : x(o.x), y(o.y), z(o.z), 
//           dx(d.x), dy(d.y), dz(d.z) {}
          
//     Vec3d getOrigin() const { return Vec3d(x, y, z); }
//     Vec3d getDirection() const { return Vec3d(dx, dy, dz); }
// };
struct Ray {
    float x, y, z;    // origin
    float dx, dy, dz; // direction

    HD Ray() : x(0), y(0), z(0), dx(0), dy(0), dz(0) {}

    HD Ray(const Vec3d& o, const Vec3d& d)
        : x(o.x), y(o.y), z(o.z),
          dx(d.x), dy(d.y), dz(d.z) {}

    HD Vec3d getOrigin() const {
        return Vec3d(x, y, z);
    }

    HD Vec3d getDirection() const {
        return Vec3d(dx, dy, dz);
    }
};


struct Light {
    Vec3d positionOrdir; //or is direction if it's not point light
    bool isPoint;
    float intensity;
    float c1, c2, c3;
};
struct Camera {
    Vec3d viewDir, upDir;
    Vec3d eye;
    double vfov_degrees, w, h;
    double vfov_rad() const {
        return vfov_degrees * (M_PI / 180.0);
    }
};
// struct Color {
//     double r, g, b;
//     Color(double r = 0, double g = 0, double b = 0) : r(r), g(g), b(b) {}
    
//     Color operator*(double s) const { return Color(r * s, g * s, b * s); }
//     Color operator+(const Color& other) const { return Color(r + other.r, g + other.g, b + other.b); }

//     // Clamp color values to [0,1]
//     Color clamp() const {
//         return Color(
//             std::max(0.0, std::min(1.0, r)),
//             std::max(0.0, std::min(1.0, g)),
//             std::max(0.0, std::min(1.0, b))
//         );
//     }
//     Color operator*(const Color& other) const {
//         return Color(r * other.r, g * other.g, b * other.b);
//     }
//     friend std::ostream& operator<<(std::ostream& os, const Color& color) {
//         os << "(" << color.r << ", " << color.g << ", " << color.b << ")";
//         return os;
//     }
// };
struct Color {
    double r, g, b;

    HD Color(double r = 0.0, double g = 0.0, double b = 0.0)
        : r(r), g(g), b(b) {}

    HD Color operator*(double s) const {
        return Color(r * s, g * s, b * s);
    }

    HD Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b);
    }

    HD Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b);
    }

    // Clamp color values to [0, 1] without using std::max/min
    // so that it is safe for device code as well.
    HD Color clamp() const {
        double rr = r;
        double gg = g;
        double bb = b;

        if (rr < 0.0) rr = 0.0;
        if (rr > 1.0) rr = 1.0;
        if (gg < 0.0) gg = 0.0;
        if (gg > 1.0) gg = 1.0;
        if (bb < 0.0) bb = 0.0;
        if (bb > 1.0) bb = 1.0;

        return Color(rr, gg, bb);
    }

    // Host-only debug print
    friend std::ostream& operator<<(std::ostream& os, const Color& color) {
        os << "(" << color.r << ", " << color.g << ", " << color.b << ")";
        return os;
    }
};

// struct MaterialColor {
//     Color color;        // Base color
//     Color specular;
//     double ka, kd, ks;   
//     double shininess;
//     double alpha, ior;
//     MaterialColor(
//         const Color& c = Color(), 
//         const Color& s = Color(), 
//         double ka = 0, double kd = 0, double ks = 0, double shininess = 0, double alpha=0, double ior=0
//     ) : color(c),  specular(s), ka(ka), kd(kd), ks(ks), shininess(shininess), alpha(alpha), ior(ior) {}

// };
struct MaterialColor {
    Color  color;        // Base color
    Color  specular;
    double ka, kd, ks;
    double shininess;
    double alpha;        // Transparency
    double ior;          // Index of refraction

    HD MaterialColor(
        const Color& c = Color(),
        const Color& s = Color(),
        double ka = 0.0, double kd = 0.0, double ks = 0.0,
        double shininess = 0.0, double alpha = 0.0, double ior = 1.0
    )
        : color(c),
          specular(s),
          ka(ka), kd(kd), ks(ks),
          shininess(shininess),
          alpha(alpha),
          ior(ior) {}
};


struct Texture {
    int width, height;
    std::vector<Color> pixels;

    bool loadFromPPM(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open texture file " << filename << std::endl;
            return false;
        }

        std::string format;
        file >> format;
        if (format != "P3") {
            std::cerr << "Error: Unsupported PPM format in " << filename << std::endl;
            return false;
        }

        file >> width >> height;
        int maxVal;
        file >> maxVal;

        pixels.resize(width * height);

        for (int i = 0; i < width * height; ++i) {
            int r, g, b;
            file >> r >> g >> b;
            pixels[i] = { r / float(maxVal), g / float(maxVal), b / float(maxVal) };
        }

        file.close();
        return true;
    }

    Color getPixel(int u, int v) const {
        if (u < 0) u = 0;
        if (v < 0) v = 0;
        if (u >= width) u = width - 1;
        if (v >= height) v = height - 1;
        return pixels[v * width + u];  
    }
};
//this is for expand the code to various types of shapes
struct Objects {
    virtual ~Objects() = default;
    virtual MaterialColor getColor() const = 0;
    Vec3d center;
    static std::vector<Vec3d> vertices;


    Vec3d getCenter(){
        return center;
    }
};
struct Sphere : public Objects {
    Vec3d center;
    double radius;
    MaterialColor material; 
    Texture texture;
    bool hasTexture;
    Sphere(Vec3d c, double r, MaterialColor mat, Texture t, bool hast)  
        : center(c), radius(r), material(mat), texture(t),hasTexture(hast) {}

    MaterialColor getColor() const override {
        return material;  // Return MaterialColor
    }
};

struct Cylinder : public Objects {
    Vec3d center;
    Vec3d dir;
    double radius,length;
    MaterialColor color;
    Cylinder(Vec3d c, Vec3d d,double r,double l, Color col) : center(c), dir(d),radius(r), length(l),color(col) {}
    MaterialColor getColor() const override {
        return color;
    }
};



struct Scene {
    Camera camera;
    Color bkgcolor;
    double bkgior;
    std::vector<std::unique_ptr<Light>> lights;
    MaterialColor temp_mtlcolor;
    std::vector<std::unique_ptr<Objects>> objects;
    std::vector<Vec3d> vertices;
    std::vector<Vec3d> normals;
    std::vector<Vec2d> textureCoords;
    int textureIndex = -1;
    std::vector<Texture> textures;
};

struct Triangle : public Objects {
    Vec3d v0, v1, v2;  // vertices
    Vec3d n0, n1, n2;  // normals (for smooth shading)
    Vec3d planeNormal; // plane normal for flat shading
    Vec2d vt0, vt1, vt2; 
    MaterialColor material;
    bool isSmooth;
    Color col;
    bool hasTexture;

    Texture texture;
    Triangle(const Vec3d& a, const Vec3d& b, const Vec3d& c,
        const Vec3d& a_, const Vec3d& b_, const Vec3d& c_,   
        const Vec2d& a__, const Vec2d& b__, const Vec2d& c__,   
        const MaterialColor& mat = MaterialColor(), bool isS = false, bool hast = false, const Texture& tex = Texture())
   : v0(a), v1(b), v2(c),
     n0(a_), n1(b_), n2(c_),
     vt0(a__), vt1(b__), vt2(c__),
     material(mat), isSmooth(isS), hasTexture(hast),texture(tex) {  
   computePlaneNormal();
}


    MaterialColor getColor() const override {
        return material;
    }

    // flat shading normal calculation
    void computePlaneNormal() {
        Vec3d e1 = v1 - v0;
        Vec3d e2 = v2 - v0;
        planeNormal = e1.cross(e2).norm();
    }//forgot to use it may refactor later

};


#endif