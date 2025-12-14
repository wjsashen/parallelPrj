readme.txt

compile with make for cpu code

./raytracer1d example1.txt



for gpu code needing rtx:
Compile: nvcc -O3 -lineinfo -gencode arch=compute_90,code=sm_120 raycast_ssr_cuda_bvh_binned.cu parse.cpp -o app_v1 -std=c++14 -D_USE_MATH_DEFINES
I use x64 native under win11, delete this in parse.cpp when using cpp14
#if __cplusplus <= 201103L
namespace std {
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

Profile ncu --set full --target-processes all -o v1_report ./app_v1 bunny.in

