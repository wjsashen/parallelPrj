readme.txt

compile with Makefile

./raycast_xxx example1.txt/.in

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

Profile with ncu/nsys command: ncu --set full --target-processes all -o v1_report ./app_v1 bunny.in
nv-nsight-cu-cli ./raycast_ssr_cuda_bvh dragon.in

Profile with Python tool:
$ ./profiling_tool
This will generate executable profiling_results.csv for a simple example input at increasing size

$ ./profiling_tool <filename> <filename> <...>
This will profile all executables w/ specific set of files, output to profiling_results.csv

