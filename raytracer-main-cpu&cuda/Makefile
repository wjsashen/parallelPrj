# CPU compiler
CXX = g++
CXXFLAGS = -Wall -std=c++11

# GPU compiler
NVCC = nvcc
NVCCFLAGS = -std=c++11

# Target executable name (Windows vs. others)
ifeq ($(OS),Windows_NT)
    TARGET = raytracer1d.exe
else
    TARGET = raytracer1d
endif

# CPU source files - use raycast_ssr.cpp instead of raycast.cpp
SRCS = raycast_ssr.cpp parse.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target: build only the CPU version when running "make" with no arguments
all: $(TARGET)

# Link CPU executable from object files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Generic rule to compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# GPU target: "make raycast_cuda" builds raycast_cuda from raycast_cuda.cu and parse.cpp
raycast_cuda: raycast_cuda.cu parse.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ raycast_cuda.cu parse.cpp

# GPU target: "make raycast_ssr_cuda" builds raycast_ssr_cuda from raycast_ssr_cuda.cu and parse.cpp
raycast_ssr_cuda: raycast_ssr_cuda.cu parse.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ raycast_ssr_cuda.cu parse.cpp

# Test GPU target with BVH: "make raycast_ssr_cuda_bvh" builds raycast_ssr_cuda_bvh from raycast_ssr_cuda_bvh.cu and parse.cpp
raycast_ssr_cuda_bvh: raycast_ssr_cuda_bvh.cu parse.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ raycast_ssr_cuda_bvh.cu parse.cpp

#shared memory
raycast_ssr_cuda_bvh_sharedMem: raycast_ssr_cuda_bvh_sharedMem.cu parse.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ raycast_ssr_cuda_bvh_sharedMem.cu parse.cpp
#bin
raycast_ssr_cuda_bvh_binned: raycast_ssr_cuda_bvh_binned.cu parse.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ raycast_ssr_cuda_bvh_binned.cu parse.cpp







# Cross-platform clean rule
clean:
	@echo "Cleaning up..."
ifeq ($(OS),Windows_NT)
	del /F /Q $(OBJS) $(TARGET) raycast_cuda.exe raycast_ssr_cuda.exe 2> nul || exit 0
else
	rm -f $(OBJS) $(TARGET) raycast_cuda raycast_ssr_cuda raycast_ssr_cuda_bvh raycast_ssr_cuda_bvh_binned raycast_ssr_cuda_bvh_sharedMem
endif

.PHONY: all clean raycast_cuda raycast_ssr_cuda
.PHONY: all_cuda
all_cuda: clean raycast_cuda raycast_ssr_cuda raycast_ssr_cuda_bvh raycast_ssr_cuda_bvh_binned raycast_ssr_cuda_bvh_sharedMem
