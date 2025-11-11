readme.txt

compile with make for cpu code

./raytracer1d example1.txt



for gpu code 
nvcc -o raycast_cuda raycast_cuda.cu parse.cpp -std=c++11

./raycast_cuda example1.txt                                                                                    
Scene info: 3 spheres, 0 triangles, 1 lights
Light 0: pos(-3, 15, 10), intensity=1, isPoint=1
Sphere 0: center(-1.1, 0, -4), radius=1
Material: color(0.2, 1, 0.2), ka=0.2, kd=0.6
Rendering 512x512 image on GPU...
GPU Rendering took 0.0003808 seconds
Complete! Image saved as 'example1_perspective.ppm'.
[wan02258@ece-gpulab05 ~/raytracer-main-cpu&cuda]$ ./raycast_cuda showcase.txt
Error on line 1: Unexpected extra parameters
Line content: # Simple test scene for CUDA ray tracer
Error on line 13: Unexpected extra parameters
Line content: # Green sphere on the left
Error on line 17: Unexpected extra parameters
Line content: # Blue sphere on the right
Error on line 21: Unexpected extra parameters
Line content: # Red sphere in the middle
Scene info: 3 spheres, 0 triangles, 1 lights
Light 0: pos(3, 5, 0), intensity=1, isPoint=1
Sphere 0: center(-1.5, 0, -6), radius=1
Material: color(0.2, 1, 0.2), ka=0.2, kd=0.6
Rendering 512x512 image on GPU...
GPU Rendering took 0.00028224 seconds
Complete! Image saved as 'showcase_perspective.ppm'.