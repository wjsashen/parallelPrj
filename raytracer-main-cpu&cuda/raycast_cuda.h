#ifndef RAYCAST_CUDA_H
#define RAYCAST_CUDA_H

// This header is a thin wrapper around raycast.h for CUDA builds.
// We do NOT redefine Vec3d, Vec2d, Ray, etc. here to avoid
// duplicate type definitions. All geometry and scene types come
// from raycast.h.

#include "raycast.h"

#endif // RAYCAST_CUDA_H
