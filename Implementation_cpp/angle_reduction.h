#ifndef ANGLE_REDUCTION_H
#define ANGLE_REDUCTION_H

#include <immintrin.h>  // Include AVX intrinsics
#include <cmath>        // For M_PI

// Define M_PI if not already defined
#ifndef M_PID
#define M_PID 3.14159265358979323846
#endif

// Struct for storing reduced angles and corresponding signs
struct Vec2 {
    __m256d red;  // Reduced angle
    __m256d sign; // Corresponding sign (1.0 or -1.0)
};

// Function declarations
inline __m256d abs256(__m256d x);
Vec2 reduce_angle_sin_SIMD(__m256d ang);
Vec2 reduce_angle_cos_SIMD(__m256d ang);
Vec2 reduce_angle_tan_SIMD(__m256d ang);

#endif  // ANGLE_REDUCTION_H
