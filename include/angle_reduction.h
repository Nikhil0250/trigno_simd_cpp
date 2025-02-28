// Header guard to prevent multiple inclusions of this header file during compilation.
// Ensures definitions are only included once in the final executable.
#ifndef ANGLE_REDUCTION_H
#define ANGLE_REDUCTION_H

// Include Intel Intrinsics header for SIMD operations.
// Provides AVX instructions for vectorized computations using __m256d type.
#include <immintrin.h>

// Include C++ math library for access to M_PI constant.
// Used for mathematical operations and constants in angle reduction.
#include <cmath>

// Define M_PID as a high-precision approximation of π if not already defined.
// Used consistently across angle reduction functions for accuracy.
#ifndef M_PID
#define M_PID 3.14159265358979323846
#endif

// Structure for storing reduced angles and their corresponding signs.
// Used to return both the reduced angle and its sign adjustment from reduction functions.
struct Vec2 {
    __m256d red;  // Reduced angle in radians, stored as a 256-bit vector of four doubles
    __m256d sign; // Sign adjustment (1.0 or -1.0) for each reduced angle, stored as a 256-bit vector
};

// Inline function to compute the absolute value of a 256-bit vector of doubles.
// Used as a utility in angle reduction to handle sign adjustments.
inline __m256d abs256(__m256d x);

// Function to reduce angles for sine computation using SIMD instructions.
// Takes a 256-bit vector of angles and returns a Vec2 with reduced angles and signs.
Vec2 reduce_angle_sin_SIMD(__m256d ang);

// Function to reduce angles for cosine computation using SIMD instructions.
// Takes a 256-bit vector of angles and returns a Vec2 with reduced angles and signs.
Vec2 reduce_angle_cos_SIMD(__m256d ang);

// Function to reduce angles for tangent computation using SIMD instructions.
// Takes a 256-bit vector of angles and returns a Vec2 with reduced angles and signs.
Vec2 reduce_angle_tan_SIMD(__m256d ang);

// End of header guard, closing the conditional compilation block.
#endif  // ANGLE_REDUCTION_H