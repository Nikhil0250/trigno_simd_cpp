// Header guard to prevent multiple inclusions of this header file during compilation.
// Ensures definitions are only included once in the final executable.
#ifndef TAYLORSIMD_H
#define TAYLORSIMD_H

// Include Intel Intrinsics header for SIMD operations.
// Provides AVX instructions for vectorized computations using __m256d type.
#include <immintrin.h>

// Include custom header file for angle reduction functions.
// Contains utilities for normalizing angles used in trigonometric calculations.
#include "angle_reduction.h"

// Function to compute sine using a 5-term Taylor series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of sine approximations.
__m256d taylorSin_SIMD_5(__m256d x_vec);

// Function to compute cosine using a 5-term Taylor series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of cosine approximations.
__m256d taylorCos_SIMD_5(__m256d x_vec);

// Function to compute tangent using a 5-term Maclaurin series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of tangent approximations.
__m256d taylorTan_SIMD_5(__m256d x_vec);

// Function to compute sine using a 7-term Taylor series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of sine approximations.
__m256d taylorSin_SIMD_7(__m256d x_vec);

// Function to compute cosine using a 7-term Taylor series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of cosine approximations.
__m256d taylorCos_SIMD_7(__m256d x_vec);

// Function to compute tangent using a 7-term Maclaurin series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of tangent approximations.
__m256d taylorTan_SIMD_7(__m256d x_vec);

// Function to compute sine using a 9-term Taylor series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of sine approximations.
__m256d taylorSin_SIMD_9(__m256d x_vec);

// Function to compute cosine using a 9-term Taylor series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of cosine approximations.
__m256d taylorCos_SIMD_9(__m256d x_vec);

// Function to compute tangent using a 9-term Maclaurin series with SIMD instructions.
// Takes a 256-bit vector of angles and returns a vector of tangent approximations.
__m256d taylorTan_SIMD_9(__m256d x_vec);

// Scalar wrapper function to compute sine using a 5-term Taylor series.
// Takes a double-precision angle in radians and returns its sine value.
double taylorSin5(double x);

// Scalar wrapper function to compute cosine using a 5-term Taylor series.
// Takes a double-precision angle in radians and returns its cosine value.
double taylorCos5(double x);

// Scalar wrapper function to compute tangent using a 5-term Maclaurin series.
// Takes a double-precision angle in radians and returns its tangent value.
double maclaurinTan5(double x);

// Scalar wrapper function to compute sine using a 7-term Taylor series.
// Takes a double-precision angle in radians and returns its sine value.
double taylorSin7(double x);

// Scalar wrapper function to compute cosine using a 7-term Taylor series.
// Takes a double-precision angle in radians and returns its cosine value.
double taylorCos7(double x);

// Scalar wrapper function to compute tangent using a 7-term Maclaurin series.
// Takes a double-precision angle in radians and returns its tangent value.
double maclaurinTan7(double x);

// Scalar wrapper function to compute sine using a 9-term Taylor series.
// Takes a double-precision angle in radians and returns its sine value.
double taylorSin9(double x);

// Scalar wrapper function to compute cosine using a 9-term Taylor series.
// Takes a double-precision angle in radians and returns its cosine value.
double taylorCos9(double x);

// Scalar wrapper function to compute tangent using a 9-term Maclaurin series.
// Takes a double-precision angle in radians and returns its tangent value.
double maclaurinTan9(double x);

// End of header guard, closing the conditional compilation block.
#endif  // TAYLORSIMD_H