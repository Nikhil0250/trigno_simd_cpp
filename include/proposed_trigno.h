// Header guard to prevent multiple inclusions of this header file during compilation.
// Ensures definitions are only included once in the final executable.
#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

// Include Intel Intrinsics header for SIMD operations.
// Provides AVX instructions for vectorized computations using __m256d type.
#include<immintrin.h>

// Include custom header file for angle reduction functions.
// Contains utilities for normalizing angles used in trigonometric calculations.
#include"angle_reduction.h"

// Structure to hold parameters for the sine helper function.
// These parameters (a, b, x, y, z) are coefficients for a polynomial approximation of sine.
struct SinHelperParams {
    double a;  // Coefficient for linear term in numerator
    double b;  // Constant term in numerator
    double x;  // Coefficient for quadratic term in denominator
    double y;  // Coefficient for linear term in denominator
    double z;  // Constant term in denominator
};

// Structure to hold parameters for the cosine helper function.
// These parameters (c, d, x, y, z) are coefficients for a polynomial approximation of cosine.
struct CosHelperParams {
    double c;  // Coefficient for linear term in numerator
    double d;  // Constant term in numerator
    double x;  // Coefficient for quadratic term in denominator
    double y;  // Coefficient for linear term in denominator
    double z;  // Constant term in denominator
};

// Structure to hold parameters for the tangent helper function.
// Contains coefficients (a, b, c, d) for a rational approximation of tangent.
struct TanHelperParams {
    double a;  // Coefficient for linear term in numerator
    double b;  // Constant term in numerator
    double c;  // Coefficient for linear term in denominator
    double d;  // Constant term in denominator
};

// External declaration of arrays holding precomputed parameters for trigonometric functions.
// Each array has 7 elements, defined in the corresponding .cpp file for different angle ranges.
extern SinHelperParams sinparams[7];  // Array of sine parameters for various angle ranges
extern CosHelperParams cosparams[7];  // Array of cosine parameters for various angle ranges
extern TanHelperParams tanparams[7];  // Array of tangent parameters for various angle ranges

// Inline function to compute a fast inverse square root using SIMD instructions.
// Takes a 256-bit vector of doubles and returns the inverse square root of each element.
inline __m256d fastInverseSqrt_SIMD(__m256d number);

// Inline SIMD helper function for sine computation.
// Uses coefficients (a, b) and denominator terms (x, y, z) with the angle to approximate sine.
inline __m256d sin_helper_SIMD(__m256d a, __m256d b, __m256d x, __m256d y, __m256d z, __m256d ang);

// Inline SIMD helper function for cosine computation.
// Uses coefficients (c, d) and denominator terms (x, y, z) with the angle to approximate cosine.
inline __m256d cos_helper_SIMD(__m256d c, __m256d d, __m256d x, __m256d y, __m256d z, __m256d ang);

// Inline SIMD helper function for tangent computation.
// Uses numerator coefficients (a, b) and denominator coefficients (c, d) with the angle for tangent.
inline __m256d tan_helper_SIMD(__m256d a, __m256d b, __m256d c, __m256d d, __m256d ang);

// Public interface function to compute the sine of an angle.
// Takes a double-precision angle (in radians) and returns its sine value.
double proposed_sin(double ang);

// Public interface function to compute the cosine of an angle.
// Takes a double-precision angle (in radians) and returns its cosine value.
double proposed_cos(double ang);

// Public interface function to compute the tangent of an angle.
// Takes a double-precision angle (in radians) and returns its tangent value.
double proposed_tan(double ang);

// End of header guard, closing the conditional compilation block.
#endif  // MYFUNCTIONS_H