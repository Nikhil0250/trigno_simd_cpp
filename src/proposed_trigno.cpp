// Include the corresponding header file for function declarations and type definitions.
// Ensures consistency between function definitions and their declarations.
#include "proposed_trigno.h"

// Include limits header for handling special cases like infinity in tangent computation.
#include<limits>

// Include Intel Intrinsics header for SIMD operations.
// Provides AVX instructions for vectorized computations (redundant but included for clarity).
#include <immintrin.h>

// Array of precomputed parameters for sine approximation.
// Each element corresponds to a specific angle range, with the last being fallback for out-of-bounds cases.
SinHelperParams sinparams[7] = {
    {630.2535746, 0.0, 400502.374645736, -72421.86530064, 399424.0},      // Parameters for range [0, 0.261)
    {572.9577951, 10.0, 380805.536587892, -289687.4612814, 431749.0},     // Parameters for range [0.261, 0.522)
    {343.7746771, 46.0, 200251.187385321, -278342.89691, 294797.0},       // Parameters for range [0.522, 0.783)
    {572.9577951, 217.0, 883074.908162424, -1616428.5317114, 1614593.0},  // Parameters for range [0.783, 1.044)
    {229.1831181, 297.0, 380805.536587892, -906648.4149306, 916309.0},    // Parameters for range [1.044, 1.305)
    {57.29577951, 542.0, 400502.374645736, -1185793.45271916, 1273864.0}, // Parameters for range [1.305, RANGE_MAX)
    {57.29577951, 542.0, 400502.374645736, -1185793.45271916, 1273864.0}  // Out-of-bounds handling
};

// Array of precomputed parameters for cosine approximation.
// Each element corresponds to a specific angle range, with the last being fallback for out-of-bounds cases.
CosHelperParams cosparams[7] = {
    {-57.29577951, 632.0, 400502.374645736, -72421.86530064, 399424.0},      // Parameters for range [0, 0.261)
    {-229.1831181, 657.0, 380805.536587892, -289687.4612814, 431749.0},      // Parameters for range [0.261, 0.522)
    {-286.4788976, 541.0, 200251.187385321, -278342.89691, 294797.0},        // Parameters for range [0.522, 0.783)
    {-744.8451337, 1252.0, 883074.908162424, -1616428.5317114, 1614593.0},   // Parameters for range [0.783, 1.044)
    {-572.9577951, 910.0, 380805.536587892, -906648.4149306, 916309.0},      // Parameters for range [1.044, 1.305)
    {-630.2535746, 990.0, 400502.374645736, -1185793.45271916, 1273864.0},   // Parameters for range [1.305, RANGE_MAX)
    {-630.2535746, 990.0, 400502.374645736, -1185793.45271916, 1273864.0}    // Out-of-bounds handling
};

// Array of precomputed parameters for tangent approximation.
// Each element corresponds to a specific angle range, with the last being fallback for out-of-bounds cases.
TanHelperParams tanparams[7] = {
    {630.2535746, 0.0, -57.29577951, 632.0},      // Parameters for range [0, 0.261)
    {572.9577951, 10.0, -229.1831181, 657.0},     // Parameters for range [0.261, 0.522)
    {343.7746771, 46.0, -286.4788976, 541.0},     // Parameters for range [0.522, 0.783)
    {572.9577951, 217.0, -744.8451337, 1252.0},   // Parameters for range [0.783, 1.044)
    {229.1831181, 297.0, -572.9577951, 910.0},    // Parameters for range [1.044, 1.305)
    {57.29577951, 542.0, -630.2535746, 990.0},    // Parameters for range [1.305, RANGE_MAX)
    {57.29577951, 542.0, -630.2535746, 990.0}     // Out-of-bounds handling
};

//=============================================================================
// SIMD Helpers
//=============================================================================

// Inline function to compute a fast inverse square root using SIMD instructions.
// Implements the "magic number" method with one Newton-Raphson iteration for refinement.
inline __m256d fastInverseSqrt_SIMD(__m256d number) {
    const __m256d threeHalfs = _mm256_set1_pd(1.5);  // Constant vector with 1.5 for Newton-Raphson
    __m256d x2 = _mm256_mul_pd(number, _mm256_set1_pd(0.5));  // Half of the input for iteration
    
    // Reinterpret the double vector as 64-bit integers for initial approximation
    __m256i i = _mm256_castpd_si256(number);
    i = _mm256_sub_epi64(_mm256_set1_epi64x(0x5fe6eb50c7b537a9), _mm256_srli_epi64(i, 1));  // Magic number subtraction
    __m256d y = _mm256_castsi256_pd(i);  // Reinterpret back to double
    
    // One iteration of Newton-Raphson to refine the approximation
    y = _mm256_mul_pd(y, _mm256_sub_pd(threeHalfs,
          _mm256_mul_pd(x2, _mm256_mul_pd(y, y))));
    return y;  // Return the refined inverse square root
}

//=============================================================================
// Fused Multiply-Add based helper functions
//=============================================================================

// Inline SIMD helper function for sine computation.
// Computes (a*ang + b) * fastInverseSqrt(x*ang^2 + y*ang + z) for polynomial approximation.
inline __m256d sin_helper_SIMD(__m256d a, __m256d b,
    __m256d x, __m256d y, __m256d z,
    __m256d ang) {
    __m256d poly  = _mm256_fmadd_pd(a, ang, b);  // Compute numerator: a*ang + b
    __m256d ang2  = _mm256_mul_pd(ang, ang);     // Compute ang^2
    __m256d inner = _mm256_fmadd_pd(x, ang2, _mm256_fmadd_pd(y, ang, z));  // Compute denominator: x*ang^2 + y*ang + z
    __m256d invSqrt = fastInverseSqrt_SIMD(inner);  // Compute inverse square root of denominator
    return _mm256_mul_pd(poly, invSqrt);  // Return numerator * inverse square root
}

// Inline SIMD helper function for cosine computation.
// Computes (c*ang + d) * fastInverseSqrt(x*ang^2 + y*ang + z) for polynomial approximation.
inline __m256d cos_helper_SIMD(__m256d c, __m256d d,
    __m256d x, __m256d y, __m256d z,
    __m256d ang) {
    __m256d poly  = _mm256_fmadd_pd(c, ang, d);  // Compute numerator: c*ang + d
    __m256d ang2  = _mm256_mul_pd(ang, ang);     // Compute ang^2
    __m256d inner = _mm256_fmadd_pd(x, ang2, _mm256_fmadd_pd(y, ang, z));  // Compute denominator: x*ang^2 + y*ang + z
    __m256d invSqrt = fastInverseSqrt_SIMD(inner);  // Compute inverse square root of denominator
    return _mm256_mul_pd(poly, invSqrt);  // Return numerator * inverse square root
}

// Inline SIMD helper function for tangent computation.
// Computes ((a*ang) + b) * [fastInverseSqrt(c*ang + d)]^2 for rational approximation.
inline __m256d tan_helper_SIMD(__m256d a, __m256d b,
    __m256d c, __m256d d,
    __m256d ang) {
    __m256d poly = _mm256_fmadd_pd(a, ang, b);  // Compute numerator: a*ang + b
    __m256d tmp  = fastInverseSqrt_SIMD(_mm256_fmadd_pd(c, ang, d));  // Compute inverse square root of c*ang + d
    return _mm256_mul_pd(poly, _mm256_mul_pd(tmp, tmp));  // Return numerator * (inverse square root)^2
}

//=============================================================================
// Exposed Functions (scalar API that internally uses SIMD)
//=============================================================================

// Function to compute the sine of an angle using a SIMD-based approximation.
// Takes a scalar angle in radians and returns its sine value.
double proposed_sin(double ang) {
    __m256d ang_vec = _mm256_set1_pd(ang);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);  // Reduce angle using SIMD sine reduction
    double red_scalar;  // Scalar variable to hold the reduced angle
    _mm256_storeu_pd(&red_scalar, reduced.red);  // Extract reduced angle (only first element used)
    int idx = static_cast<int>(red_scalar / 0.261);  // Compute parameter index based on interval width 0.261
    SinHelperParams p = sinparams[idx];  // Select parameters for the computed index
    
    // Broadcast parameters to 256-bit vectors
    __m256d a = _mm256_set1_pd(p.a);
    __m256d b = _mm256_set1_pd(p.b);
    __m256d x = _mm256_set1_pd(p.x);
    __m256d y = _mm256_set1_pd(p.y);
    __m256d z = _mm256_set1_pd(p.z);
    
    __m256d res_vec = sin_helper_SIMD(a, b, x, y, z, reduced.red);  // Compute sine using SIMD helper
    res_vec = _mm256_mul_pd(res_vec, reduced.sign);  // Apply sign adjustment from angle reduction
    
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, res_vec);  // Extract result (only first element used)
    return result;  // Return the computed sine value
}

// Function to compute the cosine of an angle using a SIMD-based approximation.
// Takes a scalar angle in radians and returns its cosine value.
double proposed_cos(double ang) {
    __m256d ang_vec = _mm256_set1_pd(ang);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);  // Reduce angle using SIMD cosine reduction
    double red_scalar;  // Scalar variable to hold the reduced angle
    _mm256_storeu_pd(&red_scalar, reduced.red);  // Extract reduced angle (only first element used)
    int idx = static_cast<int>(red_scalar / 0.261);  // Compute parameter index based on interval width 0.261
    CosHelperParams p = cosparams[idx];  // Select parameters for the computed index
    
    // Broadcast parameters to 256-bit vectors
    __m256d c = _mm256_set1_pd(p.c);
    __m256d d = _mm256_set1_pd(p.d);
    __m256d x = _mm256_set1_pd(p.x);
    __m256d y = _mm256_set1_pd(p.y);
    __m256d z = _mm256_set1_pd(p.z);
    
    __m256d res_vec = cos_helper_SIMD(c, d, x, y, z, reduced.red);  // Compute cosine using SIMD helper
    res_vec = _mm256_mul_pd(res_vec, reduced.sign);  // Apply sign adjustment from angle reduction
    
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, res_vec);  // Extract result (only first element used)
    return result;  // Return the computed cosine value
}

// Function to compute the tangent of an angle using a SIMD-based approximation.
// Takes a scalar angle in radians and returns its tangent value.
double proposed_tan(double ang) {
    __m256d ang_vec = _mm256_set1_pd(ang);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);  // Reduce angle using SIMD tangent reduction
    double red_scalar;  // Scalar variable to hold the reduced angle
    _mm256_storeu_pd(&red_scalar, reduced.red);  // Extract reduced angle (only first element used)
    
    // Check if the reduced angle is near π/2 to handle singularity
    if (fabs(red_scalar - M_PID / 2.0) < 1e-3)
        return std::numeric_limits<double>::infinity();  // Return infinity for near-vertical asymptote
    
    int idx = static_cast<int>(red_scalar / 0.261);  // Compute parameter index based on interval width 0.261
    TanHelperParams p = tanparams[idx];  // Select parameters for the computed index
    
    // Broadcast parameters to 256-bit vectors
    __m256d a = _mm256_set1_pd(p.a);
    __m256d b = _mm256_set1_pd(p.b);
    __m256d c = _mm256_set1_pd(p.c);
    __m256d d = _mm256_set1_pd(p.d);
    
    __m256d res_vec = tan_helper_SIMD(a, b, c, d, reduced.red);  // Compute tangent using SIMD helper
    res_vec = _mm256_mul_pd(res_vec, reduced.sign);  // Apply sign adjustment from angle reduction
    
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, res_vec);  // Extract result (only first element used)
    return result;  // Return the computed tangent value
}